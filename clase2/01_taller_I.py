# %% [markdown]
# # Clase 2 · Taller I · Entrenar ABMIL
#
# La clase pasada promediamos los parches de cada lámina. Hoy dejamos que el modelo **aprenda cuánto pesa cada parche**
# (multiple instance learning con atención, Ilse et al. 2018) y lo comparamos con el promedio, en la misma partición.

# %%
from pathlib import Path
import time
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

DATA = Path("data/panda")
OUT = Path("output/clase2"); OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(8)

# %% [markdown]
# ## 1. Las bolsas: una lámina = un conjunto de parches

# %%
train = pd.read_csv(DATA / "train.csv")
LAMINA_C1 = train.sample(1, random_state=42).image_id.values[0]   # la lámina de la clase pasada: la guardamos para el Taller II

ids = sorted(p.stem for p in (DATA / "features_virchow2").glob("*.h5"))
meta = pd.DataFrame({"image_id": ids}).merge(train, on="image_id")
meta = meta[meta.image_id != LAMINA_C1].reset_index(drop=True)

def n_parches(i):
    with h5py.File(DATA / "features_virchow2" / f"{i}.h5") as f:
        return f["features"].shape[0]
meta["n_parches"] = [n_parches(i) for i in meta.image_id]
print(len(meta), "láminas")
meta.n_parches.describe()

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(meta.n_parches, bins=50, ax=axs[0]); axs[0].set_xlabel("Parches por lámina")
sns.countplot(meta, x="isup_grade", ax=axs[1]); axs[1].set_xlabel("ISUP")
plt.tight_layout(); plt.show()

# %% [markdown]
# Las bolsas tienen **tamaños distintos**: por eso se entrena de a una lámina (batch = 1).
#
# ## 2. Partición: entrenamiento 70 % · validación 15 % · prueba 15 % (estratificada por grado)
# En PANDA hay una biopsia por paciente, así que partir por lámina equivale a partir por paciente.

# %%
tr, resto = train_test_split(meta, test_size=0.3, stratify=meta.isup_grade, random_state=0)
va, te = train_test_split(resto, test_size=0.5, stratify=resto.isup_grade, random_state=0)
print(len(tr), len(va), len(te))
pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])[["image_id", "isup_grade", "split"]] \
    .to_csv(OUT / "particion.csv", index=False)

# %% [markdown]
# ## 3. Línea base de la clase pasada: promedio + regresión logística

# %%
def promedio(ids):
    X = []
    for i in ids:
        with h5py.File(DATA / "slide_features_mean-virchow2" / f"{i}.h5") as f:
            X.append(f["features"][:].ravel())
    return np.array(X)

base = Pipeline([("pca", PCA(0.9, random_state=0)), ("sc", StandardScaler()),
                 ("lr", LogisticRegression(C=0.1, max_iter=3000))])
base.fit(promedio(tr.image_id), tr.isup_grade)
p_base = base.predict_proba(promedio(te.image_id))
res = {"Promedio": dict(kappa=cohen_kappa_score(te.isup_grade, p_base.argmax(1), weights="quadratic"),
                        auc=roc_auc_score(te.isup_grade, p_base, multi_class="ovr"))}
res

# %% [markdown]
# ## 4. ABMIL en ~20 líneas
#
# Para cada parche *hᵢ*: puntaje → softmax → peso *aᵢ* (suman 1). La lámina es *z = Σ aᵢ hᵢ*, y encima va un clasificador.

# %%
class ABMIL(nn.Module):
    def __init__(self, d_in=2560, d=256, n_clases=6, dropout=0.25):
        super().__init__()
        self.proy = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(dropout))
        self.V = nn.Linear(d, 128)      # "gated attention": tanh(V h) * sigmoid(U h)
        self.U = nn.Linear(d, 128)
        self.w = nn.Linear(128, 1)
        self.clasificador = nn.Linear(d, n_clases)

    def forward(self, h):                       # h: (N parches, 2560)
        h = self.proy(h)                        # (N, 256)
        puntaje = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))   # (N, 1)
        a = torch.softmax(puntaje, dim=0)       # pesos que suman 1
        z = (a * h).sum(dim=0)                  # vector de la lámina
        return self.clasificador(z), a.squeeze(1)

modelo = ABMIL()
print(sum(p.numel() for p in modelo.parameters()) / 1e6, "M parámetros")

# %% [markdown]
# ## 5. Entrenar
# Las features no caben todas en memoria (~11 GB): se leen de disco lámina por lámina.
# Al entrenar se toma una muestra de hasta 1.024 parches por lámina (más rápido y actúa como aumentación).

# %%
class Bolsas(torch.utils.data.Dataset):
    def __init__(self, df, max_parches=None):
        self.ids, self.y, self.max = df.image_id.values, df.isup_grade.values, max_parches
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, k):
        with h5py.File(DATA / "features_virchow2" / f"{self.ids[k]}.h5") as f:
            h = torch.from_numpy(f["features"][:]).float()
        if self.max and len(h) > self.max:
            h = h[torch.randperm(len(h))[:self.max]]
        return h, self.y[k]

def cargador(df, max_parches=None, shuffle=False):
    return torch.utils.data.DataLoader(Bolsas(df, max_parches), batch_size=None, shuffle=shuffle, num_workers=4)

@torch.no_grad()
def predecir(modelo, df):
    modelo.eval()
    return np.array([torch.softmax(modelo(h)[0], 0).numpy() for h, _ in cargador(df)])

opt = torch.optim.AdamW(modelo.parameters(), lr=1e-4, weight_decay=1e-4)
perdida = nn.CrossEntropyLoss()
historia, mejor = [], -1
for epoca in range(15):
    t = time.time(); modelo.train(); total = 0
    for h, y in cargador(tr, max_parches=1024, shuffle=True):
        logits, _ = modelo(h)
        l = perdida(logits[None], torch.tensor([y]))
        opt.zero_grad(); l.backward(); opt.step(); total += l.item()
    p_va = predecir(modelo, va)
    kappa = cohen_kappa_score(va.isup_grade, p_va.argmax(1), weights="quadratic")
    historia.append(dict(epoca=epoca, perdida=total / len(tr), kappa_val=kappa))
    if kappa > mejor:
        mejor = kappa; torch.save(modelo.state_dict(), OUT / "abmil.pt")
    print(f"época {epoca:2d} · pérdida {total / len(tr):.3f} · kappa val {kappa:.3f} · {time.time() - t:.0f} s")

# %%
h_ = pd.DataFrame(historia)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(h_.epoca, h_.perdida, marker="o"); ax[0].set_title("Pérdida de entrenamiento")
ax[1].plot(h_.epoca, h_.kappa_val, marker="o", color="C1"); ax[1].set_title("Kappa en validación")
plt.show()

# %% [markdown]
# ## 6. Comparar en prueba

# %%
modelo.load_state_dict(torch.load(OUT / "abmil.pt"))
p_abmil = predecir(modelo, te)
res["ABMIL"] = dict(kappa=cohen_kappa_score(te.isup_grade, p_abmil.argmax(1), weights="quadratic"),
                    auc=roc_auc_score(te.isup_grade, p_abmil, multi_class="ovr"))
pd.DataFrame(res).T.round(3)

# %%
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
r = pd.DataFrame(res).T
r.plot.bar(ax=axs[0], rot=0, color=["#4F9DA6", "#D9692E"]); axs[0].set_ylim(0, 1); axs[0].set_title("Prueba")
for ax, (nombre, p) in zip(axs[1:], [("Promedio", p_base), ("ABMIL", p_abmil)]):
    ConfusionMatrixDisplay(confusion_matrix(te.isup_grade, p.argmax(1))).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(nombre); ax.set_xlabel("ISUP predicho"); ax.set_ylabel("ISUP real")
plt.tight_layout(); plt.savefig(OUT / "resultado_c2_t1.jpg", dpi=150); plt.show()

# %% [markdown]
# ¿Mejora ABMIL? ¿En qué grados? Recordar: con features de un buen modelo fundacional, el promedio ya es una línea base fuerte.
#
# ## Opcional
# - Cambiar la atención por un máximo (`z = h.max(0).values`). ¿Qué pasa?
# - Entrenar con la mitad de `tr`. ¿Cuál de los dos métodos sufre más?
