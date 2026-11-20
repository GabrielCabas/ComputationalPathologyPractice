# %% [markdown]
# # Clase 2 · Taller II · ¿Dónde miró el modelo?
#
# Usamos el ABMIL del Taller I sobre la biopsia de la clase pasada (que el modelo nunca vio al entrenar)
# y pintamos los pesos de atención sobre la lámina.

# %%
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from trident import OpenSlideWSI

DATA = Path("data/panda")
OUT = Path("output/clase2")

class ABMIL(nn.Module):   # igual que en el Taller I
    def __init__(self, d_in=2560, d=256, n_clases=6, dropout=0.25):
        super().__init__()
        self.proy = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(dropout))
        self.V = nn.Linear(d, 128); self.U = nn.Linear(d, 128); self.w = nn.Linear(128, 1)
        self.clasificador = nn.Linear(d, n_clases)
    def forward(self, h):
        h = self.proy(h)
        a = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), dim=0)
        return self.clasificador((a * h).sum(dim=0)), a.squeeze(1)

modelo = ABMIL()
modelo.load_state_dict(torch.load(OUT / "abmil.pt")); modelo.eval()

# %% [markdown]
# ## 1. La lámina de la clase pasada

# %%
train = pd.read_csv(DATA / "train.csv")
fila = train.sample(1, random_state=42)
LAMINA = fila.image_id.values[0]
display(fila)

SLIDE = DATA / "train_images" / f"{LAMINA}.tiff"
if not SLIDE.exists():                                    # copia incluida en el repositorio
    SLIDE = Path("../muestras") / f"{LAMINA}.tiff"
slide = OpenSlideWSI(slide_path=str(SLIDE), lazy_init=False)
with h5py.File(f"output/trident/20x_224px_0px_overlap/features_virchow2/{LAMINA}.h5") as f:
    feats, coords = f["features"][:], f["coords"][:]
with h5py.File(f"output/trident/20x_224px_0px_overlap/patches/{LAMINA}_patches.h5") as f:
    LADO0 = int(f["coords"].attrs["patch_size_level0"])
print(feats.shape)

# %% [markdown]
# ## 2. Predicción

# %%
with torch.no_grad():
    logits, a = modelo(torch.from_numpy(feats).float())
prob = torch.softmax(logits, 0).numpy(); a = a.numpy()
print("ISUP real:", fila.isup_grade.values[0], "· predicho:", prob.argmax())
plt.bar(range(6), prob); plt.xlabel("ISUP"); plt.ylabel("Probabilidad"); plt.show()

# %% [markdown]
# ## 3. Heatmap de atención
# Los pesos suman 1; para pintarlos usamos su **rango** (percentil), como hacen CLAM y Trident.

# %%
rango = a.argsort().argsort() / (len(a) - 1)
thumb = slide.get_thumbnail((1024, 1024))
esc = thumb.size[0] / slide.width

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs[0].imshow(thumb); axs[0].set_title("Lámina"); axs[0].axis("off")
axs[1].imshow(thumb)
lado = LADO0 * esc
for (x, y), v in zip(coords, rango):
    axs[1].add_patch(plt.Rectangle((x * esc, y * esc), lado, lado, color=plt.cm.coolwarm(v), alpha=0.6, lw=0))
axs[1].set_title("Atención (rojo = más peso)"); axs[1].axis("off")
plt.tight_layout(); plt.savefig(OUT / "heatmap_c2.jpg", dpi=150); plt.show()

# %% [markdown]
# ## 4. Los parches más y menos atendidos

# %%
def mostrar(idx, titulo):
    fig, axs = plt.subplots(1, len(idx), figsize=(2 * len(idx), 2.4))
    for ax, i in zip(axs, idx):
        x, y = coords[i]
        ax.imshow(slide.read_region((int(x), int(y)), 0, (LADO0, LADO0)).convert("RGB")); ax.axis("off")
    plt.suptitle(titulo); plt.tight_layout(); plt.show()

orden = np.argsort(a)
mostrar(orden[::-1][:10], "Más atendidos")
mostrar(orden[:10], "Menos atendidos")

# %% [markdown]
# ## 5. ¿Coincide con los grupos de k-means de la clase pasada?

# %%
cluster = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(feats)
pd.DataFrame({"cluster": cluster, "atencion": rango}).groupby("cluster").atencion.agg(["count", "mean"]).round(2)

# %% [markdown]
# ¿Qué grupo concentra la atención? ¿Qué tejido era ese grupo en el Taller I de la clase pasada?
#
# ## Opcional: una lámina benigna (ISUP 0)
# Descargar una lámina ISUP 0 del conjunto de prueba (`output/clase2/particion.csv`) con
# `kaggle competitions download -c prostate-cancer-grade-assessment -f train_images/<id>.tiff -p data/panda/train_images`,
# y usar sus features de `data/panda/features_virchow2/<id>.h5`. ¿Dónde mira ahora el modelo?
