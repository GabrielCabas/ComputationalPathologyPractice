# %% [markdown]
# # Taller II · Predecir el grado ISUP desde la H&E
#
# Ya procesamos una lámina. ¿Y el 99,99 % restante? En el HPC de la UOH preprocesamos las ~10.600 láminas de PANDA
# con Trident (Virchow2, 224 px a 20x). Como en un programa de cocina, traemos el plato ya preparado.
#
# El problema: cada lámina tiene **cientos o miles de parches** y **una sola etiqueta**. La forma más simple de obtener
# una matriz *X* (una fila por lámina) es **promediar** los parches de cada lámina. Después: *Xw = y*.

# %%
from pathlib import Path
import h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, cohen_kappa_score)

DATA = Path("data/panda")
train = pd.read_csv(DATA / "train.csv")

# %% [markdown]
# ## 1. Cargar las features
# Para no agotar la memoria usamos un **subconjunto** de láminas, con la misma proporción de grados y de centros
# que el total. Si tu computador tiene memoria de sobra, sube `N_LAMINAS` (o usa `None` para todas, ~10.600).

# %%
N_LAMINAS = 2000
N_JOBS = 2          # procesos en paralelo: cada uno copia los datos; más procesos = más memoria

carpeta = DATA / "slide_features_mean-virchow2"
disponibles = train[train.image_id.isin({p.stem for p in carpeta.glob("*.h5")})]
if N_LAMINAS and N_LAMINAS < len(disponibles):
    disponibles, _ = train_test_split(disponibles, train_size=N_LAMINAS, random_state=0,
                                      stratify=disponibles.isup_grade.astype(str) + disponibles.data_provider)

ids = sorted(disponibles.image_id)
X = np.empty((len(ids), 2560), dtype=np.float32)
for i, image_id in enumerate(ids):
    with h5py.File(carpeta / f"{image_id}.h5") as f:
        X[i] = f["features"][:].ravel()
meta = pd.DataFrame({"image_id": ids}).merge(train, on="image_id", how="left")
y = meta.isup_grade.values
print("X:", X.shape, "· y:", y.shape)
sns.countplot(x=y); plt.xlabel("ISUP"); plt.show()

# %% [markdown]
# ## 2. ¿Se ordenan las láminas por grado? PCA coloreado por ISUP

# %%
pcs = PCA(2, random_state=0).fit_transform(X)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=y, palette="viridis", s=6, ax=axs[0]); axs[0].set_title("Coloreado por ISUP")
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=meta.data_provider, s=6, ax=axs[1]); axs[1].set_title("Coloreado por centro")
plt.show()

# %% [markdown]
# Ojo con el panel derecho: si las láminas se separan por **centro** (Radboud vs Karolinska), el modelo podría aprender
# el hospital en vez de la biología (sesgo de sitio).

# %% [markdown]
# ## 3. Línea base: PCA + regresión logística, validación cruzada

# %%
pipe = Pipeline([("pca", PCA(0.9, random_state=0)), ("lr", LogisticRegression(max_iter=2000))])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, scoring="roc_auc_ovr", cv=cv, n_jobs=N_JOBS)
print(f"AUC (uno contra el resto): {scores.mean():.3f} ± {scores.std():.3f}")

# %% [markdown]
# ## 4. Separar entrenamiento y prueba, y optimizar hiperparámetros
# Optimizamos `l1_ratio` (0 = ridge, 1 = lasso) y `C` (fuerza de la regularización) con GridSearchCV,
# **solo con el conjunto de entrenamiento**.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

pipe = Pipeline([("pca", PCA(0.9, random_state=0)), ("sc", StandardScaler()),
                 ("lr", LogisticRegression(solver="saga", max_iter=3000))])
grid = GridSearchCV(pipe, {"lr__l1_ratio": [0, 0.5, 1], "lr__C": [0.01, 0.1, 1]},
                    scoring="roc_auc_ovr", cv=3, n_jobs=N_JOBS)
grid.fit(X_train, y_train)
print("Mejores parámetros:", grid.best_params_, f"· AUC CV: {grid.best_score_:.3f}")

# %% [markdown]
# ## 5. Evaluar en prueba

# %%
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Kappa cuadrático (métrica de PANDA): {cohen_kappa_score(y_test, y_pred, weights='quadratic'):.3f}")

# %%
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
plt.xlabel("ISUP predicho"); plt.ylabel("ISUP real"); plt.show()

# %% [markdown]
# Los errores, ¿están lejos de la diagonal o en grados vecinos? El kappa cuadrático castiga más los errores lejanos.

# %%
def plot_roc_curve(y_test, y_prob):
    clases = np.arange(y_prob.shape[1])
    y_bin = label_binarize(y_test, classes=clases)
    plt.figure(figsize=(7, 6))
    for i in clases:
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"ISUP {i} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Tasa de falsos positivos"); plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC uno contra el resto"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

plot_roc_curve(y_test, grid.predict_proba(X_test))

# %%
# Figura para las diapositivas: matriz de confusión + curvas ROC
Path("output/clase1").mkdir(parents=True, exist_ok=True)
y_prob = grid.predict_proba(X_test)
y_bin = label_binarize(y_test, classes=np.arange(6))
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=axs[0], cmap="Blues", colorbar=False)
axs[0].set_xlabel("ISUP predicho"); axs[0].set_ylabel("ISUP real")
for i in range(6):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    axs[1].plot(fpr, tpr, label=f"ISUP {i} (AUC = {auc(fpr, tpr):.2f})")
axs[1].plot([0, 1], [0, 1], "--", color="gray"); axs[1].legend(); axs[1].set_xlabel("FPR"); axs[1].set_ylabel("TPR")
plt.tight_layout(); plt.savefig("output/clase1/resultado_c1_t2.jpg", dpi=150); plt.show()

# %% [markdown]
# ## Opcional
# - Entrenar con un centro y evaluar en el otro (`meta.data_provider`). ¿Cuánto cae el desempeño?
# - Tarea binaria: benigno (ISUP 0) vs cáncer (ISUP ≥ 1). ¿Es más fácil?
