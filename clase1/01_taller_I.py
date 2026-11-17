# %% [markdown]
# # Taller I · De la lámina a las features con Trident
#
# Objetivo: recorrer el pipeline estándar sobre **una biopsia de próstata** de PANDA:
# abrir la WSI → segmentar el tejido → generar parches → extraer embeddings con un modelo fundacional (**Virchow2**)
# → explorar qué "vio" el modelo, sin usar ninguna etiqueta.
#
# Virchow2 (Paige) tiene acceso controlado: pedir permiso en https://huggingface.co/paige-ai/Virchow2 y crear un token de lectura
# en https://huggingface.co/settings/tokens. El inicio de sesión se hace en la celda siguiente.

# %%
import time
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import seaborn as sns
from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory

DATA = Path("data/panda")
JOB = Path("output/trident")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# %% [markdown]
# ## 0. Iniciar sesión en Hugging Face
# Si no hay sesión, aparece un cuadro: pega tu token (empieza con `hf_`) y presiona **Login**. Se guarda en el computador,
# así que basta hacerlo una vez. **No escribas el token en el código**: el notebook se comparte.

# %%
from huggingface_hub import notebook_login, whoami
try:
    print("Sesión iniciada como", whoami()["name"])
except Exception:
    notebook_login()

# %% [markdown]
# ## 1. El dataset
# Usaremos `image_id` e `isup_grade`. El puntaje de Gleason es la otra forma de expresar lo mismo, así que no lo usamos.

# %%
train = pd.read_csv(DATA / "train.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(train, x="data_provider", ax=axs[0])
sns.countplot(train, x="isup_grade", ax=axs[1])
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Abrir una lámina

# %%
muestra = train.sample(1, random_state=42)
display(muestra)
LAMINA = muestra.image_id.values[0]
SLIDE = DATA / "train_images" / f"{LAMINA}.tiff"

slide = OpenSlideWSI(slide_path=str(SLIDE), lazy_init=False)
print("Dimensiones (nivel 0):", slide.width, "x", slide.height, "px")
print("µm por píxel:", slide.mpp, "· magnificación:", slide.mag)
print("Niveles:", slide.level_count, slide.level_dimensions)

# %%
thumb = slide.get_thumbnail((1024, 1024))
plt.figure(figsize=(8, 6)); plt.imshow(thumb); plt.axis("off"); plt.title("Miniatura"); plt.show()

# %% [markdown]
# ## 3. Segmentar el tejido
# En Kaggle (GPU) usamos HEST. En CPU es lento, así que aquí usamos **GrandQC** (segundos). Para probar HEST: `"hest"`.

# %%
t = time.time()
seg_model = segmentation_model_factory("grandqc")
if DEVICE == "cpu":
    seg_model.precision = torch.float32   # Trident usa float16, que en CPU es muy lento
geojson = slide.segment_tissue(segmentation_model=seg_model, target_mag=seg_model.target_mag, job_dir=str(JOB), device=DEVICE)
print(f"Segmentación: {time.time() - t:.0f} s")
plt.figure(figsize=(8, 6)); plt.imshow(plt.imread(JOB / "contours" / f"{LAMINA}.jpg")); plt.axis("off"); plt.show()

# %% [markdown]
# ## 4. Parches de 224 × 224 px a 20x
# Virchow2 se entrenó con parches de 224 px a 20x: usamos la misma escala.

# %%
coords_path = slide.extract_tissue_coords(target_mag=20, patch_size=224, save_coords=str(JOB / "20x_224px_0px_overlap"))
with h5py.File(coords_path) as f:
    coords = f["coords"][:]
    attrs = dict(f["coords"].attrs)
LADO0 = int(attrs["patch_size_level0"])
print(len(coords), "parches · lado en el nivel 0:", LADO0, "px")

viz = slide.visualize_coords(coords_path=coords_path, save_patch_viz=str(JOB / "20x_224px_0px_overlap" / "visualization"))
plt.figure(figsize=(8, 6)); plt.imshow(plt.imread(viz)); plt.axis("off"); plt.show()

# %% [markdown]
# ## 5. Features con Virchow2 (ViT-H, 2560 dimensiones)
# Cada parche → un vector de 2560 números. **El modelo no se entrena: solo se usa.**
# En CPU tarda varios minutos (es un modelo de 632 M de parámetros): lanzarlo y seguir conversando.

# %%
encoder = encoder_factory("virchow2")
encoder.eval().to(DEVICE)
if DEVICE == "cpu":
    encoder.precision = torch.float32
t = time.time()
feats_path = slide.extract_patch_features(patch_encoder=encoder, coords_path=coords_path,
                                          save_features=str(JOB / "20x_224px_0px_overlap" / "features_virchow2"),
                                          device=DEVICE, batch_limit=32)
print(f"Features: {time.time() - t:.0f} s")

# %%
with h5py.File(feats_path) as f:
    print(list(f.keys()))
    features, coords = f["features"][:], f["coords"][:]
print("features:", features.shape, "· coords:", coords.shape)

# %% [markdown]
# ## 6. ¿Podemos resumir 2560 dimensiones? PCA
# PCA(0.8): las componentes necesarias para explicar el 80 % de la varianza.

# %%
from sklearn.decomposition import PCA
pca = PCA(0.8, random_state=0)
transformed = pd.DataFrame(pca.fit_transform(features))
print(transformed.shape[1], "componentes explican el 80 % de la varianza")
sns.pairplot(transformed[transformed.columns[:3]], plot_kws=dict(s=8)); plt.show()

# %%
var = pca.explained_variance_ratio_
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(var) + 1), var, label="por componente")
plt.plot(range(1, len(var) + 1), np.cumsum(var), color="C1", marker="o", ms=3, label="acumulada")
plt.xlabel("Componente"); plt.ylabel("Varianza explicada"); plt.legend(); plt.show()

# %% [markdown]
# ## 7. Agrupar los parches sin etiquetas (k-means, 4 grupos)

# %%
from sklearn.cluster import KMeans
K = 4
cluster = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(features)
transformed["cluster"] = cluster
sns.scatterplot(transformed, x=0, y=1, hue="cluster", palette="tab10", s=15)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.show()

# %%
# Los grupos pintados sobre la lámina
colores = plt.cm.tab10(np.arange(K))
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs[0].imshow(thumb); axs[0].axis("off"); axs[0].set_title("Lámina")
axs[1].scatter(coords[:, 0], coords[:, 1], c=colores[cluster], s=10, marker="s")
axs[1].set_xlim(0, slide.width); axs[1].set_ylim(slide.height, 0)
axs[1].set_aspect("equal"); axs[1].axis("off"); axs[1].set_title("Clusters sobre la lámina")
Path("output/clase1").mkdir(parents=True, exist_ok=True)
plt.savefig("output/clase1/kmeans.jpg", dpi=150, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 8. ¿Qué tejido es cada grupo?

# %%
def plot_patches_sample(df, title, n=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, (_, row) in zip(axes.flat, df.sample(min(n, len(df)), random_state=42).iterrows()):
        ax.imshow(slide.read_region((int(row.x), int(row.y)), 0, (LADO0, LADO0)).convert("RGB"))
    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle(title); plt.tight_layout(); plt.show()

parches = pd.DataFrame(coords, columns=["x", "y"]).assign(cluster=cluster)
for k in range(K):
    plot_patches_sample(parches[parches.cluster == k], f"Cluster {k} · {(cluster == k).sum()} parches")

# %% [markdown]
# ## 9. De parches a lámina: promediar
# Para el Taller II necesitamos **un vector por lámina**. Lo más simple: el promedio de los parches
# (en Trident: `--slide_encoder mean-virchow2`).

# %%
slide_vec = features.mean(axis=0)
print(slide_vec.shape)

# %% [markdown]
# ## Lo mismo, desde la terminal (así se procesaron las 10.600 láminas en el HPC)
# ```bash
# run_batch_of_slides --task all --wsi_dir data/panda/train_images --job_dir output/trident_batch \
#     --segmenter hest --patch_encoder virchow2 --mag 20 --patch_size 224 --gpu 0
# run_batch_of_slides --task feat --wsi_dir data/panda/train_images --job_dir output/trident_batch \
#     --slide_encoder mean-virchow2 --mag 20 --patch_size 224 --gpu 0
# ```
