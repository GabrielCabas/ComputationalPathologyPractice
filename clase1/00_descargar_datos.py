# %% [markdown]
# # 00 · Descarga de datos
#
# Usamos **PANDA** (Prostate cANcer graDe Assessment, Kaggle 2020): ~10.600 biopsias de próstata (H&E) de
# Radboud (Países Bajos) y Karolinska (Suecia), cada una con su **grado ISUP** (0 = benigno · 1–5 = cáncer, de menor a mayor gravedad).
#
# Se descarga:
# 1. `train.csv`: etiquetas de cada lámina (competencia PANDA).
# 2. **Una lámina completa** (.tiff) para el Taller I. Viene incluida en el repositorio (`muestras/`); solo se descarga si falta.
# 3. **Features Virchow2 promediadas por lámina**, ya calculadas con Trident en el HPC de la UOH
#    (dataset de Kaggle `gabrielcabas/panda-trident-preprocessed`, ~1,7 GB). Se usan en el Taller II.
#
# Requisitos (una sola vez):
# - Cuenta en Kaggle e iniciar sesión: `kaggle auth login` en la terminal (o token de kaggle.com → Settings → API →
#   *Generate New Token*, guardado en `~/.kaggle/access_token`).
# - Aceptar las reglas de la competencia: https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/rules

# %%
import shutil, subprocess, zipfile
from pathlib import Path
import pandas as pd

DATA = Path("data/panda")
DATA.mkdir(parents=True, exist_ok=True)
COMP = "prostate-cancer-grade-assessment"

def kaggle(*args):
    print("$ kaggle", " ".join(args))
    subprocess.run(["kaggle", *args], check=True)

def descomprimir(carpeta):
    for z in Path(carpeta).glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(carpeta)
        z.unlink()

# %% [markdown]
# ## 1. Etiquetas

# %%
if not (DATA / "train.csv").exists():
    kaggle("competitions", "download", "-c", COMP, "-f", "train.csv", "-p", str(DATA))
    descomprimir(DATA)
train = pd.read_csv(DATA / "train.csv")
train.head()

# %% [markdown]
# ## 2. Lámina del Taller I
# La misma del notebook de Kaggle: `train.sample(1, random_state=42)`.

# %%
LAMINA = train.sample(1, random_state=42).image_id.values[0]
destino = DATA / "train_images"
destino.mkdir(parents=True, exist_ok=True)
en_repo = Path("../muestras") / f"{LAMINA}.tiff"      # el repositorio ya trae esta lámina
if not (destino / f"{LAMINA}.tiff").exists():
    if en_repo.exists():
        shutil.copy(en_repo, destino)
    else:
        kaggle("competitions", "download", "-c", COMP, "-f", f"train_images/{LAMINA}.tiff", "-p", str(destino))
        descomprimir(destino)
print(LAMINA, sorted(p.name for p in destino.iterdir()))

# %% [markdown]
# ## 3. Features de todas las láminas (Taller II)

# %%
if not (DATA / "slide_features_mean-virchow2").exists():
    kaggle("datasets", "download", "gabrielcabas/panda-trident-preprocessed", "-p", str(DATA), "--unzip")
print(len(list((DATA / "slide_features_mean-virchow2").glob("*.h5"))), "láminas con features")
