# %% [markdown]
# # Clase 2 · 00 · Descarga de datos
#
# Para entrenar ABMIL necesitamos las features **por parche** (no el promedio). Están en el dataset de Kaggle
# `gabrielcabas/panda-preprocessed`: ~1.550 biopsias de PANDA procesadas con Trident (Virchow2, 224 px a 20x), ~11 GB.
#
# Requisitos: haber corrido el notebook 00 de la Clase 1 (`train.csv`) y haber iniciado sesión en Kaggle (`kaggle auth login`).

# %%
import json, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

REF = "gabrielcabas/panda-preprocessed"
CARPETA = "20x_224px_0px_overlap/features_virchow2"
DEST = Path("data/panda/features_virchow2")
DEST.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()

# %%
# Lista de archivos del dataset (API pública, paginada)
def listar(ref):
    archivos, token = [], None
    while True:
        url = f"https://www.kaggle.com/api/v1/datasets/list/{ref}?pageSize=200" + (f"&pageToken={token}" if token else "")
        r = json.load(urllib.request.urlopen(url))
        archivos += [f["name"] for f in r.get("datasetFiles", [])]
        token = r.get("nextPageToken")
        if not token:
            return archivos

h5s = [a for a in listar(REF) if a.startswith(CARPETA + "/")]
print(len(h5s), "láminas con features por parche")

# %%
def bajar(nombre):
    destino = DEST / Path(nombre).name
    if destino.exists():
        return
    api.dataset_download_file(REF, nombre, path=str(DEST), quiet=True)
    z = DEST / (Path(nombre).name + ".zip")
    if z.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(DEST)
        z.unlink()

with ThreadPoolExecutor(8) as ex:
    list(ex.map(bajar, h5s))
print(len(list(DEST.glob("*.h5"))), "archivos en", DEST)
