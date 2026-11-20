# %% [markdown]
# # Clase 2 · 00 · Descarga de datos
#
# Para entrenar ABMIL necesitamos las features **por parche** (no el promedio). Están en el dataset de Kaggle
# `gabrielcabas/panda-preprocessed`: ~1.550 biopsias de PANDA procesadas con Trident (Virchow2, 224 px a 20x), ~11 GB.
#
# Usamos un **subconjunto equilibrado por grado ISUP** (`N_LAMINAS`). Los archivos se bajan **en paralelo pero con un
# ritmo máximo**: si Kaggle responde que son demasiadas solicitudes (error 429), todos los hilos esperan juntos y siguen.
# Si la descarga se interrumpe, basta con volver a correr la celda: salta los archivos que ya están completos.
#
# Requisitos: haber corrido el notebook 00 de la Clase 1 (`train.csv`) y haber iniciado sesión en Kaggle (`kaggle auth login`).

# %%
import json, random, shutil, tempfile, threading, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import h5py, pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

REF = "gabrielcabas/panda-preprocessed"
CARPETA = "20x_224px_0px_overlap/features_virchow2"
DATA = Path("data/panda")
DEST = DATA / "features_virchow2"            # aquí queda cada lámina como <image_id>.h5
DEST.mkdir(parents=True, exist_ok=True)

N_LAMINAS = 400        # láminas a usar (equilibradas por grado). None = todas (~1.550, ~11 GB)
HILOS = 4              # descargas simultáneas
POR_SEGUNDO = 2.0      # máximo de solicitudes nuevas por segundo (entre todos los hilos)
REINTENTOS = 8
PAUSA = 1.0            # pausa entre páginas al listar

api = KaggleApi()
api.authenticate()

# %% [markdown]
# ## 1. ¿Qué láminas hay en el dataset?

# %%
def listar(ref):
    """Lista los archivos del dataset (API pública, paginada, con reintentos)."""
    archivos, token = [], None
    while True:
        url = f"https://www.kaggle.com/api/v1/datasets/list/{ref}?pageSize=200" + (f"&pageToken={token}" if token else "")
        for intento in range(REINTENTOS):
            try:
                r = json.load(urllib.request.urlopen(url, timeout=60))
                break
            except Exception as e:
                espera = 5 * 2 ** intento
                print(f"Error al listar ({e}); reintento en {espera} s")
                time.sleep(espera)
        else:
            raise RuntimeError("No se pudo listar el dataset")
        archivos += [f["name"] for f in r.get("datasetFiles", [])]
        token = r.get("nextPageToken")
        if not token:
            return archivos
        time.sleep(PAUSA)

disponibles = {Path(a).stem for a in listar(REF) if a.startswith(CARPETA + "/") and a.endswith(".h5")}
print(len(disponibles), "láminas con features por parche")

# %% [markdown]
# ## 2. Elegir el subconjunto
# Siempre el mismo (semilla fija). La lámina de la Clase 1 se excluye: la usamos en el Taller II.

# %%
train = pd.read_csv(DATA / "train.csv")
LAMINA_C1 = train.sample(1, random_state=42).image_id.values[0]
meta = train[train.image_id.isin(disponibles) & (train.image_id != LAMINA_C1)]
if N_LAMINAS:
    por_grado = N_LAMINAS // meta.isup_grade.nunique()
    meta = pd.concat([d.sample(min(len(d), por_grado), random_state=0) for _, d in meta.groupby("isup_grade")])
ids = sorted(meta.image_id)
print(len(ids), "láminas ·", meta.isup_grade.value_counts().sort_index().to_dict())

# %% [markdown]
# ## 3. Descargar
#
# Cada archivo se baja a una carpeta temporal y, solo cuando está completo y se puede abrir, se mueve a
# `data/panda/features_virchow2/<image_id>.h5`. Así nunca queda un archivo a medias en el destino, y lo que ya está
# completo no se vuelve a pedir.

# %%
class Ritmo:
    """Limita las solicitudes por segundo entre todos los hilos y coordina una pausa común ante un 429."""
    def __init__(self, por_segundo):
        self.intervalo, self.siguiente, self.pausa_hasta = 1 / por_segundo, 0.0, 0.0
        self.lock = threading.Lock()
    def esperar_turno(self):
        with self.lock:
            ahora = time.monotonic()
            turno = max(ahora, self.siguiente, self.pausa_hasta)
            self.siguiente = turno + self.intervalo
        time.sleep(max(0.0, turno - time.monotonic()))
    def pausar(self, segundos):
        with self.lock:
            self.pausa_hasta = max(self.pausa_hasta, time.monotonic() + segundos)

ritmo = Ritmo(POR_SEGUNDO)

def ordenar_existentes():
    """Mueve a DEST/<id>.h5 los archivos que una versión anterior dejó en subcarpetas, y borra restos de descargas."""
    for h5 in DEST.rglob("*.h5"):
        if h5.parent != DEST and not (DEST / h5.name).exists():
            shutil.move(str(h5), DEST / h5.name)
    for sub in [p for p in DEST.iterdir() if p.is_dir()]:
        shutil.rmtree(sub)
    for resto in list(DEST.glob("*.zip")) + list(DEST.glob("*.tmp")):
        resto.unlink()

def completo(ruta):
    try:
        with h5py.File(ruta) as f:
            return f["features"].shape[0] > 0
    except Exception:
        return False

def bajar(image_id):
    destino = DEST / f"{image_id}.h5"
    if destino.exists() and completo(destino):
        return "ya estaba"
    for intento in range(REINTENTOS):
        ritmo.esperar_turno()
        with tempfile.TemporaryDirectory(dir=DEST.parent) as tmp:
            try:
                api.dataset_download_file(REF, f"{CARPETA}/{image_id}.h5", path=tmp, quiet=True)
                for z in Path(tmp).rglob("*.zip"):
                    with zipfile.ZipFile(z) as f:
                        f.extractall(tmp)
                bajado = next(Path(tmp).rglob(f"{image_id}.h5"), None)
                if bajado is None or not completo(bajado):
                    raise IOError("archivo incompleto")
                temporal = destino.with_name(f"{image_id}.{threading.get_ident()}.tmp")
                shutil.move(str(bajado), temporal)
                temporal.replace(destino)                  # se renombra recién al final
                return "bajado"
            except Exception as e:
                es_429 = "429" in str(e) or "Too Many Requests" in str(e)
                espera = (20 if es_429 else 5) * 2 ** min(intento, 4) + random.uniform(0, 3)
                if es_429:
                    ritmo.pausar(espera)                   # todos los hilos esperan
                print(f"  {image_id[:8]}…: {'429, demasiadas solicitudes' if es_429 else str(e)[:60]} · "
                      f"reintento {intento + 1}/{REINTENTOS} en {espera:.0f} s")
                if not es_429:
                    time.sleep(espera)
    return "falló"

ordenar_existentes()
pendientes = [i for i in ids if not ((DEST / f"{i}.h5").exists() and completo(DEST / f"{i}.h5"))]
print(len(ids) - len(pendientes), "ya estaban ·", len(pendientes), "por bajar")

t0, estado = time.time(), {}
with ThreadPoolExecutor(HILOS) as ex:
    futuros = {ex.submit(bajar, i): i for i in pendientes}
    for k, fut in enumerate(as_completed(futuros), 1):
        estado[futuros[fut]] = fut.result()
        if k % 25 == 0 or k == len(pendientes):
            print(f"{k}/{len(pendientes)} · {time.time() - t0:.0f} s · bajados: {list(estado.values()).count('bajado')}"
                  f" · fallaron: {list(estado.values()).count('falló')}")

fallaron = [i for i, e in estado.items() if e == "falló"]
listos = sum((DEST / f"{i}.h5").exists() for i in ids)
print(f"{listos}/{len(ids)} láminas listas en {DEST}", "· vuelve a correr esta celda para reintentar las que fallaron" if fallaron else "")
