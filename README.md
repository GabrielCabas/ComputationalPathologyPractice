# Práctico de Histopatología computacional

Magíster en Informática Médica.

Recorremos el pipeline estándar de patología computacional con **modelos fundacionales** usando
[Trident](https://github.com/mahmoodlab/TRIDENT) (Mahmood Lab): lámina completa (WSI) → segmentación del tejido →
parches → features con **Virchow2** → promedio por lámina → predicción del **grado ISUP** (promedio en la clase 1, ABMIL en la clase 2) en biopsias de próstata
([PANDA](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment)).

## Contenido

Dos clases, cada una con su carpeta. Los notebooks se corren **en orden**; los `.py` son el mismo contenido en texto.

| Archivo | Qué hace |
|---|---|
| `clase1/00_descargar_datos.ipynb` | `train.csv`, una lámina de PANDA y las features Virchow2 promediadas de ~10.600 láminas (~1,7 GB) |
| `clase1/01_taller_I.ipynb` | Abrir una WSI, segmentar, cortar tiles, extraer features con Virchow2, PCA y k-means |
| `clase1/02_taller_II.ipynb` | PCA + regresión logística sobre todas las láminas, GridSearchCV, métricas, matriz de confusión y curvas ROC |
| `clase2/00_descargar_datos.ipynb` | Features Virchow2 **por parche** de ~1.550 láminas (~11 GB, `gabrielcabas/panda-preprocessed`) |
| `clase2/01_taller_I.ipynb` | ABMIL en PyTorch vs promedio + regresión logística, misma partición |
| `clase2/02_taller_II.ipynb` | Heatmap de atención y parches más atendidos sobre la lámina de la clase 1 |
| `environment.yml` | Entorno de Python con Trident (CPU) |

## Antes de la clase

1. **Kaggle**: crear cuenta, aceptar las reglas de https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/rules
   e iniciar sesión con `kaggle auth login` (o generar un token en Settings → API → *Generate New Token* y guardarlo en
   `~/.kaggle/access_token`). Ojo: ese archivo es solo para el token de **Kaggle**, no el de Hugging Face.
2. **Hugging Face**: pedir acceso a https://huggingface.co/paige-ai/Virchow2, crear un token de lectura
   (https://huggingface.co/settings/tokens). Se registra en la primera celda del Taller I (o con `hf auth login` en la terminal).
   Solo lo usa el Taller I de la Clase 1 (extraer features con Virchow2).
3. Entorno:

```bash
micromamba create -f environment.yml -y      # o: conda env create -f environment.yml
micromamba activate patologia
hf auth login
jupyter lab
```

## Notas

- **Sin GPU funciona**, pero Trident usa float16 por defecto, que en CPU es muy lento: los notebooks fuerzan float32.
- La segmentación usa **GrandQC** en CPU (segundos). HEST es mejor pero tarda minutos sin GPU.
- Las features del Práctico II se extrajeron con Trident en el HPC de la UOH (`gabrielcabas/panda-trident-preprocessed`).
- También existe la versión en Kaggle con GPU: https://www.kaggle.com/code/gabrielcabas/cmm-summer-course-2026

## Lámina de prueba

`muestras/e8baa3bb9dcfb9cef5ca599d62bb8046.tiff` es la biopsia de PANDA que se usa en el Taller I de la Clase 1 y en el
Taller II de la Clase 2 (`train.sample(1, random_state=42)`). El notebook `clase1/00_descargar_datos` la copia a
`data/panda/train_images/`; si no estuviera, la descarga de Kaggle.

Datos de PANDA: Bulten et al. (2022), *Nat Med* 28:154–163, https://www.kaggle.com/competitions/prostate-cancer-grade-assessment.
Uso académico y no comercial, según los términos de la competencia.

