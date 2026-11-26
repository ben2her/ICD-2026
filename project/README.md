# Similitud de fotos para unificar carpetas

Proyecto final individual (clasificación / aprendizaje automático). Detecta y agrupa fotos duplicadas o casi duplicadas
y compara cuatro representaciones de costo creciente: **pHash** (línea base), **histograma de color**, **LBP** y
**embeddings de ResNet-50** preentrenada, con agrupamiento jerárquico aglomerativo.

## Descripción

- **Verdad de referencia:** aumento de datos. Cada foto original genera 5 variantes con 3 transformaciones cada una
  (recorte, rotación, reescalado, JPEG, brillo, contraste, saturación, tono, balance de color). Original y variantes comparten `group_id`.
- **Evaluación:** precisión y exhaustividad sobre pares (más F1, ROC-AUC, AP), recuperación (mAP, top-1, R-precisión)
  y agrupamiento (ARI, homogeneidad, completitud, V-measure). Folds por grupo (`StratifiedGroupKFold`, K=5), umbral calibrado solo con entrenamiento
  e intervalos bootstrap por grupos.
- **Resultado principal** (100 fotos, 600 imágenes, 179,700 pares, 1,500 duplicados): ResNet-50 es la mejor en ranking
  (mAP 0.883, F1 0.719); pHash es la más barata y empata en agrupamiento (ARI 0.687 frente a 0.712); color y LBP no son competitivos.
  Detalle y limitaciones en la sección 19 del notebook.

## Origen de los datos

Conjunto de imágenes **Lewita** (100 originales), fotografiado por la maestra en docencia Adriana Guadalupe Leree-Villalejos, quien comparte el crédito del conjunto. **No se incluye en el repositorio** (por autoría y privacidad, y por tamaño si superara 50 MB); pide acceso a la autora del conjunto.
Para reproducir:

1. Crea la carpeta `mis_fotos/` junto al notebook y coloca tus imágenes (`.jpg`, `.jpeg`, `.png`, `.webp`; se admiten subcarpetas).
2. Deja `MODO = "carpeta"` en la celda de configuración.
3. Sin fotos propias, usa `MODO = "sintetico"`: el notebook genera figuras geométricas y todo el flujo corre igual
   (los números cambiarán, pues los datos son otros).

Las carpetas `originales/`, `dataset/` y `cache/` las crea el notebook; `resultados/` guarda tablas, figuras y `run_info.json`.

Su uso requiere permiso previo de los coautores del conjunto, y no pueden redistribuirse. Para solicitar acceso, contactar a los coautores del conjunto de imágenes: Lic Benjamín Hernández-Herrera [bhernandez@cicese.edu.mx](mailto:bhernandez@cicese.edu.mx) y M.D. Adriana Guadalupe Leree-Villalejos [000049213@ens.xochicalco.edu.mx](mailto:000049213@ens.xochicalco.edu.mx)

## Requisitos

- Python 3.11.
- Librerías: `requirements.txt` (versiones mínimas) y `requirements-lock.txt` (versiones exactas de la corrida; la genera la última celda de código).
- ResNet-50 usa los pesos preentrenados de `torchvision` (requiere internet la primera vez). Funciona en CPU; con GPU es más rápido.

```bash
conda create -n fotos python=3.11
conda activate fotos
pip install -r requirements.txt
jupyter lab
```

## Cómo ejecutar

Abre `proyecto_similitud_fotos_benjamin.ipynb` y usa *Restart Kernel and Run All*. Todo el trabajo, incluida la extracción de ResNet-50,
ocurre dentro del notebook; no hay scripts de entrenamiento aparte. Semillas fijas (`SEED = 42`, derivadas `+1`, `+2`, `+3`) y rutas relativas.
Después puedes comprobar la entrega:

```bash
python verificar_notebook.py proyecto_similitud_fotos_v6.ipynb
```

## Estructura de archivos

```
.
├── proyecto_similitud_fotos_benjamin.ipynb   # notebook completo, ejecutado, con salidas
├── README.md
├── requirements.txt                    # versiones mínimas
├── requirements-lock.txt               # versiones exactas (se genera al ejecutar)
├── verificar_notebook.py               # revisa contadores, errores y salidas
├── .gitignore
├── mis_fotos/                          # (no se sube) tus originales
├── originales/  dataset/  cache/       # (no se suben) generados por el notebook
└── resultados/                         # tablas, figuras y run_info.json
```

## Uso de IA generativa

Declarado en la última celda del notebook.
