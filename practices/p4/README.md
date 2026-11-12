# Práctica 4 — Ensambles sobre clasificadores bien ajustados

Pregunta: ¿cuánto ganan los ensambles sobre los cinco clasificadores base de la Práctica 3 cuando estos ya están bien ajustados?

## Conjuntos de datos

**1. Deserción estudiantil (Práctica 3).** Estudiantes de licenciatura del Instituto Politécnico de Portalegre (Portugal), [UCI dataset 697](https://archive.ics.uci.edu/dataset/697). 3539 estudiantes de entrenamiento y 30 atributos disponibles al cierre del primer semestre (9 nominales codificados como enteros, 8 binarios y 13 numéricos). Variable objetivo `Target`: `dropout` (32 %), `enrolled` (18 %), `graduate` (50 %). Evaluación con `StratifiedKFold(5, shuffle=True, random_state=42)`, los mismos pliegues de la Práctica 3.

**2. Bank Marketing (Tarea 4).** `bank-additional-full.csv`: 41 188 clientes de una campaña de telemarketing de un banco portugués, con atributos del cliente, del contacto y del contexto socioeconómico. Objetivo `y` (¿contrata un depósito a plazo?): `yes` 11.3 %, `no` 88.7 %. Se elimina la columna `duration` (no se conoce antes de la llamada). Evaluación con `StratifiedKFold(5, shuffle=True, random_state=2026)`. Fuente: Moro, Cortez y Rita (2014), _A Data-Driven Approach to Predict the Success of Bank Telemarketing_, Decision Support Systems.

## Estructura del repositorio

```
.
├── datos/
│   ├── desercion-entrenamiento.csv
│   ├── desercion-prueba.csv              (de la Práctica 3; esta práctica no lo usa)
│   └── /bank-adittional/bank/adittional/bank-additional-full.csv
├── ICD_practica4_ensambles.ipynb         # notebook principal
├── benjamin-hernandez_modelo.joblib      # modelo final de Bank Marketing (generado por el notebook)
├── usar_modelo.py                        # aplica el .joblib a un CSV
├── salidas/                              # figuras y tablas generadas por el notebook
├── requirements.txt
└── README.md
```

## Requisitos

- Python ≥ 3.10 (los resultados se obtuvieron en Google Colab con Python 3.13).
- Versiones usadas: numpy 2.4.6, pandas 2.2.3, scipy 1.16.3, **scikit-learn 1.6.1**, xgboost 3.4.1, lightgbm 4.6.0, joblib 1.6.0, matplotlib 3.10.0. Se necesita **scikit-learn ≥ 1.5** (`TunedThresholdClassifierCV`).
- Para cargar el `.joblib` conviene usar la misma versión de scikit-learn, xgboost y lightgbm con la que se guardó (un modelo serializado puede fallar o comportarse distinto con otras versiones).

```bash
pip install -r requirements.txt
```

## Cómo ejecutar el notebook

1. Coloca los CSV en `datos/` (en Colab: sube la carpeta o monta Drive y ajusta `DATA_DIR`).
2. Ejecuta el notebook con **Restart & Run All**. En Colab con 2 CPU tarda del orden de una hora (las búsquedas de deserción toman 4–7 minutos cada una y las de Bank 1.5–6 minutos cada una).
3. El notebook guarda tablas y figuras en `salidas/` y el modelo final en `NOMBRE_modelo.joblib` (la variable `NOMBRE` está en la celda de configuración).

Parámetros opcionales (variables de entorno): `N_ITER` (configuraciones por búsqueda en deserción, 60 por defecto), `N_ITER_BANK` (50), `SEARCH_FRAC` (fracción de Bank usada para buscar hiperparámetros, 0.3). El mínimo exigido es 50 configuraciones por búsqueda.

## Cómo usar el archivo `.joblib`

El archivo es el mejor modelo de **Bank Marketing**: un `Pipeline` completo entrenado con las 41 188 filas (preprocesamiento + clasificador + umbral de decisión ajustado). Recibe un `DataFrame` con las **19 columnas** de `bank-additional-full.csv` **sin `duration` ni `y`** y devuelve `1` (`yes`) o `0` (`no`).

**Opción A — script incluido (la más segura).**

```bash
python usar_modelo.py benjamin-hernandez_modelo.joblib bank-additional-full.csv predicciones_bank.csv
```

Lee el CSV (separador `;`, ignora `duration` y `y` si vienen), aplica el modelo y escribe `predicciones_bank.csv` con las columnas `prediccion` (1/0) y `prediccion_etiqueta` (yes/no). Si el CSV trae la columna `y`, además imprime el F1 macro y la sensibilidad de `yes`.

**Opción B — desde Python o un notebook.** El modelo usa una función (`fe_bank`) y una clase (`WeightedBoost`) definidas en el notebook de entrenamiento. Si el archivo se guardó con la versión actual del notebook (que usa `cloudpickle`), esas definiciones viajan dentro del archivo y basta con:

```python
import joblib, pandas as pd

modelo = joblib.load("benjamin-hernandez_modelo.joblib")
df = pd.read_csv("datos/bank-additional-full.csv", sep=";")
X = df.drop(columns=["y", "duration"])          # mismas 19 columnas del entrenamiento
pred = modelo.predict(X)                        # 1 = yes, 0 = no
proba = modelo.predict_proba(X)[:, 1]           # probabilidad de yes (el umbral ya está dentro de predict)
```

Si el archivo se guardó con `joblib.dump` (versiones anteriores del notebook), falla con `AttributeError: Can't get attribute 'fe_bank' on <module '__main__'>`. En ese caso copia las definiciones de `fe_bank` y `WeightedBoost` de `usar_modelo.py` en la sesión **antes** de llamar a `joblib.load`, o usa la opción A.

Notas: `predict` aplica el umbral que maximiza el F1 macro (no 0.5); el modelo necesita las bibliotecas `xgboost` y `lightgbm` instaladas aunque el modelo guardado use solo una de ellas, y los nombres de las columnas deben coincidir con los del archivo original.

## Resultados

**Deserción** (F1 macro, media ± desviación entre 5 pliegues, mismos pliegues para todos; sens. = sensibilidad de `enrolled`):

| Modelo                                 | F1 macro          | Sens. enrolled | Entrenamiento por pliegue |
| -------------------------------------- | ----------------- | -------------- | ------------------------- |
| **XGBoost (optimizado)**               | **0.696 ± 0.011** | 0.627          | 1.7 s                     |
| LightGBM (optimizado)                  | 0.689 ± 0.016     | 0.465          | 3.0 s                     |
| Stacking (meta: regresión logística)   | 0.687 ± 0.008     | 0.617          | 24.7 s                    |
| Votación dura (5 base)                 | 0.681 ± 0.013     | 0.542          | 3.2 s                     |
| Random Forest (optimizado)             | 0.681 ± 0.008     | 0.471          | 2.2 s                     |
| Regresión logística (base, Práctica 3) | 0.679 ± 0.011     | 0.575          | 2.1 s                     |
| Votación suave (5 base)                | 0.678 ± 0.011     | 0.452          | 4.3 s                     |
| SVM (base, Práctica 3)                 | 0.676 ± 0.008     | 0.578          | 1.3 s                     |
| Extra-Trees (optimizado)               | 0.668 ± 0.009     | 0.441          | 6.7 s                     |
| Bagging (400 árboles sin límite)       | 0.660 ± 0.022     | 0.340          | 14.7 s                    |
| Árbol (base)                           | 0.644 ± 0.015     | 0.570          | 0.1 s                     |
| AdaBoost (400 rondas, tocones)         | 0.638 ± 0.010     | 0.272          | 3.9 s                     |
| Naive Bayes (base)                     | 0.625 ± 0.014     | 0.487          | 0.1 s                     |
| Árbol sin límite de profundidad        | 0.591 ± 0.027     | 0.373          | 0.1 s                     |
| kNN (base)                             | 0.560 ± 0.027     | 0.217          | 0.0 s                     |
| Tocón (profundidad 1)                  | 0.492 ± 0.012     | 0.000          | 0.0 s                     |

**Bank Marketing** (sin `duration`, `random_state=2026`; sens. = sensibilidad de `yes`):

| Modelo                                                 | F1 macro            | Sens. yes | Entrenamiento por pliegue |
| ------------------------------------------------------ | ------------------- | --------- | ------------------------- |
| **LightGBM + umbral** (modelo guardado)                | **0.7178 ± 0.0061** | 0.538     | 12.5 s                    |
| XGBoost + umbral                                       | 0.7178 ± 0.0047     | 0.547     | 9.5 s                     |
| Votación suave + umbral                                | 0.7178 ± 0.0066     | 0.546     | 72.8 s                    |
| Random Forest + umbral                                 | 0.7173 ± 0.0069     | 0.547     | 36.2 s                    |
| HistGradientBoosting + umbral                          | 0.7141 ± 0.0057     | 0.537     | 18.7 s                    |
| Extra-Trees + umbral                                   | 0.7141 ± 0.0061     | 0.552     | 34.3 s                    |
| Regresión logística + umbral                           | 0.7071 ± 0.0056     | 0.533     | 3.0 s                     |
| Extra-Trees (sin umbral)                               | 0.6902 ± 0.0059     | 0.614     | 12.2 s                    |
| LightGBM / XGBoost / Random Forest / HistGB sin umbral | 0.655–0.662         | 0.25–0.27 | 2.3–12.3 s                |
| Votación suave (sin umbral)                            | 0.6526 ± 0.0110     | 0.245     | 22.7 s                    |
| Regresión logística (sin umbral)                       | 0.6410 ± 0.0101     | 0.225     | 0.5 s                     |

El stacking de Bank no aparece porque la ejecución guardada falló por un detalle de validación cruzada (pliegues internos sin mezclar sobre datos ordenados por fecha). El código del notebook ya está corregido con `shuffle=True`; al volver a ejecutarlo se obtiene su resultado real.

Conclusión resumida: con los clasificadores base ya bien ajustados, los ensambles ganan poco (+0.017 de F1 macro en deserción; +0.011 en Bank). XGBoost/LightGBM con umbral ajustado ofrece el mejor compromiso entre desempeño y costo. En Bank, ajustar el umbral de decisión aportó más (entre +0.056 y +0.066) que cambiar de algoritmo. El detalle está en la sección de conclusiones del notebook.

## Reproducibilidad

Semillas fijas (42 para deserción y 2026 para Bank), rutas relativas, mismos pliegues para todos los modelos de cada conjunto de datos. Los tiempos dependen de la máquina.
