# Práctica 3 — Predicción de deserción estudiantil

Clasificador que, al cierre del primer semestre, predice la situación de un estudiante al final de la duración normal de su carrera: `graduate`, `dropout` o `enrolled`.

## Dataset

Estudiantes de licenciatura del Instituto Politécnico de Portalegre (Portugal), ingresados entre los ciclos 2008/09 y 2018/19. Fuente: [UCI Machine Learning Repository, dataset 697](https://archive.ics.uci.edu/dataset/697) (descripción de cada atributo y significado de los códigos).

| Archivo | Filas | Contenido |
|---|---|---|
| `datos/desercion-entrenamiento.csv` | 3539 | `id`, 36 atributos y `Target` |
| `datos/desercion-prueba.csv` | 885 | `id` y 30 atributos del primer semestre (sin `Target` ni atributos del segundo semestre) |

- Variable objetivo: `Target` ∈ {`dropout`, `enrolled`, `graduate`}.
- Atributos: 9 nominales codificados como enteros (p. ej. `Course`, ocupaciones y calificaciones de los padres), 8 binarios y el resto numéricos. Los 6 atributos `Curricular units 2nd sem (...)` solo existen en el entrenamiento y se usan únicamente para estimar el beneficio de esperar al segundo semestre.

## Qué hace el notebook

`practica3_desercion.ipynb` compara árbol de decisión, Naive Bayes, kNN, SVM y regresión logística, cada uno con su preprocesamiento dentro de un `Pipeline`, optimizados con `RandomizedSearchCV` (F1 macro, validación cruzada estratificada de 5 pliegues, **los mismos pliegues** para los cinco, ≥ 50 configuraciones por clasificador). Incluye curvas de meseta, tabla comparativa (F1 macro media ± desv. est., sensibilidad y especificidad por clase, tiempo), matriz de confusión del mejor modelo, análisis de `dropout`, comparación primer vs. segundo semestre, modelo final y predicciones.

## Estructura

```
.
├── datos/
│   ├── desercion-entrenamiento.csv
│   └── desercion-prueba.csv
├── practica3_desercion.ipynb
├── predicciones.csv          # generado por el notebook (id, Target)
├── salidas/                  # figuras y tablas generadas por el notebook
├── requirements.txt
└── README.md
```

## Requisitos

- Python ≥ 3.10
- Librerías: ver `requirements.txt`. Las versiones exactas con las que se obtuvieron los resultados se imprimen en la primera celda del notebook.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

1. Coloca los dos CSV en `datos/`.
2. Abre el notebook (`jupyter lab` o `jupyter notebook`) y ejecuta **Restart & Run All**.
3. Al terminar se escribe `predicciones.csv` en la raíz del repositorio, con exactamente dos columnas (`id`, `Target`) y 885 filas.

Opcional: `N_ITER=100 jupyter lab` aumenta el número de configuraciones por clasificador (por defecto 60; mínimo exigido 50).

## Reproducibilidad

- Semilla fija (`SEED = 42`) en la validación cruzada, las búsquedas y los modelos.
- Rutas relativas.
- El tiempo de búsqueda depende de la máquina (SVM y regresión logística son los más lentos).


