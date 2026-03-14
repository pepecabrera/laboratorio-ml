# 🧠 Librería de Machine Learning (Desde Cero)

Este proyecto es una implementación educativa y orientada a objetos de algoritmos clásicos de Machine Learning construida completamente desde cero en Python. El objetivo de esta librería es comprender las matemáticas y la lógica interna ("bajo el capó") de los modelos de aprendizaje automático, sin depender de librerías externas de ML como Scikit-Learn.

El proyecto soporta tanto tareas de **Clasificación** como de **Regresión**, e incluye herramientas para el preprocesamiento de datos, entrenamiento de modelos, evaluación de métricas y selección de atributos.

---

## 🏗️ Arquitectura del Proyecto

El código está estructurado en módulos altamente cohesivos utilizando los principios de la Programación Orientada a Objetos (Herencia, Polimorfismo y Clases Abstractas).

### 1. `registro.py` (Unidad básica de datos)
Gestiona la información a nivel de fila individual.
* **`Registro` (Abstracta):** Almacena una lista de números reales (`atributos`) y provee métodos matemáticos como `distancia_euclidea` y `distancia_manhattan`.
* **`RegistroClasificacion`:** Hereda de `Registro`. Su `objetivo` (etiqueta a predecir) es una cadena de texto (`str`), ej: `"Iris-setosa"`.
* **`RegistroRegresion`:** Hereda de `Registro`. Su `objetivo` es un valor numérico continuo (`float`), ej: `250000.0` (precio).

### 2. `dataset.py` (Gestor de colecciones)
Gestiona conjuntos de registros y operaciones sobre la matriz de datos.
* **`DataSet` (Abstracta):** Almacena una lista de objetos `Registro`.
  * *Funciones clave:* `crear_subconjunto()` para dividir datos, `calcular_min_max()` (hace la transpuesta de la matriz para hallar extremos) y `eliminar_atributos()` para descartar columnas basura.
* **`DataSetClasificacion` / `DataSetRegresion`:** Implementaciones específicas según la naturaleza del problema.

### 3. `preprocesado.py` (Normalización de datos)
Evita que atributos con escalas grandes (ej. sueldo) dominen sobre atributos pequeños (ej. edad).
* **`Preprocesamiento` (Abstracta):** Define el contrato con `ajustar(dataset)` y `transformar_dataSet(dataset)`.
* **`NormalizadorMaxMin`:** Escala todos los valores al rango [0, 1].
* **`NormalizadorZ_Score`:** Estandariza los datos a media 0 y desviación típica 1 (ideal para outliers).
> **Nota de diseño:** Los normalizadores previenen el *Data Leakage* aplicando las transformaciones al conjunto de Test basándose exclusivamente en los parámetros aprendidos del conjunto de Train.

### 4. `modelos.py` (Algoritmos Predictivos)
El núcleo inteligente del proyecto. Todos heredan de la clase base `Modelo` e implementan los métodos `entrenar(dataset)` y `predecir(registro)`.
* **`Clasificador_kNN` y `Regresor_kNN`:** Algoritmo de los k-Vecinos Más Cercanos. (Incluye un mecanismo de seguridad para que un registro se omita a sí mismo si se evalúa dentro de su propio conjunto de entrenamiento).
* **`Clasificador_Centroide`:** Calcula el punto medio (centroide) de cada clase y clasifica según la distancia mínima a estos centros.
* **`RegresionLinealMultiple`:** Implementa una red neuronal de una sola capa mediante el cálculo de pesos (`w`) y sesgo (`b`), optimizados iterativamente para predecir valores continuos.

### 5. `validacion.py` (Evaluación y Optimización)
Evalúa cómo de buenos son los modelos prediciendo datos que nunca han visto.
* **Evaluadores (`ValidacionClasificacion` / `ValidacionRegresion`):** * *Funciones clave:* `validacion_simple` (Hold-out test) y `validacion_cruzada` (k-Fold).
  * *Métricas dinámicas soportadas:* Accuracy y Error Rate (Clasificación) | MAE, MSE y RMSE (Regresión).
* **`SeleccionAtributos`:** Incluye el método `seleccion_correlacion` que utiliza el Coeficiente de Correlación de Pearson para identificar y filtrar el Top % de atributos más útiles, reduciendo el ruido del modelo.

---

## 🚀 Ejemplo de Uso Completo

Aquí tienes un flujo de trabajo típico usando la librería para predecir el precio de las casas en Boston:

```python
# 1. Importar los módulos necesarios
from src.dataset import DataSetRegresion
from src.preprocesado import NormalizadorZ_Score
from src.modelos import Regresor_kNN
from src.validacion import ValidacionRegresion, SeleccionAtributos

# 2. Cargar los datos (Asumiendo que existe un lector de CSV previamente implementado)
dataset_boston = DataSetRegresion("datos/boston.csv")

# 3. (Opcional) Selección de Atributos: Quedarse con el 50% mejor
seleccionador = SeleccionAtributos()
mejores_indices = seleccionador.seleccion_correlacion(dataset_boston, p=0.5)
indices_basura = [i for i in range(len(dataset_boston.registros[0].atributos)) if i not in mejores_indices]
boston_limpio = dataset_boston.eliminar_atributos(indices_basura)

# 4. Configurar el Modelo y el Evaluador
modelo_knn = Regresor_kNN(k=5, distancia="manhattan")
evaluador = ValidacionRegresion(metrica="rmse") # Raíz del Error Cuadrático Medio
normalizador = NormalizadorZ_Score()

# 5. Ejecutar la validación (80% Train, 20% Test)
error_rmse = evaluador.validacion_simple(
    modelo=modelo_knn, 
    dataset=boston_limpio, 
    porcentaje=0.8, 
    normalizador=normalizador
)

print(f"El error RMSE del modelo es: {error_rmse:.4f}")
```


## 📂 Estructura del Proyecto

```text
📁 Proyecto_Machine_Learning/
├── 📄 README.md                                    # Documentación principal del proyecto
│
├── 📁 datos/                                       # Directorio de datasets (archivos de datos)
│   ├── 📁 BreastCancerWisconsinDiagnostic/         # Daros de cancer de mama en Wisconsin (concurso)
│   ├── 📁 ConcreteCompressiveStrength/             # Datos de la fuerza compresiva del cemento (concurso)             
│   ├── 📁 wine/                                    # Datos en bruto de vinos (Clasificación)
│   ├── 📄 BostonHousing.csv                        # Precios de casas en Boston (Regresión)
│   ├── 📄 diabetes.csv                             # Diagnóstico de pacientes (Clasificación)
│   └── 📄 iris.xlsx                                # Especies de flores (Clasificación)
│
├── 📁 src/                                         # Código fuente (El "motor" de la librería)
│   ├── 📄 factoria.py                              # Lector de datos tanto CSV como Excel             
│   ├── 📄 dataset.py                               # Manejo de tablas, particiones y filtros
│   ├── 📄 modelos.py                               # Algoritmos predictivos (kNN, Centroide, RLM)
│   ├── 📄 preprocesado.py                          # Normalizadores de escala (MaxMin, Z-Score)
│   ├── 📄 registro.py                              # Estructura de filas (Clasificación y Regresión)
│   └── 📄 validacion.py                            # Evaluación de métricas, k-Fold y Pearson
│
└── 📁 tests/                                       # Batería de pruebas y ejecución
    ├── 📄 test1.py
    ├── 📄 test2.py
    ├── 📄 test3.py
    └── 📄 test4.py                                 # Script principal de ejecución de las etapas

El apartado concurso está reservado para una actividad especial ajena al proyecto principal.
```