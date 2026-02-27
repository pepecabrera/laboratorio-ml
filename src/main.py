import random
from factoria import FactoriaCSV
from modelos import Clasificador_kNN

def evaluar_dataset(dataset, nombre_dataset):
    """
    Evalúa un dataset con distintas configuraciones del clasificador kNN.
    """
    print(f"\n{'='*60}")
    print(f"📊 EVALUANDO DATASET: {nombre_dataset.upper()}")
    print(f"{'='*60}")

    # Aunque el PDF dice que la validación formal se verá en el próximo lab,
    # hacemos una partición aleatoria muy básica (80% / 20%) para poder 
    # calcular una precisión (accuracy) que tenga sentido.
    registros = dataset.registros.copy()
    random.seed(42)  # Semilla para que los resultados sean reproducibles
    random.shuffle(registros)

    corte = int(len(registros) * 0.8)
    train_ds = dataset.crear_subconjunto(registros[:corte])
    test_ds  = dataset.crear_subconjunto(registros[corte:])

    print(f"Total registros: {len(registros)} -> Entrenamiento: {len(train_ds.registros)} | Test: {len(test_ds.registros)}")
    print("-" * 60)

    # Combinaciones a probar solicitadas en la práctica
    valores_k = [1, 3, 5, 7, 9]
    distancias = ["euclídea", "manhattan"]

    mejor_accuracy = -1.0
    mejor_config = ""

    for k in valores_k:
        for dist in distancias:
            # 1. Instanciar y entrenar
            modelo = Clasificador_kNN(k=k, distancia=dist)
            modelo.entrenar(train_ds)

            # 2. Predecir y calcular aciertos
            aciertos = 0
            for reg_test in test_ds.registros:
                prediccion = modelo.predecir(reg_test)
                if prediccion == reg_test.objetivo:
                    aciertos += 1

            # 3. Calcular bondad (Accuracy)
            accuracy = aciertos / len(test_ds.registros)
            print(f"k={k:2} | distancia={dist:10} -> Accuracy: {accuracy:.4f} ({aciertos}/{len(test_ds.registros)})")

            # Guardar el mejor resultado
            if accuracy > mejor_accuracy:
                mejor_accuracy = accuracy
                mejor_config = f"k={k}, distancia={dist}"

    print("-" * 60)
    print(f"🏆 MEJOR CONFIGURACIÓN: {mejor_config} (Accuracy: {mejor_accuracy:.4f})")

    # =========================================================================
    # PRUEBA CON DISTANCIA PONDERADA
    # =========================================================================
    # Generamos unos pesos de ejemplo (damos el doble de peso al primer atributo)
    num_atributos = len(registros[0].atributos)
    pesos_prueba = [1.0] * num_atributos
    pesos_prueba[0] = 2.0 

    modelo_pond = Clasificador_kNN(k=3, distancia="ponderada", pesos=pesos_prueba)
    modelo_pond.entrenar(train_ds)
    aciertos_pond = sum(1 for r in test_ds.registros if modelo_pond.predecir(r) == r.objetivo)
    acc_pond = aciertos_pond / len(test_ds.registros)
    
    print(f"\n⚖️  Prueba Ponderada (k=3, pesos={pesos_prueba}) -> Accuracy: {acc_pond:.4f}")


def main():
    # Diccionario con las rutas de los ficheros de prueba
    archivos = {
        "iris": "../datos/iris.csv",       # Ajusta esta ruta según tu estructura exacta
        "wine": "../datos/wine.csv",
        "diabetes": "../datos/diabetes.csv"
    }

    for nombre, ruta in archivos.items():
        try:
            # Creamos el dataset usando nuestra Factoría. 
            # Asumimos que la etiqueta objetivo está en la última columna (-1)
            dataset = FactoriaCSV.crear_dataset_clasificacion(ruta, indice_objetivo=-1)
            evaluar_dataset(dataset, nombre)
        except FileNotFoundError:
            print(f"\n[AVISO] No se encontró el archivo '{ruta}'.")
            print("Asegúrate de tener los CSV en la carpeta correspondiente.")
        except Exception as e:
            print(f"\n[ERROR] Problema procesando {ruta}: {e}")

    # =========================================================================
    # RESPUESTA A LA PREGUNTA DEL PROFESOR SOBRE k=1 Y EL OVERFITTING
    # =========================================================================
    print("\n" + "="*80)
    print("🧠 REFLEXIÓN TEÓRICA: ¿Qué pasa cuando k=1? (El problema del Overfitting)")
    print("="*80)
    texto_overfitting = """
    Cuando configuramos el algoritmo kNN con k=1, el modelo se vuelve extremadamente 
    sensible al ruido. Al tomar la decisión basándose en un único vecino, el algoritmo 
    literalmente "memoriza" el conjunto de datos de entrenamiento en lugar de aprender 
    los patrones generales.

    Si hiciéramos una prueba intentando predecir los mismos datos que hemos usado 
    para entrenar, k=1 nos daría un 100% de aciertos (porque el vecino más cercano a un 
    punto es él mismo). Sin embargo, al probarlo con datos nuevos (nuestro subconjunto 
    de test), su rendimiento suele caer drásticamente porque las fronteras de decisión 
    son demasiado abruptas y están ajustadas a casos atípicos.
    
    A este fenómeno se le conoce como OVERFITTING o Sobreajuste.
    
    ¿Cómo se resuelve?
    1. Aumentando el valor de 'k' (k=3, 5, 7...) para suavizar las fronteras de decisión 
       y requerir un consenso entre varios vecinos.
    2. Implementando técnicas de Validación Cruzada (Cross-Validation).
    3. Normalizando los datos (como veremos en el próximo laboratorio) para evitar que 
       las dimensiones con escalas más grandes dominen la distancia.
    """
    print(texto_overfitting)


if __name__ == "__main__":
    main()