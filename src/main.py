import random
from factoria import FactoriaCSV, FactoriaXLS
from modelos import Clasificador_kNN

def evaluar_dataset(dataset, nombre_dataset):
    """
    Evalúa un dataset con distintas configuraciones del clasificador kNN.
    """
    print(f"\n{'='*60}")
    print(f"📊 EVALUANDO DATASET: {nombre_dataset.upper()}")
    print(f"{'='*60}")

    registros = dataset.registros.copy()
    random.seed(42)  
    random.shuffle(registros)

    corte = int(len(registros) * 0.8)
    train_ds = dataset.crear_subconjunto(registros[:corte])
    test_ds  = dataset.crear_subconjunto(registros[corte:])

    print(f"Total registros: {len(registros)} -> Entrenamiento: {len(train_ds.registros)} | Test: {len(test_ds.registros)}")
    print("-" * 60)

    valores_k = [1, 3, 5, 7, 9]
    distancias = ["euclídea", "manhattan"]

    mejor_accuracy = -1.0
    mejor_config = ""

    for k in valores_k:
        for dist in distancias:
            modelo = Clasificador_kNN(k=k, distancia=dist)
            modelo.entrenar(train_ds)

            aciertos = 0
            for reg_test in test_ds.registros:
                prediccion = modelo.predecir(reg_test)
                if prediccion == reg_test.objetivo:
                    aciertos += 1

            accuracy = aciertos / len(test_ds.registros)
            print(f"k={k:2} | distancia={dist:10} -> Accuracy: {accuracy:.4f} ({aciertos}/{len(test_ds.registros)})")

            if accuracy > mejor_accuracy:
                mejor_accuracy = accuracy
                mejor_config = f"k={k}, distancia={dist}"

    print("-" * 60)
    print(f"🏆 MEJOR CONFIGURACIÓN: {mejor_config} (Accuracy: {mejor_accuracy:.4f})")

def main():
    # Rutas actualizadas con tus nuevos nombres y carpetas
    archivos = {
        "iris": "datos/iris.xlsx",
        "diabetes": "datos/diabetes.csv",
        "boston": "datos/BostonHousing.csv",
        "wine": "datos/wine/wine.data"  # Ahora busca dentro de la carpeta wine
    }

    for nombre, ruta in archivos.items():
        try:
            # wine tiene la etiqueta en la columna 0, el resto en la -1 (última)
            idx_objetivo = 0 if nombre == "wine" else -1

            if ruta.endswith('.csv') or ruta.endswith('.data'):
                dataset = FactoriaCSV.crear_dataset_clasificacion(ruta, indice_objetivo=idx_objetivo)
            elif ruta.endswith('.xlsx') or ruta.endswith('.xls'):
                dataset = FactoriaXLS.crear_dataset_clasificacion(ruta, indice_objetivo=idx_objetivo)
            else:
                print(f"\n[AVISO] Formato de archivo no soportado para: {ruta}")
                continue
                
            evaluar_dataset(dataset, nombre)
            
        except FileNotFoundError:
            print(f"\n[AVISO] No se encontró el archivo '{ruta}'.")
            print("Asegúrate de que estás ejecutando el script desde la carpeta raíz de tu proyecto.")
        except Exception as e:
            print(f"\n[ERROR] Problema procesando {ruta}: {e}")

    print("\n" + "="*80)
    print("🧠 REFLEXIÓN TEÓRICA: ¿Qué pasa cuando k=1? (El problema del Overfitting)")
    print("="*80)
    texto_overfitting = """
    Cuando configuramos el algoritmo kNN con k=1, el modelo se vuelve extremadamente 
    sensible al ruido. Al tomar la decisión basándose en un único vecino, el algoritmo 
    literalmente "memoriza" el conjunto de datos de entrenamiento en lugar de aprender 
    los patrones generales.

    Con datos nuevos, su rendimiento suele caer drásticamente porque las fronteras 
    de decisión son demasiado abruptas y están ajustadas a casos atípicos. 
    A este fenómeno se le conoce como OVERFITTING o Sobreajuste.
    """
    print(texto_overfitting)

if __name__ == "__main__":
    main()