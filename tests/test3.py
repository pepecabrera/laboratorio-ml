import sys
import os

# Esto añade la carpeta superior a la lista de sitios donde Python busca archivos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.factoria import FactoriaCSV, FactoriaXLS
from src.modelos import Clasificador_kNN, Clasificador_centroide, Regresor_kNN, Regresor_lineal_multiple
from src.preprocesado import NormalizadorMaxMin, NormalizadorZ_Score
from src.validacion import ValidacionClasificacion, ValidacionRegresion

def cargar_datos():
    """Función auxiliar para cargar los datasets usando las factorías estáticas."""
    datasets = {}
    
    # 1. Iris (Clasificación, objetivo al final: -1)
    datasets["iris"] = FactoriaXLS.crear_dataset_clasificacion("datos/iris.xlsx", indice_objetivo=-1)
    
    # 2. Diabetes (Clasificación, objetivo al final: -1)
    datasets["diabetes"] = FactoriaCSV.crear_dataset_clasificacion("datos/diabetes.csv", indice_objetivo=-1)
    
    # 3. Wine (Clasificación, objetivo al principio: 0)
    datasets["wine"] = FactoriaCSV.crear_dataset_clasificacion("datos/wine/wine.data", indice_objetivo=0)
    
    # 4. Boston (Regresión, objetivo al final: -1)
    # ¡IMPORTANTE! Aquí llamamos a crear_dataset_regresion para que el objetivo sea un float
    datasets["boston"] = FactoriaCSV.crear_dataset_regresion("datos/BostonHousing.csv", indice_objetivo=-1)
    
    return datasets

def main():
    print("\n" + "="*60)
    print("🚀 INICIANDO BATERÍA DE PRUEBAS: ETAPAS 6, 7 y 8")
    print("="*60)
    
    datasets = cargar_datos()
    val_clasificacion = ValidacionClasificacion()
    val_regresion = ValidacionRegresion()

    # ---------------------------------------------------------
    # PRUEBA 1: Clasificación (Wine) - El poder de Normalizar
    # ---------------------------------------------------------
    print("\n🍷 EVALUANDO DATASET: WINE (Clasificación)")
    print("-" * 60)
    dataset_wine = datasets["wine"]
    modelo_knn = Clasificador_kNN(k=3, distancia="euclidea")
    
    # Sin normalizar
    acc_sin = val_clasificacion.validacion_simple(modelo_knn, dataset_wine, 0.8)
    print(f"kNN (k=3, Euclídea) SIN normalizar        -> Accuracy: {acc_sin:.4f}")
    
    # Con MaxMin
    acc_maxmin = val_clasificacion.validacion_simple(modelo_knn, dataset_wine, 0.8, NormalizadorMaxMin())
    print(f"kNN (k=3, Euclídea) CON MaxMin            -> Accuracy: {acc_maxmin:.4f}")
    
    # Con Z-Score
    acc_zscore = val_clasificacion.validacion_simple(modelo_knn, dataset_wine, 0.8, NormalizadorZ_Score())
    print(f"kNN (k=3, Euclídea) CON Z-Score           -> Accuracy: {acc_zscore:.4f}")

    # ---------------------------------------------------------
    # PRUEBA 2: Clasificación (Iris & Diabetes) - Nuevos Modelos
    # ---------------------------------------------------------
    print("\n🌸🩸 EVALUANDO DATASET: IRIS Y DIABETES (Clasificación Avanzada)")
    print("-" * 60)
    modelo_centroide = Clasificador_centroide(distancia="euclidea")
    
    # Validación cruzada (5 bolsas) en Iris con Centroides
    acc_iris_centroide = val_clasificacion.validacion_cruzada(modelo_centroide, datasets["iris"], m=5)
    print(f"IRIS - Clasificador Centroide (CV m=5)    -> Accuracy Media: {acc_iris_centroide:.4f}")
    
    # Validación cruzada (5 bolsas) en Diabetes con Centroides y Z-Score
    acc_diab_centroide = val_clasificacion.validacion_cruzada(modelo_centroide, datasets["diabetes"], m=5, normalizador=NormalizadorZ_Score())
    print(f"DIABETES - Centroide + Z-Score (CV m=5)   -> Accuracy Media: {acc_diab_centroide:.4f}")

    # ---------------------------------------------------------
    # PRUEBA 3: Regresión (Boston) - Modelos Numéricos
    # ---------------------------------------------------------
    print("\n🏠 EVALUANDO DATASET: BOSTON (Regresión - MAE)")
    print("Nota: En MAE (Error Absoluto Medio), ¡cuanto más bajo, MEJOR!")
    print("-" * 60)
    dataset_boston = datasets["boston"]
    
    # Regresor kNN
    reg_knn = Regresor_kNN(k=5, distancia="manhattan")
    mae_knn = val_regresion.validacion_simple(reg_knn, dataset_boston, 0.8, NormalizadorZ_Score())
    print(f"Regresor kNN (k=5, Manhattan) + Z-Score   -> MAE: {mae_knn:.4f}")
    
    # Regresor Lineal Múltiple (Descenso de Gradiente)
    reg_lineal = Regresor_lineal_multiple(learning_rate=0.01, epocas=500)
    mae_lineal = val_regresion.validacion_simple(reg_lineal, dataset_boston, 0.8, NormalizadorZ_Score())
    print(f"Regresor Lineal Múltiple + Z-Score        -> MAE: {mae_lineal:.4f}")

    print("\n" + "="*60)
    print("✅ PRUEBAS FINALIZADAS")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()