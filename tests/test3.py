import sys
import os

# Esto añade la carpeta superior a la lista de sitios donde Python busca archivos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.factoria import FactoriaCSV, FactoriaXLS
from src.modelos import Clasificador_kNN, Clasificador_centroide, Regresor_kNN, Regresor_lineal_multiple
from src.preprocesado import NormalizadorMaxMin, NormalizadorZ_Score
from src.validacion import ValidacionClasificacion, ValidacionRegresion, SeleccionAtributos

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

    # ---------------------------------------------------------
    # PRUEBA 4: NUEVAS MÉTRICAS (ETAPA 9)
    # ---------------------------------------------------------
    print("\n📈 EVALUANDO NUEVAS MÉTRICAS (ETAPA 9)")
    print("-" * 60)
    
    # Probando Tasa de Error en Diabetes
    val_clas_error = ValidacionClasificacion(metrica="error_rate")
    tasa_error = val_clas_error.validacion_cruzada(modelo_centroide, datasets["diabetes"], m=5, normalizador=NormalizadorZ_Score())
    print(f"DIABETES - Centroide + Z-Score -> Tasa de Error: {tasa_error:.4f}")
    
    # Probando RMSE en Boston
    val_reg_rmse = ValidacionRegresion(metrica="rmse")
    rmse_boston = val_reg_rmse.validacion_simple(reg_knn, datasets["boston"], 0.8, NormalizadorZ_Score())
    print(f"BOSTON - kNN (k=5) + Z-Score -> RMSE (Raíz Error Cuadrático): {rmse_boston:.4f}")
    
    # ---------------------------------------------------------
    # PRUEBA 5: SELECCIÓN DE ATRIBUTOS (ETAPA 10)
    # ---------------------------------------------------------
    print("\n🔪 EVALUANDO SELECCIÓN DE ATRIBUTOS (ETAPA 10)")
    print("-" * 60)
    
    dataset_boston = datasets["boston"]
    num_atributos_original = len(dataset_boston.registros[0].atributos)
    
    # 1. Instanciamos el seleccionador
    seleccionador = SeleccionAtributos()
    
    # 2. Pedimos que nos devuelva el 50% (p=0.5) de los atributos más correlados con el precio
    mejores_indices = seleccionador.seleccion_correlacion(dataset_boston, 0.5)
    print(f"Total atributos originales en Boston: {num_atributos_original}")
    print(f"Índices de los mejores atributos (Top 50%): {mejores_indices}")
    
    # 3. Calculamos cuáles son los índices "basura" (los que NO están en la lista de mejores)
    indices_a_eliminar = [i for i in range(num_atributos_original) if i not in mejores_indices]
    print(f"Índices que vamos a eliminar: {indices_a_eliminar}")
    
    # 4. Creamos el nuevo DataSet recortado usando nuestro método de la Etapa 10
    boston_reducido = dataset_boston.eliminar_atributos(indices_a_eliminar)
    num_atributos_nuevos = len(boston_reducido.registros[0].atributos)
    print(f"Total atributos tras recorte: {num_atributos_nuevos}")
    
    # 5. Evaluamos el mismo modelo kNN, comparando el dataset completo vs el reducido
    reg_knn_boston = Regresor_kNN(k=5, distancia="manhattan")
    val_reg_mae = ValidacionRegresion(metrica="mae")
    
    # Usamos la misma semilla aleatoria para que el corte Train/Test sea idéntico y la comparativa sea justa
    import random
    random.seed(42)
    mae_boston_completo = val_reg_mae.validacion_simple(reg_knn_boston, dataset_boston, 0.8, NormalizadorZ_Score())
    
    random.seed(42) # Reiniciamos la semilla para que corte exactamente igual
    mae_boston_reducido = val_reg_mae.validacion_simple(reg_knn_boston, boston_reducido, 0.8, NormalizadorZ_Score())
    
    print(f"\nMAE kNN (Boston Completo - {num_atributos_original} col): {mae_boston_completo:.4f}")
    print(f"MAE kNN (Boston Reducido - {num_atributos_nuevos} col): {mae_boston_reducido:.4f}")
    
    if mae_boston_reducido <= mae_boston_completo:
        print("🎯 ¡ÉXITO! El modelo es igual o más preciso, ¡y además calcula el doble de rápido!")
    else:
        print("ℹ️ NOTA: El error ha subido ligeramente, pero a cambio el modelo es el doble de rápido y ocupa menos memoria.")
    
    print("\n" + "="*60)
    print("✅ PRUEBAS FINALIZADAS")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()