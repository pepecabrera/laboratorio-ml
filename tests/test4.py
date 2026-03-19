import sys
import os
import random

# Añadir carpeta superior al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.factoria import FactoriaCSV, FactoriaXLS
from src.modelos import Clasificador_kNN, Clasificador_centroide, Regresor_kNN, Regresor_lineal_multiple
from src.preprocesado import NormalizadorMaxMin, NormalizadorZ_Score
from src.validacion import ValidacionClasificacion, ValidacionRegresion, SeleccionAtributos

def cargar_datos():
    """Carga de datasets usando las factorías estáticas."""
    datasets = {}
    datasets["iris"] = FactoriaXLS.crear_dataset_clasificacion("datos/iris.xlsx", indice_objetivo=-1)
    datasets["diabetes"] = FactoriaCSV.crear_dataset_clasificacion("datos/diabetes.csv", indice_objetivo=-1)
    datasets["wine"] = FactoriaCSV.crear_dataset_clasificacion("datos/wine/wine.data", indice_objetivo=0)
    datasets["boston"] = FactoriaCSV.crear_dataset_regresion("datos/BostonHousing.csv", indice_objetivo=-1)
    return datasets

def main():
    print("\n" + "="*70)
    print("🧪 TEST 4: ANÁLISIS DE MÉTRICAS (E9) Y OPTIMIZACIÓN DE ATRIBUTOS (E10)")
    print("Nota: Ejecución sin barajado (Shuffle OFF) según configuración actual.")
    print("="*70)
    
    datasets = cargar_datos()
    
    # ---------------------------------------------------------
    # 1. VERIFICACIÓN DE MÉTRICAS COMPLEMENTARIAS (ETAPA 9)
    # ---------------------------------------------------------
    print("\n🔹 [MÉTRICAS] Clasificación en Diabetes (Accuracy vs Error Rate)")
    ds_diab = datasets["diabetes"]
    val_acc = ValidacionClasificacion(metrica="accuracy")
    val_err = ValidacionClasificacion(metrica="error_rate")
    centroide = Clasificador_centroide(distancia="euclidea")
    
    # Probamos con un split distinto (70%) para ver cómo varía sin shuffle
    acc = val_acc.validacion_simple(centroide, ds_diab, 0.7, NormalizadorZ_Score())
    err = val_err.validacion_simple(centroide, ds_diab, 0.7, NormalizadorZ_Score())
    
    print(f"   > Accuracy:    {acc:.4f}")
    print(f"   > Error Rate:  {err:.4f}")
    print(f"   > Verificación (Acc + Err = 1): {'✅' if round(acc+err, 1) == 1.0 else '❌'}")

    # ---------------------------------------------------------
    # 2. PROFUNDIZANDO EN REGRESIÓN: MAE vs MSE vs RMSE
    # ---------------------------------------------------------
    print("\n🔹 [MÉTRICAS] Regresión en Boston (Comparativa de Sensibilidad)")
    ds_boston = datasets["boston"]
    
    # Instanciamos validaciones con las 3 métricas nuevas
    v_mae = ValidacionRegresion(metrica="mae")
    v_mse = ValidacionRegresion(metrica="mse")
    v_rmse = ValidacionRegresion(metrica="rmse")
    
    rlm = Regresor_kNN(k = 5)
    
    # Ejecutamos con el mismo dataset para comparar magnitudes
    mae_val = v_mae.validacion_simple(rlm, ds_boston, 0.8, NormalizadorMaxMin())
    mse_val = v_mse.validacion_simple(rlm, ds_boston, 0.8, NormalizadorMaxMin())
    rmse_val = v_rmse.validacion_simple(rlm, ds_boston, 0.8, NormalizadorMaxMin())
    
    print(f"   > MAE (Error lineal):            {mae_val:.4f}")
    print(f"   > MSE (Error cuadrático):        {mse_val:.4f}")
    print(f"   > RMSE (Raíz del cuadrático):    {rmse_val:.4f}")
    print("   Nota: RMSE siempre debe ser >= MAE. El RMSE penaliza más los errores grandes.")

    # ---------------------------------------------------------
    # 3. SELECCIÓN DE ATRIBUTOS Y RENDIMIENTO (ETAPA 10)
    # ---------------------------------------------------------
    print("\n🔹 [OPTIMIZACIÓN] Selección de Atributos por Pearson")
    seleccionador = SeleccionAtributos()
    
    # Queremos ver qué pasa si somos muy agresivos: solo el 30% mejor (p=0.3)
    idx_top = seleccionador.seleccion_correlacion(ds_boston, 0.3)
    print(f"   > Atributos originales: 13")
    print(f"   > Atributos tras selección (p=0.3): {len(idx_top)} indices -> {idx_top}")
    
    # Crear el dataset reducido
    indices_borrar = [i for i in range(13) if i not in idx_top]
    ds_boston_top = ds_boston.eliminar_atributos(indices_borrar)
    
    # Comparar kNN vs RLM en el dataset reducido
    # Al no haber shuffle, kNN podría seguir dando 0 si los clones están al final
    knn = Regresor_kNN(k=3)
    
    res_knn = v_mae.validacion_simple(knn, ds_boston_top, 0.8, NormalizadorZ_Score())
    res_rlm = v_mae.validacion_simple(rlm, ds_boston_top, 0.8, NormalizadorZ_Score())
    
    print(f"   > MAE kNN (Reducido): {res_knn:.4f}")
    print(f"   > MAE RLM (Reducido): {res_rlm:.4f}")
    
    # ---------------------------------------------------------
    # 4. VALIDACIÓN CRUZADA: ¿CÓMO AFECTA LA FALTA DE SHUFFLE?
    # ---------------------------------------------------------
    print("\n🔹 [VALIDACIÓN] Cross-Validation (m=5) sin barajado")
    # Al no haber shuffle, las "m" bolsas son bloques contiguos del archivo.
    # Si el archivo está ordenado, la CV dará resultados muy pobres.
    cv_acc = val_acc.validacion_cruzada(centroide, datasets["wine"], m=5, normalizador=NormalizadorMaxMin())
    print(f"   > WINE - Accuracy Media CV: {cv_acc:.4f}")
    print("   Info: Si este valor es muy bajo respecto a la validación simple,")
    print("   indica que las clases en el CSV están agrupadas por bloques.")

    print("\n" + "="*70)
    print("✅ TEST 4 FINALIZADO")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()