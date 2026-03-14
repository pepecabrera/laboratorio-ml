import sys
import os
import time
import random

# Añadimos la raíz del proyecto al PATH para que encuentre 'src' y 'datos'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.factoria import *
from src.modelos import *
from src.preprocesado import *
from src.validacion import *

def ejecutar_montecarlo(num_iteraciones=60):
    print("="*70)
    print(f"🎲 INICIANDO MONTECARLO SEARCH ({num_iteraciones} iteraciones) 🎲")
    print("="*70)
    
    start_time = time.time()

    # Espacio de búsqueda amplio (al ser aleatorio, podemos permitirnos más opciones)
    valores_k = [1, 3, 5, 7, 9, 11, 15, 19, 21, 25] 
    preprocesadores = [(None, "Sin Normalizar"), (NormalizadorMaxMin(), "MaxMin"), (NormalizadorZ_Score(), "Z-Score")]
    pesos_distancia = ["uniforme", "distancia"]
    distancias = ["euclidea", "manhattan"]
    porcentajes_seleccion = [1.0, 0.8, 0.6]

    selector = SeleccionAtributos()

    # =========================================================================
    # 1. BREAST CANCER (Clasificación)
    # =========================================================================
    print("\n[1/2] 🧬 Analizando Breast Cancer (Montecarlo)...")
    ruta_wdbc = "datos/BreastCancerWisconsinDiagnostic/wdbc.data"
    ds_cancer_bruto = FactoriaCSV.crear_dataset_clasificacion(ruta_wdbc, indice_objetivo=1)
    ds_cancer = ds_cancer_bruto.eliminar_atributos([0]) # Bye ID
    val_clas = ValidacionClasificacion(metrica="accuracy")

    mejor_acc = -1.0
    mejor_comb_cancer = ""

    for i in range(num_iteraciones):
        # 1. Elección Aleatoria
        k = random.choice(valores_k)
        dist = random.choice(distancias)
        peso = random.choice(pesos_distancia)
        prep_obj, prep_nombre = random.choice(preprocesadores)
        p_sel = random.choice(porcentajes_seleccion)

        # 2. Selección de atributos
        if p_sel < 1.0:
            mejores_idx = selector.seleccion_correlacion(ds_cancer, p=p_sel)
            indices_borrar = [idx for idx in range(len(ds_cancer.registros[0].atributos)) if idx not in mejores_idx]
            ds_actual = ds_cancer.eliminar_atributos(indices_borrar)
            txt_sel = f"Top {int(p_sel*100)}%"
        else:
            ds_actual = ds_cancer
            txt_sel = "100%"

        # 3. K-Fold CV
        modelo_knn = Clasificador_kNN(k=k, distancia=dist, pesos=peso)
        try:
            acc = val_clas.validacion_cruzada(modelo_knn, ds_actual, m=5, normalizador=prep_obj)
            if acc > mejor_acc:
                mejor_acc = acc
                mejor_comb_cancer = f"k={k} | Dist={dist} | Pesos={peso} | Prep={prep_nombre} | Filtro={txt_sel}"
                print(f"   🚀 Nuevo récord: {mejor_acc:.4f} -> {mejor_comb_cancer}")
        except Exception:
            pass

    print(f"\n🏆 GANADOR CLASIFICACIÓN MONTECARLO (Accuracy: {mejor_acc:.4f})")
    print(f"   -> {mejor_comb_cancer}")

    # =========================================================================
    # 2. CONCRETE STRENGTH (Regresión)
    # =========================================================================
    print("\n[2/2] 🏗️ Analizando Concrete Strength (Montecarlo)...")
    ruta_concrete = "datos/ConcreteCompressiveStrength/Concrete_Data.xls"
    # IMPORTANTE: Recordamos que el precio real estaba en la penúltima columna (-2)
    ds_concrete = FactoriaOldXLS.crear_dataset_regresion(ruta_concrete) 
    val_reg = ValidacionRegresion(metrica="mse")

    mejor_mse = float('inf')
    mejor_comb_concrete = ""

    for i in range(num_iteraciones):
        k = random.choice(valores_k)
        dist = random.choice(distancias)
        peso = random.choice(pesos_distancia)
        prep_obj, prep_nombre = random.choice(preprocesadores)
        p_sel = random.choice(porcentajes_seleccion)

        if p_sel < 1.0:
            mejores_idx = selector.seleccion_correlacion(ds_concrete, p=p_sel)
            indices_borrar = [idx for idx in range(len(ds_concrete.registros[0].atributos)) if idx not in mejores_idx]
            ds_actual = ds_concrete.eliminar_atributos(indices_borrar)
            txt_sel = f"Top {int(p_sel*100)}%"
        else:
            ds_actual = ds_concrete
            txt_sel = "100%"

        modelo_reg = Regresor_kNN(k=k, distancia=dist, pesos=peso)
        try:
            mse = val_reg.validacion_cruzada(modelo_reg, ds_actual, m=5, normalizador=prep_obj)
            if mse < mejor_mse:
                mejor_mse = mse
                mejor_comb_concrete = f"k={k} | Dist={dist} | Pesos={peso} | Prep={prep_nombre} | Filtro={txt_sel}"
                print(f"   📉 Nuevo récord: {mejor_mse:.4f} -> {mejor_comb_concrete}")
        except Exception:
            pass

    print(f"\n🏆 GANADOR REGRESIÓN MONTECARLO (MSE: {mejor_mse:.4f})")
    print(f"   -> {mejor_comb_concrete}")
    
    print(f"\n⏱️ Tiempo total Montecarlo: {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    ejecutar_montecarlo()