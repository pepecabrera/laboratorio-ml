import sys
import os
import time

# TRUCO SENIOR: Añadimos la carpeta raíz del proyecto al PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importaciones masivas exigidas por el enunciado
from src.factoria import *
from src.modelos import *
from src.preprocesado import *
from src.validacion import *

def ejecutar_grid_search_avanzado():
    print("="*80)
    print("🔥 INICIANDO GRID SEARCH DE ÉLITE (OPTIMIZACIÓN EXTREMA) 🔥")
    print("="*80)
    
    start_time = time.time()

    # 1. ESPACIO DE HIPERPARÁMETROS EXPANDIDO
    valores_k = [1, 3, 5, 7, 9, 11, 15, 19, 23] # Ampliado para captar mejor el consenso
    preprocesadores = [None, NormalizadorMaxMin(), NormalizadorZ_Score()]
    nombres_prep = ["Sin Normalizar", "MaxMin", "Z-Score"]
    pesos_distancia = ["uniforme", "distancia"]
    distancias = ["euclidea", "manhattan"] # ¡CLAVE! Manhattan es vital en altas dimensiones
    
    # Fracciones de características a probar (Feature Selection)
    porcentajes_seleccion = [1.0, 0.75, 0.5] 

    # =========================================================================
    # 1. RETO DE CLASIFICACIÓN: BREAST CANCER (WDBC)
    # =========================================================================
    print("\n[1/2] 🧬 Analizando Breast Cancer Wisconsin Diagnostic...")
    ruta_wdbc = "datos/BreastCancerWisconsinDiagnostic/wdbc.data"
    ds_cancer_bruto = FactoriaCSV.crear_dataset_clasificacion(ruta_wdbc, indice_objetivo=1)
    ds_cancer_limpio = ds_cancer_bruto.eliminar_atributos([0]) # Bye ID
    
    val_clas = ValidacionClasificacion(metrica="accuracy")
    selector = SeleccionAtributos() # Instanciamos tu selector de la Etapa 10
    
    mejor_acc = -1.0
    mejor_comb_cancer = ""

    # Bucle Masivo 5D (Seleccion x K x Distancia x Pesos x Preprocesamiento)
    for p_sel in porcentajes_seleccion:
        # Aplicamos Feature Selection dinámico
        if p_sel < 1.0:
            mejores_idx = selector.seleccion_correlacion(ds_cancer_limpio, p=p_sel)
            indices_borrar = [i for i in range(len(ds_cancer_limpio.registros[0].atributos)) if i not in mejores_idx]
            ds_actual = ds_cancer_limpio.eliminar_atributos(indices_borrar)
            txt_sel = f"Top {int(p_sel*100)}% Atributos"
        else:
            ds_actual = ds_cancer_limpio
            txt_sel = "100% Atributos"

        for k in valores_k:
            for dist in distancias:
                for peso in pesos_distancia:
                    modelo_knn = Clasificador_kNN(k=k, distancia=dist, pesos=peso) 
                    
                    for i, prep in enumerate(preprocesadores):
                        try:
                            # CV 5-fold (SIN SHUFFLE)
                            acc = val_clas.validacion_cruzada(modelo_knn, ds_actual, m=5, normalizador=prep)
                            
                            if acc > mejor_acc:
                                mejor_acc = acc
                                mejor_comb_cancer = f"k={k} | Dist={dist.upper()} | Pesos={peso} | Prep={nombres_prep[i]} | Filtro={txt_sel}"
                                # Si vemos un resultado de élite, lo notificamos en vivo
                                if acc >= 0.975:
                                    print(f"   🚀 ¡Pico detectado! Accuracy: {acc:.4f} -> {mejor_comb_cancer}")
                        except Exception as e:
                            pass

    print("\n" + "*"*60)
    print(f"🏆 GANADOR CLASIFICACIÓN (Accuracy: {mejor_acc:.4f})")
    print(f"   -> {mejor_comb_cancer}")
    print("*"*60)

    # =========================================================================
    # 2. RETO DE REGRESIÓN: CONCRETE COMPRESSIVE STRENGTH
    # =========================================================================
    print("\n[2/2] 🏗️ Analizando Concrete Compressive Strength...")
    ruta_concrete = "datos/ConcreteCompressiveStrength/Concrete_Data.xls"
    ds_concrete = FactoriaOldXLS.crear_dataset_regresion(ruta_concrete, indice_objetivo=-1)
    
    val_reg = ValidacionRegresion(metrica="mse")
    
    mejor_mse = float('inf')
    mejor_comb_concrete = ""

    # Para regresión, iteramos igual pero buscando el menor MSE
    for p_sel in porcentajes_seleccion:
        if p_sel < 1.0:
            mejores_idx = selector.seleccion_correlacion(ds_concrete, p=p_sel)
            indices_borrar = [i for i in range(len(ds_concrete.registros[0].atributos)) if i not in mejores_idx]
            ds_actual_reg = ds_concrete.eliminar_atributos(indices_borrar)
            txt_sel = f"Top {int(p_sel*100)}% Atributos"
        else:
            ds_actual_reg = ds_concrete
            txt_sel = "100% Atributos"

        for k in valores_k:
            for dist in distancias:
                for peso in pesos_distancia:
                    modelo_reg = Regresor_kNN(k=k, distancia=dist, pesos=peso)
                    
                    for i, prep in enumerate(preprocesadores):
                        try:
                            mse = val_reg.validacion_cruzada(modelo_reg, ds_actual_reg, m=5, normalizador=prep)
                            if mse < mejor_mse:
                                mejor_mse = mse
                                mejor_comb_concrete = f"k={k} | Dist={dist.upper()} | Pesos={peso} | Prep={nombres_prep[i]} | Filtro={txt_sel}"
                        except Exception:
                            pass

    print("\n" + "*"*60)
    print(f"🏆 GANADOR REGRESIÓN (MSE: {mejor_mse:.4f})")
    print(f"   -> {mejor_comb_concrete}")
    print("*"*60)
    
    tiempo = time.time() - start_time
    print(f"\n✅ OPOSICIÓN SUPERADA - Tiempo de cómputo: {tiempo:.2f} segundos")

if __name__ == "__main__":
    ejecutar_grid_search_avanzado()