import random
from registro import Registro

def main():
    print("--- INICIANDO TEST DE LA CLASE REGISTRO ---")
    
    # 1. Crear aleatoriamente 10 objetos Registro de dimensión 4
    random.seed(42)  # Fijamos la semilla para reproducibilidad
    lista_registros = []
    num_atributos = 4
    
    for _ in range(10):
        atributos_aleatorios = [random.uniform(0.0, 10.0) for _ in range(num_atributos)]
        lista_registros.append(Registro(atributos_aleatorios))

    print(f"\nSe han creado {len(lista_registros)} registros:")
    for i, reg in enumerate(lista_registros):
        print(f"Índice {i}: {reg}")

    # =========================================================================
    # NUEVA FUNCIONALIDAD: Cálculo dinámico de Mínimos y Máximos del Dataset
    # =========================================================================
    print("\n--- CÁLCULO DE MÍNIMOS Y MÁXIMOS REALES ---")
    
    # Extraemos todos los atributos en una matriz (lista de listas)
    matriz_atributos = [reg.atributos for reg in lista_registros]
    
    # Usamos zip(*matriz) para agrupar los datos por columnas (atributos)
    columnas = list(zip(*matriz_atributos))
    
    # Calculamos el mínimo y máximo real para cada columna
    minimos_reales = [min(col) for col in columnas]
    maximos_reales = [max(col) for col in columnas]
    
    print(f"Mínimos calculados por dimensión: {[round(x, 2) for x in minimos_reales]}")
    print(f"Máximos calculados por dimensión: {[round(x, 2) for x in maximos_reales]}")

    # =========================================================================
    # Pruebas de los métodos de la clase Registro
    # =========================================================================
    
    reg_objetivo = lista_registros[0]
    otro_reg = lista_registros[1]

    # 2. Probar cálculos de distancia
    print("\n--- PRUEBAS DE DISTANCIA (Índice 0 vs Índice 1) ---")
    dist_euc = reg_objetivo.calcula_distancia(otro_reg, tipo="euclídea")
    dist_man = reg_objetivo.calcula_distancia(otro_reg, tipo="manhattan")
    
    pesos_prueba = [0.5, 0.2, 0.2, 0.1]
    dist_pond = reg_objetivo.calcula_distancia(otro_reg, tipo="ponderada", pesos=pesos_prueba)

    print(f"Euclídea: {dist_euc:.4f}")
    print(f"Manhattan: {dist_man:.4f}")
    print(f"Ponderada (pesos {pesos_prueba}): {dist_pond:.4f}")

    # 3. Probar normalización con los valores dinámicos
    print("\n--- PRUEBA DE NORMALIZACIÓN (Índice 0) ---")
    reg_normalizado = reg_objetivo.normalizar(minimos_reales, maximos_reales)
    print(f"Original:    {reg_objetivo}")
    print(f"Normalizado: {reg_normalizado}")
    print("Nota: Observa cómo los valores normalizados ahora están estrictamente entre 0.0 y 1.0")

    # 4. Probar k_vecinos
    print("\n--- PRUEBA DE K-VECINOS ---")
    k = 3
    # Buscamos los 3 vecinos más cercanos al registro 0
    indices_cercanos = reg_objetivo.k_vecinos(lista_registros, k, tipo="euclídea")
    print(f"Los índices de los {k} vecinos más cercanos al Registro 0 son: {indices_cercanos}")
    
    # 5. Prueba de validación de errores (Descomenta para probar)
    # print("\n--- PRUEBA DE MANEJO DE ERRORES ---")
    # registro_erroneo = Registro([1.0, 2.0, 3.0]) # Dimensión 3 en vez de 4
    # reg_objetivo.calcula_distancia(registro_erroneo) # Lanzará ValueError

if __name__ == "__main__":
    main()