import random
import math
from abc import ABC, abstractmethod
from typing import List, Any, Optional
from .dataset import DataSet
from .modelos import Modelo
from .preprocesado import Preprocesamiento

class Validacion(ABC):
    """
    Clase abstracta para validar la capacidad de generalización de un modelo.
    """
    def __init__(self, metrica: str):
        self.metrica = metrica.lower()

    @abstractmethod
    def calcular_metricas(self, predichos: List[Any], reales: List[Any]) -> float:
        pass

    def validacion_simple(self, modelo: Modelo, dataset: DataSet, porcentaje: float, normalizador: Optional[Preprocesamiento] = None) -> float:
        registros = dataset.registros.copy()
        #random.shuffle(registros)

        corte = int(len(registros) * porcentaje)
        train_ds = dataset.crear_subconjunto(registros[:corte])
        test_ds = dataset.crear_subconjunto(registros[corte:])

        if normalizador is not None:
            normalizador.ajustar(train_ds)
            train_ds = normalizador.transformar_dataSet(train_ds)
            test_ds = normalizador.transformar_dataSet(test_ds)

        modelo.entrenar(train_ds)

        predichos = [modelo.predecir(reg) for reg in test_ds.registros]
        reales = [reg.objetivo for reg in test_ds.registros]

        return self.calcular_metricas(predichos, reales)

    def validacion_cruzada(self, modelo: Modelo, dataset: DataSet, m: int, normalizador: Optional[Preprocesamiento] = None) -> float:
        registros = dataset.registros.copy()
        #random.shuffle(registros)

        n = len(registros)
        tam_bolsa = n // m
        bolsas = []

        for i in range(m):
            inicio = i * tam_bolsa
            fin = (i + 1) * tam_bolsa if i < m - 1 else n
            bolsas.append(registros[inicio:fin])

        metricas_iteraciones = []

        for i in range(m):
            test_regs = bolsas[i]
            train_regs = []
            for j in range(m):
                if j != i:
                    train_regs.extend(bolsas[j])

            train_ds = dataset.crear_subconjunto(train_regs)
            test_ds = dataset.crear_subconjunto(test_regs)

            if normalizador is not None:
                normalizador.ajustar(train_ds)
                train_ds = normalizador.transformar_dataSet(train_ds)
                test_ds = normalizador.transformar_dataSet(test_ds)

            modelo.entrenar(train_ds)

            predichos = [modelo.predecir(reg) for reg in test_ds.registros]
            reales = [reg.objetivo for reg in test_ds.registros]

            metricas_iteraciones.append(self.calcular_metricas(predichos, reales))

        return sum(metricas_iteraciones) / m


class ValidacionClasificacion(Validacion):
    """
    Validador para problemas de clasificación.
    Métricas soportadas: 'accuracy' (Precisión), 'error_rate' (Tasa de error).
    """
    def __init__(self, metrica: str = "accuracy"):
        super().__init__(metrica)

    def calcular_metricas(self, predichos: List[str], reales: List[str]) -> float:
        if not predichos or len(predichos) != len(reales):
            raise ValueError("Las listas de predicciones y reales deben tener la misma longitud.")

        aciertos = sum(1 for p, r in zip(predichos, reales) if p == r)
        accuracy = aciertos / len(reales)

        if self.metrica == "error_rate":
            return 1.0 - accuracy
        
        return accuracy # Por defecto


class ValidacionRegresion(Validacion):
    """
    Validador para problemas de regresión.
    Métricas soportadas: 'mae' (Error Absoluto Medio), 'mse' (Error Cuadrático Medio), 'rmse' (Raíz del MSE).
    """
    def __init__(self, metrica: str = "mae"):
        super().__init__(metrica)

    def calcular_metricas(self, predichos: List[float], reales: List[float]) -> float:
        if not predichos or len(predichos) != len(reales):
            raise ValueError("Las listas de predicciones y reales deben tener la misma longitud.")

        n = len(reales)
        
        if self.metrica == "mse":
            return sum((r - p) ** 2 for r, p in zip(reales, predichos)) / n
            
        elif self.metrica == "rmse":
            return math.sqrt(sum((r - p) ** 2 for r, p in zip(reales, predichos)) / n)
            
        else: # MAE por defecto
            return sum(abs(r - p) for r, p in zip(reales, predichos)) / n


class SeleccionAtributos:
    """
    Clase encargada de analizar y seleccionar los mejores atributos de un DataSet.
    """
    
    def seleccion_correlacion(self, dataset: DataSet, p: float) -> List[int]:
        """
        Calcula la correlación de Pearson de cada atributo con el objetivo.
        Devuelve los índices del 'p' porcentaje (0.0 a 1.0) de los atributos MÁS correlados.
        """
        if not dataset.registros:
            return []
        if not (0.0 < p <= 1.0):
            raise ValueError("El porcentaje 'p' debe estar entre 0.0 (exclusivo) y 1.0 (inclusivo).")

        n_atributos = len(dataset.registros[0].atributos)
        n_registros = len(dataset.registros)

        # 1. Extraer los valores objetivo (Y).
        try:
            # Intento 1: Asumimos que es Regresión (números puros)
            valores_y = [float(reg.objetivo) for reg in dataset.registros]
        except ValueError:
            # Intento 2: TRUCO SENIOR (Correlación Biserial Puntual)
            # Si son etiquetas de texto (Clasificación), extraemos las clases únicas
            clases_unicas = list(set(reg.objetivo for reg in dataset.registros))
            
            if len(clases_unicas) == 2:
                # Si es clasificación binaria ('M' y 'B'), mapeamos a 0.0 y 1.0
                mapa_clases = {clases_unicas[0]: 0.0, clases_unicas[1]: 1.0}
                valores_y = [mapa_clases[reg.objetivo] for reg in dataset.registros]
            else:
                # Si hay 3 o más clases, Pearson no sirve.
                raise ValueError("Pearson solo soporta Regresión o Clasificación Binaria (2 clases).")

        # 2. OPTIMIZACIÓN: Calcular Y una sola vez
        media_y = sum(valores_y) / n_registros
        suma_sq_y = sum((y - media_y) ** 2 for y in valores_y)
        
        correlaciones = []

        # 3. Calcular Pearson para cada columna (X)
        for i in range(n_atributos):
            valores_x = [reg.atributos[i] for reg in dataset.registros]
            
            media_x = sum(valores_x) / n_registros
            suma_sq_x = sum((x - media_x) ** 2 for x in valores_x)
            
            numerador = sum((x - media_x) * (y - media_y) for x, y in zip(valores_x, valores_y))
            denominador = math.sqrt(suma_sq_x * suma_sq_y)

            # Evitar división por cero si una columna tiene todos sus valores iguales (varianza 0)
            if denominador == 0:
                r = 0.0
            else:
                r = numerador / denominador

            # Guardamos el índice y el valor absoluto de 'r' (nos importan tanto los positivos como negativos fuertes)
            correlaciones.append((i, abs(r)))

        # 4. Ordenar los atributos de mayor a menor correlación
        correlaciones.sort(key=lambda x: x[1], reverse=True)

        # 5. Calcular cuántos atributos nos quedamos según el porcentaje 'p'
        num_a_seleccionar = max(1, int(n_atributos * p))

        # 6. Extraer los índices de los ganadores y devolverlos
        indices_ganadores = [tupla[0] for tupla in correlaciones[:num_a_seleccionar]]
        
        return indices_ganadores