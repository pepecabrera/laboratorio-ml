import random
from abc import ABC, abstractmethod
from typing import List, Any
from .dataset import DataSet
from .modelos import Modelo
from .preprocesado import Preprocesamiento

class Validacion(ABC):
    """
    Clase abstracta para validar la capacidad de generalización de un modelo 
    de Machine Learning utilizando distintas estrategias de partición.
    """

    @abstractmethod
    def calcular_metricas(self, predichos: List[Any], reales: List[Any]) -> float:
        """
        Calcula la métrica de rendimiento comparando las predicciones con los valores reales.
        Debe ser implementado por las clases hijas según el tipo de problema (clasificación/regresión).
        """
        pass

    def validacion_simple(self, modelo: Modelo, dataset: DataSet, porcentaje: float, normalizador: Preprocesamiento = None) -> float:
        """
        Aplica la técnica Hold-out dividiendo el dataset en entrenamiento y test.
        Entrena el modelo y devuelve la métrica calculada sobre las predicciones del test.
        """
        # 1. Barajamos los datos para evitar sesgos de orden
        registros = dataset.registros.copy()
        random.shuffle(registros)

        # 2. Dividimos en train y test
        corte = int(len(registros) * porcentaje)
        train_ds = dataset.crear_subconjunto(registros[:corte])
        test_ds = dataset.crear_subconjunto(registros[corte:])

        # 3. Aplicamos preprocesamiento si existe
        if normalizador is not None:
            # ¡IMPORTANTE! El normalizador SOLO se ajusta con los datos de entrenamiento
            # para evitar que el modelo "vea" el futuro (los datos de test).
            normalizador.ajustar(train_ds)
            train_ds = normalizador.transformar_dataSet(train_ds)
            test_ds = normalizador.transformar_dataSet(test_ds)

        # 4. Entrenamos el modelo
        modelo.entrenar(train_ds)

        # 5. Predecimos y extraemos valores reales
        predichos = []
        reales = []
        for reg in test_ds.registros:
            predichos.append(modelo.predecir(reg))
            reales.append(reg.objetivo)

        # 6. Calculamos y devolvemos la métrica
        return self.calcular_metricas(predichos, reales)

    def validacion_cruzada(self, modelo: Modelo, dataset: DataSet, m: int, normalizador: Preprocesamiento = None) -> float:
        """
        Aplica k-Fold Cross-Validation dividiendo el dataset en m bolsas.
        Devuelve la media de la métrica obtenida en las m iteraciones.
        """
        registros = dataset.registros.copy()
        random.shuffle(registros)

        n = len(registros)
        tam_bolsa = n // m
        bolsas = []

        # Crear las m bolsas
        for i in range(m):
            inicio = i * tam_bolsa
            # La última bolsa se queda con el resto de elementos si la división no es exacta
            fin = (i + 1) * tam_bolsa if i < m - 1 else n
            bolsas.append(registros[inicio:fin])

        metricas_iteraciones = []

        # Realizar el proceso m veces
        for i in range(m):
            # La bolsa i es el test, el resto (sumadas) son el train
            test_regs = bolsas[i]
            train_regs = []
            for j in range(m):
                if j != i:
                    train_regs.extend(bolsas[j])

            train_ds = dataset.crear_subconjunto(train_regs)
            test_ds = dataset.crear_subconjunto(test_regs)

            # Preprocesamiento en cada iteración
            if normalizador is not None:
                normalizador.ajustar(train_ds)
                train_ds = normalizador.transformar_dataSet(train_ds)
                test_ds = normalizador.transformar_dataSet(test_ds)

            modelo.entrenar(train_ds)

            # Predicciones
            predichos = [modelo.predecir(reg) for reg in test_ds.registros]
            reales = [reg.objetivo for reg in test_ds.registros]

            # Calcular y guardar métrica de esta iteración
            metrica_actual = self.calcular_metricas(predichos, reales)
            metricas_iteraciones.append(metrica_actual)

        # Devolver la media de las m métricas
        return sum(metricas_iteraciones) / m


class ValidacionClasificacion(Validacion):
    """
    Clase concreta para validar problemas de clasificación.
    Usa la precisión (Accuracy) como métrica por defecto.
    """

    def calcular_metricas(self, predichos: List[str], reales: List[str]) -> float:
        """
        Calcula la Accuracy: (Número de aciertos) / (Total de registros).
        """
        if not predichos or len(predichos) != len(reales):
            raise ValueError("Las listas de predicciones y reales deben tener la misma longitud y no estar vacías.")

        aciertos = sum(1 for p, r in zip(predichos, reales) if p == r)
        return aciertos / len(reales)


class ValidacionRegresion(Validacion):
    """
    Clase concreta para validar problemas de regresión.
    Usa el Error Absoluto Medio (MAE) como métrica por defecto.
    """

    def calcular_metricas(self, predichos: List[float], reales: List[float]) -> float:
        """
        Calcula el MAE (Mean Absolute Error): Media de las diferencias absolutas.
        """
        if not predichos or len(predichos) != len(reales):
            raise ValueError("Las listas de predicciones y reales deben tener la misma longitud y no estar vacías.")

        # Sumamos el valor absoluto de la diferencia entre la predicción y la realidad
        error_total = sum(abs(p - r) for p, r in zip(predichos, reales))
        return error_total / len(reales)