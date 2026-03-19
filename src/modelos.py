from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random
from .dataset import DataSet
from .registro import Registro

class Modelo(ABC):
    """
    Clase abstracta que define la interfaz para cualquier modelo de Machine Learning.
    """
    def __init__(self):
        # Se inicializa a None. Se llenará cuando se llame a entrenar()
        self.datos_entrenamiento: DataSet = None

    def entrenar(self, dataset: DataSet):
        """
        Entrena el modelo guardando el conjunto de datos de entrenamiento.
        """
        self.datos_entrenamiento = dataset

    @abstractmethod
    def predecir(self, registro: Registro) -> Any:
        """
        Método abstracto para predecir la clase (clasificación) o valor (regresión).
        Obliga a las clases hijas a implementarlo.
        """
        pass


class Clasificador_kNN(Modelo):
    """
    Modelo de clasificación basado en los k vecinos más cercanos (k-Nearest Neighbors).
    """
    def __init__(self, k: int, distancia: str = "euclidea", pesos: List[float] = None):
        super().__init__()  # Llama al constructor de Modelo para inicializar datos_entrenamiento
        self.k = k
        self.distancia = distancia
        self.pesos = pesos

    def predecir(self, registro_test: Registro) -> str:
        """
        Predice la etiqueta del registro buscando sus k vecinos más cercanos 
        en el dataset de entrenamiento y aplicando votación por mayoría.
        """
        if self.datos_entrenamiento is None or not self.datos_entrenamiento.registros:
            raise ValueError("El modelo debe ser entrenado (método entrenar) antes de poder predecir.")

        # 1. Obtener los índices de los k vecinos más cercanos usando el método de la clase Registro
        indices_vecinos = registro_test.k_vecinos(
            lista_registros=self.datos_entrenamiento.registros,
            k=self.k,
            tipo=self.distancia,
            pesos=self.pesos
        )

        # 2. Sistema de votaciones
        votos = {}
        # Diccionario para saber en qué orden de cercanía apareció una etiqueta por primera vez
        primera_aparicion = {} 

        for orden, indice in enumerate(indices_vecinos):
            vecino = self.datos_entrenamiento.registros[indice]
            etiqueta = vecino.objetivo
            
            # Registramos el voto
            if etiqueta not in votos:
                votos[etiqueta] = 0
                # Como k_vecinos devuelve los índices ordenados de menor a mayor distancia,
                # el 'orden' (0, 1, 2...) nos dice la cercanía. 0 es el más cercano.
                primera_aparicion[etiqueta] = orden 
                
            votos[etiqueta] += 1

        # =================================================================================
        # RECORDATORIO DE LA REGLA DE DESEMPATE (A petición tuya):
        # Si dos etiquetas tienen los mismos votos, gana la que haya tenido un vecino 
        # a menor distancia (es decir, la que tenga el menor índice en 'primera_aparicion').
        # =================================================================================
        
        # Ordenamos las etiquetas por: 
        # 1º: Mayor número de votos (ponemos -votos[etiq] para ordenar descendentemente)
        # 2º: En caso de empate, menor 'primera_aparicion' (ascendente, gana el más cercano)
        etiquetas_ordenadas = sorted(
            votos.keys(), 
            key=lambda etiq: (-votos[etiq], primera_aparicion[etiq])
        )

        # La etiqueta ganadora será la primera de la lista ordenada
        etiqueta_ganadora = etiquetas_ordenadas[0]
        
        return etiqueta_ganadora


class Regresor_kNN(Modelo):
    """
    Modelo de regresión basado en los k vecinos más cercanos.
    Predice el valor calculando la media de los valores de sus k vecinos.
    """
    def __init__(self, k: int = 3, distancia: str = "euclidea", pesos: str = "uniforme"):
        super().__init__()
        self.k = k
        self.distancia = distancia
        self.pesos = pesos

    def entrenar(self, dataset: DataSet):
        """Memoriza los datos de entrenamiento."""
        self.datos_entrenamiento = dataset

    def predecir(self, registro: Registro) -> float:
        """Devuelve la media del atributo objetivo de los k vecinos más cercanos."""
        # Ahora sabemos que esto devuelve índices (ej: [12, 45, 3])
        indices_vecinos = registro.k_vecinos(self.datos_entrenamiento.registros, self.k, self.distancia)
        
        # Buscamos cada registro en nuestra lista de entrenamiento usando el índice
        suma_valores = 0.0
        for i in indices_vecinos:
            vecino_real = self.datos_entrenamiento.registros[i]
            suma_valores += float(vecino_real.objetivo) 
        
        return suma_valores / self.k


class Clasificador_centroide(Modelo):
    """
    Clasificador que calcula el centroide (punto medio) de cada clase.
    Asigna al nuevo registro la etiqueta del centroide más cercano.
    """
    def __init__(self, distancia: str = "euclidea"):
        super().__init__()
        self.distancia = distancia
        # Diccionario para guardar {etiqueta_clase: Registro_centroide}
        self.centroides: Dict[str, Registro] = {}

    def entrenar(self, dataset: DataSet):
        """Agrupa los datos por etiqueta y calcula el centroide de cada grupo."""
        self.centroides = {}
        
        if not dataset.registros:
            return

        grupos = {}
        for reg in dataset.registros:
            etiqueta = reg.objetivo
            if etiqueta not in grupos:
                grupos[etiqueta] = []
            grupos[etiqueta].append(reg.atributos)

        # Calculamos el centroide de cada etiqueta
        for etiqueta, matriz_atributos in grupos.items():
            n = len(matriz_atributos)
            # Transponemos la matriz para iterar por columnas (atributos)
            columnas = list(zip(*matriz_atributos))
            
            # La media de cada columna forma los atributos del centroide
            centroide_atributos = [sum(col) / n for col in columnas]
            
            # Instanciamos un nuevo Registro (de la misma clase) para representar el centroide
            clase_registro = dataset.registros[0].__class__
            self.centroides[etiqueta] = clase_registro(centroide_atributos, etiqueta)

    def predecir(self, registro: Registro) -> str:
        """Calcula la distancia a todos los centroides y devuelve la etiqueta del más cercano."""
        mejor_etiqueta = None
        menor_distancia = float('inf')

        for etiqueta, centroide in self.centroides.items():
            dist = registro.calcula_distancia(centroide, self.distancia)
            if dist < menor_distancia:
                menor_distancia = dist
                mejor_etiqueta = etiqueta

        return mejor_etiqueta


class Regresor_lineal_multiple(Modelo):
    """
    Modelo de Regresión Lineal Múltiple que utiliza el Descenso de Gradiente 
    (aproximación heurística) para encontrar los pesos de la recta de regresión.
    """
    def __init__(self, learning_rate: float = 0.01, epocas: int = 1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.epocas = epocas
        self.pesos: List[float] = []
        self.sesgo: float = 0.0  # La intersección (w_0)

    def entrenar(self, dataset: DataSet):
        """Ajusta los pesos iterativamente minimizando el error de predicción."""
        if not dataset.registros:
            return

        n_atributos = len(dataset.registros[0].atributos)
        n_registros = len(dataset.registros)
        
        # Inicializamos pesos y sesgo con valores aleatorios pequeños
        self.pesos = [random.uniform(-0.1, 0.1) for _ in range(n_atributos)]
        self.sesgo = random.uniform(-0.1, 0.1)

        # Bucle heurístico de optimización (Descenso de Gradiente)
        for _ in range(self.epocas):
            derivadas_pesos = [0.0] * n_atributos
            derivada_sesgo = 0.0

            for reg in dataset.registros:
                # Predecimos con los pesos actuales
                y_pred = self.predecir(reg)
                y_real = float(reg.objetivo)
                
                # Calculamos el error residual
                error = y_pred - y_real

                # Acumulamos los gradientes
                derivada_sesgo += error
                for i in range(n_atributos):
                    derivadas_pesos[i] += error * reg.atributos[i]

            # Actualizamos los pesos moviéndonos en dirección contraria al gradiente
            self.sesgo -= self.learning_rate * (derivada_sesgo / n_registros)
            for i in range(n_atributos):
                self.pesos[i] -= self.learning_rate * (derivadas_pesos[i] / n_registros)

    def predecir(self, registro: Registro) -> float:
        """Calcula y = sesgo + (w_1 * x_1) + ... + (w_n * x_n)."""
        if not self.pesos:
            return 0.0
            
        prediccion = self.sesgo
        for w, x in zip(self.pesos, registro.atributos):
            prediccion += w * x
            
        return prediccion