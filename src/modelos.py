from abc import ABC, abstractmethod
from typing import List, Any
from dataset import DataSet
from registro import Registro

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
    def __init__(self, k: int, distancia: str = "euclídea", pesos: List[float] = None):
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