from abc import ABC, abstractmethod
from typing import List
from .dataset import DataSet
from .registro import Registro

class Preprocesamiento(ABC):
    """
    Clase abstracta base para definir operaciones de preprocesado de datos.
    """
    
    @abstractmethod
    def ajustar(self, dataset: DataSet):
        """Calcula y almacena los parámetros necesarios (ej. min/max o media/desv)."""
        pass
        
    @abstractmethod
    def transformar_registro(self, registro: Registro) -> Registro:
        """Aplica la transformación a un único registro y lo devuelve."""
        pass
        
    @abstractmethod
    def transformar_dataSet(self, dataset: DataSet) -> DataSet:
        """Aplica la transformación a todo un dataset y devuelve uno nuevo."""
        pass


class NormalizadorMaxMin(Preprocesamiento):
    """
    Normaliza los atributos al rango [0, 1] usando los valores máximos y mínimos.
    """
    
    def __init__(self):
        """Inicializa las listas de mínimos y máximos a vacío."""
        self.maximos: List[float] = []
        self.minimos: List[float] = []
        
    def ajustar(self, dataset: DataSet):
        """Calcula los mínimos y máximos a partir del DataSet dado."""
        self.minimos, self.maximos = dataset.calcular_min_max()
        
    def transformar_registro(self, registro: Registro) -> Registro:
        """Devuelve un nuevo registro normalizado."""
        # Nota: Asumo que registro.normalizar() recibe (minimos, maximos)
        return registro.normalizar(self.minimos, self.maximos)
        
    def transformar_dataSet(self, dataset: DataSet) -> DataSet:
        """Devuelve un nuevo DataSet con todos sus registros normalizados."""
        nuevos_registros = [self.transformar_registro(reg) for reg in dataset.registros]
        return dataset.crear_subconjunto(nuevos_registros)


class NormalizadorZ_Score(Preprocesamiento):
    """
    Estandariza los atributos usando la media y la desviación típica (Z-Score).
    """
    
    def __init__(self):
        """Inicializa las listas de medias y desviaciones a vacío."""
        self.medias: List[float] = []
        self.desviaciones: List[float] = []
        
    def ajustar(self, dataset: DataSet):
        """Calcula las medias y desviaciones típicas a partir del DataSet dado."""
        self.medias, self.desviaciones = dataset.calcular_medias_desviaciones()
        
    def transformar_registro(self, registro: Registro) -> Registro:
        """Devuelve un nuevo registro estandarizado."""
        return registro.estandarizar(self.medias, self.desviaciones)
        
    def transformar_dataSet(self, dataset: DataSet) -> DataSet:
        """Devuelve un nuevo DataSet con todos sus registros estandarizados."""
        nuevos_registros = [self.transformar_registro(reg) for reg in dataset.registros]
        return dataset.crear_subconjunto(nuevos_registros)