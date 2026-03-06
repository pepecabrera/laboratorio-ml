from abc import ABC, abstractmethod
from typing import List, Tuple
from .registro import Registro, RegistroClasificacion, RegistroRegresion

class DataSet(ABC):
    """
    Clase abstracta contenedora de un conjunto de objetos Registro.
    No puede ser instanciada directamente.
    """

    def __init__(self):
        """Inicializa las listas vacías para los registros y los nombres de los atributos."""
        self.registros: List[Registro] = []
        self.nombres_atributos: List[str] = []

    def set_cabeceras(self, lista_str: List[str]):
        """
        Asigna todos los elementos de la lista menos el último a nombres_atributos.
        (El último suele corresponder a la etiqueta/objetivo).
        """
        if not lista_str:
            raise ValueError("La lista de cabeceras no puede estar vacía.")
        # [:-1] coge todos los elementos desde el principio hasta el penúltimo
        self.nombres_atributos = lista_str[:-1]

    @abstractmethod
    def agregar_registro(self, registro: Registro):
        """
        Método abstracto. Obliga a las clases hijas a implementarlo
        con su propia lógica de validación.
        """
        pass

    def calcular_min_max(self) -> Tuple[List[float], List[float]]:
        """
        Devuelve dos listas con los valores máximos y mínimos para cada dimensión 
        de los atributos en la lista de registros.
        """
        if not self.registros:
            raise ValueError("No hay registros en el DataSet para calcular mínimos y máximos.")

        # Extraemos todos los atributos en una matriz (lista de listas)
        matriz_atributos = [reg.atributos for reg in self.registros]
        
        # zip(*matriz) agrupa los datos por columnas
        columnas = list(zip(*matriz_atributos))
        
        # Calculamos el mínimo y máximo de cada columna
        minimos = [min(col) for col in columnas]
        maximos = [max(col) for col in columnas]
        
        return minimos, maximos

    def crear_subconjunto(self, lista_registros: List[Registro]) -> 'DataSet':
        """
        Devuelve un nuevo objeto DataSet (de la clase hija correspondiente)
        con la lista de registros pasada como parámetro y las mismas cabeceras.
        """
        # self.__class__() instanciará DataSetClasificacion o DataSetRegresion
        nuevo_dataset = self.__class__()
        # Copiamos los registros y las cabeceras
        nuevo_dataset.registros = lista_registros.copy()
        nuevo_dataset.nombres_atributos = self.nombres_atributos.copy()
        
        return nuevo_dataset


class DataSetClasificacion(DataSet):
    """
    DataSet específico para problemas de clasificación.
    """
    
    def agregar_registro(self, registro: Registro):
        """
        Añade un objeto a la lista de registros si es de tipo RegistroClasificacion.
        """
        # Validamos que el objeto sea estrictamente del tipo correcto
        if not isinstance(registro, RegistroClasificacion):
            raise TypeError(f"Se esperaba un RegistroClasificacion, pero se recibió {type(registro).__name__}.")
            
        self.registros.append(registro)


class DataSetRegresion(DataSet):
    """
    DataSet específico para problemas de regresión.
    """
    
    def agregar_registro(self, registro: Registro):
        """
        Añade un objeto a la lista de registros si es de tipo RegistroRegresion.
        """
        # Validamos que el objeto sea estrictamente del tipo correcto
        if not isinstance(registro, RegistroRegresion):
            raise TypeError(f"Se esperaba un RegistroRegresion, pero se recibió {type(registro).__name__}.")
            
        self.registros.append(registro)