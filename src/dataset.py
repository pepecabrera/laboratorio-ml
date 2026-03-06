import math
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
    
    def calcular_medias_desviaciones(self) -> Tuple[List[float], List[float]]:
        """
        Calcula y devuelve las medias y desviaciones típicas de cada columna de atributos.
        """
        if not self.registros:
            raise ValueError("No hay registros en el DataSet para calcular medias y desviaciones.")

        matriz_atributos = [reg.atributos for reg in self.registros]
        columnas = list(zip(*matriz_atributos))
        
        medias = []
        desviaciones = []
        
        for col in columnas:
            # Media
            media = sum(col) / len(col)
            medias.append(media)
            
            # Desviación típica (raíz cuadrada de la varianza)
            varianza = sum((x - media) ** 2 for x in col) / len(col)
            desviaciones.append(math.sqrt(varianza))
            
        return medias, desviaciones
    
    def eliminar_atributos(self, indices_a_eliminar: List[int]) -> 'DataSet':
        """
        Devuelve un nuevo DataSet eliminando las columnas especificadas por los índices.
        Comprueba que los índices estén dentro del rango lógico de los atributos.
        """
        if not self.registros:
            return self.__class__() # Devuelve un DataSet vacío del mismo tipo

        num_atributos = len(self.registros[0].atributos)
        
        # 1. Validar que los índices son correctos
        for idx in indices_a_eliminar:
            if idx < 0 or idx >= num_atributos:
                raise ValueError(f"Índice {idx} fuera de rango. Debe estar entre 0 y {num_atributos - 1}.")

        # 2. Instanciar el nuevo DataSet
        nuevo_dataset = self.__class__()
        
        # 3. Mantener las cabeceras actualizadas (si existen)
        if hasattr(self, 'cabeceras') and self.cabeceras:
            # Filtramos las cabeceras de los atributos
            cabeceras_atributos = [cab for i, cab in enumerate(self.cabeceras[:-1]) if i not in indices_a_eliminar]
            # Añadimos la cabecera del objetivo al final
            cabeceras_atributos.append(self.cabeceras[-1])
            nuevo_dataset.set_cabeceras(cabeceras_atributos)

        # 4. Crear los nuevos registros recortados
        for reg in self.registros:
            # Nos quedamos solo con los valores cuyo índice NO esté en la lista negra
            nuevos_atributos = [val for i, val in enumerate(reg.atributos) if i not in indices_a_eliminar]
            
            # Instanciamos un nuevo registro de la misma clase (Clasificacion o Regresion)
            nuevo_registro = reg.__class__(nuevos_atributos, reg.objetivo)
            nuevo_dataset.agregar_registro(nuevo_registro)

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