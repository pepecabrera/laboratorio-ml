import math
from typing import List, Union

class Registro:
    """
    Clase base que representa un conjunto de puntos en un espacio n-dimensional.
    """
    
    def __init__(self, atributos: List[float], objetivo=None):
        """
        Constructor de la clase base.
        Se inicializa el objetivo a None por defecto para la clase base.
        """
        self.atributos = atributos
        self.objetivo = objetivo

    def _validar_dimensiones(self, otra_lista: List[float], nombre_lista: str = "la otra lista"):
        """Método auxiliar privado para validar que las dimensiones coincidan."""
        if len(self.atributos) != len(otra_lista):
            raise ValueError(
                f"Las dimensiones no coinciden. Este registro tiene dimensión {len(self.atributos)}, "
                f"pero {nombre_lista} tiene dimensión {len(otra_lista)}."
            )

    def distancia_euclidea(self, otro: 'Registro') -> float:
        """Calcula la distancia Euclidiana entre este registro y otro."""
        self._validar_dimensiones(otro.atributos, "el otro registro")
        suma_cuadrados = sum((p - q) ** 2 for p, q in zip(self.atributos, otro.atributos))
        return math.sqrt(suma_cuadrados)

    def distancia_manhattan(self, otro: 'Registro') -> float:
        """Calcula la distancia Manhattan entre este registro y otro."""
        self._validar_dimensiones(otro.atributos, "el otro registro")
        return sum(abs(p - q) for p, q in zip(self.atributos, otro.atributos))

    def distancia_ponderada(self, otro: 'Registro', pesos: List[float]) -> float:
        """Calcula la distancia ponderada utilizando una lista de pesos."""
        self._validar_dimensiones(otro.atributos, "el otro registro")
        self._validar_dimensiones(pesos, "la lista de pesos")
        
        suma_ponderada = sum(w * ((p - q) ** 2) for p, q, w in zip(self.atributos, otro.atributos, pesos))
        return math.sqrt(suma_ponderada)

    def calcula_distancia(self, otro: 'Registro', tipo: str = "euclídea", pesos: List[float] = None) -> float:
        """Calcula la distancia según el tipo especificado."""
        tipo = tipo.lower()
        if tipo == "manhattan":
            return self.distancia_manhattan(otro)
        elif tipo == "ponderada":
            if pesos is None:
                raise ValueError("Para la distancia ponderada, se debe proporcionar una lista de pesos.")
            return self.distancia_ponderada(otro, pesos)
        else:
            return self.distancia_euclidea(otro)

    def normalizar(self, minimos: List[float], maximos: List[float]) -> 'Registro':
        """
        Devuelve un nuevo objeto de la MISMA CLASE que lo invoca, 
        con los valores de los atributos normalizados.
        """
        self._validar_dimensiones(minimos, "la lista de mínimos")
        self._validar_dimensiones(maximos, "la lista de máximos")
        
        atributos_norm = []
        for x, min_val, max_val in zip(self.atributos, minimos, maximos):
            rango = max_val - min_val
            if rango == 0.0:
                atributos_norm.append(0.0) 
            else:
                x_prima = (x - min_val) / rango
                atributos_norm.append(x_prima)
                
        # Polimorfismo: self.__class__ creará un Registro, RegistroClasificacion o RegistroRegresion
        # según qué objeto haya llamado a la función.
        nuevo_registro = self.__class__(atributos_norm, self.objetivo)
        return nuevo_registro

    def k_vecinos(self, lista_registros: List['Registro'], k: int, tipo: str = "euclídea", pesos: List[float] = None) -> List[int]:
        """
        Devuelve una lista con los índices de los 'k' registros más cercanos.
        Omite al propio objeto si está en la lista.
        """
        distancias = []
        for indice, registro in enumerate(lista_registros):
            dist = self.calcula_distancia(registro, tipo, pesos)
            if registro is self:
                continue
            distancias.append((dist, indice))
            
        distancias.sort(key=lambda x: x[0])
        indices_k_vecinos = [item[1] for item in distancias[:k]]
        return indices_k_vecinos
    
    def estandarizar(self, medias: List[float], desviaciones: List[float]) -> 'Registro':
        """
        Devuelve un nuevo registro con los atributos estandarizados (Z-score).
        """
        nuevos_atributos = []
        for i, valor in enumerate(self.atributos):
            # Evitamos la división por cero si la desviación típica es 0 (todos los valores iguales)
            if desviaciones[i] == 0:
                nuevos_atributos.append(0.0)
            else:
                nuevos_atributos.append((valor - medias[i]) / desviaciones[i])
                
        # Instanciamos un nuevo registro de la misma clase (RegistroClasificacion o Regresion)
        return self.__class__(nuevos_atributos, self.objetivo)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({[round(x, 4) for x in self.atributos]})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(atributos={self.atributos}, objetivo={self.objetivo})"


# ==========================================
# ETAPA 1: CLASES HIJAS (NUEVO)
# ==========================================

class RegistroClasificacion(Registro):
    """
    Registro para problemas de clasificación, donde el objetivo es una categoría (etiqueta).
    """
    def __init__(self, atributos: List[float], objetivo: str):
        # Invocamos al constructor de la clase padre
        super().__init__(atributos, objetivo)
        
    def __repr__(self) -> str:
        # Redefinición para mostrar explícitamente que es una etiqueta (string)
        return f"RegistroClasificacion(atributos={self.atributos}, etiqueta='{self.objetivo}')"


class RegistroRegresion(Registro):
    """
    Registro para problemas de regresión, donde el objetivo es un valor numérico real.
    """
    def __init__(self, atributos: List[float], objetivo: float):
        # Invocamos al constructor de la clase padre
        super().__init__(atributos, objetivo)

    def __repr__(self) -> str:
        # Redefinición para mostrar explícitamente que es un valor numérico
        return f"RegistroRegresion(atributos={self.atributos}, valor={self.objetivo})"