import math
from typing import List

class Registro:
    """
    Representa un conjunto de puntos en un espacio n-dimensional para 
    algoritmos de Machine Learning (Clasificación Supervisada).
    """
    
    def __init__(self, atributos: List[float]):
        """
        Constructor que acepta un vector de números flotantes.
        """
        self.atributos = atributos

    def _validar_dimensiones(self, otra_lista: List[float], nombre_lista: str = "la otra lista"):
        """
        Método auxiliar privado para validar que las dimensiones coincidan
        antes de realizar operaciones matemáticas.
        """
        if len(self.atributos) != len(otra_lista):
            raise ValueError(
                f"Las dimensiones no coinciden. Este registro tiene dimensión {len(self.atributos)}, "
                f"pero {nombre_lista} tiene dimensión {len(otra_lista)}."
            )

    def distancia_euclidea(self, otro: 'Registro') -> float:
        """
        Calcula la distancia Euclidiana entre este registro y otro.
        """
        self._validar_dimensiones(otro.atributos, "el otro registro")
        suma_cuadrados = sum((p - q) ** 2 for p, q in zip(self.atributos, otro.atributos))
        return math.sqrt(suma_cuadrados)

    def distancia_manhattan(self, otro: 'Registro') -> float:
        """
        Calcula la distancia Manhattan entre este registro y otro.
        """
        self._validar_dimensiones(otro.atributos, "el otro registro")
        return sum(abs(p - q) for p, q in zip(self.atributos, otro.atributos))

    def distancia_ponderada(self, otro: 'Registro', pesos: List[float]) -> float:
        """
        Calcula la distancia ponderada utilizando una lista de pesos.
        """
        self._validar_dimensiones(otro.atributos, "el otro registro")
        self._validar_dimensiones(pesos, "la lista de pesos")
        
        suma_ponderada = sum(w * ((p - q) ** 2) for p, q, w in zip(self.atributos, otro.atributos, pesos))
        return math.sqrt(suma_ponderada)

    def calcula_distancia(self, otro: 'Registro', tipo: str = "euclídea", pesos: List[float] = None) -> float:
        """
        Calcula la distancia según el tipo especificado ("euclídea", "manhattan" o "ponderada").
        Por defecto es "euclídea".
        """
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
        Devuelve un nuevo Registro con los valores de los atributos normalizados.
        Asigna 0.0 si el rango (max - min) es 0 para evitar divisiones por cero.
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
                
        return Registro(atributos_norm)

    def k_vecinos(self, lista_registros: List['Registro'], k: int, tipo: str = "euclídea", pesos: List[float] = None) -> List[int]:
        """
        Devuelve una lista con los índices de los 'k' registros más cercanos al objeto que invoca.
        """
        distancias = []
        for indice, registro in enumerate(lista_registros):
            dist = self.calcula_distancia(registro, tipo, pesos)
            distancias.append((dist, indice))
            
        # Ordenamos por distancia (el primer elemento de la tupla)
        distancias.sort(key=lambda x: x[0])
        
        # Extraemos solo los índices de los primeros k elementos
        indices_k_vecinos = [item[1] for item in distancias[:k]]
        return indices_k_vecinos

    def __str__(self) -> str:
        """Representación amigable para imprimir (ej. con print)."""
        return f"Registro({[round(x, 4) for x in self.atributos]})"

    def __repr__(self) -> str:
        """Representación formal y detallada del objeto para depuración."""
        return f"Registro(atributos={self.atributos})"