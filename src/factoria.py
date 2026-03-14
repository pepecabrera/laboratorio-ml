import csv
from .dataset import DataSetClasificacion, DataSetRegresion
from .registro import RegistroClasificacion, RegistroRegresion

# Importamos openpyxl para Excel. 
# Nota: Debes instalarlo previamente con 'pip install openpyxl'
try:
    import openpyxl
except ImportError:
    openpyxl = None


class FactoriaCSV:
    """Clase factoría para crear DataSets a partir de archivos CSV."""

    @staticmethod
    def crear_dataset_clasificacion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetClasificacion:
        dataset = DataSetClasificacion()
        
        with open(ruta_fichero, mode='r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            
            # 1. Leer cabeceras
            cabeceras = next(reader)
            
            # Reorganizamos las cabeceras para asegurar que el objetivo queda al final
            # y así la función set_cabeceras (que ignora el último elemento) funcione bien.
            obj_header = cabeceras.pop(indice_objetivo)
            cabeceras.append(obj_header)
            dataset.set_cabeceras(cabeceras)
            
            # 2. Leer los datos
            for fila in reader:
                if not fila or not any(fila):  # Ignorar líneas vacías
                    continue
                    
                # Extraemos la etiqueta objetivo (str) y los atributos (float)
                objetivo = str(fila.pop(indice_objetivo)).strip()
                atributos = [float(x.strip()) for x in fila]
                
                # Creamos el registro y lo añadimos al DataSet
                registro = RegistroClasificacion(atributos, objetivo)
                dataset.agregar_registro(registro)
                
        return dataset

    @staticmethod
    def crear_dataset_regresion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetRegresion:
        dataset = DataSetRegresion()
        
        with open(ruta_fichero, mode='r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            
            cabeceras = next(reader)
            obj_header = cabeceras.pop(indice_objetivo)
            cabeceras.append(obj_header)
            dataset.set_cabeceras(cabeceras)
            
            for fila in reader:
                if not fila or not any(fila):
                    continue
                    
                # En regresión, el objetivo es un valor numérico (float)
                objetivo = float(fila.pop(indice_objetivo).strip())
                atributos = [float(x.strip()) for x in fila]
                
                registro = RegistroRegresion(atributos, objetivo)
                dataset.agregar_registro(registro)
                
        return dataset


class FactoriaXLS:
    """Clase factoría para crear DataSets a partir de archivos Excel."""

    @staticmethod
    def _comprobar_libreria():
        if openpyxl is None:
            raise ImportError("La librería 'openpyxl' es obligatoria para leer Excel. Ejecuta 'pip install openpyxl'.")

    @staticmethod
    def crear_dataset_clasificacion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetClasificacion:
        FactoriaXLS._comprobar_libreria()
        dataset = DataSetClasificacion()
        
        # Cargamos el libro de Excel (data_only=True para ignorar fórmulas y coger el valor final)
        wb = openpyxl.load_workbook(ruta_fichero, data_only=True)
        sheet = wb.active
        
        # Extraemos todas las filas
        filas = list(sheet.iter_rows(values_only=True))
        if not filas:
            return dataset
            
        # 1. Leer cabeceras
        cabeceras = list(filas[0])
        obj_header = cabeceras.pop(indice_objetivo)
        cabeceras.append(obj_header)
        # Nos aseguramos de que las cabeceras sean strings
        dataset.set_cabeceras([str(c) for c in cabeceras])
        
        # 2. Leer los datos
        for fila in filas[1:]:
            if not any(fila): 
                continue
                
            fila_lista = list(fila)
            objetivo = str(fila_lista.pop(indice_objetivo)).strip()
            # Los valores en Excel pueden venir como enteros o nulos, los forzamos a float
            atributos = [float(x) for x in fila_lista if x is not None]
            
            registro = RegistroClasificacion(atributos, objetivo)
            dataset.agregar_registro(registro)
            
        return dataset

    @staticmethod
    def crear_dataset_regresion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetRegresion:
        FactoriaXLS._comprobar_libreria()
        dataset = DataSetRegresion()
        
        wb = openpyxl.load_workbook(ruta_fichero, data_only=True)
        sheet = wb.active
        
        filas = list(sheet.iter_rows(values_only=True))
        if not filas:
            return dataset
            
        cabeceras = list(filas[0])
        obj_header = cabeceras.pop(indice_objetivo)
        cabeceras.append(obj_header)
        dataset.set_cabeceras([str(c) for c in cabeceras])
        
        for fila in filas[1:]:
            if not any(fila):
                continue
                
            fila_lista = list(fila)
            # En regresión, el objetivo es float
            objetivo = float(fila_lista.pop(indice_objetivo))
            atributos = [float(x) for x in fila_lista if x is not None]
            
            registro = RegistroRegresion(atributos, objetivo)
            dataset.agregar_registro(registro)
            
        return dataset
    
# Importamos xlrd para soportar archivos Excel antiguos (.xls)
try:
    import xlrd
except ImportError:
    xlrd = None


class FactoriaOldXLS:
    """Clase factoría para crear DataSets a partir de archivos Excel antiguos (.xls)."""

    @staticmethod
    def _comprobar_libreria():
        if xlrd is None:
            raise ImportError("La librería 'xlrd' es obligatoria para leer .xls. Ejecuta 'pip install xlrd'.")

    @staticmethod
    def crear_dataset_clasificacion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetClasificacion:
        FactoriaOldXLS._comprobar_libreria()
        dataset = DataSetClasificacion()

        wb = xlrd.open_workbook(ruta_fichero)
        sheet = wb.sheet_by_index(0)

        if sheet.nrows == 0:
            return dataset

        # 1. Cabeceras
        cabeceras = sheet.row_values(0)
        obj_header = cabeceras.pop(indice_objetivo)
        cabeceras.append(obj_header)
        dataset.set_cabeceras([str(c) for c in cabeceras])

        # 2. Datos
        for i in range(1, sheet.nrows):
            fila = sheet.row_values(i)

            if not any(fila):
                continue

            fila_lista = list(fila)
            objetivo = str(fila_lista.pop(indice_objetivo)).strip()
            atributos = [float(x) for x in fila_lista]

            registro = RegistroClasificacion(atributos, objetivo)
            dataset.agregar_registro(registro)

        return dataset

    @staticmethod
    def crear_dataset_regresion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetRegresion:
        FactoriaOldXLS._comprobar_libreria()
        dataset = DataSetRegresion()

        wb = xlrd.open_workbook(ruta_fichero)
        sheet = wb.sheet_by_index(0)

        if sheet.nrows == 0:
            return dataset

        # 1. Cabeceras
        cabeceras = sheet.row_values(0)
        obj_header = cabeceras.pop(indice_objetivo)
        cabeceras.append(obj_header)
        dataset.set_cabeceras([str(c) for c in cabeceras])

        # 2. Datos
        for i in range(1, sheet.nrows):
            fila = sheet.row_values(i)

            if not any(fila):
                continue

            fila_lista = list(fila)
            objetivo = float(fila_lista.pop(indice_objetivo))
            atributos = [float(x) for x in fila_lista]

            registro = RegistroRegresion(atributos, objetivo)
            dataset.agregar_registro(registro)

        return dataset