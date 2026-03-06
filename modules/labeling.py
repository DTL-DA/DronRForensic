"""
Módulo de Etiquetado de Datos
Gestiona etiquetas de marca, tipo y huella RF de drones para los espectrogramas.
"""

import numpy as np
import json
import io
import zipfile
import csv


# Marcas y tipos de drones conocidos
MARCAS_DRONES = [
    "DJI", "Parrot", "Autel", "Skydio", "Yuneec",
    "Holy Stone", "Hubsan", "Syma", "FIMI", "Potensic",
    "Eachine", "MJX", "WLtoys", "Cheerson", "JJRC",
    "Snaptain", "Ruko", "Tello", "FPV Custom", "Desconocido"
]

TIPOS_DRONES = [
    "Multirotor (Quadcopter)",
    "Multirotor (Hexacopter)",
    "Multirotor (Octocopter)",
    "Ala Fija",
    "Híbrido VTOL",
    "FPV Racing",
    "Nano/Mini",
    "Comercial/Industrial",
    "Militar/Táctico",
    "Desconocido"
]


class GestorEtiquetas:
    """Gestiona el etiquetado de espectrogramas de señales RF de drones."""

    def __init__(self):
        self.etiquetas = {}  # {nombre_archivo: {marca, tipo, huella_id, notas}}

    def etiquetar(self, nombre_archivo, marca, tipo, huella_id="", notas=""):
        self.etiquetas[nombre_archivo] = {
            'marca': marca,
            'tipo': tipo,
            'huella_id': huella_id,
            'notas': notas
        }

    def obtener_etiqueta(self, nombre_archivo):
        return self.etiquetas.get(nombre_archivo, None)

    def eliminar_etiqueta(self, nombre_archivo):
        if nombre_archivo in self.etiquetas:
            del self.etiquetas[nombre_archivo]

    def total_etiquetados(self):
        return len(self.etiquetas)

    def obtener_clases_marca(self):
        """Retorna las marcas únicas presentes en las etiquetas."""
        return sorted(set(e['marca'] for e in self.etiquetas.values()))

    def obtener_clases_tipo(self):
        """Retorna los tipos únicos presentes en las etiquetas."""
        return sorted(set(e['tipo'] for e in self.etiquetas.values()))

    def exportar_json(self):
        """Exporta etiquetas a formato JSON."""
        return json.dumps(self.etiquetas, indent=2, ensure_ascii=False)

    def importar_json(self, json_str):
        """Importa etiquetas desde JSON."""
        self.etiquetas.update(json.loads(json_str))

    def exportar_csv(self):
        """Exporta etiquetas a CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['archivo', 'marca', 'tipo', 'huella_id', 'notas'])
        for nombre, datos in sorted(self.etiquetas.items()):
            writer.writerow([
                nombre, datos['marca'], datos['tipo'],
                datos.get('huella_id', ''), datos.get('notas', '')
            ])
        return output.getvalue()

    def importar_csv(self, csv_str):
        """Importa etiquetas desde CSV."""
        reader = csv.DictReader(io.StringIO(csv_str))
        for row in reader:
            self.etiquetar(
                row['archivo'], row['marca'], row['tipo'],
                row.get('huella_id', ''), row.get('notas', '')
            )


def preparar_dataset(resultados, gestor_etiquetas, target='marca'):
    """
    Prepara imágenes y labels para entrenamiento CNN.
    
    Args:
        resultados: Lista de resultados de procesar_lote().
        gestor_etiquetas: Instancia de GestorEtiquetas con las etiquetas.
        target: 'marca' o 'tipo' (qué clasificar).
    
    Returns:
        X: np.array de imágenes (N, H, W, 1).
        y: np.array de labels codificados (N,).
        clases: Lista de nombres de clases.
        nombres: Lista de nombres de archivo.
    """
    imagenes = []
    labels = []
    nombres = []

    # Recopilar todas las clases
    if target == 'marca':
        todas_clases = gestor_etiquetas.obtener_clases_marca()
    else:
        todas_clases = gestor_etiquetas.obtener_clases_tipo()

    clase_a_idx = {c: i for i, c in enumerate(todas_clases)}

    for r in resultados:
        if r.get('error'):
            continue
        
        etiqueta = gestor_etiquetas.obtener_etiqueta(r['nombre'])
        if etiqueta is None:
            continue

        valor = etiqueta[target]
        if valor not in clase_a_idx:
            continue

        imagenes.append(r['imagen_cnn'])
        labels.append(clase_a_idx[valor])
        nombres.append(r['nombre'])

    if not imagenes:
        return None, None, todas_clases, nombres

    X = np.array(imagenes, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return X, y, todas_clases, nombres


def crear_zip_dataset(resultados, gestor_etiquetas, target='marca'):
    """
    Crea un archivo ZIP con el dataset listo para entrenar:
    - Carpetas por clase con las imágenes de espectrogramas
    - Archivo labels.csv con las etiquetas
    - Archivo metadata.json con información del dataset
    """
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        etiquetas_csv = gestor_etiquetas.exportar_csv()
        zf.writestr('labels.csv', etiquetas_csv)

        conteo_clases = {}

        for r in resultados:
            if r.get('error'):
                continue
            
            etiqueta = gestor_etiquetas.obtener_etiqueta(r['nombre'])
            if etiqueta is None:
                continue

            clase = etiqueta[target].replace('/', '_').replace(' ', '_')
            nombre_base = r['nombre'].rsplit('.', 1)[0]

            # Guardar imagen como .npy
            ruta = f"dataset/{clase}/{nombre_base}.npy"
            img_bytes = io.BytesIO()
            np.save(img_bytes, r['imagen_cnn'])
            zf.writestr(ruta, img_bytes.getvalue())

            conteo_clases[clase] = conteo_clases.get(clase, 0) + 1

        # Metadata
        metadata = {
            'total_muestras': sum(conteo_clases.values()),
            'clases': conteo_clases,
            'num_clases': len(conteo_clases),
            'target': target,
            'imagen_shape': [128, 128, 1]
        }
        zf.writestr('metadata.json', json.dumps(metadata, indent=2, ensure_ascii=False))

    buffer.seek(0)
    return buffer
