"""
Módulo de Procesamiento de Señales RF - STFT y Espectrogramas
Convierte señales RF crudas en espectrogramas usando la Transformada de Fourier
de Ventana Corta (Short-Time Fourier Transform).
"""

import numpy as np
from scipy.signal import stft, windows
from scipy.io import wavfile, loadmat
import io
import json


def cargar_senal(archivo_bytes, nombre_archivo):
    """
    Carga señales RF desde diferentes formatos de archivo.
    Soporta: CSV, NPY, WAV, JSON, TXT, BIN (IQ raw).
    
    Returns:
        senal (np.ndarray): Señal como array 1D o 2D (para IQ).
        fs (float): Frecuencia de muestreo (estimada si no está disponible).
    """
    ext = nombre_archivo.lower().rsplit('.', 1)[-1] if '.' in nombre_archivo else ''
    fs = None
    senal = None

    if ext == 'csv':
        contenido = archivo_bytes.decode('utf-8')
        lineas = contenido.strip().split('\n')
        
        # Detectar si la primera línea es header
        primera = lineas[0].strip()
        tiene_header = False
        try:
            float(primera.split(',')[0].strip())
        except ValueError:
            tiene_header = True

        if tiene_header:
            # Buscar 'fs' o 'sample_rate' en el header
            headers = [h.strip().lower() for h in primera.split(',')]
            if 'fs' in headers or 'sample_rate' in headers:
                idx = headers.index('fs') if 'fs' in headers else headers.index('sample_rate')
                try:
                    fs = float(lineas[1].split(',')[idx].strip())
                except (ValueError, IndexError):
                    pass
            lineas = lineas[1:]

        datos = []
        for linea in lineas:
            valores = [float(v.strip()) for v in linea.split(',') if v.strip()]
            datos.append(valores)
        
        datos = np.array(datos)
        if datos.ndim == 2 and datos.shape[1] == 1:
            senal = datos.flatten()
        elif datos.ndim == 2 and datos.shape[1] == 2:
            # I/Q data
            senal = datos[:, 0] + 1j * datos[:, 1]
        elif datos.ndim == 2 and datos.shape[1] >= 3:
            # Múltiples señales, tomar primera columna de datos
            senal = datos[:, -1] if fs else datos[:, 0]
        else:
            senal = datos.flatten()

    elif ext == 'npy':
        senal = np.load(io.BytesIO(archivo_bytes))
        if senal.ndim == 2 and senal.shape[1] == 2:
            senal = senal[:, 0] + 1j * senal[:, 1]
        elif senal.ndim > 1:
            senal = senal.flatten()

    elif ext == 'npz':
        data = np.load(io.BytesIO(archivo_bytes))
        keys = list(data.keys())
        senal = data[keys[0]]
        if 'fs' in data:
            fs = float(data['fs'])
        if senal.ndim > 1:
            senal = senal.flatten()

    elif ext == 'wav':
        fs_wav, senal = wavfile.read(io.BytesIO(archivo_bytes))
        fs = float(fs_wav)
        if senal.ndim > 1:
            senal = senal[:, 0]
        senal = senal.astype(np.float64)

    elif ext == 'json':
        contenido = json.loads(archivo_bytes.decode('utf-8'))
        if isinstance(contenido, dict):
            if 'signal' in contenido:
                senal = np.array(contenido['signal'])
            elif 'data' in contenido:
                senal = np.array(contenido['data'])
            elif 'iq' in contenido:
                iq = np.array(contenido['iq'])
                if iq.ndim == 2:
                    senal = iq[:, 0] + 1j * iq[:, 1]
                else:
                    senal = iq
            if 'fs' in contenido:
                fs = float(contenido['fs'])
            elif 'sample_rate' in contenido:
                fs = float(contenido['sample_rate'])
        elif isinstance(contenido, list):
            senal = np.array(contenido)

    elif ext == 'txt':
        contenido = archivo_bytes.decode('utf-8')
        valores = []
        for linea in contenido.strip().split('\n'):
            linea = linea.strip()
            if linea and not linea.startswith('#'):
                partes = linea.replace('\t', ',').split(',')
                for p in partes:
                    p = p.strip()
                    if p:
                        try:
                            valores.append(float(p))
                        except ValueError:
                            pass
        senal = np.array(valores)

    elif ext == 'mat':
        mat_data = loadmat(io.BytesIO(archivo_bytes))
        # Buscar la variable de señal (ignorar claves internas de MATLAB)
        claves = [k for k in mat_data.keys() if not k.startswith('__')]
        # Priorizar claves comunes de señal
        clave_senal = None
        for nombre in ('signal', 'data', 'rf', 'iq', 'x', 'y', 's', 'senal'):
            for k in claves:
                if k.lower() == nombre:
                    clave_senal = k
                    break
            if clave_senal:
                break
        if clave_senal is None and claves:
            clave_senal = claves[0]
        if clave_senal is None:
            raise ValueError("No se encontró ninguna variable de señal en el archivo .mat")
        arr = np.array(mat_data[clave_senal]).squeeze()
        # Detectar I/Q en columnas
        if arr.ndim == 2 and arr.shape[1] == 2:
            senal = arr[:, 0] + 1j * arr[:, 1]
        elif arr.ndim == 2:
            senal = arr.flatten()
        else:
            senal = arr.flatten()
        # Buscar frecuencia de muestreo
        for fs_key in ('fs', 'Fs', 'sample_rate', 'SampleRate', 'samplerate'):
            if fs_key in mat_data:
                fs = float(np.array(mat_data[fs_key]).flatten()[0])
                break

    elif ext in ('bin', 'raw', 'iq'):
        # Raw IQ: pares float32 intercalados (I, Q, I, Q, ...)
        raw = np.frombuffer(archivo_bytes, dtype=np.float32)
        if len(raw) % 2 == 0:
            senal = raw[0::2] + 1j * raw[1::2]
        else:
            senal = raw

    else:
        # Intentar como CSV genérico
        try:
            contenido = archivo_bytes.decode('utf-8')
            valores = []
            for linea in contenido.strip().split('\n'):
                for v in linea.strip().split(','):
                    v = v.strip()
                    if v:
                        try:
                            valores.append(float(v))
                        except ValueError:
                            pass
            senal = np.array(valores)
        except Exception:
            raise ValueError(f"Formato no soportado: .{ext}")

    if senal is None or len(senal) == 0:
        raise ValueError("No se pudo extraer la señal del archivo.")

    # Frecuencia de muestreo por defecto para señales RF de drones
    if fs is None:
        fs = 2.4e6  # 2.4 MHz (banda común de drones)

    return senal, fs


def generar_espectrograma(senal, fs, nperseg=256, noverlap=None, ventana='hann', nfft=None, max_samples=5_000_000):
    """
    Genera un espectrograma usando STFT (Short-Time Fourier Transform).
    
    Args:
        senal: Señal de entrada (real o compleja).
        fs: Frecuencia de muestreo (Hz).
        nperseg: Número de muestras por segmento de la ventana.
        noverlap: Número de muestras de solapamiento (default: nperseg // 2).
        ventana: Tipo de ventana ('hann', 'hamming', 'blackman', 'kaiser', etc.).
        nfft: Puntos de la FFT (default: nperseg).
        max_samples: Máximo de muestras a procesar (evita desbordamiento de memoria).
    
    Returns:
        f: Frecuencias del espectrograma.
        t: Tiempos del espectrograma.
        Sxx: Magnitud del espectrograma en dB.
        Sxx_linear: Magnitud lineal (para la imagen CNN).
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    # Limitar longitud de la señal para evitar desbordamiento de memoria
    if max_samples and len(senal) > max_samples:
        senal = senal[:max_samples]

    # Normalizar señal
    senal_real = np.real(senal) if np.iscomplexobj(senal) else senal
    if np.std(senal_real) > 0:
        senal_real = (senal_real - np.mean(senal_real)) / np.std(senal_real)

    f, t, Zxx = stft(senal_real, fs=fs, window=ventana, nperseg=nperseg,
                      noverlap=noverlap, nfft=nfft)

    Sxx_linear = np.abs(Zxx)
    # Convertir a dB con floor para evitar log(0)
    Sxx_db = 10 * np.log10(Sxx_linear ** 2 + 1e-12)

    return f, t, Sxx_db, Sxx_linear


def espectrograma_a_imagen(Sxx, tamano=(128, 128)):
    """
    Convierte la matriz del espectrograma a una imagen normalizada
    lista para alimentar una CNN.
    
    Args:
        Sxx: Matriz del espectrograma (frecuencia x tiempo).
        tamano: Tupla (alto, ancho) de la imagen de salida.
    
    Returns:
        imagen: Array normalizado [0, 1] de forma (alto, ancho, 1).
    """
    from PIL import Image

    # Normalizar a [0, 255]
    sxx_min = Sxx.min()
    sxx_max = Sxx.max()
    if sxx_max - sxx_min > 0:
        img_data = ((Sxx - sxx_min) / (sxx_max - sxx_min) * 255).astype(np.uint8)
    else:
        img_data = np.zeros_like(Sxx, dtype=np.uint8)

    img = Image.fromarray(img_data)
    img = img.resize((tamano[1], tamano[0]), Image.BILINEAR)

    # Normalizar a [0, 1] para CNN
    imagen = np.array(img, dtype=np.float32) / 255.0
    imagen = imagen.reshape(tamano[0], tamano[1], 1)

    return imagen


def procesar_lote(archivos, fs_override=None, nperseg=256, noverlap=None,
                  ventana='hann', tamano_img=(128, 128)):
    """
    Procesa un lote de archivos de señales RF y genera espectrogramas.
    
    Returns:
        resultados: Lista de dicts con {nombre, senal, fs, f, t, Sxx_db, imagen_cnn}.
    """
    resultados = []
    for nombre, datos_bytes in archivos:
        try:
            senal, fs = cargar_senal(datos_bytes, nombre)
            if fs_override:
                fs = fs_override

            f, t, Sxx_db, Sxx_linear = generar_espectrograma(
                senal, fs, nperseg=nperseg, noverlap=noverlap, ventana=ventana
            )

            imagen = espectrograma_a_imagen(Sxx_db, tamano=tamano_img)

            resultados.append({
                'nombre': nombre,
                'senal': senal,
                'fs': fs,
                'f': f,
                't': t,
                'Sxx_db': Sxx_db,
                'imagen_cnn': imagen,
                'error': None
            })
        except Exception as e:
            resultados.append({
                'nombre': nombre,
                'error': str(e)
            })

    return resultados
