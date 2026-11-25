"""Conexión de cada técnica con la interfaz.

Una función por análisis. Las que terminan en `correr_` corren en el hilo de fondo: no tocan
Tkinter ni abren ventanas, y devuelven un `Resultado` con la carpeta de salida y las figuras
que la ventana tiene que abrir (solo cuando se analizó una sola imagen). La carga de la imagen
SEM y el selector de ROI son interactivos y corren en el hilo principal (`preparar_imagen_sem`,
`elegir_y_guardar_roi`).

Los módulos se importan dentro de cada función para que la ventana abra rápido.

Autor: Ángel A. Pérez
"""

import gc
import time
from collections.abc import Callable
from pathlib import Path

from .ejecucion import Resultado, pngs_nuevos

# Raíz del proyecto (este archivo está en <raíz>/src/interfaz/tecnicas.py)
RAIZ: Path = Path(__file__).resolve().parents[2]

# Carpetas de salida: las mismas de siempre
SALIDA_MOKE: Path = RAIZ / "data" / "moke"
SALIDA_SEM: Path = RAIZ / "data" / "sem" / "roi"   


def correr_moke(parametros: dict) -> Resultado:
    """Analiza una carpeta MOKE con `analizadorMOKE` (siempre carpeta: no abre figuras).

    Args:
        parametros: De `validar_opciones('MOKE', ...)`: carpeta, fondo, escala y unidad.

    Returns:
        Resultado con la carpeta `resultados/`.
    """
    from moke import analizadorMOKE

    print(f"MOKE: {parametros['carpeta']} (fondo: {parametros['fondo'].name})")
    filas = analizadorMOKE(parametros["fondo"], parametros["carpeta"], salida=SALIDA_MOKE,
                           escala=parametros["escala"], unidad=parametros["unidad"])
    analizadas = len(filas) - 1  # la primera fila es el encabezado
    print(f"Imágenes con dominio detectado: {analizadas}")
    dudosas = [fila[0] for fila in filas[1:] if fila[-1]]
    if dudosas:
        print(f"⚠ Contraste bajo frente al ruido (dudoso) en: {', '.join(dudosas)}")
    print(f"Tabla: {SALIDA_MOKE / 'medidas_moke.csv'}")
    return Resultado(salida=SALIDA_MOKE)


def correr_tem(parametros: dict, pedir_dato: Callable[[str, float], float]) -> Resultado:
    """Analiza una imagen o una carpeta TEM, como `scripts/analizar_tem.py`.

    Las figuras se guardan sin mostrarse; con una sola imagen se devuelven para abrirlas.
    Si a un patrón de difracción le faltan metadatos, se piden con `pedir_dato`.

    Args:
        parametros: De `validar_opciones('TEM', ...)`.
        pedir_dato: Función (texto, valor por defecto) -> número que muestra un diálogo.

    Returns:
        Resultado con la carpeta de figuras y CSV (data/samples/).
    """
    from tem import analizar_ruta

    inicio = time.time()
    if parametros["fondo"] is not None and parametros["fases"] is not None:
        print("Se indicó un fondo: se miden todas las demás fases y 'Fases' se ignora.")
    salida = analizar_ruta(
        parametros["ruta"], filtro=parametros["filtro"], umbral=parametros["umbral"],
        fases=parametros["fases"], fondo=parametros["fondo"],
        difraccion=parametros["difraccion"],
        ancho_instrumental=parametros["ancho_instrumental"],
        mostrar=False, pedir_dato=pedir_dato,
    )
    if salida["n_errores"]:
        print(f"⚠ {salida['n_errores']} de {salida['n_imagenes']} imágenes no se pudieron analizar.")
    figuras = pngs_nuevos(salida["figuras"], inicio) if parametros["ruta"].is_file() else []
    return Resultado(salida=salida["figuras"], figuras=figuras)


def preparar_imagen_sem(imagen: Path):
    """Carga una imagen SEM, lee su metadata y quita la barra de datos (como sem_carga.py).

    Args:
        imagen: Ruta a la imagen .tif.

    Returns:
        (imagen sin barra de datos, metadata).
    """
    from sem.carga import cargar_imagen, leer_metadata, recortar_barra_datos

    imagen_completa = cargar_imagen(imagen)
    metadata = leer_metadata(imagen)
    recortada = recortar_barra_datos(imagen_completa, metadata)
    print(f"Imagen: {imagen.name}")
    print(f"  Fabricante: {metadata.get('fabricante', 'desconocido')}")
    print(f"  Shape original: {imagen_completa.shape} -> sin barra de datos: {recortada.shape}")
    if "tamano_pixel_m" in metadata:
        print(f"  Tamaño de píxel: {metadata['tamano_pixel_m'] * 1e9:.3g} nm")
    return recortada, metadata


def elegir_y_guardar_roi(imagen_sem: Path, recortada, metadata: dict,
                         sufijo: str | None) -> Path | None:
    """Abre el selector de ROI y guarda la ROI elegida. Corre en el hilo principal.

    matplotlib usa 'Agg' en la interfaz (el análisis corre en otro hilo); mientras el
    selector está abierto se cambia a 'TkAgg', que abre ventanas, y después se vuelve.

    Args:
        imagen_sem: Ruta de la imagen original (para el nombre de la carpeta).
        recortada: Imagen sin barra de datos, de `preparar_imagen_sem`.
        metadata: Metadata de `preparar_imagen_sem`.
        sufijo: Sufijo de la ROI (ej. '_1'); None = el primer '_N' libre.

    Returns:
        Ruta del .npy de la ROI guardada, o None si se cerró el selector sin elegir.
    """
    import matplotlib.pyplot as plt

    from sem.carga import guardar_roi, seleccionar_roi

    print("Seleccioná el ROI y presioná Enter (o cerrá la ventana) para confirmar.")
    plt.switch_backend("TkAgg")
    try:
        coordenadas = seleccionar_roi(recortada, titulo=imagen_sem.name)
    finally:
        plt.switch_backend("Agg")
        # Los objetos Tk de la ventana del selector se liberan acá, en el hilo principal; si
        # los liberara el recolector desde el hilo de fondo, Tkinter daría RuntimeError.
        gc.collect()
    if coordenadas is None:
        print("⚠️ No se seleccionó ningún ROI; no se guardó nada.")
        return None
    print(f"  ROI (x1, y1, x2, y2): {coordenadas}")
    carpeta = guardar_roi(recortada, imagen_sem.name, coordenadas, SALIDA_SEM, metadata,
                          sufijo=sufijo)
    print(f"✅ ROI guardada en: {carpeta}")
    return carpeta / f"roi_{carpeta.name}.npy"


def correr_sem_roi(ruta_roi: Path, intensidad_pozos: float) -> Resultado:
    """Segmenta una ROI recién guardada y devuelve su figura para abrirla.

    Args:
        ruta_roi: .npy de la ROI.
        intensidad_pozos: Peso del término de pozos del filtro de choque.

    Returns:
        Resultado con la carpeta de la ROI y la figura de la segmentación.
    """
    from sem.bse.resultados import segmentar_y_guardar

    carpeta = segmentar_y_guardar(ruta_roi, intensidad_pozos=intensidad_pozos)
    return Resultado(salida=carpeta, figuras=[carpeta / f"segmentacion_{ruta_roi.stem}.png"])


def buscar_rois(carpeta: Path) -> list[Path]:
    """ROIs guardadas (`roi_*.npy`) dentro de una carpeta y sus subcarpetas, ordenadas."""
    return sorted(Path(carpeta).rglob("roi_*.npy"))


def correr_sem_rois(carpeta: Path, intensidad_pozos: float) -> Resultado:
    """Segmenta todas las ROIs guardadas de una carpeta; si una falla, sigue con la siguiente.

    Args:
        carpeta: Carpeta donde buscar `roi_*.npy` (también en subcarpetas).
        intensidad_pozos: Peso del término de pozos del filtro de choque.

    Returns:
        Resultado con la carpeta elegida (cada segmentación queda junto a su ROI).

    Raises:
        ValueError: Si no hay ninguna ROI en la carpeta.
    """
    from sem.bse.resultados import segmentar_y_guardar

    rois = buscar_rois(carpeta)
    if not rois:
        raise ValueError(f"No hay ROIs (roi_*.npy) en {carpeta} ni en sus subcarpetas.")
    print(f"Segmentando {len(rois)} ROIs de {carpeta} (sin abrir ventanas)...")
    errores = 0
    for i, roi in enumerate(rois, start=1):
        print(f"\n[{i}/{len(rois)}] {roi.relative_to(carpeta)}")
        try:
            segmentar_y_guardar(roi, intensidad_pozos=intensidad_pozos)
        except Exception as error:  # una ROI mala no frena el resto
            print(f"  → ERROR: {error}")
            errores += 1
    if errores:
        print(f"\n⚠ {errores} de {len(rois)} ROIs no se pudieron segmentar.")
    print("Cada segmentación quedó junto a su ROI (segmentacion_<nombre>.npy, .csv y .png).")
    return Resultado(salida=Path(carpeta))
