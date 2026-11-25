"""Validación de las opciones que se cargan en la interfaz.

Convierte los textos de los campos de la ventana en los parámetros que reciben las funciones
de cada técnica. No usa Tkinter, así que se puede probar sin abrir ventanas.

Autor: Ángel A. Pérez
"""

from pathlib import Path

TECNICAS: tuple[str, ...] = ("MOKE", "SEM", "TEM")

# Opciones de TEM: las mismas que la sección CONFIGURACIÓN de scripts/analizar_tem.py
FILTROS_TEM: tuple[str, ...] = ("auto", "gaussian", "median", "bilateral", "wiener")
UMBRALES_TEM: tuple[str, ...] = ("auto", "otsu", "gmm", "triangle", "adaptive")
DIFRACCION_TEM: dict[str, bool | None] = {
    "Detectar sola": None,
    "Analizar siempre como difracción": True,
    "Nunca": False,
}
FORMATOS_TEM: tuple[str, ...] = (".dm3", ".mrc", ".tif", ".tiff", ".png", ".jpg")

# Modos de SEM
MODO_SEM_IMAGEN = "Imagen y ROI"
MODO_SEM_ROIS = "ROIs guardadas"
MODOS_SEM: tuple[str, ...] = (MODO_SEM_IMAGEN, MODO_SEM_ROIS)

# Valores por defecto de cada campo (los mismos que en los módulos y scripts)
POR_DEFECTO: dict[str, dict[str, str]] = {
    "MOKE": {"carpeta": "", "fondo": "", "escala": "1", "unidad": "px"},
    "SEM": {"modo": MODO_SEM_IMAGEN, "imagen": "", "carpeta": "", "sufijo": "",
            "intensidad_pozos": "2"},
    "TEM": {"ruta": "", "filtro": "auto", "umbral": "auto", "fases": "", "fondo": "",
            "difraccion": "Detectar sola", "ancho_instrumental": "0"},
}


def validar_opciones(tecnica: str, campos: dict[str, str]) -> dict:
    """Valida los campos de una técnica y los convierte en parámetros.

    Args:
        tecnica: 'MOKE', 'SEM' o 'TEM'.
        campos: Texto de cada campo de la ventana (las claves de POR_DEFECTO[tecnica]).

    Returns:
        Diccionario de parámetros listo para la función de la técnica:
        - MOKE: carpeta (Path), fondo (Path), escala (float), unidad (str).
        - SEM: modo; con 'Imagen y ROI', imagen (Path) y sufijo (str o None); con
          'ROIs guardadas', carpeta (Path); siempre intensidad_pozos (float).
        - TEM: ruta (Path), filtro, umbral, fases (list[int] o None), fondo (int o None),
          difraccion (True, False o None) y ancho_instrumental (float).

    Raises:
        ValueError: Con un mensaje para el usuario si algún campo es inválido.
    """
    if tecnica == "MOKE":
        return _validar_moke(campos)
    if tecnica == "SEM":
        return _validar_sem(campos)
    if tecnica == "TEM":
        return _validar_tem(campos)
    raise ValueError(f"Técnica desconocida: {tecnica!r}. Opciones: {', '.join(TECNICAS)}")


def _validar_moke(campos: dict[str, str]) -> dict:
    """Valida las opciones de MOKE."""
    carpeta = _carpeta(campos["carpeta"], "la carpeta de imágenes")
    fondo = _archivo(campos["fondo"], "el archivo de fondo")
    escala = _numero(campos["escala"], "Escala", positivo=True)
    unidad = campos["unidad"].strip()
    if not unidad:
        raise ValueError("Indicá una unidad (por ejemplo px, nm o um).")
    return {"carpeta": carpeta, "fondo": fondo, "escala": escala, "unidad": unidad}


def _validar_sem(campos: dict[str, str]) -> dict:
    """Valida las opciones de SEM."""
    modo = campos["modo"]
    intensidad = _numero(campos["intensidad_pozos"], "Intensidad de pozos", minimo=0.0)
    if modo == MODO_SEM_IMAGEN:
        imagen = _archivo(campos["imagen"], "la imagen SEM")
        if imagen.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"La imagen SEM tiene que ser .tif: {imagen.name}")
        sufijo = campos["sufijo"].strip() or None
        if sufijo is not None and any(c in sufijo for c in '/\\:*?"<>|'):
            raise ValueError(f"El sufijo no puede tener los caracteres / \\ : * ? \" < > |: {sufijo!r}")
        return {"modo": modo, "imagen": imagen, "sufijo": sufijo, "intensidad_pozos": intensidad}
    if modo == MODO_SEM_ROIS:
        carpeta = _carpeta(campos["carpeta"], "la carpeta de ROIs")
        return {"modo": modo, "carpeta": carpeta, "intensidad_pozos": intensidad}
    raise ValueError(f"Modo de SEM desconocido: {modo!r}")


def _validar_tem(campos: dict[str, str]) -> dict:
    """Valida las opciones de TEM."""
    texto = campos["ruta"].strip()
    if not texto:
        raise ValueError("Elegí una imagen o una carpeta TEM.")
    ruta = Path(texto).expanduser()
    if not ruta.exists():
        raise ValueError(f"No existe: {ruta}")
    if ruta.is_file() and ruta.suffix.lower() not in FORMATOS_TEM:
        raise ValueError(f"Formato no soportado: {ruta.name}. Válidos: {', '.join(FORMATOS_TEM)}")
    if campos["filtro"] not in FILTROS_TEM:
        raise ValueError(f"Filtro desconocido: {campos['filtro']!r}")
    if campos["umbral"] not in UMBRALES_TEM:
        raise ValueError(f"Umbral desconocido: {campos['umbral']!r}")
    if campos["difraccion"] not in DIFRACCION_TEM:
        raise ValueError(f"Opción de difracción desconocida: {campos['difraccion']!r}")
    return {
        "ruta": ruta,
        "filtro": campos["filtro"],
        "umbral": campos["umbral"],
        "fases": _lista_fases(campos["fases"]),
        "fondo": _entero_opcional(campos["fondo"], "Fondo"),
        "difraccion": DIFRACCION_TEM[campos["difraccion"]],
        "ancho_instrumental": _numero(campos["ancho_instrumental"], "Ancho instrumental",
                                      minimo=0.0),
    }


def _carpeta(texto: str, que: str) -> Path:
    """Convierte el texto en una carpeta existente."""
    if not texto.strip():
        raise ValueError(f"Elegí {que}.")
    ruta = Path(texto.strip()).expanduser()
    if not ruta.is_dir():
        raise ValueError(f"No existe la carpeta: {ruta}")
    return ruta


def _archivo(texto: str, que: str) -> Path:
    """Convierte el texto en un archivo existente."""
    if not texto.strip():
        raise ValueError(f"Elegí {que}.")
    ruta = Path(texto.strip()).expanduser()
    if not ruta.is_file():
        raise ValueError(f"No existe el archivo: {ruta}")
    return ruta


def _numero(texto: str, nombre: str, positivo: bool = False,
            minimo: float | None = None) -> float:
    """Convierte el texto en un número (acepta coma decimal)."""
    try:
        valor = float(texto.strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"{nombre} tiene que ser un número: {texto!r}") from None
    if valor != valor or valor in (float("inf"), float("-inf")):
        raise ValueError(f"{nombre} tiene que ser un número finito: {texto!r}")
    if positivo and valor <= 0:
        raise ValueError(f"{nombre} tiene que ser mayor que 0: {texto!r}")
    if minimo is not None and valor < minimo:
        raise ValueError(f"{nombre} no puede ser menor que {minimo:g}: {texto!r}")
    return valor


def _entero_opcional(texto: str, nombre: str) -> int | None:
    """Convierte el texto en un entero ≥ 0, o None si está vacío."""
    texto = texto.strip()
    if not texto:
        return None
    if not texto.isdecimal():
        raise ValueError(f"{nombre} tiene que ser una etiqueta entera (0, 1, 2…) o quedar vacío: "
                         f"{texto!r}")
    return int(texto)


def _lista_fases(texto: str) -> list[int] | None:
    """Convierte '1,2' en [1, 2]; vacío es None (todas menos el fondo)."""
    texto = texto.strip()
    if not texto:
        return None
    partes = [p.strip() for p in texto.replace(";", ",").split(",")]
    if not all(p.isdecimal() for p in partes):
        raise ValueError(f"Fases tiene que ser una lista de etiquetas enteras, como 1,2: {texto!r}")
    return [int(p) for p in partes]
