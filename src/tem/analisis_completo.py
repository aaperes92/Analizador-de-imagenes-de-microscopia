"""Análisis TEM completo de una imagen o de una carpeta.

Corre las cuatro etapas del módulo TEM (preprocesamiento, segmentación, morfología y análisis
de fases) o, si la imagen es un patrón de difracción, la Etapa 4.5. Guarda las figuras de cada
etapa y un resumen en CSV (los patrones de difracción van en un CSV aparte,
<nombre>_resumen_difraccion.csv).

Es la lógica de `scripts/analizar_tem.py`, con las opciones como parámetros en lugar de
variables globales, para poder usarla también desde la interfaz gráfica. El script sigue
funcionando igual: solo llama a `analizar_ruta` con su sección CONFIGURACIÓN.

Autor: Daniel Romero (daniel.romero@ib.edu.ar)
"""

from __future__ import annotations

import contextlib
import csv
import io
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .diffraction import (
    is_diffraction_pattern,
    pipeline_diffraction,
    read_camera_metadata,
)
from .fraction import phase_names, phase_title
from .loader import DEFAULT_OUTPUT_DIR, SUPPORTED_FORMATS, load_tem_image, pipeline_tem
from .morphology import pipeline_morphology
from .phase_analysis import pipeline_phase_analysis
from .segmentation import pipeline_segmentation


def es_difraccion(path: Path, difraccion: bool | None = None) -> tuple[bool, object]:
    """Decide si la imagen se analiza como patrón de difracción.

    Args:
        path: Ruta a la imagen.
        difraccion: None = detectar, True = siempre, False = nunca.

    Returns:
        (True si es difracción, imagen cargada o None).
    """
    if difraccion is False:
        return False, None
    with contextlib.redirect_stdout(io.StringIO()):
        image = load_tem_image(path)
        if difraccion:
            return True, image
        meta = read_camera_metadata(path, interactive=False)
        return is_diffraction_pattern(image, meta, verbose=False), image


def analizar(
    path: Path,
    mostrar: bool,
    filtro: str = "auto",
    umbral: str = "auto",
    fases: list[int] | None = None,
    fondo: int | None = None,
    difraccion: bool | None = None,
    ancho_instrumental: float = 0.0,
    carpeta_salida: str | Path | None = None,
    mensajes: bool | None = None,
    interactivo: bool | None = None,
    pedir_dato: Callable[[str, float], float] | None = None,
) -> dict:
    """Corre las cuatro etapas (o la Etapa 4.5) sobre una imagen.

    Args:
        path: Ruta a la imagen.
        mostrar: Si es True, muestra las figuras en pantalla.
        filtro: Etapa 1 — 'auto', 'gaussian', 'median', 'bilateral' o 'wiener'.
        umbral: Etapa 2 — 'auto', 'otsu', 'gmm', 'triangle' o 'adaptive'.
        fases: Etapa 3 — fases a medir; None = todas menos la mayoritaria.
        fondo: Etapas 3 y 4 — etiqueta del fondo; None = la fase mayoritaria. Si se indica,
            la Etapa 3 mide todas las demás y `fases` se ignora.
        difraccion: None = detectar, True = siempre difracción, False = nunca.
        ancho_instrumental: Ancho instrumental (px) que se resta en Scherrer.
        carpeta_salida: Carpeta de figuras; None = data/samples/.
        mensajes: Si se imprimen los mensajes de cada etapa; None = igual que `mostrar`.
        interactivo: Si en difracción se piden los metadatos que faltan y se corrige el
            centro a mano; None = igual que `mostrar`.
        pedir_dato: Función (texto, valor por defecto) -> número para pedir los metadatos sin
            teclado; además, si se pasa, no se abre la ventana del centro (que necesita el
            hilo principal).

    Returns:
        Diccionario con los resultados de cada etapa ('etapa1' a 'etapa4') o con
        'difraccion' (Etapa 4.5), y el tiempo total en segundos ('segundos').
    """
    if mensajes is None:
        mensajes = mostrar
    if interactivo is None:
        interactivo = mostrar
    inicio = time.perf_counter()
    salida = sys.stdout if mensajes else io.StringIO()
    es_dif, imagen = es_difraccion(path, difraccion)
    if es_dif:
        with contextlib.redirect_stdout(salida):
            resultado = pipeline_diffraction(
                path, image=imagen, force=True, interactive=interactivo,
                instrumental_fwhm_px=ancho_instrumental,
                output_dir=carpeta_salida, show=mostrar,
                pedir_dato=pedir_dato,
                corregir_centro=interactivo and pedir_dato is None)
        return {"difraccion": resultado,
                "segundos": time.perf_counter() - inicio}
    with contextlib.redirect_stdout(salida):
        print("\n" + "=" * 70)
        print(f"ETAPA 1 — Carga y preprocesamiento: {path.name}")
        print("=" * 70)
        etapa1 = pipeline_tem(path, method=filtro,
                              output_dir=carpeta_salida, show=mostrar)

        print("\n" + "=" * 70)
        print("ETAPA 2 — Segmentación de fases")
        print("=" * 70)
        etapa2 = pipeline_segmentation(etapa1, method=umbral,
                                       output_dir=carpeta_salida,
                                       show=mostrar)

        print("\n" + "=" * 70)
        print("ETAPA 3 — Análisis morfológico")
        print("=" * 70)
        if fondo is not None:
            # Todas las fases menos el fondo indicado
            fases = list(phase_names(etapa2["mask"], fondo)[1].values())
        etapa3 = pipeline_morphology(etapa2, phases=fases,
                                     output_dir=carpeta_salida,
                                     show=mostrar)

        print("\n" + "=" * 70)
        print("ETAPA 4 — Análisis de fases")
        print("=" * 70)
        etapa4 = pipeline_phase_analysis(etapa3, background=fondo,
                                         output_dir=carpeta_salida,
                                         show=mostrar)
    return {"etapa1": etapa1, "etapa2": etapa2, "etapa3": etapa3,
            "etapa4": etapa4, "segundos": time.perf_counter() - inicio}


def filas_resumen(path: Path, r: dict) -> list[dict]:
    """Arma una fila de resumen por fase analizada.

    Args:
        path: Ruta a la imagen.
        r: Resultado de analizar().

    Returns:
        Lista de diccionarios (una fila por fase de la Etapa 3). La
        columna 'fase' usa el nombre de la Etapa 4 (Fase 1, Fase 2...)
        y 'etiqueta' la etiqueta de la máscara.
    """
    e2, e3, e4 = r["etapa2"], r["etapa3"], r["etapa4"]
    # Etiqueta de la máscara → nombre de la Etapa 4
    nombres = {v["etiqueta_mascara"]: k
               for k, v in e4["area_fractions"].items()
               if k.startswith("fase_")}
    filas = []
    for fase, size in e3["size_stats"].items():
        etiqueta = int(fase.split("_")[1])
        f4 = nombres.get(etiqueta)
        orient = e3["orientation_stats"][fase]
        ajuste = size["ajuste_lognormal"]
        fila = {
            "archivo": path.name,
            "filtro": r["etapa1"]["method_used"],
            "n_fases": e2["n_phases"],
            "umbral": e2["threshold_method"],
            "fase": phase_title(f4) if f4 else fase,
            "etiqueta": etiqueta,
            # % sobre la imagen completa (con fondo e interfaces)
            "area_pct_total": round(
                e4["area_fractions"][f4]["area_pct_con_fondo"] if f4
                else 100 * e2["stats"][fase]["area_px"]
                / e2["mask"].size, 2),
            "n_regiones": size["n_regiones"],
            "area_media_px": _redondear(size["area_media_px"]),
            "area_mediana_px": _redondear(size["area_mediana_px"]),
            "lognormal_p": (_redondear(ajuste["ks_pvalue"], 3)
                            if ajuste else ""),
            "forma": e3["shape_stats"][fase]["forma_predominante"] or "",
            "indice_S": _redondear(orient["indice_orden_S"], 2),
            "orientacion": orient["clasificacion"] or "",
        }
        if f4:
            dens = e4["number_density"][f4]
            hop = e4["clustering"][f4]
            inh = e4["inhomogeneities"][f4]
            fila.update({
                "vol_pct_est_total": round(
                    e4["volume_fractions"][f4]["vol_frac_pct_con_fondo"], 2),
                "densidad_1000px2": _redondear(
                    dens["densidad_por_1000px2"], 4),
                "clark_evans_R": _redondear(dens["indice_clark_evans"], 2),
                "rdf": e4["rdf"][f4]["clasificacion"],
                "distribucion_densidad": e4["density_map"][f4][
                    "clasificacion"],
                "hopkins_H": _redondear(hop["hopkins_statistic"], 2),
                "hopkins": hop["hopkins_clasificacion"],
                "grupos_dbscan": (hop["n_clusters_dbscan"]
                                  if hop["n_clusters_dbscan"] is not None
                                  else ""),
                "inhomogeneidades": inh["n_alta"] + inh["n_baja"],
                "gradientes_significativos": len(e4["spatial_gradients"][
                    f4]["gradientes_significativos"]),
            })
        fila["segundos"] = round(r["segundos"], 1)
        filas.append(fila)
    return filas


def _redondear(valor: float | None,
               decimales: int = 0) -> int | float | str:
    """Redondea un valor (entero si decimales es 0) o devuelve '' si es
    None."""
    if valor is None:
        return ""
    return int(round(valor)) if decimales == 0 else round(valor, decimales)


def fila_difraccion(path: Path, r: dict) -> dict:
    """Arma la fila de resumen de un patrón de difracción.

    Args:
        path: Ruta a la imagen.
        r: Resultado de analizar() con 'difraccion'.

    Returns:
        Diccionario con archivo, tipo, calibración, centro, cantidad de
        picos o spots, los d (Å), el tamaño de Scherrer medio y, para
        spots, la simetría, la fracción indexada y la red.
    """
    d = r["difraccion"]
    res = d["results"]
    meta = d["metadata"]
    fila = {
        "archivo": path.name,
        "tipo": d["pattern_type"],
        "calibracion_inv_A_px": round(meta["calibracion_inv_angs_px"], 7),
        "fuente_calibracion": meta["fuente"],
        "centro_y": "" if d["center"] is None else round(d["center"][0], 1),
        "centro_x": "" if d["center"] is None else round(d["center"][1], 1),
        "n_picos_o_spots": "",
        "d_A": "",
        "scherrer_medio_nm": "",
        "simetria_friedel": "",
        "fraccion_indexada": "",
        "red_a_A": "",
        "red_b_A": "",
        "red_gamma_deg": "",
    }
    if "anillos" in res or "perfil_1d" in res:
        picos = (res["anillos"] if "anillos" in res
                 else res["perfil_1d"]["picos"])
        cristal = (res.get("cristalito") if "anillos" in res
                   else res["perfil_1d"]["cristalito"])
        fila["n_picos_o_spots"] = picos["n_anillos"]
        fila["d_A"] = " ".join(f"{a['d_angs']:.3f}"
                               for a in picos["anillos"])
        if cristal and cristal["D_medio_nm"] is not None:
            fila["scherrer_medio_nm"] = round(cristal["D_medio_nm"], 1)
    else:
        spots, geo = res["spots"], res["geometria"]
        fila["n_picos_o_spots"] = spots["n_spots"]
        fila["d_A"] = " ".join(f"{f['d_angs']:.3f}"
                               for f in geo["familias"]
                               if f["d_angs"] and f["n_spots"] >= 3)
        fila["simetria_friedel"] = _redondear(spots["indice_simetria"], 2)
        fila["fraccion_indexada"] = _redondear(geo["fraccion_indexada"], 2)
        red = geo["red"]
        if red is not None and red["a_angs"] is not None:
            fila["red_a_A"] = round(red["a_angs"], 3)
            fila["red_b_A"] = round(red["b_angs"], 3)
            fila["red_gamma_deg"] = round(red["gamma_deg"], 1)
    fila["segundos"] = round(r["segundos"], 1)
    return fila


def imprimir_difraccion(filas: list[dict]) -> None:
    """Imprime el resumen de los patrones de difracción.

    Args:
        filas: Filas de fila_difraccion().
    """
    print(f"\n{'archivo (difracción)':30} {'tipo':>9} {'n':>4} "
          f"{'Scherrer':>9} {'calibración':>12}  d (Å)")
    for f in filas:
        nm = f["scherrer_medio_nm"]
        print(f"{f['archivo'][:30]:30} {f['tipo']:>9} "
              f"{f['n_picos_o_spots']:>4} "
              f"{(f'{nm} nm' if nm != '' else '-'):>9} "
              f"{f['fuente_calibracion']:>12}  {f['d_A'][:60]}")


def imprimir_tabla(filas: list[dict]) -> None:
    """Imprime el resumen como tabla.

    Args:
        filas: Filas de filas_resumen().
    """
    print(f"\n{'archivo':36} {'umbral':>8} {'fase':7} {'área%':>6} "
          f"{'n':>4} {'media px':>9} {'forma':>10} {'S':>5} "
          f"{'orientación':>12} {'RDF':>12} {'Hopkins':>12}")
    for f in filas:
        print(f"{f['archivo']:36} {f['umbral']:>8} {f['fase']:7} "
              f"{f['area_pct_total']:>6} {f['n_regiones']:>4} "
              f"{str(f['area_media_px']):>9} {f['forma']:>10} "
              f"{str(f['indice_S']):>5} {f['orientacion']:>12} "
              f"{f.get('rdf', '-'):>12} {f.get('hopkins', '-'):>12}")


def guardar_csv(filas: list[dict], nombre: str,
                sufijo: str = "resumen_tem",
                carpeta_salida: str | Path | None = None) -> Path:
    """Guarda el resumen en un CSV (se abre bien con Excel).

    Args:
        filas: Filas de filas_resumen().
        nombre: Nombre base del archivo.
        sufijo: Final del nombre ('resumen_tem' o 'resumen_difraccion').
        carpeta_salida: Carpeta del CSV; None = data/samples/.

    Returns:
        Ruta del CSV.
    """
    carpeta = Path(carpeta_salida) if carpeta_salida else DEFAULT_OUTPUT_DIR
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"{nombre}_{sufijo}.csv"
    # utf-8-sig, ';' y coma decimal para que Excel en español lo abra
    # con los números como números
    con_coma = [
        {k: str(v).replace(".", ",") if isinstance(v, float) else v
         for k, v in fila.items()}
        for fila in filas
    ]
    with destino.open("w", newline="", encoding="utf-8-sig") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(filas[0]),
                                delimiter=";")
        writer.writeheader()
        writer.writerows(con_coma)
    return destino


def analizar_ruta(
    ruta: str | Path,
    filtro: str = "auto",
    umbral: str = "auto",
    fases: list[int] | None = None,
    fondo: int | None = None,
    difraccion: bool | None = None,
    ancho_instrumental: float = 0.0,
    mostrar: bool = True,
    carpeta_salida: str | Path | None = None,
    pedir_dato: Callable[[str, float], float] | None = None,
) -> dict:
    """Analiza una imagen o todas las imágenes de una carpeta e imprime el resumen.

    Con una imagen, si `mostrar` es True (o se pasa `pedir_dato`), imprime todos los mensajes
    de cada etapa; si `mostrar` es True, además muestra las figuras. Con una carpeta procesa todas las imágenes sin abrir ventanas, imprime el
    avance ([i/N]) y una tabla, y sigue con la siguiente si una imagen falla. En los dos casos
    guarda las figuras y el CSV de resumen (el de difracción, aparte).

    Args:
        ruta: Imagen o carpeta (.dm3, .mrc, .tif, .tiff, .png, .jpg).
        filtro, umbral, fases, fondo, difraccion, ancho_instrumental: Ver `analizar`.
        mostrar: Si se muestran las figuras en pantalla (solo con una imagen).
        carpeta_salida: Carpeta de figuras y CSV; None = data/samples/.
        pedir_dato: Función (texto, valor por defecto) -> número para pedir los metadatos de
            difracción que faltan sin usar el teclado. Si se pasa, se piden también con una
            carpeta. None = por teclado con una imagen (si `mostrar`) y valores por defecto
            con una carpeta, como siempre hizo el script.

    Returns:
        {'csv': rutas de los CSV guardados, 'figuras': carpeta de las figuras,
         'n_imagenes': imágenes encontradas, 'n_errores': imágenes que fallaron}.

    Raises:
        FileNotFoundError: Si la ruta no existe.
        ValueError: Si la carpeta no tiene imágenes soportadas.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe: {ruta}")
    carpeta_figuras = Path(carpeta_salida) if carpeta_salida else DEFAULT_OUTPUT_DIR
    opciones = dict(filtro=filtro, umbral=umbral, fases=fases, fondo=fondo,
                    difraccion=difraccion, ancho_instrumental=ancho_instrumental,
                    carpeta_salida=carpeta_salida, pedir_dato=pedir_dato)
    guardados: list[Path] = []

    if ruta.is_file():
        # Como el script: los mensajes de cada etapa se imprimen si se muestran las figuras
        # (o si los lee la interfaz, que pasa pedir_dato)
        con_interfaz = pedir_dato is not None
        resultado = analizar(ruta, mostrar=mostrar, mensajes=mostrar or con_interfaz,
                             interactivo=mostrar or con_interfaz, **opciones)
        print("\n" + "=" * 70)
        print(f"RESUMEN ({resultado['segundos']:.1f} s)")
        print("=" * 70)
        if "difraccion" in resultado:
            if resultado["difraccion"]["es_difraccion"]:
                fila = fila_difraccion(ruta, resultado)
                imprimir_difraccion([fila])
                guardados.append(guardar_csv([fila], ruta.stem, "resumen_difraccion",
                                             carpeta_salida))
                print(f"\nCSV: {guardados[-1]}")
        else:
            filas = filas_resumen(ruta, resultado)
            imprimir_tabla(filas)
            if filas:
                guardados.append(guardar_csv(filas, ruta.stem, carpeta_salida=carpeta_salida))
                print(f"\nCSV: {guardados[-1]}")
        print(f"Figuras: {carpeta_figuras}")
        return {"csv": guardados, "figuras": carpeta_figuras, "n_imagenes": 1,
                "n_errores": 0}

    imagenes = sorted(p for p in ruta.iterdir()
                      if p.suffix.lower() in SUPPORTED_FORMATS)
    if not imagenes:
        raise ValueError(f"No hay imágenes soportadas en {ruta}")
    print(f"Analizando {len(imagenes)} imágenes de {ruta} "
          "(sin abrir ventanas)...")
    filas: list[dict] = []
    filas_dif: list[dict] = []
    errores = 0
    for i, imagen in enumerate(imagenes, start=1):
        print(f"  [{i}/{len(imagenes)}] {imagen.name}", end="", flush=True)
        try:
            resultado = analizar(imagen, mostrar=False, mensajes=False,
                                 interactivo=pedir_dato is not None, **opciones)
        except Exception as error:  # una imagen mala no frena el resto
            print(f"  → ERROR: {error}")
            errores += 1
            continue
        if "difraccion" in resultado:
            filas_dif.append(fila_difraccion(imagen, resultado))
            tipo = f"difracción: {resultado['difraccion']['pattern_type']}"
        else:
            filas.extend(filas_resumen(imagen, resultado))
            tipo = "Etapas 1-4"
        print(f"  ({tipo}, {resultado['segundos']:.1f} s)")
    if filas:
        imprimir_tabla(filas)
        guardados.append(guardar_csv(filas, ruta.name, carpeta_salida=carpeta_salida))
        print(f"\nCSV: {guardados[-1]}")
    if filas_dif:
        imprimir_difraccion(filas_dif)
        guardados.append(guardar_csv(filas_dif, ruta.name, "resumen_difraccion",
                                     carpeta_salida))
        print(f"\nCSV: {guardados[-1]}")
    print(f"Figuras: {carpeta_figuras}")
    return {"csv": guardados, "figuras": carpeta_figuras, "n_imagenes": len(imagenes),
            "n_errores": errores}
