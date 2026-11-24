"""Detección de dominios MOKE con una imagen de fondo.

El main original se puede usar sin cambios:
    python main.py moke background.png evolucion_dominio --escala 0.5 --unidad um

background.png puede estar en la carpeta analizada o indicarse mediante una ruta.
La diferencia se normaliza por el fondo: (imagen - fondo) / fondo.
"""

import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as P
from scipy import ndimage as ndi
from skimage import filters, io, measure, morphology, segmentation, transform
from skimage.util import img_as_float

def leer_gris(ruta):
    return img_as_float(io.imread(ruta, as_gray=True))

def detectar_dominio(img, semilla=None, referencia=None, escala=4, sigma=12,
                     margen=80, suavizado=24, orden=2):
    """Detecta un dominio oscuro; referencia es una imagen sin el dominio.

    Las distancias se expresan en píxeles de la imagen original. Aquí escala
    es el factor de reducción para acelerar el cálculo, no la calibración
    física que recibe analizadorMOKE.
    """
    img = np.asarray(img, dtype=float)
    if img.ndim != 2:
        raise ValueError("La imagen debe ser una matriz 2D en escala de grises.")
    if not np.isfinite(img).all():
        raise ValueError("La imagen contiene valores no finitos.")

    chica = transform.rescale(img, 1 / escala, anti_aliasing=True)
    disco = morphology.disk(suavizado // escala)

    ref = None
    if referencia is not None:
        referencia = np.asarray(referencia, dtype=float)
        if referencia.shape != img.shape:
            raise ValueError(
                f"La imagen tiene tamaño {img.shape}, pero el fondo tiene "
                f"{referencia.shape}. Deben coincidir."
            )
        if not np.isfinite(referencia).all():
            raise ValueError("El fondo contiene valores no finitos.")
        ref = transform.rescale(referencia, 1 / escala, anti_aliasing=True)

    # 1) La apertura se obtiene del fondo, cuando está disponible.
    # No se busca en la resta, que tiene valores próximos a cero o negativos.
    imagen_apertura = ref if ref is not None else chica
    apertura = imagen_apertura > filters.threshold_otsu(imagen_apertura)
    etiquetas = measure.label(apertura)
    areas = np.bincount(etiquetas.ravel())[1:]
    if areas.size == 0:
        raise ValueError("No se encontró la apertura iluminada.")
    apertura = ndi.binary_fill_holes(etiquetas == areas.argmax() + 1)
    interior = morphology.erosion(apertura, morphology.disk(margen // escala))
    if not interior.any():
        raise ValueError("El margen elimina toda la apertura. Reducí margen.")

    # 2) El mismo suavizado para la imagen y la referencia.
    suave = filters.gaussian(chica, sigma / escala)

    # 3) Fondo medido o estimado mediante un polinomio.
    filas, cols = np.mgrid[: chica.shape[0], : chica.shape[1]]
    if ref is not None:
        fondo_img = filters.gaussian(ref, sigma / escala)
    else:
        V = P.polyvander2d(
            filas / chica.shape[0] - 0.5,
            cols / chica.shape[1] - 0.5,
            [orden, orden],
        )
        Vi, zi = V[interior], suave[interior]
        peso = np.ones(zi.size)
        for _ in range(15):
            coef, *_ = np.linalg.lstsq(Vi * peso[:, None], zi * peso, rcond=None)
            peso = np.where(zi < Vi @ coef, 0.2, 1.0)
        fondo_img = V @ coef

    if (fondo_img[interior] <= 0).any():
        raise ValueError("El fondo debe ser positivo dentro de la apertura.")

    # Resta en float: los píxeles negativos NO se recortan ni desbordan.
    resta = suave - fondo_img
    plana = np.zeros_like(suave)
    plana[interior] = resta[interior] / fondo_img[interior]
    # Fondo ~0; dominio negativo. -0.12 significa un 12 % más oscuro.

    # 4) Zona inicial.
    if semilla is not None:
        zona = np.hypot(
            filas - semilla[0] / escala, cols - semilla[1] / escala
        ) <= suavizado / escala
        zona &= interior
    else:
        oscuro = interior & (plana < filters.threshold_otsu(plana[interior]))
        etiquetas = segmentation.clear_border(
            measure.label(morphology.opening(oscuro, disco)), mask=interior
        )
        zona = etiquetas == np.bincount(
            etiquetas.ravel(), minlength=2
        )[1:].argmax() + 1
    if not zona.any():
        raise ValueError("No se encontró el dominio. Probá indicar una semilla dentro del dominio.")

    # 5) Umbral entre los niveles del dominio y del fondo.
    fondo = np.percentile(plana[interior], 95)
    for _ in range(3):
        nivel = np.median(plana[zona])
        for _ in range(3):
            umbral = (fondo + nivel) / 2
            valores_fondo = plana[interior & (plana > umbral)]
            if valores_fondo.size == 0:
                raise ValueError("No se pudo separar el dominio del fondo.")
            fondo = np.median(valores_fondo)
        umbral = (fondo + nivel) / 2
        etiquetas = measure.label(
            morphology.opening(interior & (plana < umbral), disco)
        )
        cuenta = np.bincount(etiquetas[zona], minlength=2)[1:]
        if cuenta.sum() == 0:
            raise ValueError("La semilla no cae en una zona oscura. Probá con otro punto.")
        zona = ndi.binary_fill_holes(etiquetas == cuenta.argmax() + 1)
    dominio = morphology.closing(zona, disco)
    ruido = 1.4826 * np.median(
        np.abs(plana[interior & (plana > umbral)] - fondo)
    )

    if abs(nivel - fondo) < 3 * ruido:
        warnings.warn("El contraste es bajo comparado con el ruido: revisá el resultado.")
    if (dominio & ~morphology.erosion(interior, morphology.disk(2))).any():
        warnings.warn("El dominio llega al borde de la apertura: el contorno puede estar cortado.")

    # 6) Contorno y círculo ajustado.
    contorno = max(measure.find_contours(dominio.astype(float), 0.5), key=len)
    contorno = (contorno + 0.5) * escala - 0.5
    f, c = contorno[:, 0], contorno[:, 1]
    A = np.column_stack([2 * f, 2 * c, np.ones_like(f)])
    (f0, c0, k), *_ = np.linalg.lstsq(A, f**2 + c**2, rcond=None)

    return {
        "contorno": contorno,
        "centro": (float(f0), float(c0)),
        "radio": float(np.sqrt(k + f0**2 + c0**2)),
        "area": int(dominio.sum()) * escala**2,
        "contraste": float(nivel - fondo),
        "ruido": float(ruido),
        "mascara": transform.resize(dominio, img.shape, order=0),
    }

def mostrar_dominio(img, res, ax=None):
    """Muestra la imagen original con el contorno y el círculo ajustado."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    claro = img[img > filters.threshold_otsu(img)]
    if claro.size == 0:
        claro = img.ravel()
    ax.imshow(
        img, cmap="gray", vmin=np.percentile(claro, 1),
        vmax=np.percentile(claro, 99),
    )
    ax.plot(res["contorno"][:, 1], res["contorno"][:, 0], "r-", lw=1, label="contorno")
    f0, c0 = res["centro"]
    ax.add_patch(plt.Circle(
        (c0, f0), res["radio"], fill=False, color="cyan", ls="--",
        label="círculo ajustado",
    ))
    ax.plot(c0, f0, "c+")
    ax.set_title(f"centro = ({f0:.0f}, {c0:.0f})   radio = {res['radio']:.0f} px")
    ax.legend(loc="upper right")
    ax.axis("off")
    return ax

def analizadorMOKE(background, carpeta, salida="resultados", escala=1.0, unidad="px",
                   graficar=False):
    """Analiza PNG numerados, excluyendo el fondo y usándolo como referencia.

    background puede ser un nombre dentro de carpeta o una ruta al archivo.
    Las imágenes y el fondo deben tener igual tamaño y estar alineados.
    La última imagen numerada debe contener el dominio de mayor tamaño.
    escala es el tamaño físico de un píxel, expresado en unidad.
    """
    carpeta, salida = Path(carpeta).expanduser(), Path(salida).expanduser()
    if not carpeta.is_dir():
        raise NotADirectoryError(f"No existe la carpeta de imágenes: {carpeta}")

    # Primero se busca el nombre dentro de la carpeta de imágenes.
    archivo_fondo = Path(background).expanduser()
    if not archivo_fondo.is_absolute():
        candidato = carpeta / archivo_fondo
        if candidato.is_file():
            archivo_fondo = candidato
    if not archivo_fondo.is_file():
        raise FileNotFoundError(f"No se encontró la imagen de fondo: {background}")
    archivo_fondo = archivo_fondo.resolve()
    fondo = leer_gris(archivo_fondo)
    # Excluir el fondo ANTES de int(f.stem): puede llamarse background.png.
    archivos = [
        f for f in carpeta.iterdir()
        if f.is_file() and (f.suffix.lower() == ".png" or f.suffix.lower() == ".jpg")
        and f.resolve() != archivo_fondo
    ]
    print("Archivos encontrados:", [f.name for f in archivos])
    if not archivos:
        raise ValueError("No hay imágenes PNG o JPG para analizar, aparte del fondo.")
    if any(not f.stem.isdecimal() for f in archivos):
        
        raise ValueError("Las imágenes a analizar deben llamarse 0.png, 1.png, 2.png, etc.")
    archivos.sort(key=lambda f: int(f.stem))
    salida.mkdir(parents=True, exist_ok=True)

    # La referencia también se usa para encontrar la semilla inicial.
    ultima = leer_gris(archivos[-1])
    semilla = detectar_dominio(ultima, referencia=fondo)["centro"]

    filas = [["archivo", "fila_centro_px", "col_centro_px", "radio_px",
              f"radio_{unidad}", f"area_{unidad}2", "contraste", "dudoso"]]
    for f in archivos:
        img = leer_gris(f)
        try:
            res = detectar_dominio(img, semilla=semilla, referencia=fondo)
        except ValueError as error:
            print(f.name, "-> no se detectó dominio:", error)
            continue

        ax = mostrar_dominio(img, res)

        # después
        ax.figure.savefig(salida / f"dominio_{f.stem}.png", dpi=150, bbox_inches="tight")
        if graficar:
            plt.show()
        plt.close(ax.figure)

        dudoso = abs(res["contraste"]) < 3 * res["ruido"]
        filas.append([
            f.name, round(res["centro"][0], 1), round(res["centro"][1], 1),
            round(res["radio"], 1), round(escala * res["radio"], 1),
            round(escala**2 * res["area"], 1), round(res["contraste"], 3), dudoso,
        ])

    with open(salida / "medidas_moke.csv", "w", newline="", encoding="utf-8") as tabla:
        csv.writer(tabla).writerows(filas)
    return filas
