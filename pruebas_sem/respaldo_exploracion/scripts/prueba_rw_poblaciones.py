"""Random walker con semillas por población (sintética parecida a la ROI 3 y ROI 3 real).

1. Poblaciones (cantidad y gris) con el paisaje filtrado, igual que en 22_:
   actividad = gaussiana(sobel, 25 px); paisaje = log(1 + histograma 2D 128 x 128); cada fila menos
   su versión suavizada (gaussiana 6 bins); perfil = suma de filas; picos (prominencia 5 %).
2. Semillas: píxeles a menos de TOL_SEMILLA del gris de una población.
3. El resto lo decide skimage.segmentation.random_walker, siguiendo los bordes de la imagen.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap
from scipy import ndimage, signal

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA_ACTIVIDAD = 25
BINS = 128
SIGMA_FILTRO = 6
TOL_SEMILLA = 0.015
COLORES = ["#000000", "#8c564b", "#1f5fbf", "#e8d8a0", "#ff7f0e", "#2ca02c", "#9467bd"]
GRISES_REALES = np.array([0.0, 0.45, 0.58, 0.70, 0.85])
NOMBRES_REALES = ["negro", "zona gris", "fase oscura", "matriz", "partículas"]


def imagen_sintetica(lado=650, semilla=3):
    rng = np.random.default_rng(semilla)
    y, x = np.mgrid[0:lado, 0:lado] / lado
    img = np.full((lado, lado), 0.70)
    estrias = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (2, 12))
    estrias = ndimage.rotate(estrias, 35, reshape=False, mode="reflect")
    img += 0.02 * estrias / estrias.std()
    zona_patron = (x + y) < 0.75
    campo = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), 6)
    red = np.abs(campo / campo.std()) < 0.25
    verdad = np.full((lado, lado), 3)
    img[zona_patron & red] = 0.58
    verdad[zona_patron & red] = 2
    particulas = ski.data.binary_blobs(lado, blob_size_fraction=0.01, volume_fraction=0.02, rng=semilla)
    img[particulas] = 0.85
    verdad[particulas] = 4
    d = (x + y) - 1.55
    franja, gris = np.abs(d) < 0.03, d >= 0.03
    img[franja], verdad[franja] = 0.0, 0
    img[gris], verdad[gris] = 0.45, 1
    img = ndimage.gaussian_filter(img, 1.5) + rng.normal(0, 0.005, img.shape)
    return np.clip(img, 0, 1), verdad


def grises_de_poblaciones(img):
    gris = (img - img.min()) / (img.max() - img.min())
    actividad = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACTIVIDAD)
    actividad = (actividad - actividad.min()) / (actividad.max() - actividad.min())
    conteos, _, _ = np.histogram2d(actividad.ravel(), gris.ravel(), bins=BINS, range=((0, 1), (0, 1)))
    paisaje = np.log1p(conteos)
    filtrado = np.maximum(paisaje - ndimage.gaussian_filter1d(paisaje, SIGMA_FILTRO, axis=1, mode="constant"), 0)
    perfil = filtrado.sum(axis=0)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], perfil, [0.0])), prominence=0.05 * perfil.max(),
                                 distance=3)
    centros = (np.arange(BINS) + 0.5) / BINS
    return img.min() + centros[picos - 1] * (img.max() - img.min())


sintetica, verdad = imagen_sintetica()
roi3 = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy"))

for nombre, img, ver in (("sintetica", sintetica, verdad), ("ROI3", roi3, None)):
    grises = grises_de_poblaciones(img)
    n = len(grises)
    distancias = np.abs(img[..., None] - grises[None, None, :])
    semillas = np.argmin(distancias, axis=-1) + 1
    semillas[distancias.min(axis=-1) > TOL_SEMILLA] = 0
    datos = (img - img.min()) / (img.max() - img.min())
    resultado = ski.segmentation.random_walker(datos, semillas, mode="cg_j") - 1
    fracciones = [np.mean(resultado == c) for c in range(n)]
    print(f"{nombre}: {n} poblaciones en {np.round(grises, 3)}; sin semilla {np.mean(semillas == 0):.1%}; "
          f"áreas finales {np.round(fracciones, 3)}")
    if ver is not None:
        real_de = np.argmin(np.abs(grises[:, None] - GRISES_REALES[None, :]), axis=1)
        prediccion = real_de[resultado]
        for r, nom in enumerate(NOMBRES_REALES):
            m = ver == r
            print(f"   {nom:<12} acierto {np.mean(prediccion[m] == r):.1%} de sus píxeles")
        print(f"   acierto total {np.mean(prediccion == ver):.1%}")

    cmap = ListedColormap(COLORES[:n])
    paneles = 4 if ver is not None else 3
    fig, ejes = plt.subplots(1, paneles, figsize=(8.5 * paneles, 8.5))
    ejes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ejes[0].set_title("imagen sintética" if ver is not None else "ROI 3 suavizada", fontsize=16)
    ejes[0].set_axis_off()
    ax = ejes[1]
    ax.imshow(np.ones(img.shape + (3,)))
    ax.imshow(np.ma.masked_where(semillas == 0, semillas - 1), cmap=cmap, vmin=-0.5, vmax=n - 0.5,
              interpolation="nearest")
    ax.set_title(f"SEMILLAS (blanco = sin semilla, {np.mean(semillas == 0):.0%})\n"
                 "poblaciones: " + ", ".join(f"{g:.2f}" for g in grises), fontsize=15)
    ax.set_axis_off()
    ax = ejes[2]
    ax.imshow(resultado, cmap=cmap, vmin=-0.5, vmax=n - 0.5, interpolation="nearest")
    ax.set_title("RESULTADO del random walker (sólidos)", fontsize=16)
    ax.set_axis_off()
    if ver is not None:
        ax = ejes[3]
        ax.imshow(ver, cmap=ListedColormap(COLORES[:5]), vmin=-0.5, vmax=4.5, interpolation="nearest")
        ax.set_title("VERDAD de la sintética", fontsize=16)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(SALIDA / f"23_random_walker_semillas_por_poblacion_{nombre}.png", dpi=45)
    plt.close(fig)
print("ok")
