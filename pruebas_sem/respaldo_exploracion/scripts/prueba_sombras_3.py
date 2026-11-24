"""Prueba 3: iluminación por fases usando solo el interior de cada fase, N=3 contra N=4.

Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/sombras/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

sys.path.insert(0, "src")
from sem.carga import cargar_imagen, leer_metadata, recortar_barra_datos
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/sombras")
RADIO_FRANJA = 2           # px que se descartan a cada lado de las transiciones entre fases
POBLACIONES = (3, 4)


def clasificar(imagen, n):
    return np.digitize(imagen, ski.filters.threshold_multiotsu(imagen, classes=n))


def separabilidad(imagen, n):
    """Fracción de la varianza explicada por las clases (1 = clases perfectamente separadas)."""
    clases = clasificar(imagen, n)
    medias = np.array([imagen[clases == k].mean() for k in range(n)])
    return 1 - np.mean((imagen - medias[clases]) ** 2) / imagen.var()


def espesor_fase_dominante(clases, n):
    k = np.argmax([np.sum(clases == k) for k in range(n)])
    esqueleto, distancia = ski.morphology.medial_axis(clases == k, return_distance=True)
    return 2 * float(np.median(distancia[esqueleto]))


def corregir_por_fases(imagen, n, sigma_ventana):
    """Devuelve (corregida, iluminación, clases, interior)."""
    clases = clasificar(imagen, n)
    grises = np.array([np.median(imagen[clases == k]) for k in range(n)])
    # Solo píxeles bien adentro de cada fase: se descarta una franja en las transiciones
    disco = ski.morphology.disk(RADIO_FRANJA)
    interior = np.zeros(imagen.shape, bool)
    for k in range(n):
        interior |= ski.morphology.binary_erosion(clases == k, disco)
    peso = interior.astype(np.float32)
    numerador = ski.filters.gaussian(peso * imagen, sigma_ventana)
    denominador = ski.filters.gaussian(peso * grises[clases], sigma_ventana)
    iluminacion = numerador / np.maximum(denominador, 1e-6)
    return imagen / iluminacion, iluminacion, clases, interior


casos = {f"ROI_{k}": np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy") for k in (1, 2, 3, 4)}
ruta = Path("imagenes/SEM/BSE/BSE_1.tif")
casos["BSE_1_completa"] = recortar_barra_datos(cargar_imagen(ruta), leer_metadata(ruta))

resultados = {}
for nombre, imagen in casos.items():
    suave = suavizar(imagen)
    sep = {n: separabilidad(suave, n) for n in (2, 3, 4, 5)}
    clases3 = clasificar(suave, 3)
    sigma_v = 1.5 * espesor_fase_dominante(clases3, 3)
    resultados[nombre] = {"suave": suave, "sigma": sigma_v}
    linea = f"{nombre}: ventana σ={sigma_v:.0f}px  separabilidad N=2..5: " + \
        "  ".join(f"{n}:{s:.3f}" for n, s in sep.items())
    for n in POBLACIONES:
        corregida, ilum, clases, interior = corregir_por_fases(suave, n, sigma_v)
        resultados[nombre][n] = (corregida, ilum, clases)
        linea += f"\n    N={n}: iluminación percentiles 1-99 = {np.percentile(ilum, 1):.3f}-" \
                 f"{np.percentile(ilum, 99):.3f}  píxeles interiores = {interior.mean():.0%}"
    print(linea)

# Figura 1: resultado completo. Gris en todo; misma escala de gris para antes y después.
columnas = ["suavizada (antes)", "fases N=3", "corregida N=3", "fases N=4", "corregida N=4", "histogramas"]
fig, ejes = plt.subplots(len(casos), 6, figsize=(28, 4.7 * len(casos)))
for i, (nombre, r) in enumerate(resultados.items()):
    suave = r["suave"]
    lo, hi = np.percentile(suave, (0.5, 99.5))
    paneles = [suave, r[3][2], r[3][0], r[4][2], r[4][0]]
    for j, panel in enumerate(paneles):
        ax = ejes[i, j]
        if j in (1, 3):
            ax.imshow(panel, cmap="gray")
        else:
            ax.imshow(panel, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"{nombre} — {columnas[j]}" if j == 0 else columnas[j], fontsize=12)
        ax.set_axis_off()
    ax = ejes[i, 5]
    for img, etiqueta, color in ((suave, "antes", "0.5"), (r[3][0], "N=3", "C0"), (r[4][0], "N=4", "C3")):
        ax.hist(img.ravel(), bins=256, range=(0, 1), histtype="step", color=color, label=etiqueta)
    ax.set_yscale("log")
    ax.set_xlabel("intensidad")
    ax.set_title(f"histogramas (ventana σ={r['sigma']:.0f}px)", fontsize=12)
    ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(SALIDA / "03_correccion_por_fases_N3_vs_N4_BSE_1.png", dpi=50)

# Figura 2: recortes ampliados donde hay manchas o bordes negros, contraste estirado igual en los tres.
recortes = [
    ("ROI_4", "mancha central", (130, 170, 230)),
    ("ROI_3", "borde de la zona negra", (430, 380, 230)),
    ("BSE_1_completa", "manchas a la derecha", (330, 1700, 330)),
    ("BSE_1_completa", "zona con patrón", (500, 900, 330)),
]
fig, ejes = plt.subplots(len(recortes), 3, figsize=(15, 5 * len(recortes)))
for i, (nombre, desc, (y, x, lado)) in enumerate(recortes):
    r = resultados[nombre]
    trozos = [r["suave"], r[3][0], r[4][0]]
    trozos = [t[y:y + lado, x:x + lado] for t in trozos]
    lo, hi = np.percentile(trozos[0], (1, 99))
    for j, (t, titulo) in enumerate(zip(trozos, ("antes", "corregida N=3", "corregida N=4"))):
        ejes[i, j].imshow(t, cmap="gray", vmin=lo, vmax=hi)
        ejes[i, j].set_title(f"{nombre} ({desc})\n{titulo}" if j == 0 else titulo, fontsize=12)
        ejes[i, j].set_axis_off()
fig.tight_layout()
fig.savefig(SALIDA / "04_recortes_ampliados_N3_vs_N4_BSE_1.png", dpi=60)
print("ok")
