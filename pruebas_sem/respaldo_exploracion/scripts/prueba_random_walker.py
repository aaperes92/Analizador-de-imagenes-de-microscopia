"""Random walker con semillas automáticas (ROIs 1 y 3).

Semillas:
- zonas chicas del nivel 1 (picos y valles del histograma): cada una es semilla de sí misma;
- zona principal: contraste local (sigma 4 px, promedio solo con la zona) comparado con el
  ruido de la imagen suavizada (sigma_ruido, estimar_ruido): < -3 sigma -> fase oscura,
  > +3 sigma -> fase clara. El resto queda sin semilla y lo decide el random walker.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap
from scipy import ndimage, signal

sys.path.insert(0, "src")
from sem.suavizado import estimar_ruido, suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA_VENTANA = 4
K_SIGMAS = 3
BINS, SIGMA_HISTOGRAMA, PROMINENCIA_MINIMA, DISTANCIA_MINIMA = 256, 2.0, 0.05, 8
MASA_MINIMA, VALLE_MAXIMO = 0.01, 0.75


def normalizar(valores):
    bajo, alto = np.percentile(valores, (0.1, 99.9))
    return np.clip((valores - bajo) / (alto - bajo), 0, 1)


def valles_entre(h, picos):
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def masas(h, picos):
    bordes = [0, *valles_entre(h, picos), len(h)]
    return [h[bordes[i]:bordes[i + 1]].sum() / h.sum() for i in range(len(picos))]


def zonas_por_valles(img):
    conteos, bordes = np.histogram(img, bins=BINS, range=(0, 1))
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ndimage.gaussian_filter1d(conteos.astype(float), SIGMA_HISTOGRAMA)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], h, [0.0])),
                                 prominence=PROMINENCIA_MINIMA * h.max(), distance=DISTANCIA_MINIMA)
    picos = [int(p) - 1 for p in picos]
    while len(picos) > 1:
        m = masas(h, picos)
        peor = int(np.argmin(m))
        if m[peor] < MASA_MINIMA:
            picos.pop(peor)
            continue
        valles = valles_entre(h, picos)
        rel = [h[v] / min(h[a], h[b]) for v, a, b in zip(valles, picos, picos[1:])]
        i = int(np.argmax(rel))
        if rel[i] <= VALLE_MAXIMO:
            break
        picos.pop(i if h[picos[i]] < h[picos[i + 1]] else i + 1)
    return np.digitize(img, centros[valles_entre(h, picos)]), len(picos)


def contraste_local(img, mascara, sigma):
    peso = mascara.astype(float)
    promedio = ndimage.gaussian_filter(img * peso, sigma) / np.maximum(ndimage.gaussian_filter(peso, sigma), 1e-6)
    return img - promedio


for k in (1, 3):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    sigma_ruido = estimar_ruido(roi)       # ruido de la ROI ORIGINAL (semillas estrictas)
    img = normalizar(suave)
    zonas, nz = zonas_por_valles(img)
    areas = [np.mean(zonas == z) for z in range(nz)]
    principal = int(np.argmax(areas))

    # Etiquetas: 1 = fase oscura del patrón, 2 = fase clara del patrón, 3.. = otras zonas del nivel 1
    nombres = ["fase oscura del patrón", "fase clara del patrón"]
    colores = ["#1f8fff", "#ffd23f"]
    semillas = np.zeros(img.shape, int)
    siguiente = 3
    for z in range(nz):
        if z == principal:
            continue
        semillas[zonas == z] = siguiente
        nombres.append(f"zona {z} del nivel 1 ({'negra' if z == 0 else 'gris'})")
        colores.append(["#08306b", "#a05a00", "#2ca02c"][min(z, 2)])
        siguiente += 1
    mascara = zonas == principal
    c = contraste_local(suave, mascara, SIGMA_VENTANA)
    # Fase clara: por GRIS, píxeles más claros que la mediana de la zona por más de 3 sigma
    mediana_zona = np.median(suave[mascara])
    semillas[mascara & (suave > mediana_zona + K_SIGMAS * sigma_ruido)] = 2
    # Fase oscura: por contraste local (más oscuro que su entorno por más de 3 sigma); tiene prioridad
    semillas[mascara & (c < -K_SIGMAS * sigma_ruido)] = 1
    sin_semilla = np.mean(semillas == 0)

    t0 = time.perf_counter()
    resultado = ski.segmentation.random_walker(img, semillas, mode="cg_j")
    dt = time.perf_counter() - t0
    n = int(resultado.max())
    fracciones = [np.mean(resultado == e) for e in range(1, n + 1)]
    print(f"ROI_{k}: σ ruido (ROI original) ={sigma_ruido:.4f} -> umbral semillas ±{K_SIGMAS * sigma_ruido:.4f}; "
          f"sin semilla = {sin_semilla:.1%}; random walker {dt:.0f}s")
    for nombre, f in zip(nombres, fracciones):
        print(f"   {nombre:<32} {f:.1%}")

    fig, ejes = plt.subplots(1, 3, figsize=(28, 9), gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})
    cmap = ListedColormap(colores[:n])
    lo, hi = np.percentile(img, (0.5, 99.5))
    ax = ejes[0]
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.imshow(np.ma.masked_where(semillas == 0, semillas), cmap=cmap, vmin=0.5, vmax=n + 0.5,
              alpha=0.6, interpolation="nearest")
    ax.set_title(f"ROI {k} — SEMILLAS (seguro); sin color = lo decide el random walker\n"
                 f"(sin semilla: {sin_semilla:.0%} de la ROI)", fontsize=15)
    ax.set_axis_off()
    ax = ejes[1]
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.imshow(resultado, cmap=cmap, vmin=0.5, vmax=n + 0.5, alpha=0.45, interpolation="nearest")
    ax.set_title("RESULTADO del random walker (todas las regiones pintadas)", fontsize=15)
    ax.set_axis_off()
    ax = ejes[2]
    ax.barh(range(n), [100 * f for f in fracciones], color=colores[:n], edgecolor="k")
    ax.set_yticks(range(n), nombres[:n], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("área (%)", fontsize=13)
    ax.set_title("fracción de área", fontsize=15)
    for i, f in enumerate(fracciones):
        ax.text(100 * f + 1, i, f"{100 * f:.1f}%", va="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(SALIDA / f"10_random_walker_semillas_claras_por_gris_ROI{k}.png", dpi=45)
    plt.close(fig)
print("ok")
