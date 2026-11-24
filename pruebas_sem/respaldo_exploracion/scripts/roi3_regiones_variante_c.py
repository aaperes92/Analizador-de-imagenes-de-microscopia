"""ROI 3: variante c (máscara de enfoque + filtro de choque) aplicada por región de análisis.

1. Regiones de análisis como en 28-30: zonas por gris (picos y valles), franja de borde entre
   zonas, actividad (gaussiana(sobel), 25 px) sin cruzar límites, y dentro de cada zona, clases de
   actividad por picos y valles.
2. Variante c sobre la ROI suavizada: máscara de enfoque (radio = degradé estimado, cantidad 1) y
   filtro de choque (sigma interno 1 px).
3. En CADA región, con solo sus píxeles (sin la franja de borde): histograma de la imagen procesada,
   suavizado 5 casillas; los dos picos con más píxeles. Se separa en dos poblaciones solo si el
   valle entre ellos baja del 75 % del pico más bajo (mismo criterio que la fusión de picos); si no,
   la región es una sola población. Además, cada población debe tener al menos el 5 % de la región.
   El filtro de choque frena cuando el cambio medio por paso es menor que 0.1 x ruido.
4. La franja de borde toma la etiqueta del píxel clasificado más cercano.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.patches import Patch
from scipy import ndimage, signal

sys.path.insert(0, "src")
from sem.suavizado import estimar_ruido, suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA_ACTIVIDAD = 25
SIGMA_ACT_PAISAJE = 3
AREA_MINIMA_COMPONENTE = 0.01
BINS_H, SIGMA_H, PROMINENCIA, DISTANCIA, MASA_MINIMA, VALLE_MAXIMO = 256, 2.0, 0.05, 8, 0.01, 0.75
SUAVIZADO_HIST = 5
MASA_MINIMA_POBLACION = 0.05   # cada población, al menos el 5 % de su región
COLORES_REGION = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#d62728"]
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])

# --- funciones copiadas de scripts anteriores ---

def escalar(v):
    return (v - v.min()) / max(v.max() - v.min(), 1e-12)


def valles_entre(h, picos):
    return [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]


def masas(h, picos):
    bordes = [0, *valles_entre(h, picos), len(h)]
    return [h[bordes[i]:bordes[i + 1]].sum() / h.sum() for i in range(len(picos))]


def cortes_por_valles(valores01):
    """Cortes (en 0-1) entre los picos reales del histograma de los valores."""
    conteos, bordes = np.histogram(valores01, bins=BINS_H, range=(0, 1))
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ndimage.gaussian_filter1d(conteos.astype(float), SIGMA_H)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], h, [0.0])), prominence=PROMINENCIA * h.max(),
                                 distance=DISTANCIA)
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
    return centros[valles_entre(h, picos)]


def zonas_por_gris(gris01):
    zonas = np.digitize(gris01, cortes_por_valles(gris01.ravel()))
    # Componentes chicas -> se reasignan a la zona grande más cercana
    limpia = np.full(zonas.shape, -1)
    for z in np.unique(zonas):
        grandes = ski.morphology.remove_small_objects(zonas == z, int(AREA_MINIMA_COMPONENTE * zonas.size))
        limpia[grandes] = z
    _, (fi, co) = ndimage.distance_transform_edt(limpia < 0, return_indices=True)
    limpia = limpia[fi, co]
    _, renumerada = np.unique(limpia, return_inverse=True)
    return renumerada.reshape(zonas.shape)


def franja_de_borde(gris01, zonas):
    """Píxeles de transición junto al límite entre zonas.

    Para cada zona: distancia de cada píxel al límite con otra zona; mediana del gris a cada
    distancia. El ancho de la franja es la primera distancia donde el gris ya recorrió el 90 % del
    camino desde el valor en el límite hasta el valor típico del interior de la zona.
    """
    borde = np.zeros(zonas.shape, bool)
    anchos = {}
    for z in np.unique(zonas):
        mascara = zonas == z
        dist = ndimage.distance_transform_edt(mascara)
        dmax = int(dist.max())
        interior = gris01[dist >= max(3, 0.5 * dmax)]
        referencia = np.median(interior) if interior.size else np.median(gris01[mascara])
        en_limite = np.median(gris01[mascara & (dist <= 1)])
        ancho = 1
        for d in range(1, min(dmax, 40) + 1):
            anillo = mascara & (dist > d - 1) & (dist <= d)
            if anillo.any() and abs(np.median(gris01[anillo]) - referencia) <= 0.1 * abs(en_limite - referencia):
                ancho = d
                break
        anchos[int(z)] = ancho
        borde |= mascara & (dist <= ancho)
    return borde, anchos


def actividad_sin_cruzar(img, zonas, borde):
    gradiente = ski.filters.sobel(img)
    actividad = np.zeros(img.shape)
    for z in np.unique(zonas):
        mascara = zonas == z
        valido = (ski.morphology.erosion(mascara, ski.morphology.footprint_rectangle((3, 3))) & ~borde).astype(float)
        promedio = ski.filters.gaussian(gradiente * valido, SIGMA_ACTIVIDAD) / np.maximum(
            ski.filters.gaussian(valido, SIGMA_ACTIVIDAD), 1e-6)
        actividad[mascara] = promedio[mascara]
    return actividad


def estimar_sigma_borde(img, muestras=1500, semilla=0):
    gy, gx = ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1)
    maximos = ski.feature.peak_local_max(np.hypot(gx, gy), min_distance=3, threshold_rel=0.2, exclude_border=12)
    rng = np.random.default_rng(semilla)
    if len(maximos) > muestras:
        maximos = maximos[rng.choice(len(maximos), muestras, replace=False)]
    t = np.linspace(-10, 10, 201)
    anchos = []
    for y, x in maximos:
        n = np.hypot(gy[y, x], gx[y, x])
        perfil = ndimage.map_coordinates(img, [y + t * gy[y, x] / n, x + t * gx[y, x] / n], order=1)
        p = (perfil - perfil.min()) / max(perfil.max() - perfil.min(), 1e-9)
        if p[0] > p[-1]:
            p = p[::-1]
        if p[t < -5].max() > 0.12 or p[t > 5].min() < 0.88 or np.min(np.diff(p)) < -0.03:
            continue
        i10, i90 = np.argmax(p >= 0.1), np.argmax(p >= 0.9)
        if i90 > i10:
            anchos.append(t[i90] - t[i10])
    return (float(np.median(anchos)) / 2.563 if len(anchos) >= 10 else 1.0), len(anchos)


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def choque(img, sigma_interno, dt=0.25, tolerancia=1e-3, max_pasos=200):
    I = img.astype(float).copy()
    rango = np.ptp(img)
    for paso in range(max_pasos):
        S = ndimage.gaussian_filter(I, sigma_interno)
        Sy, Sx = np.gradient(S)
        Syy, Syx = np.gradient(Sy)
        _, Sxx = np.gradient(Sx)
        D2 = (Sx ** 2 * Sxx + 2 * Sx * Sy * Syx + Sy ** 2 * Syy) / (Sx ** 2 + Sy ** 2 + 1e-12)
        P = np.pad(I, 1, mode="edge")
        Ix = _minmod(P[1:-1, 2:] - I, I - P[1:-1, :-2])
        Iy = _minmod(P[2:, 1:-1] - I, I - P[:-2, 1:-1])
        delta = -dt * np.sign(D2) * np.hypot(Ix, Iy)
        I += delta
        if np.mean(np.abs(delta)) / rango < tolerancia:
            break
    return I, paso + 1




def dos_poblaciones(valores):
    """Decide si los valores tienen dos poblaciones y dónde cortar."""
    rango = tuple(np.percentile(valores, (0.1, 99.9)))
    conteos, bordes = np.histogram(valores, bins=256, range=rango)
    centros = (bordes[:-1] + bordes[1:]) / 2
    h = ndimage.gaussian_filter1d(conteos.astype(float), SUAVIZADO_HIST)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], h, [0.0])))
    picos = list(picos - 1)
    info = dict(centros=centros, conteos=conteos, h=h, picos=picos, dos=[], corte=None, relacion=None)
    if len(picos) < 2:
        return info, "un solo pico: una población"
    valles = [a + int(np.argmin(h[a:b + 1])) for a, b in zip(picos, picos[1:])]
    limites = [0, *valles, len(h)]
    masas_p = [h[limites[i]:limites[i + 1]].sum() for i in range(len(picos))]
    a, b = (picos[i] for i in sorted(np.argsort(masas_p)[-2:]))
    v = a + int(np.argmin(h[a:b + 1]))
    relacion = h[v] / min(h[a], h[b])
    info.update(dos=[a, b], relacion=relacion)
    menor = min(np.mean(valores <= centros[v]), np.mean(valores > centros[v]))
    if menor < MASA_MINIMA_POBLACION:
        return info, f"la población menor tendría {menor:.1%} de la región (< {MASA_MINIMA_POBLACION:.0%}): una población"
    if relacion < VALLE_MAXIMO:
        info["corte"] = centros[v]
        return info, f"valle al {relacion:.0%} del pico menor: dos poblaciones"
    return info, f"valle al {relacion:.0%} del pico menor (≥ {VALLE_MAXIMO:.0%}): una población"


# ---------------- 1. Regiones de análisis
suave = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy"))
gris = escalar(suave)
zonas = zonas_por_gris(gris)
borde, anchos = franja_de_borde(gris, zonas)
actividad = actividad_sin_cruzar(suave, zonas, borde)
regiones = np.full(suave.shape, -1)
descripcion = []
for z in np.unique(zonas):
    mascara = (zonas == z) & ~borde
    a = escalar(actividad[mascara])
    clases = np.digitize(a, cortes_por_valles(a))
    for c in np.unique(clases):
        sub = np.zeros(suave.shape, bool)
        sub[mascara] = clases == c
        regiones[sub] = len(descripcion)
        descripcion.append(f"zona de gris {z}, actividad {c}")
nr = len(descripcion)
print(f"{nr} regiones; franja de borde {borde.mean():.1%} (anchos {anchos})")

# ---------------- 2. Variante c
sigma_deg, usados = estimar_sigma_borde(suave)
realzada = ski.filters.unsharp_mask(suave, radius=sigma_deg, amount=1.0, preserve_range=True)
sigma_ruido = estimar_ruido(suave)
procesada, pasos = choque(realzada, 1.0, tolerancia=0.1 * sigma_ruido / np.ptp(realzada))   # frena con cambio < 0.1 σ
print(f"degradé estimado {sigma_deg:.2f} px ({usados} bordes aislados); ruido {sigma_ruido:.4f}; filtro de choque {pasos} pasos")

# ---------------- 3. Poblaciones por región
final = np.full(suave.shape, -1)
colores_finales, leyenda, filas = [], [], []
for r in range(nr):
    mascara = regiones == r
    info, decision = dos_poblaciones(procesada[mascara])
    if info["corte"] is None:
        etiq = np.zeros(mascara.sum(), int)
    else:
        etiq = (procesada[mascara] > info["corte"]).astype(int)
    n = etiq.max() + 1
    base = np.array(to_rgb(COLORES_REGION[r]))
    tonos = [tuple(base * 0.45)] if n == 1 else [tuple(base * 0.45), tuple(base + (1 - base) * 0.55)]
    if n == 1:
        tonos = [tuple(base)]
    for p in range(n):
        sub = np.zeros(suave.shape, bool)
        sub[mascara] = etiq == p
        final[sub] = len(colores_finales)
        colores_finales.append(tonos[p])
        nombre_p = "única" if n == 1 else ("oscura" if p == 0 else "clara")
        leyenda.append(f"región {r} · población {nombre_p} ({np.mean(etiq == p):.0%} de la región)")
    print(f"región {r} ({descripcion[r]}, {mascara.mean():.1%} de la ROI): {decision}"
          + (f"; corte {info['corte']:.4f}; oscura {np.mean(etiq == 0):.1%} de la región" if n == 2 else ""))
    filas.append((r, mascara, info, decision, etiq, tonos))
_, (fi, co) = ndimage.distance_transform_edt(final < 0, return_indices=True)
final = final[fi, co]

lo, hi = np.percentile(suave, (0.5, 99.5))
lo_p, hi_p = np.percentile(procesada, (0.5, 99.5))

# ---------------- Figura 54: regiones y variante c
fig, ejes = plt.subplots(1, 3, figsize=(36, 11))
ax = ejes[0]
im = ax.imshow(suave, cmap="gray", vmin=lo, vmax=hi)
cb = fig.colorbar(im, ax=ax, fraction=0.046)
cb.set_label("gris", fontsize=18)
cb.ax.tick_params(labelsize=15)
ax.set_title("ROI 3 suavizada", fontsize=24)
ax = ejes[1]
ax.imshow(np.ones(suave.shape), cmap="gray", vmin=0, vmax=1)
ax.imshow(np.ma.masked_where(regiones < 0, regiones), cmap=ListedColormap(COLORES_REGION[:nr]),
          vmin=-0.5, vmax=nr - 0.5, interpolation="nearest")
ax.legend(handles=[Patch(color=COLORES_REGION[r], label=f"región {r} ({np.mean(regiones == r):.0%})")
                   for r in range(nr)] + [Patch(color="white", ec="k", label="franja de borde")],
          loc="lower left", fontsize=18)
ax.set_title("regiones de análisis", fontsize=24)
ax = ejes[2]
im = ax.imshow(procesada, cmap="gray", vmin=lo_p, vmax=hi_p)
cb = fig.colorbar(im, ax=ax, fraction=0.046)
cb.set_label("gris", fontsize=18)
cb.ax.tick_params(labelsize=15)
ax.set_title(f"variante c: enfoque (radio {sigma_deg:.2f} px) + choque ({pasos} pasos)", fontsize=24)
for ax in ejes:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(SALIDA / "54_ROI3_regiones_y_variante_c.png", dpi=30)
plt.close(fig)

# ---------------- Figura 55: cada región por separado
fig, ejes = plt.subplots(nr, 4, figsize=(44, 11 * nr), squeeze=False,
                         gridspec_kw={"width_ratios": [1, 1.25, 1.2, 1]})
for fila, (r, mascara, info, decision, etiq, tonos) in enumerate(filas):
    ax = ejes[fila, 0]
    ax.imshow(np.where(mascara, procesada, lo_p), cmap="gray", vmin=lo_p, vmax=hi_p)
    capa = np.zeros(suave.shape + (4,))
    capa[mascara] = (*to_rgb(COLORES_REGION[r]), 0.25)
    capa[~mascara] = (0, 0, 0, 1)
    ax.imshow(capa)
    ax.set_title(f"región {r} ({mascara.mean():.0%} de la ROI)\nimagen procesada; afuera en negro",
                 fontsize=24, color=COLORES_REGION[r])
    ax.set_axis_off()

    ax = ejes[fila, 1]
    c = info["centros"]
    ax.bar(c, info["conteos"], width=c[1] - c[0], color="0.75", label="histograma")
    ax.plot(c, info["h"], color="k", lw=2.5, label=f"suavizado ({SUAVIZADO_HIST} casillas)")
    if info["picos"]:
        ax.plot(c[info["picos"]], info["h"][info["picos"]], "o", color="0.4", ms=9, label="picos")
    if info["dos"]:
        ax.plot(c[info["dos"]], info["h"][info["dos"]], "o", color="C2", ms=15, mec="k", label="los 2 con más píxeles")
    if info["corte"] is not None:
        ax.axvline(info["corte"], color="C3", lw=3.5, ls="--", label=f"corte {info['corte']:.3f}")
    ax.legend(fontsize=16)
    ax.set_xlabel("gris (imagen procesada)", fontsize=19)
    ax.set_ylabel("cantidad de píxeles", fontsize=19)
    ax.tick_params(labelsize=15)
    ax.set_title(f"histograma de la región\n{decision}", fontsize=21)

    ax = ejes[fila, 2]
    act = ski.filters.gaussian(ski.filters.sobel(procesada), SIGMA_ACT_PAISAJE)
    act = escalar(act[mascara])
    rango = tuple(np.percentile(procesada[mascara], (0.1, 99.9)))
    conteos2, _, _ = np.histogram2d(act, procesada[mascara], bins=160, range=((0, 1), rango))
    im = ax.imshow(np.log1p(conteos2), origin="lower", aspect="auto", cmap=CMAP_PAISAJE,
                   extent=(rango[0], rango[1], 0, 1))
    if info["corte"] is not None:
        ax.axvline(info["corte"], color="k", lw=2.5, ls="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("log(1 + cantidad de píxeles)", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    ax.set_xlabel("gris (imagen procesada)", fontsize=19)
    ax.set_ylabel(f"actividad (σ={SIGMA_ACT_PAISAJE}px, rango de la región)", fontsize=19)
    ax.tick_params(labelsize=15)
    ax.set_title("paisaje gris-actividad de la región\n(línea = corte)", fontsize=22)

    ax = ejes[fila, 3]
    mapa = np.full(suave.shape, -1)
    mapa[mascara] = etiq
    ax.imshow(np.zeros(suave.shape), cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(mapa < 0, mapa), cmap=ListedColormap(tonos), vmin=-0.5,
              vmax=len(tonos) - 0.5, interpolation="nearest")
    n = len(tonos)
    ax.legend(handles=[Patch(color=tonos[p], label=("única" if n == 1 else ["oscura", "clara"][p])
                             + f" {np.mean(etiq == p):.0%}") for p in range(n)], loc="lower left", fontsize=17)
    ax.set_title(f"resultado de la región {r}", fontsize=24)
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(SALIDA / "55_ROI3_variante_c_por_region.png", dpi=26)
plt.close(fig)

# ---------------- Figura 56: resultado final y contornos sobre la original
fig, ejes = plt.subplots(1, 3, figsize=(40, 12), gridspec_kw={"width_ratios": [1, 1, 1]})
ax = ejes[0]
ax.imshow(suave, cmap="gray", vmin=lo, vmax=hi)
ax.set_title("ROI 3 suavizada", fontsize=24)
ax = ejes[1]
ax.imshow(final, cmap=ListedColormap(colores_finales), vmin=-0.5, vmax=len(colores_finales) - 0.5,
          interpolation="nearest")
ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colores_finales, leyenda)],
          loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=17)
ax.set_title("SEGMENTACIÓN FINAL (región · población)", fontsize=24)
ax = ejes[2]
ax.imshow(suave, cmap="gray", vmin=lo, vmax=hi)
contorno = ski.segmentation.find_boundaries(final, mode="inner")
ax.imshow(np.ma.masked_where(~contorno, contorno), cmap=ListedColormap(["#ff2020"]), interpolation="nearest")
ax.set_title("contornos de la segmentación sobre la ROI", fontsize=24)
for ax in ejes:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(SALIDA / "56_ROI3_variante_c_resultado_final.png", dpi=30, bbox_inches="tight")
print("ok")
