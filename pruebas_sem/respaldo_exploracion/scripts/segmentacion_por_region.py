"""Segmentación por región: zonas por gris -> actividad sin cruzar límites -> regiones -> poblaciones.

1. Zonas por gris: picos y valles del histograma de gris (prominencia 5 %, distancia 8, masa 1 %,
   valle < 75 %). Las manchitas de una zona con menos de AREA_MINIMA_COMPONENTE de la imagen se
   reasignan a la zona vecina más cercana (así quedan solo las zonas grandes, como lo negro).
2. Actividad sin cruzar límites: |Sobel| solo donde el vecindario 3x3 es de una misma zona;
   promedio gaussiano (25 px) con solo los píxeles de la propia zona.
3. Regiones: dentro de cada zona, picos y valles del histograma de su actividad.
4. Poblaciones de cada región: paisaje gris-actividad de la región, filtrado (cada fila menos su
   versión suavizada, 6 bins); perfil = suma de filas; picos (prominencia 5 %, distancia 3 bins).
   Cortes en los mínimos del histograma de gris de la región entre picos consecutivos; se
   fusionan poblaciones con menos del 1 % de la región. Etiqueta final = (región, población).
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
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA_ACTIVIDAD = 25
BINS = 128
SIGMA_FILTRO = 6
AREA_MINIMA_COMPONENTE = 0.01
MASA_MINIMA_POBLACION = 0.01
BINS_H, SIGMA_H, PROMINENCIA, DISTANCIA, MASA_MINIMA, VALLE_MAXIMO = 256, 2.0, 0.05, 8, 0.01, 0.75
COLORES_REGION = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#d62728", "#bcbd22", "#8c564b"]
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])


def escalar(v):
    return (v - v.min()) / max(v.max() - v.min(), 1e-12)


def imagen_sintetica(lado=650, semilla=3):
    rng = np.random.default_rng(semilla)
    y, x = np.mgrid[0:lado, 0:lado] / lado
    img = np.full((lado, lado), 0.70)
    estrias = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (2, 12))
    estrias = ndimage.rotate(estrias, 35, reshape=False, mode="reflect")
    img += 0.02 * estrias / estrias.std()
    zona_patron = (x + y) < 0.75
    campo = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), 6)
    red = zona_patron & (np.abs(campo / campo.std()) < 0.25)
    verdad = np.full((lado, lado), 3)
    img[red], verdad[red] = 0.58, 2
    particulas = ski.data.binary_blobs(lado, blob_size_fraction=0.01, volume_fraction=0.02, rng=semilla)
    img[particulas], verdad[particulas] = 0.85, 4
    d = (x + y) - 1.55
    franja, gris = np.abs(d) < 0.03, d >= 0.03
    img[franja], verdad[franja] = 0.0, 0
    img[gris], verdad[gris] = 0.45, 1
    img = ndimage.gaussian_filter(img, 1.5) + rng.normal(0, 0.005, img.shape)
    return np.clip(img, 0, 1), verdad


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


def poblaciones_de_region(gris01, actividad, mascara):
    """Devuelve (paisaje, filtrado, perfil, grises de los picos, cortes, etiquetas dentro de la región)."""
    g = gris01[mascara]
    a = escalar(actividad[mascara])
    conteos, _, _ = np.histogram2d(a, g, bins=BINS, range=((0, 1), (0, 1)))
    paisaje = np.log1p(conteos)
    filtrado = np.maximum(paisaje - ndimage.gaussian_filter1d(paisaje, SIGMA_FILTRO, axis=1, mode="constant"), 0)
    perfil = filtrado.sum(axis=0)
    picos, _ = signal.find_peaks(np.concatenate(([0.0], perfil, [0.0])), prominence=0.05 * perfil.max(),
                                 distance=3)
    picos = list(picos - 1)
    hist_gris = ndimage.gaussian_filter1d(np.histogram(g, bins=BINS, range=(0, 1))[0].astype(float), 1)
    centros = (np.arange(BINS) + 0.5) / BINS
    while True:
        cortes = []
        for a_, b_ in zip(picos, picos[1:]):
            v = a_ + int(np.argmin(hist_gris[a_:b_ + 1]))
            # Sin mínimo interior (población "hombro"): corte en el punto medio entre los picos
            cortes.append(centros[v] if a_ < v < b_ else (centros[a_] + centros[b_]) / 2)
        cortes = np.array(cortes)
        etiquetas = np.digitize(g, cortes)
        fr = np.bincount(etiquetas, minlength=len(picos)) / g.size
        if len(picos) <= 1 or fr.min() >= MASA_MINIMA_POBLACION:
            break
        picos.pop(int(np.argmin(fr)))
    return paisaje, filtrado, perfil, centros[picos], cortes, etiquetas


def tonos(color, n):
    """n tonos de un color, de oscuro a claro."""
    base = np.array(to_rgb(color))
    if n == 1:
        return [tuple(base)]
    return [tuple(np.clip(base * (0.35 + 0.65 * t) + (1 - base) * max(0, t - 0.6), 0, 1))
            for t in np.linspace(0, 1, n)]


casos = {
    "ROI3": (suavizar(np.load("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy")), None),
    "sintetica": imagen_sintetica(),
}
for nombre, (img, verdad) in casos.items():
    gris = escalar(img)
    zonas = zonas_por_gris(gris)
    borde, anchos = franja_de_borde(gris, zonas)
    actividad = actividad_sin_cruzar(img, zonas, borde)
    print(f"\n{nombre}: ancho de la franja de borde por zona (px) = {anchos}; borde = {borde.mean():.1%} de la imagen")

    # Regiones = (zona, clase de actividad dentro de la zona), solo con píxeles que NO son borde
    regiones = np.full(img.shape, -1)
    descripcion = []
    for z in np.unique(zonas):
        mascara = (zonas == z) & ~borde
        a = escalar(actividad[mascara])
        clases = np.digitize(a, cortes_por_valles(a))
        for c in np.unique(clases):
            sub = np.zeros(img.shape, bool)
            sub[mascara] = clases == c
            regiones[sub] = len(descripcion)
            descripcion.append(f"zona de gris {z}, actividad {c}")
    nr = len(descripcion)
    print(f"\n{nombre}: {len(np.unique(zonas))} zona(s) por gris; {nr} región(es)")

    # ---- Figura A: cómo se arman las regiones
    fig, ejes = plt.subplots(1, 4, figsize=(34, 8.5))
    ejes[0].imshow(gris, cmap="gray")
    ejes[0].set_title(f"{nombre}: imagen suavizada", fontsize=17)
    nz = len(np.unique(zonas))
    ejes[1].imshow(zonas, cmap="viridis", interpolation="nearest")
    ejes[1].set_title(f"1) {nz} zona(s) por GRIS (sin manchitas sueltas)", fontsize=17)
    im = ejes[2].imshow(actividad, cmap="magma")
    ejes[2].set_title("2) actividad SIN cruzar límites de zona", fontsize=17)
    fig.colorbar(im, ax=ejes[2], fraction=0.046)
    ejes[3].imshow(np.ones(img.shape), cmap="gray", vmin=0, vmax=1)
    ejes[3].imshow(np.ma.masked_where(regiones < 0, regiones), cmap=ListedColormap(COLORES_REGION[:nr]),
                   vmin=-0.5, vmax=nr - 0.5, interpolation="nearest")
    ejes[3].set_title(f"3) {nr} región(es) de análisis\n(blanco = franja de borde, se asigna al final)", fontsize=17)
    ejes[3].legend(handles=[Patch(color=COLORES_REGION[r], label=f"región {r}") for r in range(nr)],
                   loc="lower left", fontsize=12)
    for ax in ejes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(SALIDA / f"28_regiones_con_borde_{nombre}.png", dpi=42)
    plt.close(fig)

    # ---- Figura B: cada región por separado
    final = np.full(img.shape, -1)
    colores_finales, leyenda = [], []
    fig, ejes = plt.subplots(nr, 4, figsize=(34, 7.6 * nr), squeeze=False)
    for r in range(nr):
        mascara = regiones == r
        paisaje, filtrado, perfil, picos_g, cortes, etiquetas = poblaciones_de_region(gris, actividad, mascara)
        np_ = len(picos_g)
        ts = tonos(COLORES_REGION[r], np_)
        print(f"  región {r} ({descripcion[r]}, {mascara.mean():.1%} de la imagen): {np_} población(es) en gris "
              f"{np.round(picos_g, 3)}, cortes {np.round(cortes, 3)}, áreas "
              f"{np.round(np.bincount(etiquetas, minlength=np_) / etiquetas.size, 3)}")
        for p in range(np_):
            idx = len(colores_finales)
            sub = np.zeros(img.shape, bool)
            sub[mascara] = etiquetas == p
            final[sub] = idx
            colores_finales.append(ts[p])
            leyenda.append(f"región {r} · población {p} (gris {picos_g[p]:.2f})")

        # panel 1: imagen, afuera negro, región con su color semitransparente
        ax = ejes[r, 0]
        ax.imshow(np.where(mascara, gris, 0), cmap="gray", vmin=0, vmax=1)
        capa = np.zeros(img.shape + (4,))
        capa[mascara] = (*to_rgb(COLORES_REGION[r]), 0.35)
        ax.imshow(capa)
        ax.set_title(f"región {r} ({mascara.mean():.0%} de la imagen; afuera en negro)", fontsize=17,
                     color=COLORES_REGION[r])
        ax.set_axis_off()
        for ax, datos, titulo in ((ejes[r, 1], paisaje, "paisaje ORIGINAL de la región"),
                                  (ejes[r, 2], filtrado, "paisaje FILTRADO de la región")):
            im = ax.imshow(datos, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap=CMAP_PAISAJE)
            for t in cortes:
                ax.axvline(t, color="k", lw=1.5, ls="--")
            ax.set_xlabel("gris (0-1, escala de toda la imagen)", fontsize=14)
            ax.set_ylabel("actividad (rango de la región)", fontsize=14)
            ax.set_title(f"{titulo}\n(línea negra = cortes entre poblaciones)", fontsize=16)
            fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + píxeles)")
        ax = ejes[r, 3]
        mapa = np.full(img.shape, -1)
        mapa[mascara] = etiquetas
        ax.imshow(np.zeros(img.shape), cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(mapa < 0, mapa), cmap=ListedColormap(ts), vmin=-0.5, vmax=np_ - 0.5,
                  interpolation="nearest")
        ax.set_title(f"segmentación de la región {r}: {np_} población(es)", fontsize=17)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(SALIDA / f"29_paisaje_y_segmentacion_por_region_con_borde_{nombre}.png", dpi=40)
    plt.close(fig)

    # Píxeles de la franja de borde: toman la etiqueta del píxel ya clasificado más cercano
    _, (fi, co) = ndimage.distance_transform_edt(final < 0, return_indices=True)
    final = final[fi, co]

    # ---- Figura C: resultado final, todas las regiones juntas
    paneles = 3 if verdad is not None else 2
    fig, ejes = plt.subplots(1, paneles, figsize=(11 * paneles, 10))
    ejes[0].imshow(gris, cmap="gray")
    ejes[0].set_title(f"{nombre}: imagen suavizada", fontsize=18)
    ejes[1].imshow(final, cmap=ListedColormap(colores_finales), vmin=-0.5, vmax=len(colores_finales) - 0.5,
                   interpolation="nearest")
    ejes[1].set_title("SEGMENTACIÓN FINAL (región · población)\ncada región con su color; poblaciones en tonos",
                      fontsize=18)
    ejes[1].legend(handles=[Patch(color=c, label=l) for c, l in zip(colores_finales, leyenda)],
                   loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=13)
    if verdad is not None:
        ejes[2].imshow(verdad, cmap=ListedColormap(["#000000", "#8c564b", "#1f5fbf", "#e8d8a0", "#ff7f0e"]),
                       vmin=-0.5, vmax=4.5, interpolation="nearest")
        ejes[2].set_title("VERDAD de la sintética\n(negro, zona gris, fase oscura, matriz, partículas)", fontsize=18)
    for ax in ejes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(SALIDA / f"30_segmentacion_final_con_borde_{nombre}.png", dpi=42, bbox_inches="tight")
    plt.close(fig)
print("ok")
