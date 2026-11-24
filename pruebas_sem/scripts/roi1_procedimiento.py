"""ROI 1 real (BSE_1_1): el procedimiento de las simuladas de tres fases (figuras 10/11), sin tocar parámetros.

Procedimiento:
0. suavizado de la Etapa 2 (non-local means, src/sem/suavizado.py);
0b. realce: máscara de enfoque (radio = degradé estimado en bordes aislados de la suavizada,
    cantidad 1);
1. filtro de choque sobre la realzada (σ interno 1 px, dt 0.25, frena cuando el cambio medio es
   < 0.1 x ruido estimado), en dos versiones: solo, y con pozos (dos etapas, intensidad 2);
2. actividad de la imagen de PARTIDA (la suavizada): gaussiana(sobel), 3 px, normalizada 0-1;
3. paisaje: gris tras el choque x actividad de partida (128 x 128 bins);
4. normalización: cada fila (suavizada 1 bin) dividida por su máximo;
5. núcleos: bins sobre la media altura de su fila, conexos, con masa >= 1 % de los píxeles;
   gris central = mediana del gris de los píxeles del núcleo. Cada núcleo es una población;
6. pozos: inundación (watershed) desde los núcleos; cada píxel toma el pozo de su bin. Las
   poblaciones se numeran por gris (0 = la más oscura).
No hay verdad: se evalúa si los núcleos y los contornos tienen sentido sobre la imagen.
Figuras: 13a partida y suavizado, 13b realce y choque, 13c gradiente y actividad, 13d paisajes,
13e núcleos y pozos, 13f resultado (las 12* eran la misma cadena sin realce).
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/roi1/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy import ndimage

sys.path.insert(0, "src")
from sem.suavizado import estimar_ruido, suavizar

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})

ROI = Path("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy")
SALIDA = Path("pruebas_sem/roi1")
SALIDA.mkdir(parents=True, exist_ok=True)
SIGMA_ACT, BINS, FILA_MINIMA = 3, 128, 0.005
MEDIA_ALTURA, MASA_MINIMA = 0.5, 0.01
SIGMA_INTERNO, DT, INTENSIDAD_POZOS = 1.0, 0.25, 2.0
CANTIDAD_ENFOQUE = 1.0
COLORES = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
Y0, X0, LADO = 180, 200, 150                     # recorte para ver detalle
CAJA = {"facecolor": "#fffbe6", "edgecolor": "0.4", "alpha": 0.95, "boxstyle": "round"}


def estimar_sigma_borde(img, muestras=1500, semilla=0):
    """Ancho del degradé (σ de una gaussiana, px) a partir del ancho 10-90 % de bordes aislados."""
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


def actividad_de_partida(img):
    """Actividad gaussiana(sobel), 3 px, normalizada 0-1 (percentil 99.9 = 1)."""
    act = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACT)
    return np.clip(act / np.percentile(act, 99.9), 0, 1)


def paisaje(img, act, rango):
    """Índices de bin (actividad, gris) de cada píxel y conteos del paisaje."""
    i_act = np.clip((act * BINS).astype(int), 0, BINS - 1)
    i_gris = np.clip(((img - rango[0]) / (rango[1] - rango[0]) * BINS).astype(int), 0, BINS - 1)
    conteos = np.bincount(np.ravel_multi_index((i_act, i_gris), (BINS, BINS)).ravel(),
                          minlength=BINS ** 2).reshape(BINS, BINS).astype(float)
    return i_act, i_gris, conteos


def normalizar_por_fila(conteos):
    suave = ndimage.gaussian_filter(conteos, 1)
    por_fila = conteos.sum(axis=1, keepdims=True)
    validas = por_fila >= FILA_MINIMA * por_fila.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(validas, suave / suave.max(axis=1, keepdims=True), 0.0)


def nucleos_y_pozos(normalizado, conteos, n_pixeles):
    """Núcleos (media altura, masa >= 1 %) y pozos por inundación desde los núcleos."""
    marcadas = ski.measure.label(normalizado >= MEDIA_ALTURA, connectivity=2)
    masas = ndimage.sum(conteos, marcadas, index=np.arange(1, marcadas.max() + 1))
    con_masa = np.flatnonzero(masas >= MASA_MINIMA * n_pixeles) + 1
    nucleos = np.zeros_like(marcadas)
    for nueva, vieja in enumerate(con_masa, start=1):
        nucleos[marcadas == vieja] = nueva
    pozos = ski.segmentation.watershed(-normalizado, nucleos) if nucleos.max() else nucleos
    return nucleos, pozos


def choque(img, act, sigma_interno, tolerancia_abs, intensidad=0.0, cada=5, dt=DT, max_pasos=200):
    """Choque (Osher-Rudin, minmod); con intensidad > 0, segunda etapa con atracción al gris del pozo.

    Devuelve imagen, pasos e iteración en que se activó el término de pozos (None si no se activó).
    """
    I = img.astype(float).copy()
    objetivo, densidad = I.copy(), np.zeros_like(I)
    activacion = None
    for paso in range(max_pasos):
        if activacion is not None and (paso - activacion) % cada == 0:
            i_act, i_gris, conteos = paisaje(I, act, tuple(np.percentile(I, (0.1, 99.9))))
            normalizado = normalizar_por_fila(conteos)
            nucleos, pozos = nucleos_y_pozos(normalizado, conteos, I.size)
            n = nucleos.max()
            if n:
                centrales = np.asarray(ndimage.median(I, nucleos[i_act, i_gris], index=np.arange(1, n + 1)))
                objetivo = centrales[pozos[i_act, i_gris] - 1]
                densidad = normalizado[i_act, i_gris]
        S = ndimage.gaussian_filter(I, sigma_interno)
        Sy, Sx = np.gradient(S)
        Syy, Syx = np.gradient(Sy)
        _, Sxx = np.gradient(Sx)
        D2 = (Sx ** 2 * Sxx + 2 * Sx * Sy * Syx + Sy ** 2 * Syy) / (Sx ** 2 + Sy ** 2 + 1e-12)
        P = np.pad(I, 1, mode="edge")
        Ix = _minmod(P[1:-1, 2:] - I, I - P[1:-1, :-2])
        Iy = _minmod(P[2:, 1:-1] - I, I - P[:-2, 1:-1])
        delta = -dt * np.sign(D2) * np.hypot(Ix, Iy)
        if activacion is not None:
            delta += intensidad * dt * densidad * (objetivo - I)
        I += delta
        if np.mean(np.abs(delta)) < tolerancia_abs:
            if intensidad > 0 and activacion is None:
                activacion = paso + 1
                continue
            break
    return I, paso + 1, activacion


def segmentar(tras_choque, act):
    """Paisaje, núcleos, pozos y población de cada píxel (numeradas por gris central)."""
    rango = tuple(np.percentile(tras_choque, (0.1, 99.9)))
    i_act, i_gris, conteos = paisaje(tras_choque, act, rango)
    normalizado = normalizar_por_fila(conteos)
    nucleos, pozos = nucleos_y_pozos(normalizado, conteos, tras_choque.size)
    n = nucleos.max()
    centrales = np.asarray(ndimage.median(tras_choque, nucleos[i_act, i_gris], index=np.arange(1, n + 1)))
    orden = np.argsort(centrales)
    rango_de = np.empty(n, int)
    rango_de[orden] = np.arange(n)
    return {"rango": rango, "conteos": conteos, "normalizado": normalizado, "nucleos": nucleos, "pozos": pozos,
            "centrales": centrales[orden], "rango_de": rango_de, "poblacion": rango_de[pozos[i_act, i_gris] - 1],
            "en_nucleo": float(np.mean(nucleos[i_act, i_gris] > 0))}


def recorte(ax, img, titulo, vmin, vmax, cmap="gray", zoom=False):
    v = img[Y0:Y0 + LADO, X0:X0 + LADO] if zoom else img
    im = ax.imshow(v, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(titulo)
    ax.set_axis_off()
    return im


def caja(ax, texto):
    ax.text(0.98, 0.97, texto, transform=ax.transAxes, ha="right", va="top", fontsize=13, bbox=CAJA)


def guardar(fig, nombre, titulo):
    fig.suptitle(titulo, fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(SALIDA / nombre, dpi=80)
    plt.close(fig)


original = np.load(ROI, allow_pickle=False).astype(float)
suave = suavizar(original)
ruido_original, ruido_suave = estimar_ruido(original), estimar_ruido(suave)
act = actividad_de_partida(suave)
gradiente = ski.filters.sobel(suave)
sigma_deg, bordes_usados = estimar_sigma_borde(suave)
radio_enfoque = max(sigma_deg, 0.5)
realzada = ski.filters.unsharp_mask(suave, radius=radio_enfoque, amount=CANTIDAD_ENFOQUE, preserve_range=True)
print(f"degradé estimado {sigma_deg:.2f} px ({bordes_usados} bordes aislados) -> radio del enfoque {radio_enfoque:.2f}")
versiones = {}
for nombre, intensidad in (("choque solo", 0.0), (f"choque con pozos (intensidad {INTENSIDAD_POZOS:g})", INTENSIDAD_POZOS)):
    tras, pasos, activacion = choque(realzada, act, SIGMA_INTERNO, 0.1 * ruido_suave, intensidad)
    s = segmentar(tras, act)
    versiones[nombre] = {"img": tras, "pasos": pasos, "activacion": activacion, **s}
    print(f"{nombre:<34} {pasos} pasos (pozos desde {activacion}); {len(s['centrales'])} núcleos en "
          f"{np.round(s['centrales'], 3)}; en núcleo {s['en_nucleo']:.0%}; área "
          + ", ".join(f"{np.mean(s['poblacion'] == j):.1%}" for j in range(len(s['centrales']))))
print(f"ruido estimado: original {ruido_original:.4f}, suavizada {ruido_suave:.4f}")
rango_gris = tuple(np.percentile(np.concatenate([original.ravel(), suave.ravel(), realzada.ravel()]
                                                + [v["img"].ravel() for v in versiones.values()]), (0.1, 99.9)))

# 12a: partida y suavizado
fig, ejes = plt.subplots(2, 3, figsize=(28, 15), gridspec_kw={"width_ratios": [1, 1, 1.5]})
for fila, (titulo, v) in enumerate((("ROI 1 original", original), ("suavizada (Etapa 2, NLM)", suave))):
    im = recorte(ejes[fila, 0], v, titulo, *rango_gris)
    fig.colorbar(im, ax=ejes[fila, 0], fraction=0.046)
    ejes[fila, 0].add_patch(plt.Rectangle((X0, Y0), LADO, LADO, fill=False, ec="y", lw=2))
    recorte(ejes[fila, 1], v, f"{titulo}: recorte (cuadro amarillo)", *rango_gris, zoom=True)
    ax = ejes[fila, 2]
    ax.hist(v.ravel(), bins=200, range=rango_gris, color="0.3")
    ax.set_xlabel("gris")
    ax.set_ylabel("cantidad de píxeles")
    ax.set_title(f"histograma: {titulo}")
caja(ejes[1, 2], f"ruido estimado: original {ruido_original:.4f}, suavizada {ruido_suave:.4f}\n"
                 "parámetros del NLM: automáticos (src/sem/suavizado.py)")
guardar(fig, "13a_partida.png", "Paso 0: ROI 1 original y suavizada (la suavizada es la imagen de partida)")

# 12b: choque
fig, ejes = plt.subplots(2, 4, figsize=(34, 15), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
for fila, (nombre, v) in enumerate(versiones.items()):
    recorte(ejes[fila, 0], suave, "suavizada, recorte", *rango_gris, zoom=True)
    recorte(ejes[fila, 1], realzada, f"realzada (enfoque, radio {radio_enfoque:.2f} px)", *rango_gris, zoom=True)
    recorte(ejes[fila, 2], v["img"], f"{nombre}\n({v['pasos']} pasos), recorte", *rango_gris, zoom=True)
    ax = ejes[fila, 3]
    ax.hist(suave.ravel(), bins=200, range=rango_gris, histtype="step", lw=1.5, color="0.6", label="suavizada")
    ax.hist(realzada.ravel(), bins=200, range=rango_gris, histtype="step", lw=2, color="#1f77b4", label="realzada")
    ax.hist(v["img"].ravel(), bins=200, range=rango_gris, histtype="step", lw=2.5, color="k", label="tras el choque")
    ax.set_xlabel("gris")
    ax.set_ylabel("cantidad de píxeles")
    ax.legend(loc="upper left")
    ax.set_title(f"histograma: {nombre}")
    caja(ax, f"enfoque: radio = degradé estimado {sigma_deg:.2f} px, cantidad {CANTIDAD_ENFOQUE:g}\n"
             f"choque: σ interno {SIGMA_INTERNO} px, dt {DT} (fijos)\n"
             f"frena si cambio medio < 0.1 × ruido = {0.1 * ruido_suave:.5f}\n→ {v['pasos']} pasos"
             + (f"\npozos desde el paso {v['activacion']}" if v["activacion"] else ""))
guardar(fig, "13b_choque.png", "Paso 1: realce (máscara de enfoque) y filtro de choque, solo y con pozos")

# 12c: actividad
fig, ejes = plt.subplots(2, 3, figsize=(30, 16), gridspec_kw={"width_ratios": [1, 1, 1.5]})
lim_grad = np.percentile(gradiente, 99.9)
for fila, (titulo, v, lim, texto) in enumerate((
        ("gradiente: sobel(suavizada)", gradiente, lim_grad, "módulo del gradiente (sobel) de la suavizada\nsin promediar"),
        ("actividad: promedio del gradiente", act, 1, "gaussiana(sobel(suavizada), 3 px)\nnormalizada: percentil 99.9 = 1"))):
    im = recorte(ejes[fila, 0], v, titulo, 0, lim, "magma")
    fig.colorbar(im, ax=ejes[fila, 0], fraction=0.046)
    recorte(ejes[fila, 1], v, "recorte", 0, lim, "magma", zoom=True)
    ax = ejes[fila, 2]
    ax.hist(v.ravel(), bins=150, range=(0, lim), color="0.3")
    ax.axvline(np.median(v), color="#d62728", lw=2, ls="--", label=f"mediana {np.median(v):.4g}")
    ax.axvline(np.mean(v), color="#1f77b4", lw=2, ls=":", label=f"media {np.mean(v):.4g}")
    ax.set_xlabel(titulo.split(":")[0])
    ax.set_ylabel("cantidad de píxeles")
    ax.set_title(f"histograma: {titulo.split(':')[0]}")
    ax.legend(loc="center right")
    caja(ax, texto)
guardar(fig, "13c_actividad.png", "Paso 2: gradiente y actividad (promedio local del gradiente) de la imagen de partida")

# 12d: paisajes (a ciegas, normalizado, histograma) antes y después del choque
columnas = [("suavizada", suave), ("realzada (antes del choque)", realzada)] + [(n, v["img"]) for n, v in versiones.items()]
centros = rango_gris[0] + (np.arange(BINS) + 0.5) / BINS * (rango_gris[1] - rango_gris[0])
extent = (*rango_gris, 0, 1)
fig, ejes = plt.subplots(3, len(columnas), figsize=(8 * len(columnas), 21), sharex=True)
for col, (titulo, v) in enumerate(columnas):
    _, _, conteos = paisaje(v, act, rango_gris)
    ejes[0, col].imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
    ejes[0, col].set_title(f"{titulo}\nlog(1 + cuentas)")
    ejes[1, col].imshow(normalizar_por_fila(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
    ejes[1, col].set_title("fila / máximo de la fila")
    for fila in range(2):
        ejes[fila, col].set_ylabel("actividad de partida (0-1)")
    ejes[2, col].plot(centros, conteos.sum(axis=0), "k", lw=2.5)
    ejes[2, col].set_title("histograma de gris")
    ejes[2, col].set_xlabel("gris")
    ejes[2, col].set_ylabel("cantidad de píxeles")
guardar(fig, "13d_paisajes.png", "Paso 3: paisaje gris × actividad de partida, en cada paso")

# 12e: núcleos y pozos
fig, ejes = plt.subplots(1, len(versiones), figsize=(13 * len(versiones), 11))
for ax, (nombre, v) in zip(ejes, versiones.items()):
    cg = v["rango"][0] + (np.arange(BINS) + 0.5) / BINS * (v["rango"][1] - v["rango"][0])
    ca = (np.arange(BINS) + 0.5) / BINS
    im = ax.imshow(v["normalizado"], origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(*v["rango"], 0, 1))
    fig.colorbar(im, ax=ax, fraction=0.046, label="fila / máximo de la fila")
    for k in range(1, v["nucleos"].max() + 1):
        ax.contour(cg, ca, v["pozos"] == k, levels=[0.5], colors="k", linewidths=1.5)
        ax.contour(cg, ca, v["nucleos"] == k, levels=[0.5], colors=COLORES[v["rango_de"][k - 1]], linewidths=3.5)
    ax.set_xlabel("gris tras el choque")
    ax.set_ylabel("actividad de partida (0-1)")
    ax.set_title(f"{nombre}: {v['nucleos'].max()} núcleos (color), fronteras de pozos (negro)")
    caja(ax, "núcleo: bins ≥ media altura de su fila, conexos, masa ≥ 1 %\n"
             "pozos: inundación (watershed) desde los núcleos\n"
             f"gris central: {', '.join(f'{g:.3f}' for g in v['centrales'])}\n"
             f"píxeles dentro de un núcleo: {v['en_nucleo']:.0%}")
guardar(fig, "13e_nucleos_y_pozos.png", "Pasos 4 y 5: paisaje normalizado, núcleos y pozos")

# 12f: resultado
fig, ejes = plt.subplots(2, 3, figsize=(30, 16))
for fila, (nombre, v) in enumerate(versiones.items()):
    n = len(v["centrales"])
    cmap = ListedColormap(COLORES[:n])
    recorte(ejes[fila, 0], v["poblacion"], f"{nombre}\npoblaciones (por gris, 0 = la más oscura)", -0.5, n - 0.5, cmap)
    for ax, zoom in ((ejes[fila, 1], False), (ejes[fila, 2], True)):
        recorte(ax, suave, "contornos sobre la suavizada" + (" (recorte)" if zoom else ""), *rango_gris, zoom=zoom)
        pob = v["poblacion"][Y0:Y0 + LADO, X0:X0 + LADO] if zoom else v["poblacion"]
        for j in range(n - 1):
            ax.contour(pob >= j + 1, levels=[0.5], colors=COLORES[j + 1], linewidths=2 if zoom else 0.8)
    ejes[fila, 0].text(0.01, 0.01, "\n".join(f"población {j} (gris {g:.3f}): {np.mean(v['poblacion'] == j):.1%}"
                                             for j, g in enumerate(v["centrales"])),
                       transform=ejes[fila, 0].transAxes, fontsize=14, va="bottom", bbox=CAJA)
guardar(fig, "13f_resultado.png", "Paso 6: poblaciones de la ROI 1 (contorno de color = borde de la población de ese color)")
print("ok")
