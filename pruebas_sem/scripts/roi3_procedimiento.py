"""ROI 3 real (BSE_1_3): regiones de análisis y, dentro de cada una, poblaciones por núcleos y pozos.

1. Regiones de análisis (igual que segmentacion_por_region.py, figuras 28-30):
   - zonas por gris: picos y valles del histograma de la suavizada (prominencia 5 %, distancia 8 bins,
     masa 1 %, valle < 75 %); manchitas < 1 % del área pasan a la zona vecina;
   - franja de borde entre zonas: distancia donde el gris recorre el 90 % del camino al interior;
   - actividad sin cruzar límites de zona: gaussiana(sobel), 25 px;
   - dentro de cada zona, clases de actividad por picos y valles.
2. Cada región se aísla: se recorta y lo que queda fuera de ella (otras regiones y la franja de
   borde) se reemplaza por el píxel más cercano de la región, para que los filtros no vean las
   características de las demás. Sobre la región aislada:
   - ruido y degradé estimados en la región (bordes aislados cuyo máximo cae dentro de ella);
   - realce: máscara de enfoque (radio = degradé estimado, cantidad 1);
   - filtro de choque (σ interno 1 px, dt 0.25, frena con cambio medio < 0.1 x ruido de la región);
   - actividad de partida: gaussiana(sobel(región aislada)), 3 px.
3. En cada región, con solo sus píxeles (sin la franja de borde):
   - paisaje: gris tras el choque x actividad de partida (normalizada 0-1 con el percentil 99.9 de
     la región);
   - normalización por fila (fila suavizada 1 bin / su máximo);
   - núcleos: bins sobre la media altura de su fila, conexos, con masa >= 1 % de la región; cada
     núcleo es una población (no se exige valle);
   - pozos: inundación (watershed) desde los núcleos; cada píxel toma el pozo de su bin.
4. La franja de borde toma la etiqueta del píxel clasificado más cercano.
Figuras: 16a regiones, 16b zona de interés de cada región antes y después del choque,
16c evolución del paisaje de cada región, 16d histogramas de cada región en cada paso, 16e resultado
con recortes (las 15* aplicaban realce y choque a toda la ROI).
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/roi3/.
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

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13})

ROI = Path("data/sem/roi/BSE_1/BSE_1_3/roi_BSE_1_3.npy")
SALIDA = Path("pruebas_sem/roi3")
SALIDA.mkdir(parents=True, exist_ok=True)
# Regiones de análisis (criterios de TEM, como en segmentacion_por_region.py)
SIGMA_ACTIVIDAD_REGIONES, AREA_MINIMA_COMPONENTE = 25, 0.01
BINS_H, SIGMA_H, PROMINENCIA, DISTANCIA, MASA_MINIMA, VALLE_MAXIMO = 256, 2.0, 0.05, 8, 0.01, 0.75
# Poblaciones (como en roi1_procedimiento.py)
SIGMA_ACT, BINS, FILA_MINIMA, MEDIA_ALTURA, MASA_NUCLEO = 3, 128, 0.005, 0.5, 0.01
SIGMA_INTERNO, DT, CANTIDAD_ENFOQUE = 1.0, 0.25, 1.0
COLORES_REGION = ["#1f3fbf", "#e6a100", "#12a35a", "#c2185b", "#00acc1", "#6d4c41", "#7cb342", "#5e35b1"]
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
CAJA = {"facecolor": "#fffbe6", "edgecolor": "0.4", "alpha": 0.95, "boxstyle": "round"}


# --- regiones de análisis (copiadas de segmentacion_por_region.py) ---

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
    limpia = np.full(zonas.shape, -1)
    for z in np.unique(zonas):
        grandes = ski.morphology.remove_small_objects(zonas == z, max_size=int(AREA_MINIMA_COMPONENTE * zonas.size) - 1)
        limpia[grandes] = z
    _, (fi, co) = ndimage.distance_transform_edt(limpia < 0, return_indices=True)
    limpia = limpia[fi, co]
    _, renumerada = np.unique(limpia, return_inverse=True)
    return renumerada.reshape(zonas.shape)


def franja_de_borde(gris01, zonas):
    """Píxeles de transición junto al límite entre zonas (ancho: 90 % del camino al gris interior)."""
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
        promedio = ski.filters.gaussian(gradiente * valido, SIGMA_ACTIVIDAD_REGIONES) / np.maximum(
            ski.filters.gaussian(valido, SIGMA_ACTIVIDAD_REGIONES), 1e-6)
        actividad[mascara] = promedio[mascara]
    return actividad


# --- realce, choque y poblaciones (como en roi1_procedimiento.py) ---

def aislar_region(img, mascara, margen=20):
    """Recorte de la región con lo de afuera reemplazado por el píxel más cercano de la región.

    Así los filtros (realce, choque, actividad) no ven el gris ni los bordes de las otras regiones,
    ni de la franja de borde: fuera de la región la imagen continúa con los valores de su orilla.
    Devuelve el recorte relleno y el slice del recorte en la imagen completa.
    """
    filas, cols = np.nonzero(mascara)
    caja = (slice(max(filas.min() - margen, 0), filas.max() + margen + 1),
            slice(max(cols.min() - margen, 0), cols.max() + margen + 1))
    sub, m = img[caja], mascara[caja]
    _, (fi, co) = ndimage.distance_transform_edt(~m, return_indices=True)
    return sub[fi, co], caja


def estimar_sigma_borde(img, muestras=1500, semilla=0, dentro=None):
    """Ancho del degradé (σ de una gaussiana, px) a partir del ancho 10-90 % de bordes aislados.

    Con `dentro`, solo se usan bordes cuyo máximo de gradiente cae dentro de esa máscara.
    """
    gy, gx = ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1)
    maximos = ski.feature.peak_local_max(np.hypot(gx, gy), min_distance=3, threshold_rel=0.2, exclude_border=12)
    if dentro is not None:
        maximos = maximos[dentro[maximos[:, 0], maximos[:, 1]]]
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


def choque(img, sigma_interno, tolerancia_abs, dt=DT, max_pasos=200):
    """Filtro de choque (Osher-Rudin, esquema minmod)."""
    I = img.astype(float).copy()
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
        if np.mean(np.abs(delta)) < tolerancia_abs:
            break
    return I, paso + 1


def poblaciones_de_region(gris, act, n_pixeles):
    """Paisaje normalizado, núcleos, pozos y población de cada píxel de la región (por gris central)."""
    rango = tuple(np.percentile(gris, (0.1, 99.9)))
    act = np.clip(act / np.percentile(act, 99.9), 0, 1)
    i_act = np.clip((act * BINS).astype(int), 0, BINS - 1)
    i_gris = np.clip(((gris - rango[0]) / max(rango[1] - rango[0], 1e-9) * BINS).astype(int), 0, BINS - 1)
    conteos = np.bincount(np.ravel_multi_index((i_act, i_gris), (BINS, BINS)),
                          minlength=BINS ** 2).reshape(BINS, BINS).astype(float)
    suave = ndimage.gaussian_filter(conteos, 1)
    por_fila = conteos.sum(axis=1, keepdims=True)
    validas = por_fila >= FILA_MINIMA * por_fila.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        normalizado = np.where(validas, suave / suave.max(axis=1, keepdims=True), 0.0)
    marcadas = ski.measure.label(normalizado >= MEDIA_ALTURA, connectivity=2)
    m = ndimage.sum(conteos, marcadas, index=np.arange(1, marcadas.max() + 1))
    con_masa = np.flatnonzero(m >= MASA_NUCLEO * n_pixeles) + 1
    nucleos = np.zeros_like(marcadas)
    for nueva, vieja in enumerate(con_masa, start=1):
        nucleos[marcadas == vieja] = nueva
    pozos = ski.segmentation.watershed(-normalizado, nucleos)
    n = nucleos.max()
    centrales = np.asarray(ndimage.median(gris, nucleos[i_act, i_gris], index=np.arange(1, n + 1)))
    orden = np.argsort(centrales)
    rango_de = np.empty(n, int)
    rango_de[orden] = np.arange(n)
    return {"rango": rango, "normalizado": normalizado, "nucleos": nucleos, "pozos": pozos, "rango_de": rango_de,
            "centrales": centrales[orden], "poblacion": rango_de[pozos[i_act, i_gris] - 1],
            "en_nucleo": float(np.mean(nucleos[i_act, i_gris] > 0))}


def tonos(color, n):
    """n tonos de un color, de oscuro a claro."""
    base = np.array(to_rgb(color))
    if n == 1:
        return [tuple(base)]
    return [tuple(base * (0.35 + 0.65 * k / (n - 1)) + (1 - base) * 0.6 * k / (n - 1)) for k in range(n)]


def guardar(fig, nombre, titulo):
    fig.suptitle(titulo, fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(SALIDA / nombre, dpi=80)
    plt.close(fig)


# ---------------- 1. Regiones de análisis
original = np.load(ROI, allow_pickle=False).astype(float)
suave = suavizar(original)
gris01 = escalar(suave)
zonas = zonas_por_gris(gris01)
borde, anchos = franja_de_borde(gris01, zonas)
actividad25 = actividad_sin_cruzar(suave, zonas, borde)
regiones = np.full(suave.shape, -1)
descripcion = []
for z in np.unique(zonas):
    mascara = (zonas == z) & ~borde
    a = escalar(actividad25[mascara])
    clases = np.digitize(a, cortes_por_valles(a))
    for c in np.unique(clases):
        sub = np.zeros(suave.shape, bool)
        sub[mascara] = clases == c
        regiones[sub] = len(descripcion)
        descripcion.append(f"zona de gris {z}, actividad {c}")
nr = len(descripcion)
print(f"{nr} regiones; franja de borde {borde.mean():.1%} (anchos {anchos})")

# ---------------- 2 y 3. Dentro de cada región: realce, choque, actividad, paisaje, núcleos y pozos
NOMBRE_COLOR = ["azul", "amarilla", "verde", "magenta", "celeste", "marrón", "oliva", "violeta"]
nombre_region = [f"región {NOMBRE_COLOR[r % len(NOMBRE_COLOR)]}" for r in range(nr)]
color_region = [COLORES_REGION[r % len(COLORES_REGION)] for r in range(nr)]
realzada = np.full(suave.shape, np.nan)
tras_choque = np.full(suave.shape, np.nan)
parametros, aisladas = [], []
final = np.full(suave.shape, -1)
colores_finales, leyenda, resultados = [], [], []
for r in range(nr):
    mascara = regiones == r
    img_r, caja_r = aislar_region(suave, mascara)
    m_r = mascara[caja_r]
    ruido_r = estimar_ruido(img_r)
    sigma_r, usados_r = estimar_sigma_borde(img_r, dentro=m_r)
    radio_r = max(sigma_r, 0.5)
    realzada_r = ski.filters.unsharp_mask(img_r, radius=radio_r, amount=CANTIDAD_ENFOQUE, preserve_range=True)
    choque_r, pasos_r = choque(realzada_r, SIGMA_INTERNO, 0.1 * ruido_r)
    act_r = ski.filters.gaussian(ski.filters.sobel(img_r), SIGMA_ACT)
    realzada[caja_r][m_r] = realzada_r[m_r]
    tras_choque[caja_r][m_r] = choque_r[m_r]
    parametros.append((ruido_r, sigma_r, usados_r, pasos_r))
    aisladas.append((img_r, realzada_r, choque_r, m_r, act_r))
    print(f"{nombre_region[r]}: ruido {ruido_r:.4f}; degradé {sigma_r:.2f} px ({usados_r} bordes aislados); "
          f"choque {pasos_r} pasos")
    res = poblaciones_de_region(choque_r[m_r], act_r[m_r], mascara.sum())
    n = len(res["centrales"])
    tn = tonos(color_region[r], n)
    for p in range(n):
        sub = np.zeros(suave.shape, bool)
        sub[mascara] = res["poblacion"] == p
        final[sub] = len(colores_finales)
        colores_finales.append(tn[p])
        leyenda.append(f"{nombre_region[r]} · población {p + 1} (gris {res['centrales'][p]:.3f}, "
                       f"{np.mean(res['poblacion'] == p):.0%} de la región)")
    resultados.append(res)
    print(f"{nombre_region[r]} ({mascara.mean():.1%} de la ROI): {n} núcleos en "
          f"{np.round(res['centrales'], 3)}; en núcleo {res['en_nucleo']:.0%}; reparto "
          + ", ".join(f"{np.mean(res['poblacion'] == p):.0%}" for p in range(n)))
_, (fi, co) = ndimage.distance_transform_edt(final < 0, return_indices=True)
final = final[fi, co]
lo, hi = np.nanpercentile(np.concatenate([suave.ravel(), tras_choque.ravel()]), (0.5, 99.5))


def marco(ax, color):
    """Borde del panel en el color de la región."""
    for lado in ax.spines.values():
        lado.set_visible(True)
        lado.set_edgecolor(color)
        lado.set_linewidth(6)


def paisaje_log(gris, act, rango):
    """log(1 + cuentas) del paisaje gris x actividad (actividad normalizada con su percentil 99.9)."""
    a = np.clip(act / np.percentile(act, 99.9), 0, 1)
    conteos, _, _ = np.histogram2d(a, gris, bins=BINS, range=((0, 1), rango))
    return np.log1p(conteos)


def ventana_mas_llena(mascara, lado):
    """Esquina (fila, columna) de la ventana de lado x lado con más píxeles de la región."""
    lado = min(lado, *mascara.shape)
    llenado = ndimage.uniform_filter(mascara.astype(float), lado, mode="constant")
    medio = lado // 2
    interior = llenado[medio:mascara.shape[0] - lado + medio + 1, medio:mascara.shape[1] - lado + medio + 1]
    f, c = np.unravel_index(np.argmax(interior), interior.shape)
    return int(f), int(c), lado


gris_fondo = np.clip((suave - lo) / (hi - lo), 0, 1)
rgb_fondo = np.repeat(gris_fondo[..., None], 3, axis=-1)
LADO_RECORTE = 140

# 16a: regiones de análisis
fig, ejes = plt.subplots(1, 4, figsize=(36, 10))
ejes[0].imshow(suave, cmap="gray", vmin=lo, vmax=hi)
ejes[0].set_title("ROI 3 suavizada (Etapa 2)")
ejes[1].imshow(np.where(borde, -1, zonas), cmap=ListedColormap(["k", "0.55", "0.75", "0.92", "0.4", "0.3"][:zonas.max() + 2]),
               vmin=-1.5, vmax=zonas.max() + 0.5, interpolation="nearest")
ejes[1].set_title(f"zonas por gris ({zonas.max() + 1}, en grises); negro = franja de borde\nanchos {anchos} px")
im = ejes[2].imshow(actividad25, cmap="magma")
fig.colorbar(im, ax=ejes[2], fraction=0.046)
ejes[2].set_title(f"actividad sin cruzar zonas (σ = {SIGMA_ACTIVIDAD_REGIONES} px)")
ejes[3].imshow(np.where(regiones < 0, np.nan, regiones), cmap=ListedColormap(color_region),
               vmin=-0.5, vmax=nr - 0.5, interpolation="nearest")
ejes[3].set_title(f"{nr} regiones de análisis (blanco = franja de borde)")
ejes[3].legend(handles=[Patch(color=color_region[r], label=f"{nombre_region[r]}: {np.mean(regiones == r):.0%} de la ROI")
                        for r in range(nr)], loc="lower left", fontsize=14)
for ax in ejes:
    ax.set_axis_off()
guardar(fig, "16a_regiones.png", "Paso 1: regiones de análisis (zonas por gris + clases de actividad)")

# 16b: zona de interés de cada región, antes y después del realce y el choque
fig, ejes = plt.subplots(2, nr, figsize=(9.5 * nr, 19), squeeze=False)
for r in range(nr):
    mascara = regiones == r
    ruido_r, sigma_r, usados_r, pasos_r = parametros[r]
    for fila, (titulo, v) in enumerate((("antes (suavizada)", suave),
                                        (f"después del realce y el choque ({pasos_r} pasos)", tras_choque))):
        ax = ejes[fila, r]
        lo_r, hi_r = np.nanpercentile(v[mascara], (0.5, 99.5))
        g = np.clip((v - lo_r) / max(hi_r - lo_r, 1e-9), 0, 1)
        rgb = np.repeat(np.nan_to_num(g)[..., None], 3, axis=-1)
        rgb[mascara] = 0.6 * rgb[mascara] + 0.4 * np.array(to_rgb(color_region[r]))   # la región, en su color
        rgb[~mascara] = 0.0                                                            # lo demás, en negro
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{nombre_region[r]}\n{titulo}")
        ax.set_xticks([])
        ax.set_yticks([])
        marco(ax, color_region[r])
    ejes[1, r].set_xlabel(f"ruido {ruido_r:.4f} · degradé {sigma_r:.2f} px ({usados_r} bordes)\n"
                          f"realce radio {max(sigma_r, 0.5):.2f} px · choque σ {SIGMA_INTERNO} px", fontsize=15)
guardar(fig, "16b_realce_y_choque.png",
        "Paso 2: cada región (en su color; lo demás en negro) antes y después del realce y el choque, "
        "procesada por separado")

# 16c: evolución del paisaje de cada región
etapas = ("suavizada", "realzada", "tras el choque", "tras el choque, normalizado por fila")
fig, ejes = plt.subplots(nr, len(etapas), figsize=(9 * len(etapas), 8 * nr), squeeze=False)
for r, (img_r, realzada_r, choque_r, m_r, act_r) in enumerate(aisladas):
    res = resultados[r]
    rango = tuple(np.percentile(np.concatenate([img_r[m_r], choque_r[m_r]]), (0.1, 99.9)))
    for col, (titulo, v) in enumerate(zip(etapas, (img_r, realzada_r, choque_r, None))):
        ax = ejes[r, col]
        if v is None:
            ax.imshow(res["normalizado"], origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(*res["rango"], 0, 1))
            cg = res["rango"][0] + (np.arange(BINS) + 0.5) / BINS * (res["rango"][1] - res["rango"][0])
            ca = (np.arange(BINS) + 0.5) / BINS
            for k in range(1, res["nucleos"].max() + 1):
                ax.contour(cg, ca, res["nucleos"] == k, levels=[0.5], colors="k", linewidths=2.5)
            titulo = f"{titulo}\ncontorno = núcleo ({len(res['centrales'])})"
        else:
            ax.imshow(paisaje_log(v[m_r], act_r[m_r], rango), origin="lower", aspect="auto", cmap=CMAP_PAISAJE,
                      extent=(*rango, 0, 1))
        ax.set_title(f"{nombre_region[r]}: {titulo}")
        ax.set_xlabel("gris")
        ax.set_ylabel("actividad (0-1)")
        marco(ax, color_region[r])
guardar(fig, "16c_paisajes_por_region.png", "Paso 3: evolución del paisaje gris × actividad de cada región")

# 16d: todos los histogramas de cada región
fig, ejes = plt.subplots(1, nr, figsize=(10 * nr, 8), squeeze=False)
for r, (img_r, realzada_r, choque_r, m_r, act_r) in enumerate(aisladas):
    ax = ejes[0, r]
    rango = tuple(np.percentile(np.concatenate([img_r[m_r], choque_r[m_r]]), (0.5, 99.5)))
    for titulo, v, color, lw in (("suavizada", img_r, "0.6", 1.8), ("realzada", realzada_r, "#3b6fd4", 2),
                                 ("tras el choque", choque_r, "k", 2.5)):
        ax.hist(v[m_r], bins=150, range=rango, histtype="step", color=color, lw=lw, label=titulo)
    for g in resultados[r]["centrales"]:
        ax.axvline(g, color=color_region[r], lw=3, ls="--")
    ax.set_xlabel("gris")
    ax.set_ylabel("cantidad de píxeles")
    ax.set_title(f"{nombre_region[r]} (rayado = gris central de cada población)")
    ax.legend(loc="upper left")
    marco(ax, color_region[r])
guardar(fig, "16d_histogramas_por_region.png", "Histogramas de gris de cada región en cada paso")

# 16e: imagen segmentada. Cada región en su color; dentro, la población mayoritaria en tono claro y
# las demás en tonos oscuros (cuanto más oscura la población, más oscuro el tono).
segmentada = np.zeros((*suave.shape, 3))
leyenda_e = []
for r, res in enumerate(resultados):
    mascara = regiones == r
    base = np.array(to_rgb(color_region[r]))
    n = len(res["centrales"])
    reparto = np.bincount(res["poblacion"], minlength=n) / res["poblacion"].size
    mayor = int(np.argmax(reparto))
    menores = [p for p in range(n) if p != mayor]
    for p in range(n):
        if p == mayor:
            color = base + (1 - base) * 0.35
        else:
            color = base * (0.55 - 0.4 * menores.index(p) / max(len(menores), 1))
        sub = np.zeros(suave.shape, bool)
        sub[mascara] = res["poblacion"] == p
        segmentada[sub] = color
        leyenda_e.append(Patch(color=color, label=f"{nombre_region[r]}: gris {res['centrales'][p]:.3f} "
                                                   f"({reparto[p]:.0%} de la región)"))
# La franja de borde toma el color del píxel clasificado más cercano
sin_asignar = regiones < 0
_, (fi, co) = ndimage.distance_transform_edt(sin_asignar, return_indices=True)
segmentada = segmentada[fi, co]

fig, ejes = plt.subplots(1, 2, figsize=(34, 15))
ejes[0].imshow(suave, cmap="gray", vmin=lo, vmax=hi)
ejes[0].set_title("ROI 3 suavizada")
ejes[1].imshow(segmentada, interpolation="nearest")
ejes[1].set_title("segmentada: color = región; claro = población mayoritaria, oscuro = las demás")
for ax in ejes:
    ax.set_axis_off()
fig.legend(handles=leyenda_e, loc="lower center", ncol=4, fontsize=16)
fig.suptitle("Paso 4: ROI 3 segmentada", fontsize=24)
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig(SALIDA / "16e_resultado.png", dpi=80)
plt.close(fig)
print("ok")
