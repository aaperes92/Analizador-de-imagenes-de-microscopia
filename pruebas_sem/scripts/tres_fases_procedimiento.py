"""Simulada de TRES fases: el procedimiento de 06e, paso a paso y con todos los datos a la vista.

Imágenes de partida (matriz 0.70 56 %, patrón A 0.60 25 %, patrón B 0.52 19 %; ruido 0.005):
- nítida: el gris de cada fase, sin difuminar (es la imagen ideal);
- con degradé: gaussiana de 1.5 px sobre los grises (desenfoque en las interfaces).

Procedimiento (la verdad solo se usa para colorear y evaluar):
1. filtro de choque sobre la imagen de partida (σ interno 1 px, dt 0.25, frena cuando el cambio
   medio es < 0.1 x ruido estimado);
2. actividad de la imagen de PARTIDA: gaussiana(sobel), 3 px, normalizada 0-1;
3. paisaje: gris tras el choque x actividad de partida (128 x 128 bins);
4. normalización: cada fila (suavizada 1 bin) dividida por su máximo;
5. núcleos: bins sobre la media altura de su fila, conexos, con masa >= 1 % de los píxeles;
   gris central = mediana del gris de los píxeles que caen en el núcleo;
6. pozos: inundación (watershed) desde los núcleos sobre el paisaje normalizado invertido; cada
   píxel toma el pozo de su bin. Los pozos se ordenan por gris (el más oscuro = patrón B).
Con --pozos X, el paso 1 es el choque con pozos de tres_fases_choque_con_pozos.py (dos etapas:
choque solo hasta que frena, después choque + atracción hacia el gris central de cada pozo, con
intensidad X) y las figuras llevan el prefijo 11 en lugar de 10.
Figuras: 10a partida, 10b choque, 10c actividad, 10d paisajes (como 07e), 10e núcleos y pozos
(como 07c), 10f resultado.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/tres_fases/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage

sys.path.insert(0, "src")
from sem.suavizado import estimar_ruido

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})

SALIDA = Path("pruebas_sem/tres_fases")
SALIDA.mkdir(parents=True, exist_ok=True)
ALTO, ANCHO = 520, 560
GRISES = np.array([0.52, 0.60, 0.70])            # 0 = patrón B, 1 = patrón A, 2 = matriz
FASES = ["patrón B", "patrón A", "matriz"]
NOMBRES = [f"{n} ({g:.2f})" for n, g in zip(FASES, GRISES)]
COLORES = ["#1f77b4", "#2ca02c", "#ff7f0e"]
SIGMA_ACT, BINS, FILA_MINIMA = 3, 128, 0.005
MEDIA_ALTURA, MASA_MINIMA = 0.5, 0.01
SIGMA_INTERNO, DT = 1.0, 0.25
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
Y0, X0, LADO = 170, 190, 170
LEYENDA = [Patch(color=c, label=n) for c, n in zip(COLORES, NOMBRES)]
POZOS = float(sys.argv[sys.argv.index("--pozos") + 1]) if "--pozos" in sys.argv else 0.0
PREFIJO = "11" if POZOS > 0 else "10"
OPERACION = f"choque con pozos (intensidad {POZOS:g})" if POZOS > 0 else "filtro de choque"
CAJA = {"facecolor": "#fffbe6", "edgecolor": "0.4", "alpha": 0.95, "boxstyle": "round"}


def campo_alargado(rng, escala, angulo):
    lado = int(1.5 * max(ALTO, ANCHO))
    c = ndimage.rotate(ndimage.gaussian_filter(rng.normal(size=(lado, lado)), escala), angulo,
                       reshape=False, mode="reflect")
    y0, x0 = (lado - ALTO) // 2, (lado - ANCHO) // 2
    c = c[y0:y0 + ALTO, x0:x0 + ANCHO]
    return c / c.std()


def tres_fases_con_presencia(semilla=21):
    """Etiquetas de la simulada: 0 = patrón B, 1 = patrón A, 2 = matriz."""
    rng = np.random.default_rng(semilla)
    campo = campo_alargado(rng, (3.5, 9), 60)
    marcas = np.zeros(campo.shape, int)
    picos = ski.feature.peak_local_max(campo, min_distance=5)
    marcas[tuple(picos.T)] = np.arange(1, len(picos) + 1)
    islas = ski.segmentation.watershed(-campo, marcas, watershed_line=True)
    patron_a = ndimage.binary_dilation(islas == 0, structure=ski.morphology.disk(2))
    campo_b = campo_alargado(rng, (4, 10), 20)
    patron_b = ski.morphology.remove_small_objects(np.abs(campo_b) < 0.25, max_size=59)
    etiquetas = np.full(campo.shape, 2)
    etiquetas[patron_a] = 1
    etiquetas[patron_b] = 0
    return etiquetas, rng


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


def choque_con_pozos(img, act, sigma_interno, tolerancia_abs, intensidad, cada=5, dt=DT, max_pasos=200):
    """Choque en dos etapas: solo hasta que frena; después choque + atracción hacia el gris central del pozo.

    Con el término activo, cada `cada` iteraciones se recalcula el paisaje (gris actual x actividad de
    partida), sus núcleos y pozos; cada píxel es atraído hacia el gris central de su pozo con peso
    intensidad x densidad normalizada de su bin. Devuelve imagen, pasos e iteración de activación.
    """
    I = img.astype(float).copy()
    objetivo, densidad = I.copy(), np.zeros_like(I)
    activacion = None
    for paso in range(max_pasos):
        if activacion is not None and (paso - activacion) % cada == 0:
            rango = tuple(np.percentile(I, (0.1, 99.9)))
            i_act, i_gris, conteos = paisaje(I, act, rango)
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
            if activacion is None:
                activacion = paso + 1
                continue
            break
    return I, paso + 1, activacion


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
    return nucleos, ski.segmentation.watershed(-normalizado, nucleos)


def por_fase_en_paisaje(i_act, i_gris):
    plano = np.ravel_multi_index((i_act, i_gris), (BINS, BINS))
    return np.stack([np.bincount(plano[etiquetas == i], minlength=BINS ** 2).reshape(BINS, BINS) for i in range(3)])


def recorte(ax, img, titulo, vmin, vmax, cmap="gray"):
    im = ax.imshow(img[Y0:Y0 + LADO, X0:X0 + LADO], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(titulo)
    ax.set_axis_off()
    return im


def mapa_fases(ax, fases, titulo):
    recorte(ax, fases, titulo, -0.5, 2.5, ListedColormap(COLORES))


def histograma_por_fase(ax, v, rango, titulo, total=True, bins=200):
    for i in range(3):
        ax.hist(v[etiquetas == i], bins=bins, range=rango, histtype="step", lw=2.5, color=COLORES[i],
                label=f"{FASES[i]}: mediana {np.median(v[etiquetas == i]):.3f}")
    if total:
        ax.hist(v.ravel(), bins=bins, range=rango, histtype="step", lw=1.5, color="k", ls="--", label="total")
    ax.set_title(titulo)
    ax.set_ylabel("cantidad de píxeles")
    ax.legend(loc="upper left", fontsize=12)


def caja(ax, texto):
    ax.text(0.98, 0.97, texto, transform=ax.transAxes, ha="right", va="top", fontsize=13, bbox=CAJA)


def guardar(fig, nombre, titulo):
    fig.suptitle(titulo, fontsize=22)
    fig.legend(handles=LEYENDA, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(SALIDA / nombre, dpi=80)
    plt.close(fig)


etiquetas, rng = tres_fases_con_presencia()
fracciones = [np.mean(etiquetas == i) for i in range(3)]
print("fracciones verdaderas:", {n: f"{f:.1%}" for n, f in zip(FASES, fracciones)})
ruido = rng.normal(0, 0.005, etiquetas.shape)
partida = {"nítida": GRISES[etiquetas] + ruido,
           "con degradé (σ=1.5 px)": ski.filters.gaussian(GRISES[etiquetas], 1.5) + ruido}

# Procedimiento
casos = {}
for nombre, img in partida.items():
    ruido_est = estimar_ruido(img)
    act = actividad_de_partida(img)
    activacion = None
    if POZOS > 0:
        tras_choque, pasos, activacion = choque_con_pozos(img, act, SIGMA_INTERNO, 0.1 * ruido_est, POZOS)
    else:
        tras_choque, pasos = choque(img, SIGMA_INTERNO, 0.1 * ruido_est)
    rango_p = tuple(np.percentile(tras_choque, (0.1, 99.9)))
    i_act, i_gris, conteos = paisaje(tras_choque, act, rango_p)
    normalizado = normalizar_por_fila(conteos)
    nucleos, pozos = nucleos_y_pozos(normalizado, conteos, img.size)
    n = nucleos.max()
    centrales = np.asarray(ndimage.median(tras_choque, nucleos[i_act, i_gris], index=np.arange(1, n + 1)))
    orden = np.argsort(centrales)
    rango_de = np.empty(n, int)
    rango_de[orden] = np.arange(n)
    asignada = rango_de[pozos[i_act, i_gris] - 1]
    en_nucleo = np.mean(nucleos[i_act, i_gris] > 0)
    acierto = [np.mean(asignada[etiquetas == i] == i) for i in range(3)]
    confusion = np.array([[np.mean(asignada[etiquetas == i] == j) for j in range(n)] for i in range(3)])
    casos[nombre] = {"img": img, "ruido_est": ruido_est, "choque": tras_choque, "pasos": pasos, "act": act,
                     "activacion": activacion,
                     "rango": rango_p, "i_act": i_act, "i_gris": i_gris, "conteos": conteos,
                     "normalizado": normalizado, "nucleos": nucleos, "pozos": pozos, "centrales": centrales[orden],
                     "rango_de": rango_de, "asignada": asignada, "en_nucleo": en_nucleo, "acierto": acierto,
                     "confusion": confusion}
    print(f"{nombre:<24} ruido estimado {ruido_est:.4f}; {OPERACION}: {pasos} pasos (pozos desde {activacion}); núcleos en "
          f"{np.round(centrales[orden], 3)}; en núcleo {en_nucleo:.0%}; acierto "
          + ", ".join(f"{f} {a:.0%}" for f, a in zip(FASES, acierto))
          + "; área asignada " + ", ".join(f"{np.mean(asignada == j):.1%}" for j in range(n)))

rango_gris = tuple(np.percentile(np.concatenate([c[k].ravel() for c in casos.values() for k in ("img", "choque")]),
                                 (0.1, 99.9)))

# 10a: imagen de partida
fig, ejes = plt.subplots(2, 3, figsize=(26, 14), gridspec_kw={"width_ratios": [1, 1, 1.6]})
for fila, (nombre, c) in enumerate(casos.items()):
    im = recorte(ejes[fila, 0], c["img"], f"{nombre}: imagen de partida (recorte)", *rango_gris)
    fig.colorbar(im, ax=ejes[fila, 0], fraction=0.046, label="gris")
    mapa_fases(ejes[fila, 1], etiquetas, "verdad (solo para evaluar)")
    ax = ejes[fila, 2]
    histograma_por_fase(ax, c["img"], rango_gris, f"{nombre}: histograma de intensidad")
    for g, col in zip(GRISES, COLORES):
        ax.axvline(g, color=col, lw=2, ls=":")
    ax.set_xlabel("gris")
    caja(ax, "punteado de color = gris real de cada fase\n"
             f"ruido estimado en la imagen: {c['ruido_est']:.4f}\n"
             f"áreas reales: B {fracciones[0]:.0%}, A {fracciones[1]:.0%}, matriz {fracciones[2]:.0%}")
guardar(fig, f"{PREFIJO}a_partida.png", "Paso 0: imágenes de partida (arriba nítida = ideal, abajo con degradé)")

# 10b: filtro de choque
fig, ejes = plt.subplots(2, 4, figsize=(32, 14), gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
for fila, (nombre, c) in enumerate(casos.items()):
    recorte(ejes[fila, 0], c["img"], f"{nombre}\nantes del choque", *rango_gris)
    recorte(ejes[fila, 1], c["choque"], f"después ({c['pasos']} pasos)", *rango_gris)
    cambio = c["choque"] - c["img"]
    im = recorte(ejes[fila, 2], cambio, "Δgris (rojo = se aclaró,\nazul = se oscureció)", -0.06, 0.06, "RdBu_r")
    fig.colorbar(im, ax=ejes[fila, 2], fraction=0.046)
    ax = ejes[fila, 3]
    histograma_por_fase(ax, c["choque"], rango_gris, f"{nombre}: histograma después de {OPERACION}")
    ax.hist(c["img"].ravel(), bins=200, range=rango_gris, histtype="step", lw=1.5, color="0.6", label="antes")
    for g, col in zip(GRISES, COLORES):
        ax.axvline(g, color=col, lw=2, ls=":")
    ax.set_xlabel("gris")
    ax.legend(loc="upper left", fontsize=12)
    caja(ax, "operación: filtro de choque (Osher-Rudin)\n"
             "cada píxel se aleja del gris de sus vecinos,\nhacia el lado del borde al que pertenece\n"
             f"σ interno {SIGMA_INTERNO} px, dt {DT} (fijos)\n"
             f"frena si cambio medio < 0.1 × ruido = {0.1 * c['ruido_est']:.5f}\n"
             f"(sale de la imagen) → {c['pasos']} pasos"
             + (f"\n+ pozos desde el paso {c['activacion']} (intensidad {POZOS:g}):\n"
                "atracción al gris central del pozo × densidad" if POZOS > 0 else ""))
guardar(fig, f"{PREFIJO}b_choque.png", f"Paso 1: {OPERACION}")

# 10c: actividad de partida
fig, ejes = plt.subplots(2, 2, figsize=(22, 14), gridspec_kw={"width_ratios": [1, 1.6]})
for fila, (nombre, c) in enumerate(casos.items()):
    im = recorte(ejes[fila, 0], c["act"], f"{nombre}\nactividad de la imagen de partida", 0, 1, "magma")
    fig.colorbar(im, ax=ejes[fila, 0], fraction=0.046)
    ax = ejes[fila, 1]
    histograma_por_fase(ax, c["act"], (0, 1), f"{nombre}: histograma de actividad", bins=100)
    ax.set_xlabel("actividad (0-1)")
    caja(ax, "operación: gaussiana(sobel(partida), 3 px)\n"
             "normalizada: percentil 99.9 = 1\n"
             "se calcula sobre la imagen de PARTIDA\n(el choque afila bordes y la subiría en todas las fases)\n"
             "σ = 3 px fijo (~ espesor de lo más fino)")
guardar(fig, f"{PREFIJO}c_actividad.png", "Paso 2: actividad (cuántos bordes hay alrededor de cada píxel)")

# 10d: paisajes, como 07e (a ciegas, con fases reales, histograma por fase)
columnas = [("nítida (IDEAL): partida", casos["nítida"]["img"], casos["nítida"]["act"]),
            (f"nítida: tras {OPERACION}", casos["nítida"]["choque"], casos["nítida"]["act"])]
nombre_deg = "con degradé (σ=1.5 px)"
columnas += [("degradé: partida", casos[nombre_deg]["img"], casos[nombre_deg]["act"]),
             (f"degradé: tras {OPERACION}", casos[nombre_deg]["choque"], casos[nombre_deg]["act"])]
rgb = np.array([[int(col[k:k + 2], 16) / 255 for k in (1, 3, 5)] for col in COLORES])
centros_g = rango_gris[0] + (np.arange(BINS) + 0.5) / BINS * (rango_gris[1] - rango_gris[0])
extent = (*rango_gris, 0, 1)
fig, ejes = plt.subplots(3, len(columnas), figsize=(7 * len(columnas), 21), sharex=True)
for col, (titulo, v, act) in enumerate(columnas):
    i_act, i_gris, conteos = paisaje(v, act, rango_gris)
    por_fase = por_fase_en_paisaje(i_act, i_gris)
    ax = ejes[0, col]
    ax.imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
    ax.set_title(f"{titulo}\na ciegas: log(1 + cuentas)")
    ax.set_ylabel("actividad de partida (0-1)")
    ax = ejes[1, col]
    mezcla = np.einsum("fyx,fc->yxc", por_fase, rgb) / np.maximum(conteos, 1)[..., None]
    brillo = (np.log1p(conteos) / np.log1p(conteos).max())[..., None]
    ax.imshow(1 - brillo * (1 - mezcla), origin="lower", aspect="auto", extent=extent)
    ax.set_title("con las fases reales\n(color = fase, brillo = log cuentas)")
    ax.set_ylabel("actividad de partida (0-1)")
    ax = ejes[2, col]
    for i in range(3):
        ax.plot(centros_g, por_fase[i].sum(axis=0), color=COLORES[i], lw=3,
                label=f"{FASES[i]}: mediana {np.median(v[etiquetas == i]):.3f}")
    ax.plot(centros_g, conteos.sum(axis=0), "k--", lw=1.5, label="total")
    ax.set_title("histograma de gris por fase real")
    ax.set_xlabel("gris")
    ax.set_ylabel("cantidad de píxeles")
    ax.legend(loc="upper left", fontsize=12)
    for fila in range(3):
        for g, c_ in zip(GRISES, COLORES):
            ejes[fila, col].axvline(g, color=c_, lw=2, alpha=0.5, ls=":")
guardar(fig, f"{PREFIJO}d_paisajes.png",
        "Paso 3: paisaje gris × actividad de partida (punteado = gris real de cada fase)")

# 10e: normalización, núcleos y pozos, como 07c
fig, ejes = plt.subplots(1, 2, figsize=(26, 11))
for ax, (nombre, c) in zip(ejes, casos.items()):
    ext = (*c["rango"], 0, 1)
    cg = c["rango"][0] + (np.arange(BINS) + 0.5) / BINS * (c["rango"][1] - c["rango"][0])
    ca = (np.arange(BINS) + 0.5) / BINS
    im = ax.imshow(c["normalizado"], origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=ext)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fila / máximo de la fila")
    for k in range(1, c["nucleos"].max() + 1):
        ax.contour(cg, ca, c["pozos"] == k, levels=[0.5], colors="k", linewidths=1.5)
        ax.contour(cg, ca, c["nucleos"] == k, levels=[0.5], colors=COLORES[c["rango_de"][k - 1]], linewidths=3.5)
    for g, col in zip(GRISES, COLORES):
        ax.axvline(g, color=col, lw=2, alpha=0.6, ls="--")
    ax.set_xlabel("gris tras el choque")
    ax.set_ylabel("actividad de partida (0-1)")
    ax.set_title(f"{nombre}: {c['nucleos'].max()} núcleos (color, por orden de gris), fronteras de pozos (negro)")
    caja(ax, "normalización: fila (suavizada 1 bin) / su máximo\n"
             "núcleo: bins ≥ media altura de su fila, conexos,\ncon masa ≥ 1 % de los píxeles\n"
             "pozos: inundación (watershed) desde los núcleos\n"
             f"gris central: {', '.join(f'{g:.3f}' for g in c['centrales'])}\n"
             f"píxeles dentro de un núcleo: {c['en_nucleo']:.0%}")
guardar(fig, f"{PREFIJO}e_nucleos_y_pozos.png",
        "Pasos 4 y 5: paisaje normalizado, núcleos y pozos (rayado = gris real de cada fase)")

# 10f: resultado
fig, ejes = plt.subplots(2, 4, figsize=(30, 14), gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
for fila, (nombre, c) in enumerate(casos.items()):
    mapa_fases(ejes[fila, 0], c["asignada"], f"{nombre}\nresultado")
    ejes[fila, 0].legend(handles=[Patch(color=col, label=f"{f}: {a:.0%} bien")
                                  for col, f, a in zip(COLORES, FASES, c["acierto"])], loc="lower left", fontsize=13)
    error = np.where(c["asignada"] == etiquetas, -1, c["asignada"])
    recorte(ejes[fila, 1], error, f"errores ({np.mean(error >= 0):.1%} de los píxeles)\n"
                                  "color = fase asignada por error", -1.5, 2.5, ListedColormap(["0.93", *COLORES]))
    mapa_fases(ejes[fila, 2], etiquetas, "verdad")
    ax = ejes[fila, 3]
    ax.imshow(c["confusion"], cmap="Greens", vmin=0, vmax=1)
    for i in range(3):
        for j in range(c["confusion"].shape[1]):
            ax.text(j, i, f"{c['confusion'][i, j]:.0%}", ha="center", va="center", fontsize=18,
                    color="w" if c["confusion"][i, j] > 0.5 else "k")
    ax.set_xticks(range(c["confusion"].shape[1]), [f"asignada\n{f}" for f in FASES[:c["confusion"].shape[1]]])
    ax.set_yticks(range(3), [f"real {f}" for f in FASES])
    area = ", ".join(f"{np.mean(c['asignada'] == j):.0%}" for j in range(c["confusion"].shape[1]))
    ax.set_title(f"a dónde va cada fase real\nárea asignada: {area} (real: "
                 f"{', '.join(f'{f:.0%}' for f in fracciones)})")
guardar(fig, f"{PREFIJO}f_resultado.png", "Paso 6: cada píxel toma el pozo de su bin; resultado contra la verdad")
print("ok")
