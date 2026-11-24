"""Simulada de TRES fases, paso a paso: nítida contra degradé, con las operaciones en orden.

Imágenes de partida (misma simulada que tres_fases_paisaje_normalizado.py: matriz 0.70 56 %,
patrón A 0.60 25 %, patrón B 0.52 19 %):
- nítida: gris de cada fase + ruido 0.005;
- con degradé: gaussiana de 1.5 px sobre los grises (desenfoque en las interfaces) + ruido 0.005.
No se suavizan: representan la imagen tal como sale de la etapa anterior.

Operaciones, en orden:
1. enfoque: máscara de enfoque (radio = degradé estimado en bordes aislados, cantidad 1);
2. choque: filtro de choque (σ interno 1 px, frena con cambio medio < 0.1 x ruido);
3. paisaje gris-actividad (actividad = gaussiana(sobel), 3 px) de la imagen tras el choque;
4. normalización del paisaje: cada fila (suavizada 1 bin) dividida por su máximo;
5. regiones con área: bins sobre la media altura de su fila, conexos, con masa >= 1 % de los
   píxeles; vuelta a la imagen (sin región -> píxel etiquetado más cercano).
Con la opción --sin-enfoque se saltea el paso 1 (el choque se aplica directo a la ficticia).
Figuras: 04a imágenes, 04b histogramas, 04c paisajes de cada paso, 04d regiones y resultado
(prefijo 05 en lugar de 04 con --sin-enfoque).
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
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15})

SALIDA = Path("pruebas_sem/tres_fases")
SALIDA.mkdir(parents=True, exist_ok=True)
ALTO, ANCHO = 520, 560
GRISES = np.array([0.52, 0.60, 0.70])            # 0 = patrón B, 1 = patrón A, 2 = matriz
FASES = ["patrón B", "patrón A", "matriz"]
NOMBRES = [f"{n} ({g:.2f})" for n, g in zip(FASES, GRISES)]
COLORES = ["#1f77b4", "#2ca02c", "#ff7f0e"]
SIGMA_ACT, BINS, FILA_MINIMA = 3, 128, 0.005
MEDIA_ALTURA, MASA_MINIMA, CERCA_DEL_GRIS = 0.5, 0.01, 0.01
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
Y0, X0, LADO = 170, 190, 170
LEYENDA = [Patch(color=c, label=n) for c, n in zip(COLORES, NOMBRES)]
SIN_ENFOQUE = "--sin-enfoque" in sys.argv
PREFIJO = "05" if SIN_ENFOQUE else "04"
CADENA = "ficticia → choque" if SIN_ENFOQUE else "ficticia → enfoque → choque"


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


def choque(img, sigma_interno, tolerancia_abs, dt=0.25, max_pasos=200):
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


def paisaje(img, rango):
    """Conteos del paisaje gris-actividad, actividad normalizada 0-1 y bordes de los bins de gris."""
    act = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACT)
    act = (act - act.min()) / (act.max() - act.min())
    conteos, _, bordes_g = np.histogram2d(act.ravel(), img.ravel(), bins=BINS, range=((0, 1), rango))
    return conteos, act, bordes_g


def regiones_del_paisaje(img, rango):
    """Paisaje normalizado por fila, regiones con masa y segmentación por vuelta atrás."""
    conteos, act, bordes_g = paisaje(img, rango)
    suave = ndimage.gaussian_filter(conteos, 1)
    por_fila = conteos.sum(axis=1, keepdims=True)
    validas = por_fila >= FILA_MINIMA * por_fila.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        normalizado = np.where(validas, suave / suave.max(axis=1, keepdims=True), 0.0)
    marcadas = ski.measure.label(normalizado >= MEDIA_ALTURA, connectivity=2)
    masas = ndimage.sum(conteos, marcadas, index=np.arange(1, marcadas.max() + 1))
    con_masa = np.flatnonzero(masas >= MASA_MINIMA * img.size) + 1
    regiones = np.zeros_like(marcadas)
    for nueva, vieja in enumerate(con_masa, start=1):
        regiones[marcadas == vieja] = nueva
    i_act = np.clip((act * BINS).astype(int), 0, BINS - 1)
    i_gris = np.clip(((img - rango[0]) / (rango[1] - rango[0]) * BINS).astype(int), 0, BINS - 1)
    directa = regiones[i_act, i_gris]
    _, (iy, ix) = ndimage.distance_transform_edt(directa == 0, return_indices=True)
    return {"normalizado": normalizado, "regiones": regiones, "directa": directa, "segmentada": directa[iy, ix],
            "centros_g": (bordes_g[:-1] + bordes_g[1:]) / 2}


def evaluar(r, etiquetas):
    """Asocia cada región a la fase verdadera mayoritaria y mide el acierto por fase."""
    n = r["regiones"].max()
    fase_de = np.array([np.bincount(etiquetas[r["segmentada"] == k], minlength=3).argmax() for k in range(1, n + 1)])
    prediccion = fase_de[r["segmentada"] - 1]
    return fase_de, prediccion, [np.mean(prediccion[etiquetas == i] == i) for i in range(3)]


def mapa_fases(ax, fases, titulo):
    ax.imshow(fases[Y0:Y0 + LADO, X0:X0 + LADO], cmap=ListedColormap(COLORES), vmin=-0.5, vmax=2.5,
              interpolation="nearest")
    ax.set_title(titulo)
    ax.set_axis_off()


def guardar(fig, nombre, titulo, leyenda=True):
    fig.suptitle(titulo, fontsize=22)
    if leyenda:
        fig.legend(handles=LEYENDA, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04 if leyenda else 0, 1, 0.95))
    fig.savefig(SALIDA / nombre, dpi=80)
    plt.close(fig)


etiquetas, rng = tres_fases_con_presencia()
print("fracciones verdaderas:", {n: f"{np.mean(etiquetas == i):.1%}" for i, n in enumerate(FASES)})
ruido = rng.normal(0, 0.005, etiquetas.shape)
partida = {"nítida": GRISES[etiquetas] + ruido,
           "con degradé (σ=1.5 px)": ski.filters.gaussian(GRISES[etiquetas], 1.5) + ruido}

pasos = {}
for nombre, ficticia in partida.items():
    sigma_deg, usados = estimar_sigma_borde(ficticia)
    pasos[nombre] = {"1. ficticia": ficticia}
    entrada = ficticia
    if not SIN_ENFOQUE:
        entrada = ski.filters.unsharp_mask(ficticia, radius=max(sigma_deg, 0.5), amount=1.0, preserve_range=True)
        pasos[nombre][f"2. enfoque (radio {max(sigma_deg, 0.5):.2f} px)"] = entrada
    con_choque, iteraciones = choque(entrada, 1.0, 0.1 * estimar_ruido(ficticia))
    pasos[nombre][f"{len(pasos[nombre]) + 1}. choque ({iteraciones} pasos)"] = con_choque
    print(f"{nombre}: degradé estimado {sigma_deg:.2f} px ({usados} bordes aislados); choque {iteraciones} pasos")
N_PASOS = len(next(iter(pasos.values())))
rango = tuple(np.percentile(np.concatenate([v.ravel() for p in pasos.values() for v in p.values()]), (0.1, 99.9)))

# 04a: imágenes, cada operación en orden
fig, ejes = plt.subplots(2, N_PASOS + 1, figsize=(6.5 * (N_PASOS + 1), 14))
for fila, (nombre, p) in enumerate(pasos.items()):
    for col, (titulo, v) in enumerate(p.items()):
        ax = ejes[fila, col]
        ax.imshow(v[Y0:Y0 + LADO, X0:X0 + LADO], cmap="gray", vmin=rango[0], vmax=rango[1])
        ax.set_title(f"{nombre}\n{titulo}")
        ax.set_axis_off()
    mapa_fases(ejes[fila, N_PASOS], etiquetas, "verdad")
guardar(fig, f"{PREFIJO}a_imagenes_paso_a_paso.png",
        f"{CADENA}. Imágenes (recorte, misma escala de gris): arriba nítida, abajo con degradé")

# 04b: histogramas de intensidad de cada paso
fig, ejes = plt.subplots(2, N_PASOS, figsize=(8.7 * N_PASOS, 14), sharex=True)
for fila, (nombre, p) in enumerate(pasos.items()):
    for col, (titulo, v) in enumerate(p.items()):
        ax = ejes[fila, col]
        ax.hist(v.ravel(), bins=256, range=rango, color="0.3")
        for g, c in zip(GRISES, COLORES):
            ax.axvline(g, color=c, lw=4, alpha=0.6)
        cerca = [np.mean(np.abs(v[etiquetas == i] - GRISES[i]) <= CERCA_DEL_GRIS) for i in range(3)]
        ax.text(0.02, 0.97, "en su gris (±0.01):\n" + "\n".join(f"{n}: {c:.0%}" for n, c in zip(FASES, cerca)),
                transform=ax.transAxes, va="top", fontsize=15, bbox={"facecolor": "w", "alpha": 0.85})
        ax.set_title(f"{nombre}\n{titulo}")
        ax.set_xlabel("gris")
        ax.set_ylabel("cantidad de píxeles")
        print(f"   {nombre:<24} {titulo:<28} en su gris: " + ", ".join(f"{n} {c:.0%}" for n, c in zip(FASES, cerca)))
guardar(fig, f"{PREFIJO}b_histogramas_paso_a_paso.png",
        f"{CADENA}. Histogramas de intensidad: arriba nítida, abajo con degradé")

# 04c: paisaje gris-actividad (sin normalizar) de cada paso
fig, ejes = plt.subplots(2, N_PASOS, figsize=(8.7 * N_PASOS, 15))
for fila, (nombre, p) in enumerate(pasos.items()):
    for col, (titulo, v) in enumerate(p.items()):
        ax = ejes[fila, col]
        conteos, _, _ = paisaje(v, rango)
        im = ax.imshow(np.log1p(conteos), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(*rango, 0, 1))
        fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + cuentas)")
        for g, c in zip(GRISES, COLORES):
            ax.axvline(g, color=c, lw=5, alpha=0.45)
        ax.set_title(f"{nombre}\n{titulo}")
        ax.set_xlabel("gris")
        ax.set_ylabel(f"actividad (σ={SIGMA_ACT} px, 0-1)")
guardar(fig, f"{PREFIJO}c_paisajes_paso_a_paso.png",
        f"{CADENA}. Paisaje gris-actividad de cada paso: arriba nítida, abajo con degradé")

# 04d: normalización, regiones con área y vuelta a la imagen (sobre la salida del choque)
fig, ejes = plt.subplots(2, 4, figsize=(28, 14), gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
for fila, (nombre, p) in enumerate(pasos.items()):
    r = regiones_del_paisaje(list(p.values())[-1], rango)
    fase_de, prediccion, acierto = evaluar(r, etiquetas)
    ax = ejes[fila, 0]
    im = ax.imshow(r["normalizado"], origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=(*rango, 0, 1))
    fig.colorbar(im, ax=ax, fraction=0.046)
    centros_a = (np.arange(BINS) + 0.5) / BINS
    for k, f in enumerate(fase_de, start=1):
        ax.contour(r["centros_g"], centros_a, r["regiones"] == k, levels=[0.5], colors=COLORES[f], linewidths=3)
    ax.set_xlabel("gris")
    ax.set_ylabel(f"actividad (σ={SIGMA_ACT} px, 0-1)")
    ax.set_title(f"{nombre}\n4. paisaje normalizado + 5. regiones ({len(fase_de)})")
    sin_region = np.mean(r["directa"] == 0)
    directa = np.where(r["directa"] > 0, fase_de[np.maximum(r["directa"], 1) - 1], -1)
    ax = ejes[fila, 1]
    ax.imshow(directa[Y0:Y0 + LADO, X0:X0 + LADO], cmap=ListedColormap(["k", *COLORES]), vmin=-1.5, vmax=2.5,
              interpolation="nearest")
    ax.set_title(f"vuelta directa\nnegro = sin región ({sin_region:.0%})")
    ax.set_axis_off()
    mapa_fases(ejes[fila, 2], prediccion, "resultado\n(sin región → vecino más cercano)")
    ejes[fila, 2].legend(handles=[Patch(color=c, label=f"{n}: {a:.0%} bien")
                                  for c, n, a in zip(COLORES, FASES, acierto)], loc="lower left")
    mapa_fases(ejes[fila, 3], etiquetas, "verdad")
    print(f"{nombre:<24} regiones {[FASES[f] for f in fase_de]}; sin región {sin_region:.0%}; acierto "
          + ", ".join(f"{n} {a:.0%}" for n, a in zip(FASES, acierto)))
guardar(fig, f"{PREFIJO}d_regiones_paso_a_paso.png", f"{CADENA}. Sobre la salida del choque: arriba nítida, abajo con degradé")
print("ok")
