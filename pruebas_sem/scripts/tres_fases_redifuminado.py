"""Simulada de TRES fases: corrección de la segmentación por re-difuminado.

Misma simulada que tres_fases_choque_con_pozos.py (matriz 0.70 56 %, patrón A 0.60 25 %, patrón B
0.52 19 %), nítida y con degradé (gaussiana 1.5 px), ruido 0.005.

Idea: el degradé es un promedio local. Las estructuras chicas no tienen núcleo y se corren hacia el
gris de lo que las rodea; las grandes conservan su gris en el núcleo y solo el borde se corre. Una
interfaz B-matriz produce en el medio el gris de A. Si se difumina una segmentación candidata con el
mismo degradé, la correcta es la que reproduce la imagen observada (ver redifuminar_etiquetas).

Procesamiento, sin usar la verdad:
1. etiquetas iniciales: choque solo -> paisaje gris final x actividad de partida -> pozos por
   inundación (watershed); los pozos se ordenan por su gris central (0 = el más oscuro);
2. gris de cada población: gris central de cada pozo (mediana en la semilla);
3. ancho del degradé: estimado en bordes aislados de la imagen de partida;
4. re-difuminado sobre la imagen de partida (sin choque).
Además, para el degradé, se repite con el ancho real (1.5 px) para ver cuánto pesa la estimación.
Energía: residuo²/ruido² + interacción × (pares de vecinos con etiqueta distinta); se comparan
varios pesos de la interacción (0 = solo ajuste a la imagen).
Figuras: 08c resultados para cada peso, 08d acierto y residuo según el peso
(08a y 08b eran la versión sin interacción).
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/tres_fases/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap
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
MEDIA_ALTURA, MASA_MINIMA = 0.5, 0.01
INTERACCIONES = [0, 1, 3, 10, 30]
Y0, X0, LADO = 170, 190, 170
LEYENDA = [Patch(color=c, label=n) for c, n in zip(COLORES, NOMBRES)]


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


def actividad_de_partida(img):
    """Actividad gaussiana(sobel), 3 px, normalizada 0-1 (percentil 99.9 = 1)."""
    act = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACT)
    return np.clip(act / np.percentile(act, 99.9), 0, 1)


def semillas_del_paisaje(conteos, n_pixeles):
    """Regiones con masa del paisaje normalizado por fila (media altura, masa >= 1 %)."""
    suave = ndimage.gaussian_filter(conteos, 1)
    por_fila = conteos.sum(axis=1, keepdims=True)
    validas = por_fila >= FILA_MINIMA * por_fila.max()
    with np.errstate(invalid="ignore", divide="ignore"):
        normalizado = np.where(validas, suave / suave.max(axis=1, keepdims=True), 0.0)
    marcadas = ski.measure.label(normalizado >= MEDIA_ALTURA, connectivity=2)
    masas = ndimage.sum(conteos, marcadas, index=np.arange(1, marcadas.max() + 1))
    con_masa = np.flatnonzero(masas >= MASA_MINIMA * n_pixeles) + 1
    semillas = np.zeros_like(marcadas)
    for nueva, vieja in enumerate(con_masa, start=1):
        semillas[marcadas == vieja] = nueva
    return semillas, normalizado


def segmentacion_por_pozos(img_choque, act_partida):
    """Etiquetas por pozos (inundación) y gris central de cada pozo, ordenados del más oscuro al más claro."""
    i_act = np.clip((act_partida * BINS).astype(int), 0, BINS - 1)
    rango = np.percentile(img_choque, (0.1, 99.9))
    i_gris = np.clip(((img_choque - rango[0]) / (rango[1] - rango[0]) * BINS).astype(int), 0, BINS - 1)
    conteos = np.bincount(np.ravel_multi_index((i_act, i_gris), (BINS, BINS)).ravel(),
                          minlength=BINS ** 2).reshape(BINS, BINS).astype(float)
    semillas, normalizado = semillas_del_paisaje(conteos, img_choque.size)
    pozos = ski.segmentation.watershed(-normalizado, semillas)
    n = semillas.max()
    centrales = np.asarray(ndimage.median(img_choque, semillas[i_act, i_gris], index=np.arange(1, n + 1)))
    orden = np.argsort(centrales)
    rango_de = np.empty(n, int)
    rango_de[orden] = np.arange(n)
    return rango_de[pozos[i_act, i_gris] - 1], centrales[orden]


def redifuminar_etiquetas(observada: np.ndarray, etiquetas_iniciales: np.ndarray,
                          grises: np.ndarray, sigma: float, ruido: float, interaccion: float = 0.0,
                          fraccion: float = 0.1, max_iter: int = 100,
                          semilla: int = 0) -> tuple[np.ndarray, list[float]]:
    """Corrige una segmentación exigiendo que, al difuminarla, reproduzca la imagen observada.

    Modelo: observada ≈ gaussiana(grises[etiquetas], sigma) + ruido. Las estructuras finas no
    conservan su gris (no tienen núcleo) y una interfaz entre dos fases produce grises intermedios
    (por ejemplo, B 0.52 | matriz 0.70 da ~0.61, el gris de A). Comparar la segmentación
    re-difuminada con la observada distingue esos casos: una interfaz B-matriz se explica sin A;
    una pared de A, con su meseta, no.

    Solo con el ajuste a la imagen el problema es ambiguo: un salpicado de B y matriz, difuminado,
    da el mismo gris que una pared de A. Por eso se suma un término de interacción entre píxeles
    vecinos (cada píxel prefiere la población de sus 4 vecinos): un salpicado tiene muchos más
    bordes entre poblaciones que una pared compacta. Energía:
        E = Σ residuo² / ruido² + interaccion · (cantidad de pares de vecinos con etiqueta distinta)

    En cada iteración:
    1. residuo r = observada - gaussiana(grises[etiquetas], sigma);
    2. para cada píxel y cada etiqueta candidata, el cambio de energía si el píxel cambiara de
       etiqueta: ajuste (-2·d·(K * r)(p) + d²·ΣK²) / ruido², con d la diferencia de gris y K el
       núcleo gaussiano (para todos los píxeles con una convolución), más interaccion · (vecinos
       con etiqueta distinta a la candidata - vecinos con etiqueta distinta a la actual);
    3. cambian de etiqueta solo los píxeles con ΔE < 0, elegidos al azar entre los que mejoran
       (una fracción `fraccion`), para que dos vecinos no cambien a la vez con información vieja;
    4. termina cuando ningún píxel mejora o tras `max_iter` iteraciones.

    Parámetros
    ----------
    observada : imagen de partida (nítida o con degradé), sin choque.
    etiquetas_iniciales : segmentación inicial (0..K-1), por ejemplo la de choque + pozos.
    grises : gris de cada población (centros de los pozos), sale de la imagen.
    sigma : ancho del degradé (px), estimado en bordes aislados; sale de la imagen.
    ruido : desviación estándar del ruido, estimada en la imagen; pone el ajuste en unidades del
        ruido para que `interaccion` no dependa del contraste.
    interaccion : peso de cada par de vecinos con etiqueta distinta. 0 = solo ajuste. No sale de
        la imagen: se prueban varios valores.
    fraccion : fracción de los píxeles que mejoran que cambian en cada iteración.
    max_iter : máximo de iteraciones.
    semilla : semilla del generador aleatorio que elige qué píxeles cambian.

    Devuelve
    --------
    etiquetas corregidas y la evolución de la energía por iteración.
    """
    rng = np.random.default_rng(semilla)
    etiquetas = etiquetas_iniciales.copy()
    k = len(grises)
    impulso = np.zeros((41, 41))
    impulso[20, 20] = 1.0
    suma_k2 = float((ndimage.gaussian_filter(impulso, sigma) ** 2).sum())
    cruz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    vecinos_totales = ndimage.convolve(np.ones(etiquetas.shape), cruz, mode="constant")
    energia = []
    for _ in range(max_iter):
        residuo = observada - ndimage.gaussian_filter(grises[etiquetas], sigma)
        # Vecinos (de 4) con cada etiqueta, y pares de vecinos distintos para la energía
        iguales = np.stack([ndimage.convolve((etiquetas == j).astype(float), cruz, mode="constant")
                            for j in range(k)], axis=-1)
        distintos = vecinos_totales[..., None] - iguales                # si el píxel tuviera cada etiqueta
        actual = np.take_along_axis(distintos, etiquetas[..., None], axis=-1)[..., 0]
        energia.append(float((residuo ** 2).sum() / ruido ** 2 + interaccion * actual.sum() / 2))
        k_r = ndimage.gaussian_filter(residuo, sigma)
        d = grises[None, :] - grises[etiquetas][..., None]           # (alto, ancho, K)
        delta_e = (-2 * d * k_r[..., None] + d ** 2 * suma_k2) / ruido ** 2 \
            + interaccion * (distintos - actual[..., None])
        mejor = np.argmin(delta_e, axis=-1)
        mejora = np.take_along_axis(delta_e, mejor[..., None], axis=-1)[..., 0] < 0
        if not mejora.any():
            break
        cambian = mejora & (rng.random(mejora.shape) < fraccion)
        etiquetas[cambian] = mejor[cambian]
    return etiquetas, energia


def mapa_fases(ax, fases, titulo):
    ax.imshow(fases[Y0:Y0 + LADO, X0:X0 + LADO], cmap=ListedColormap(COLORES), vmin=-0.5, vmax=2.5,
              interpolation="nearest")
    ax.set_title(titulo)
    ax.set_axis_off()


def leyenda_acierto(ax, acierto):
    ax.legend(handles=[Patch(color=c, label=f"{n}: {a:.0%}") for c, n, a in zip(COLORES, FASES, acierto)],
              loc="lower left", fontsize=13)


def guardar(fig, nombre, titulo):
    fig.suptitle(titulo, fontsize=22)
    fig.legend(handles=LEYENDA, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(SALIDA / nombre, dpi=80)
    plt.close(fig)


etiquetas, rng = tres_fases_con_presencia()
print("fracciones verdaderas:", {n: f"{np.mean(etiquetas == i):.1%}" for i, n in enumerate(FASES)})
ruido = rng.normal(0, 0.005, etiquetas.shape)
partida = {"nítida": GRISES[etiquetas] + ruido,
           "con degradé (σ=1.5 px)": ski.filters.gaussian(GRISES[etiquetas], 1.5) + ruido}

casos = []
for nombre, img in partida.items():
    ruido_est = estimar_ruido(img)
    con_choque, _ = choque(img, 1.0, 0.1 * ruido_est)
    iniciales, grises = segmentacion_por_pozos(con_choque, actividad_de_partida(img))
    sigma_est, _ = estimar_sigma_borde(img)
    sigmas = [(f"σ estimado {sigma_est:.2f} px", sigma_est)]
    if "degradé" in nombre:
        sigmas.append(("σ real 1.50 px", 1.5))
    for etiqueta_sigma, sigma in sigmas:
        caso = {"nombre": nombre, "sigma": etiqueta_sigma, "img": img, "iniciales": iniciales, "grises": grises,
                "antes": [np.mean(iniciales[etiquetas == i] == i) for i in range(3)], "por_interaccion": {}}
        for beta in INTERACCIONES:
            corregidas, energia = redifuminar_etiquetas(img, iniciales, grises, sigma, ruido_est, interaccion=beta)
            residuo = img - ndimage.gaussian_filter(grises[corregidas], sigma)
            despues = [np.mean(corregidas[etiquetas == i] == i) for i in range(3)]
            caso["por_interaccion"][beta] = {"etiquetas": corregidas, "acierto": despues,
                                             "rms": float(np.sqrt(np.mean(residuo ** 2))),
                                             "iteraciones": len(energia)}
            print(f"{nombre:<24} {etiqueta_sigma:<20} interacción {beta:<4} {len(energia):>3} iter; RMS "
                  f"{caso['por_interaccion'][beta]['rms']:.4f} (ruido {ruido_est:.4f}); acierto "
                  + ", ".join(f"{a:.0%}" for a in caso["antes"]) + " -> " + ", ".join(f"{a:.0%}" for a in despues))
        casos.append(caso)

# 08c: resultados para cada peso de la interacción
columnas = len(INTERACCIONES) + 2
fig, ejes = plt.subplots(len(casos), columnas, figsize=(5.5 * columnas, 6.5 * len(casos)))
for fila, c in enumerate(casos):
    mapa_fases(ejes[fila, 0], c["iniciales"], f"{c['nombre']}\n{c['sigma']}\ninicial: choque + pozos")
    leyenda_acierto(ejes[fila, 0], c["antes"])
    for col, (beta, r) in enumerate(c["por_interaccion"].items(), start=1):
        mapa_fases(ejes[fila, col], r["etiquetas"], f"interacción {beta}\nRMS residuo {r['rms']:.4f}")
        leyenda_acierto(ejes[fila, col], r["acierto"])
    mapa_fases(ejes[fila, -1], etiquetas, "verdad")
guardar(fig, "08c_redifuminado_interaccion.png",
        "Re-difuminado con interacción entre vecinos (energía = residuo²/ruido² + interacción × bordes)")

# 08d: acierto y residuo en función del peso de la interacción
fig, ejes = plt.subplots(1, len(casos), figsize=(9 * len(casos), 8))
x = np.arange(len(INTERACCIONES))
for ax, c in zip(ejes, casos):
    for i in range(3):
        ax.plot(x, [r["acierto"][i] for r in c["por_interaccion"].values()], "-o", color=COLORES[i], lw=3,
                label=FASES[i])
        ax.axhline(c["antes"][i], color=COLORES[i], ls=":", lw=2)
    ax.set_xticks(x, [str(b) for b in INTERACCIONES])
    ax.set_xlabel("peso de la interacción")
    ax.set_ylabel("acierto por fase (punteado = inicial)")
    ax.set_ylim(0, 1.02)
    ax2 = ax.twinx()
    ax2.plot(x, [r["rms"] for r in c["por_interaccion"].values()], "k--s", lw=2, label="RMS residuo")
    ax2.set_ylabel("RMS del residuo (negro)")
    ax.set_title(f"{c['nombre']}\n{c['sigma']}")
    ax.legend(loc="lower left")
guardar(fig, "08d_redifuminado_interaccion_curvas.png", "Acierto y residuo según el peso de la interacción")
print("ok")
