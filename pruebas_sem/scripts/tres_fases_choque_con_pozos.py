"""Simulada de TRES fases: filtro de choque con un término de atracción hacia los pozos del paisaje.

Misma simulada que tres_fases_flujo_choque.py (matriz 0.70 56 %, patrón A 0.60 25 %, patrón B 0.52
19 %), nítida y con degradé (gaussiana 1.5 px), ruido 0.005.

Idea: los píxeles son partículas del paisaje gris-actividad. El choque las empuja en gris; cuando
se amontonan en un pozo, un término extra las alinea en el gris central del pozo en todas las
actividades, para que cada masa tienda a una línea vertical y no a una mancha redonda (ver
choque_con_pozos).

Se compara intensidad = 0 (choque solo, control), 0.5, 1 y 2. Al final, cada píxel se asigna como
en 06e: paisaje gris final x actividad de partida, semillas (media altura, masa >= 1 %) y pozos por
inundación (watershed).
Figuras: 07a imágenes, 07b histogramas, 07c paisajes con pozos, 07d resultados, 07e paisajes del
degradé tras cada transformación contra la imagen ideal (nítida sin procesar), a ciegas y con las
fases reales.
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
INTENSIDADES = [0.0, 0.5, 1.0, 2.0]
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
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


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def actividad_de_partida(img):
    """Actividad gaussiana(sobel), 3 px, normalizada 0-1 (percentil 99.9 = 1)."""
    act = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACT)
    return np.clip(act / np.percentile(act, 99.9), 0, 1)


def bins_de_gris(img):
    """Bin de gris de cada píxel, con el rango entre los percentiles 0.1 y 99.9 de la imagen."""
    rango = np.percentile(img, (0.1, 99.9))
    return np.clip(((img - rango[0]) / (rango[1] - rango[0]) * BINS).astype(int), 0, BINS - 1), rango


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


def pozos_del_paisaje(img, i_act):
    """Paisaje gris x actividad de partida: conteos, normalizado, semillas, pozos y bin de gris."""
    i_gris, rango = bins_de_gris(img)
    conteos = np.bincount(np.ravel_multi_index((i_act, i_gris), (BINS, BINS)).ravel(),
                          minlength=BINS ** 2).reshape(BINS, BINS).astype(float)
    semillas, normalizado = semillas_del_paisaje(conteos, img.size)
    pozos = ski.segmentation.watershed(-normalizado, semillas) if semillas.max() else semillas
    return {"conteos": conteos, "normalizado": normalizado, "semillas": semillas, "pozos": pozos,
            "i_gris": i_gris, "rango": rango}


def choque_con_pozos(img: np.ndarray, act_partida: np.ndarray, sigma_interno: float,
                     tolerancia_abs: float, intensidad: float = 1.0, cada: int = 5,
                     dt: float = 0.25, max_pasos: int = 200
                     ) -> tuple[np.ndarray, int, list[np.ndarray], int | None]:
    """Filtro de choque con un término de atracción hacia el gris central de cada pozo del paisaje.

    Al choque de Osher-Rudin (cada píxel se aleja del gris de sus vecinos, hacia el lado del borde
    al que pertenece) se le suma un término que piensa a los píxeles como partículas del paisaje
    gris-actividad: cuando se amontonan en un pozo, se alinean en el gris central de ese pozo en
    todas las actividades. Así, cada masa tiende a una línea vertical (el mismo gris en el
    interior y en el borde) en lugar de una mancha redonda.

    Dos etapas: primero el choque solo, hasta que se frena (cambio medio < tolerancia_abs); recién
    entonces se activa el término de pozos y se sigue hasta que vuelve a frenarse. Así el término
    no actúa antes de que el choque haya separado las masas (con degradé, el paisaje de partida
    tiene un solo pozo y el término comprimiría todos los grises hacia su centro).

    Con el término activo, cada `cada` iteraciones se recalculan los pozos:
    1. paisaje con el gris actual y la actividad de partida (fija, para que la actividad no se
       realimente);
    2. normalización por fila (fila / máximo de la fila);
    3. semillas: bins sobre la media altura de su fila, conexos, con masa >= 1 % de los píxeles;
    4. pozos: inundación (watershed) desde las semillas sobre el paisaje normalizado invertido;
    5. gris central de cada pozo: mediana del gris de los píxeles que caen en su semilla, sin
       importar la actividad.
    En cada iteración:
        I += -dt * signo(D2) * |∇I|                                        (choque)
           + intensidad * dt * densidad(bin del píxel) * (gris_central(pozo) - I)   (pozos)
    donde densidad es el paisaje normalizado (0-1) en el bin del píxel: cuanto más amontonado,
    más fuerte la atracción.

    Parámetros
    ----------
    img : imagen de partida (nítida o con degradé), en gris.
    act_partida : actividad de la imagen de partida (gaussiana(sobel), 3 px), normalizada 0-1.
    sigma_interno : σ (px) de la gaussiana con la que el choque estima la dirección del borde.
    tolerancia_abs : el proceso termina cuando el cambio medio por iteración es menor que esto
        (0.1 × ruido estimado, como en el choque solo).
    intensidad : peso del término de pozos frente al choque. 0 = choque solo. No sale de la
        imagen: se prueba 1.0 y se compara con 0.
    cada : cada cuántas iteraciones se recalculan los pozos.
    dt : paso de tiempo.
    max_pasos : máximo de iteraciones.

    Devuelve
    --------
    imagen procesada, cantidad de iteraciones, la lista de grises centrales de los pozos en cada
    recálculo (para ver si convergen o se corren) y la iteración en que se activó el término de
    pozos (None si no se activó).
    """
    I = img.astype(float).copy()
    i_act = np.clip((act_partida * BINS).astype(int), 0, BINS - 1)
    historial = []
    objetivo, densidad = I.copy(), np.zeros_like(I)
    activacion = None
    for paso in range(max_pasos):
        if activacion is not None and (paso - activacion) % cada == 0:
            p = pozos_del_paisaje(I, i_act)
            n = p["semillas"].max()
            if n:
                en_semilla = p["semillas"][i_act, p["i_gris"]]
                centrales = np.asarray(ndimage.median(I, en_semilla, index=np.arange(1, n + 1)))
                objetivo = centrales[p["pozos"][i_act, p["i_gris"]] - 1]
                densidad = p["normalizado"][i_act, p["i_gris"]]
                historial.append(np.sort(centrales))

        # Choque (Osher-Rudin, esquema minmod)
        S = ndimage.gaussian_filter(I, sigma_interno)
        Sy, Sx = np.gradient(S)
        Syy, Syx = np.gradient(Sy)
        _, Sxx = np.gradient(Sx)
        D2 = (Sx ** 2 * Sxx + 2 * Sx * Sy * Syx + Sy ** 2 * Syy) / (Sx ** 2 + Sy ** 2 + 1e-12)
        P = np.pad(I, 1, mode="edge")
        Ix = _minmod(P[1:-1, 2:] - I, I - P[1:-1, :-2])
        Iy = _minmod(P[2:, 1:-1] - I, I - P[:-2, 1:-1])
        delta = -dt * np.sign(D2) * np.hypot(Ix, Iy)

        # Pozos: atracción hacia el gris central del pozo, pesada por el amontonamiento
        if activacion is not None:
            delta += intensidad * dt * densidad * (objetivo - I)
        I += delta
        if np.mean(np.abs(delta)) < tolerancia_abs:
            if intensidad > 0 and activacion is None:
                activacion = paso + 1
                continue
            break
    return I, paso + 1, historial, activacion


def segmentar(img, i_act, etiquetas):
    """Asignación como en 06e: pozos por inundación; cada pozo a la fase verdadera mayoritaria."""
    p = pozos_del_paisaje(img, i_act)
    etiqueta_px = p["pozos"][i_act, p["i_gris"]]
    n = p["semillas"].max()
    fase_de = np.array([np.bincount(etiquetas[etiqueta_px == k], minlength=3).argmax() for k in range(1, n + 1)])
    prediccion = fase_de[etiqueta_px - 1]
    acierto = [np.mean(prediccion[etiquetas == i] == i) for i in range(3)]
    return p, fase_de, prediccion, acierto


def mapa_fases(ax, fases, titulo):
    ax.imshow(fases[Y0:Y0 + LADO, X0:X0 + LADO], cmap=ListedColormap(COLORES), vmin=-0.5, vmax=2.5,
              interpolation="nearest")
    ax.set_title(titulo)
    ax.set_axis_off()


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

resultados = {}
for nombre, img in partida.items():
    act = actividad_de_partida(img)
    i_act = np.clip((act * BINS).astype(int), 0, BINS - 1)
    tolerancia = 0.1 * estimar_ruido(img)
    resultados[nombre] = {}
    for k in INTENSIDADES:
        procesada, pasos, historial, activacion = choque_con_pozos(img, act, 1.0, tolerancia, intensidad=k)
        p, fase_de, prediccion, acierto = segmentar(procesada, i_act, etiquetas)
        cerca = [np.mean(np.abs(procesada[etiquetas == i] - GRISES[i]) <= CERCA_DEL_GRIS) for i in range(3)]
        resultados[nombre][k] = {"img": procesada, "pasos": pasos, "p": p, "fase_de": fase_de,
                                 "prediccion": prediccion, "acierto": acierto, "cerca": cerca}
        evolucion = " -> ".join(str(np.round(h, 3)) for h in (historial[:1] + historial[-1:])) if historial else "-"
        print(f"{nombre:<24} intensidad {k:<4} {pasos:>3} pasos (pozos desde {activacion}); en su gris "
              + ", ".join(f"{n} {c:.0%}" for n, c in zip(FASES, cerca)) + "; acierto "
              + ", ".join(f"{n} {a:.0%}" for n, a in zip(FASES, acierto)) + f"; centros {evolucion}")

columnas = len(INTENSIDADES) + 1
rango_comun = tuple(np.percentile(np.concatenate([r["img"].ravel() for c in resultados.values()
                                                  for r in c.values()]), (0.1, 99.9)))

# 07a: imágenes
fig, ejes = plt.subplots(2, columnas, figsize=(6 * columnas, 13))
for fila, (nombre, img) in enumerate(partida.items()):
    ax = ejes[fila, 0]
    ax.imshow(img[Y0:Y0 + LADO, X0:X0 + LADO], cmap="gray", vmin=rango_comun[0], vmax=rango_comun[1])
    ax.set_title(f"{nombre}\npartida")
    ax.set_axis_off()
    for col, (k, r) in enumerate(resultados[nombre].items(), start=1):
        ax = ejes[fila, col]
        ax.imshow(r["img"][Y0:Y0 + LADO, X0:X0 + LADO], cmap="gray", vmin=rango_comun[0], vmax=rango_comun[1])
        ax.set_title(f"intensidad {k} ({r['pasos']} pasos)")
        ax.set_axis_off()
guardar(fig, "07a_choque_con_pozos_imagenes.png", "Choque con pozos: imágenes (recorte, misma escala de gris)")

# 07b: histogramas
fig, ejes = plt.subplots(2, columnas, figsize=(6.5 * columnas, 13), sharex=True)
for fila, (nombre, img) in enumerate(partida.items()):
    paneles = [("partida", img, None)] + [(f"intensidad {k}", r["img"], r["cerca"]) for k, r in resultados[nombre].items()]
    for col, (titulo, v, cerca) in enumerate(paneles):
        ax = ejes[fila, col]
        ax.hist(v.ravel(), bins=256, range=rango_comun, color="0.3")
        for g, c in zip(GRISES, COLORES):
            ax.axvline(g, color=c, lw=4, alpha=0.6)
        if cerca is None:
            cerca = [np.mean(np.abs(v[etiquetas == i] - GRISES[i]) <= CERCA_DEL_GRIS) for i in range(3)]
        ax.text(0.02, 0.97, "en su gris (±0.01):\n" + "\n".join(f"{n}: {c:.0%}" for n, c in zip(FASES, cerca)),
                transform=ax.transAxes, va="top", fontsize=14, bbox={"facecolor": "w", "alpha": 0.85})
        ax.set_title(f"{nombre}\n{titulo}")
        ax.set_xlabel("gris")
guardar(fig, "07b_choque_con_pozos_histogramas.png", "Choque con pozos: histogramas de intensidad")

# 07c: paisajes normalizados con pozos
fig, ejes = plt.subplots(2, len(INTENSIDADES), figsize=(7.5 * len(INTENSIDADES), 14))
for fila, nombre in enumerate(partida):
    for col, (k, r) in enumerate(resultados[nombre].items()):
        ax, p = ejes[fila, col], r["p"]
        extent = (*p["rango"], 0, 1)
        centros_g = p["rango"][0] + (np.arange(BINS) + 0.5) / BINS * (p["rango"][1] - p["rango"][0])
        centros_a = (np.arange(BINS) + 0.5) / BINS
        ax.imshow(p["normalizado"], origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
        for j in range(1, p["semillas"].max() + 1):
            ax.contour(centros_g, centros_a, p["pozos"] == j, levels=[0.5], colors="k", linewidths=1.5)
        for j, f in enumerate(r["fase_de"], start=1):
            ax.contour(centros_g, centros_a, p["semillas"] == j, levels=[0.5], colors=COLORES[f], linewidths=3)
        for g, c in zip(GRISES, COLORES):
            ax.axvline(g, color=c, lw=3, alpha=0.4, ls="--")
        ax.set_xlabel("gris final")
        ax.set_ylabel("actividad de partida (0-1)")
        ax.set_title(f"{nombre}\nintensidad {k}: {p['semillas'].max()} pozos")
guardar(fig, "07c_choque_con_pozos_paisajes.png",
        "Paisaje normalizado (gris final × actividad de partida): semillas en color, frontera de pozos en negro")

# 07d: resultados
fig, ejes = plt.subplots(2, columnas, figsize=(6 * columnas, 13))
for fila, nombre in enumerate(partida):
    for col, (k, r) in enumerate(resultados[nombre].items()):
        mapa_fases(ejes[fila, col], r["prediccion"], f"{nombre}\nintensidad {k}")
        ejes[fila, col].legend(handles=[Patch(color=c, label=f"{n}: {a:.0%}")
                                        for c, n, a in zip(COLORES, FASES, r["acierto"])], loc="lower left",
                               fontsize=13)
    mapa_fases(ejes[fila, -1], etiquetas, "verdad")
guardar(fig, "07d_choque_con_pozos_resultados.png", "Choque con pozos: resultado (pozos por inundación) y verdad")


def paisaje_por_fase(img, act):
    """Conteos del paisaje (gris x actividad de partida) separados por fase verdadera: (3, BINS, BINS)."""
    i_act = np.clip((act * BINS).astype(int), 0, BINS - 1)
    i_gris = np.clip(((img - rango_comun[0]) / (rango_comun[1] - rango_comun[0]) * BINS).astype(int), 0, BINS - 1)
    plano = np.ravel_multi_index((i_act, i_gris), (BINS, BINS))
    return np.stack([np.bincount(plano[etiquetas == i], minlength=BINS ** 2).reshape(BINS, BINS) for i in range(3)])


# 07e: paisajes comparados con la imagen ideal (nítida, sin procesar), a ciegas y con las fases reales
ideal = partida["nítida"]
nombre_deg = "con degradé (σ=1.5 px)"
act_deg = actividad_de_partida(partida[nombre_deg])
paneles = [("IDEAL: nítida sin procesar", ideal, actividad_de_partida(ideal)),
           ("degradé: partida", partida[nombre_deg], act_deg)]
paneles += [(f"degradé: choque, intensidad {k}", r["img"], act_deg) for k, r in resultados[nombre_deg].items()]
extent = (*rango_comun, 0, 1)
centros = rango_comun[0] + (np.arange(BINS) + 0.5) / BINS * (rango_comun[1] - rango_comun[0])
rgb_fases = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in COLORES])
fig, ejes = plt.subplots(3, len(paneles), figsize=(6.5 * len(paneles), 20), sharex=True)
for col, (titulo, v, act) in enumerate(paneles):
    por_fase = paisaje_por_fase(v, act)
    total = por_fase.sum(axis=0)
    ax = ejes[0, col]
    ax.imshow(np.log1p(total), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
    ax.set_title(f"{titulo}\na ciegas: log(1 + cuentas)")
    ax.set_ylabel("actividad de partida (0-1)")

    ax = ejes[1, col]
    mezcla = np.einsum("fyx,fc->yxc", por_fase, rgb_fases) / np.maximum(total, 1)[..., None]
    brillo = (np.log1p(total) / np.log1p(total).max())[..., None]
    ax.imshow(1 - brillo * (1 - mezcla), origin="lower", aspect="auto", extent=extent)
    ax.set_title("con las fases reales\n(color = fase, brillo = log cuentas)")
    ax.set_ylabel("actividad de partida (0-1)")

    ax = ejes[2, col]
    for i in range(3):
        ax.plot(centros, por_fase[i].sum(axis=0), color=COLORES[i], lw=3,
                label=f"{FASES[i]}: mediana {np.median(v[etiquetas == i]):.3f}")
    ax.plot(centros, total.sum(axis=0), color="k", lw=1.5, ls="--", label="total")
    ax.set_title("histograma de gris por fase real")
    ax.set_xlabel("gris")
    ax.set_ylabel("cantidad de píxeles")
    ax.legend(fontsize=12, loc="upper left")
    for fila in range(3):
        for g, c in zip(GRISES, COLORES):
            ejes[fila, col].axvline(g, color=c, lw=2, alpha=0.5, ls=":")
guardar(fig, "07e_paisajes_vs_ideal.png",
        "Distribución de los píxeles en el paisaje: ideal contra degradé tras cada transformación "
        "(líneas punteadas = gris real)")
print("ok")
