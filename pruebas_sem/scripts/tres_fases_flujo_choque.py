"""Simulada de TRES fases: pozos de atracción en el paisaje a partir del flujo del filtro de choque.

Misma simulada que tres_fases_paso_a_paso.py (matriz 0.70 56 %, patrón A 0.60 25 %, patrón B 0.52
19 %), nítida y con degradé (gaussiana 1.5 px), ruido 0.005. Cadena: ficticia -> choque (sin
enfoque).

Idea: el choque empuja cada píxel hacia la masa a la que pertenece. Cada píxel va de un punto
(gris, actividad) de la ficticia a otro tras el choque; el desplazamiento medio de cada bin del
paisaje es un campo de flujo. Las regiones con masa del paisaje normalizado tras el choque (media
altura por fila, masa >= 1 %) son los atractores (semillas).
Asignación de cada píxel:
1. si tras el choque cae en una semilla, toma esa población (directo);
2. si no, se sigue el flujo bin a bin (desplazamiento medio de los píxeles que empezaron en cada
   bin) hasta caer en una semilla (por flujo);
3. si el flujo se detiene fuera de las semillas, toma la semilla más cercana en el paisaje
   (respaldo).
Ambos paisajes usan los mismos bins: gris en un rango común y actividad dividida por un máximo común.
Figuras: 06a flujo en el paisaje, 06b pozos y resultado, 06c componentes del desplazamiento
(Δgris y Δactividad por píxel, por bin y por fase verdadera), una por caso; 06d pozos por semilla
más cercana y 06e pozos por inundación (watershed), con gris tras el choque × actividad de partida.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/tres_fases/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
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
PASOS_FLUJO, BLOQUE_FLECHAS = 100, 4
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


def actividad_cruda(img):
    return ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACT)


def indices_de_bin(img, act, rango, act_max):
    """Bin de actividad (fila) y de gris (columna) de cada píxel, con bins comunes."""
    i_act = np.clip((act / act_max * BINS).astype(int), 0, BINS - 1)
    i_gris = np.clip(((img - rango[0]) / (rango[1] - rango[0]) * BINS).astype(int), 0, BINS - 1)
    return i_act, i_gris


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


def flujo_por_bin(origen, destino):
    """Desplazamiento medio (en bins) de los píxeles que empiezan en cada bin, y cuántos son."""
    plano = np.ravel_multi_index(origen, (BINS, BINS))
    n = np.bincount(plano, minlength=BINS * BINS)
    d_act = np.bincount(plano, destino[0] - origen[0], minlength=BINS * BINS) / np.maximum(n, 1)
    d_gris = np.bincount(plano, destino[1] - origen[1], minlength=BINS * BINS) / np.maximum(n, 1)
    return d_act.reshape(BINS, BINS), d_gris.reshape(BINS, BINS), n.reshape(BINS, BINS)


def pozos_por_flujo(semillas, d_act, d_gris):
    """Población de cada bin siguiendo el flujo hasta una semilla; respaldo: semilla más cercana.

    Devuelve la etiqueta de cada bin y cómo se obtuvo (1 directo, 2 por flujo, 3 respaldo).
    """
    filas, cols = np.indices((BINS, BINS))
    siguiente_f = np.clip(np.rint(filas + d_act).astype(int), 0, BINS - 1)
    siguiente_c = np.clip(np.rint(cols + d_gris).astype(int), 0, BINS - 1)
    etiqueta = semillas.copy()
    via = np.where(semillas > 0, 1, 0)
    f, c = filas.copy(), cols.copy()
    for _ in range(PASOS_FLUJO):
        f, c = siguiente_f[f, c], siguiente_c[f, c]
        llega = (etiqueta == 0) & (semillas[f, c] > 0)
        etiqueta[llega] = semillas[f, c][llega]
        via[llega] = 2
    _, (cerca_f, cerca_c) = ndimage.distance_transform_edt(semillas == 0, return_indices=True)
    resto = etiqueta == 0
    etiqueta[resto] = semillas[cerca_f[f, c], cerca_c[f, c]][resto]
    via[resto] = 3
    return etiqueta, via


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

casos = {}
for nombre, ficticia in partida.items():
    con_choque, iteraciones = choque(ficticia, 1.0, 0.1 * estimar_ruido(ficticia))
    casos[nombre] = {"ficticia": ficticia, "choque": con_choque, "iteraciones": iteraciones}
todas = np.concatenate([v[k].ravel() for v in casos.values() for k in ("ficticia", "choque")])
rango = tuple(np.percentile(todas, (0.1, 99.9)))
for caso in casos.values():
    caso["act0"], caso["act1"] = actividad_cruda(caso["ficticia"]), actividad_cruda(caso["choque"])
act_max = max(np.percentile(c[k], 99.9) for c in casos.values() for k in ("act0", "act1"))
extent = (*rango, 0, 1)
centros_g = rango[0] + (np.arange(BINS) + 0.5) / BINS * (rango[1] - rango[0])
centros_a = (np.arange(BINS) + 0.5) / BINS
ancho_bin_g = (rango[1] - rango[0]) / BINS

fig_a, ejes_a = plt.subplots(2, 2, figsize=(24, 18))
fig_b, ejes_b = plt.subplots(2, 4, figsize=(30, 14), gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
for fila, (nombre, caso) in enumerate(casos.items()):
    origen = indices_de_bin(caso["ficticia"], caso["act0"], rango, act_max)
    destino = indices_de_bin(caso["choque"], caso["act1"], rango, act_max)
    conteos0 = np.bincount(np.ravel_multi_index(origen, (BINS, BINS)).ravel(), minlength=BINS ** 2).reshape(BINS, BINS)
    conteos1 = np.bincount(np.ravel_multi_index(destino, (BINS, BINS)).ravel(), minlength=BINS ** 2).reshape(BINS, BINS)
    semillas, normalizado = semillas_del_paisaje(conteos1.astype(float), etiquetas.size)
    d_act, d_gris, n_bin = flujo_por_bin(tuple(o.ravel() for o in origen), tuple(d.ravel() for d in destino))
    etiqueta_bin, via_bin = pozos_por_flujo(semillas, d_act, d_gris)
    caso["origen"], caso["conteos0"] = origen, conteos0

    # Cada píxel parte de su bin tras el choque (directo si ya es semilla)
    etiqueta_px = etiqueta_bin[destino]
    via_px = via_bin[destino]
    n_sem = semillas.max()
    fase_de = np.array([np.bincount(etiquetas[etiqueta_px == k], minlength=3).argmax() for k in range(1, n_sem + 1)])
    prediccion = fase_de[etiqueta_px - 1]
    acierto = [np.mean(prediccion[etiquetas == i] == i) for i in range(3)]
    fracc_via = [np.mean(via_px == v) for v in (1, 2, 3)]
    print(f"{nombre:<24} choque {caso['iteraciones']} pasos; semillas {[FASES[f] for f in fase_de]}; píxeles: "
          f"directo {fracc_via[0]:.0%}, por flujo {fracc_via[1]:.0%}, respaldo {fracc_via[2]:.0%}; acierto "
          + ", ".join(f"{n} {a:.0%}" for n, a in zip(FASES, acierto)))

    # 06a: flujo sobre el paisaje de la ficticia y semillas sobre el paisaje normalizado tras el choque
    ax = ejes_a[fila, 0]
    im = ax.imshow(np.log1p(conteos0), origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
    fig_a.colorbar(im, ax=ax, fraction=0.046, label="log(1 + cuentas)")
    b = BLOQUE_FLECHAS
    n_blq = n_bin.reshape(BINS // b, b, BINS // b, b).sum(axis=(1, 3))
    da_blq = (d_act * n_bin).reshape(BINS // b, b, BINS // b, b).sum(axis=(1, 3)) / np.maximum(n_blq, 1)
    dg_blq = (d_gris * n_bin).reshape(BINS // b, b, BINS // b, b).sum(axis=(1, 3)) / np.maximum(n_blq, 1)
    visibles = n_blq >= 20
    fb, cb = np.nonzero(visibles)
    ax.quiver(centros_g[cb * b + b // 2], centros_a[fb * b + b // 2], dg_blq[visibles] * ancho_bin_g,
              da_blq[visibles] / BINS, angles="xy", scale_units="xy", scale=1, width=0.003, color="k")
    for k, f in enumerate(fase_de, start=1):
        ax.contour(centros_g, centros_a, semillas == k, levels=[0.5], colors=COLORES[f], linewidths=3)
    ax.set_xlabel("gris")
    ax.set_ylabel("actividad (máximo común, 0-1)")
    ax.set_title(f"{nombre}: paisaje de la ficticia\nflechas = desplazamiento medio por el choque "
                 f"({caso['iteraciones']} pasos); contornos = semillas")

    ax = ejes_a[fila, 1]
    im = ax.imshow(normalizado, origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
    fig_a.colorbar(im, ax=ax, fraction=0.046, label="fila / máximo de la fila")
    for k, f in enumerate(fase_de, start=1):
        ax.contour(centros_g, centros_a, semillas == k, levels=[0.5], colors=COLORES[f], linewidths=3)
    ax.set_xlabel("gris")
    ax.set_ylabel("actividad (máximo común, 0-1)")
    ax.set_title(f"{nombre}: paisaje normalizado tras el choque\nsemillas = media altura, masa ≥ 1 % "
                 f"({n_sem})")

    # 06b: pozos en el paisaje, resultado y verdad
    ax = ejes_b[fila, 0]
    rgb = np.array([to_rgb(c) for c in COLORES])[fase_de[etiqueta_bin - 1]]
    brillo = np.log1p(conteos1) / np.log1p(conteos1).max()
    ax.imshow(1 - brillo[..., None] * (1 - rgb) * 0.85 - 0.15 * (1 - rgb), origin="lower", aspect="auto",
              extent=extent)
    for k, f in enumerate(fase_de, start=1):
        ax.contour(centros_g, centros_a, semillas == k, levels=[0.5], colors="k", linewidths=2)
    ax.set_xlabel("gris")
    ax.set_ylabel("actividad (máximo común, 0-1)")
    ax.set_title(f"{nombre}: pozos (color) sobre el paisaje tras el choque\n"
                 f"contorno negro = semilla; intensidad = log cuentas")

    ax = ejes_b[fila, 1]
    via_img = np.where(via_px == 1, 0, np.where(via_px == 2, 1, 2))
    ax.imshow(via_img[Y0:Y0 + LADO, X0:X0 + LADO], cmap=ListedColormap(["0.85", "#9467bd", "k"]),
              vmin=-0.5, vmax=2.5, interpolation="nearest")
    ax.set_title(f"cómo se asignó cada píxel\ngris = directo {fracc_via[0]:.0%}, violeta = flujo "
                 f"{fracc_via[1]:.0%},\nnegro = respaldo {fracc_via[2]:.0%}")
    ax.set_axis_off()
    mapa_fases(ejes_b[fila, 2], prediccion, "resultado")
    ejes_b[fila, 2].legend(handles=[Patch(color=c, label=f"{n}: {a:.0%} bien")
                                    for c, n, a in zip(COLORES, FASES, acierto)], loc="lower left")
    mapa_fases(ejes_b[fila, 3], etiquetas, "verdad")

guardar(fig_a, "06a_flujo_del_choque.png", "ficticia → choque: flujo en el paisaje (arriba nítida, abajo con degradé)")
guardar(fig_b, "06b_pozos_resultado.png", "Pozos de atracción y resultado (arriba nítida, abajo con degradé)")

# 06c: componentes del flujo (Δgris y Δactividad) por píxel, por bin y por fase verdadera
for clave, (nombre, caso) in zip(("nitida", "degrade"), casos.items()):
    d_gris_px = caso["choque"] - caso["ficticia"]
    d_act_px = (caso["act1"] - caso["act0"]) / act_max
    plano = np.ravel_multi_index(caso["origen"], (BINS, BINS)).ravel()
    n = caso["conteos0"].ravel()
    medio = {k: np.where(n > 0, np.bincount(plano, v.ravel(), minlength=BINS ** 2) / np.maximum(n, 1), np.nan)
             .reshape(BINS, BINS) for k, v in (("gris", d_gris_px), ("act", d_act_px))}
    lim_g = np.percentile(np.abs(d_gris_px), 99)
    lim_a = np.percentile(np.abs(d_act_px), 99)

    fig, ejes = plt.subplots(2, 4, figsize=(30, 15))
    ax = ejes[0, 0]
    ax.imshow(caso["ficticia"][Y0:Y0 + LADO, X0:X0 + LADO], cmap="gray", vmin=rango[0], vmax=rango[1])
    ax.set_title(f"{nombre}: imagen de partida (recorte)")
    ax.set_axis_off()
    mapa_fases(ejes[0, 1], etiquetas, "verdad")
    for ax, v, lim, titulo, unidad in ((ejes[0, 2], d_gris_px, lim_g, "Δgris por píxel", "Δgris"),
                                       (ejes[0, 3], d_act_px, lim_a, "Δactividad por píxel", "Δactividad")):
        im = ax.imshow(v[Y0:Y0 + LADO, X0:X0 + LADO], cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax, fraction=0.046, label=unidad)
        ax.set_title(f"{titulo}\n(rojo = sube, azul = baja)")
        ax.set_axis_off()

    for ax, clave_m, lim, titulo in ((ejes[1, 0], "gris", lim_g, "Δgris medio por bin"),
                                     (ejes[1, 1], "act", lim_a, "Δactividad medio por bin")):
        im = ax.imshow(medio[clave_m], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=extent)
        fig.colorbar(im, ax=ax, fraction=0.046)
        for g, c in zip(GRISES, COLORES):
            ax.axvline(g, color=c, lw=4, alpha=0.6)
        ax.set_facecolor("0.9")
        ax.set_xlabel("gris de partida")
        ax.set_ylabel("actividad de partida (0-1)")
        ax.set_title(f"{titulo}\n(paisaje de la imagen de partida)")

    for ax, v, lim, titulo in ((ejes[1, 2], d_gris_px, lim_g, "Δgris"), (ejes[1, 3], d_act_px, lim_a, "Δactividad")):
        for i in range(3):
            ax.hist(v[etiquetas == i], bins=150, range=(-1.2 * lim, 1.2 * lim), histtype="step", lw=3,
                    color=COLORES[i], density=True, label=f"{FASES[i]}: media {v[etiquetas == i].mean():+.4f}")
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel(titulo)
        ax.set_ylabel("densidad")
        ax.legend(loc="upper left")
        ax.set_title(f"{titulo} por fase verdadera")
    print(f"{nombre:<24} Δgris medio: " + ", ".join(f"{FASES[i]} {d_gris_px[etiquetas == i].mean():+.4f}"
                                                   for i in range(3))
          + "; Δactividad medio: " + ", ".join(f"{FASES[i]} {d_act_px[etiquetas == i].mean():+.3f}" for i in range(3)))
    guardar(fig, f"06c_componentes_flujo_{clave}.png", f"{nombre}: componentes del desplazamiento por el choque")


def pozos_semilla_cercana(semillas, normalizado):
    """Cada bin toma la semilla más cercana en el paisaje (distancia en bins)."""
    _, (cerca_f, cerca_c) = ndimage.distance_transform_edt(semillas == 0, return_indices=True)
    return semillas[cerca_f, cerca_c]


def pozos_watershed(semillas, normalizado):
    """Inundación (watershed) desde las semillas sobre el paisaje normalizado invertido.

    Las masas son pozos (más píxeles = más hondo); la frontera entre pozos cae en la cresta, donde
    hay menos píxeles entre dos masas.
    """
    return ski.segmentation.watershed(-normalizado, semillas)


def figura_pozos(asignar_pozos, nombre_metodo, archivo):
    """Paisaje gris tras el choque × actividad de partida; pozos con `asignar_pozos` y resultado."""
    fig, ejes = plt.subplots(2, 4, figsize=(30, 14), gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})
    for fila, (nombre, caso) in enumerate(casos.items()):
        i_act, _ = caso["origen"]
        _, i_gris = indices_de_bin(caso["choque"], caso["act1"], rango, act_max)
        conteos = np.bincount(np.ravel_multi_index((i_act, i_gris), (BINS, BINS)).ravel(),
                              minlength=BINS ** 2).reshape(BINS, BINS)
        semillas, normalizado = semillas_del_paisaje(conteos.astype(float), etiquetas.size)
        etiqueta_bin = asignar_pozos(semillas, normalizado)
        etiqueta_px = etiqueta_bin[i_act, i_gris]
        directo = semillas[i_act, i_gris] > 0
        n_sem = semillas.max()
        fase_de = np.array([np.bincount(etiquetas[etiqueta_px == k], minlength=3).argmax()
                            for k in range(1, n_sem + 1)])
        prediccion = fase_de[etiqueta_px - 1]
        acierto = [np.mean(prediccion[etiquetas == i] == i) for i in range(3)]
        print(f"{archivo[:3]} {nombre:<20} semillas {[FASES[f] for f in fase_de]}; directo {directo.mean():.0%}, "
              f"resto {1 - directo.mean():.0%}; acierto " + ", ".join(f"{n} {a:.0%}" for n, a in zip(FASES, acierto)))

        ax = ejes[fila, 0]
        rgb = np.array([to_rgb(c) for c in COLORES])[fase_de[etiqueta_bin - 1]]
        brillo = np.log1p(conteos) / np.log1p(conteos).max()
        ax.imshow(1 - (0.15 + 0.85 * brillo[..., None]) * (1 - rgb), origin="lower", aspect="auto", extent=extent)
        for k in range(1, n_sem + 1):
            ax.contour(centros_g, centros_a, semillas == k, levels=[0.5], colors="k", linewidths=2)
        ax.set_xlabel("gris tras el choque")
        ax.set_ylabel("actividad de partida (0-1)")
        ax.set_title(f"{nombre}: pozos ({nombre_metodo})\ncontorno negro = semilla ({n_sem}); intensidad = log cuentas")

        ax = ejes[fila, 1]
        im = ax.imshow(normalizado, origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
        fig.colorbar(im, ax=ax, fraction=0.046)
        for k in range(1, n_sem + 1):
            ax.contour(centros_g, centros_a, etiqueta_bin == k, levels=[0.5], colors="k", linewidths=1.5)
        for k, f in enumerate(fase_de, start=1):
            ax.contour(centros_g, centros_a, semillas == k, levels=[0.5], colors=COLORES[f], linewidths=3)
        ax.set_xlabel("gris tras el choque")
        ax.set_ylabel("actividad de partida (0-1)")
        ax.set_title(f"paisaje normalizado por fila; negro = frontera\ndirecto {directo.mean():.0%}, "
                     f"por pozo {1 - directo.mean():.0%}")
        mapa_fases(ejes[fila, 2], prediccion, "resultado")
        ejes[fila, 2].legend(handles=[Patch(color=c, label=f"{n}: {a:.0%} bien")
                                      for c, n, a in zip(COLORES, FASES, acierto)], loc="lower left")
        mapa_fases(ejes[fila, 3], etiquetas, "verdad")
    guardar(fig, archivo, f"Pozos por {nombre_metodo}: gris tras el choque × actividad de partida "
                          "(arriba nítida, abajo con degradé)")


# 06d: cada píxel toma la semilla en la que cae; si no cae en ninguna, la semilla más cercana en el paisaje.
figura_pozos(pozos_semilla_cercana, "semilla más cercana", "06d_gris_final_actividad_inicial.png")
# 06e: igual, pero la frontera entre pozos la pone la inundación (watershed) en la cresta de densidad.
figura_pozos(pozos_watershed, "inundación (watershed)", "06e_pozos_watershed.png")
print("ok")
