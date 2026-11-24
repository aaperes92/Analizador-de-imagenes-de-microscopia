"""Simulada de TRES fases: inercia de los píxeles durante el filtro de choque.

Misma simulada que tres_fases_choque_con_pozos.py (matriz 0.70 56 %, patrón A 0.60 25 %, patrón B
0.52 19 %), nítida y con degradé (gaussiana 1.5 px), ruido 0.005. Cadena: partida -> choque solo.

Idea: en el paisaje gris final x actividad de partida, las semillas (media altura por fila, masa
>= 1 %) son núcleos atractivos, con su gris central. Un píxel que no termina dentro de un núcleo
tiene un núcleo candidato (el más cercano en gris a donde terminó). Su inercia es qué fracción del
camino hacia ese núcleo recorrió durante el choque:
    f = (gris final - gris inicial) / (gris del núcleo - gris inicial)
f ~ 1: fue con gusto y llegó; 0 < f < 1: se frenó antes (se resiste); f < 0: se alejó.
También se guarda en qué iteración se frenó (última iteración con cambio > 0.1 x ruido).
Regla propuesta: si se resiste, pertenece al núcleo vecino del lado de donde vino.

Primero se comprueba con la verdad si la inercia separa a las poblaciones (09a, 09b); después se
aplica la regla con un umbral de f por Otsu sobre los píxeles fuera de los núcleos (09c).
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
MEDIA_ALTURA, MASA_MINIMA = 0.5, 0.01
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


def choque_con_trayectoria(img, sigma_interno, tolerancia_abs, dt=0.25, max_pasos=200):
    """Filtro de choque (Osher-Rudin, minmod) que guarda el cambio de cada píxel en cada iteración."""
    I = img.astype(float).copy()
    cambios = []
    for _ in range(max_pasos):
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
        cambios.append(delta)
        if np.mean(np.abs(delta)) < tolerancia_abs:
            break
    return I, np.array(cambios)


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


def nucleos(img_final, act):
    """Núcleos (semillas) ordenados por gris central; por píxel: núcleo (0 = ninguno, 1.. = orden) y pozo."""
    i_act = np.clip((act * BINS).astype(int), 0, BINS - 1)
    rango = np.percentile(img_final, (0.1, 99.9))
    i_gris = np.clip(((img_final - rango[0]) / (rango[1] - rango[0]) * BINS).astype(int), 0, BINS - 1)
    conteos = np.bincount(np.ravel_multi_index((i_act, i_gris), (BINS, BINS)).ravel(),
                          minlength=BINS ** 2).reshape(BINS, BINS).astype(float)
    semillas, normalizado = semillas_del_paisaje(conteos, img_final.size)
    pozos = ski.segmentation.watershed(-normalizado, semillas)
    n = semillas.max()
    centrales = np.asarray(ndimage.median(img_final, semillas[i_act, i_gris], index=np.arange(1, n + 1)))
    orden = np.argsort(centrales)
    rango_de = np.zeros(n + 1, int)
    rango_de[orden + 1] = np.arange(1, n + 1)
    return {"en_nucleo": rango_de[semillas[i_act, i_gris]], "pozo": rango_de[pozos[i_act, i_gris]] - 1,
            "centrales": centrales[orden], "conteos": conteos, "semillas": rango_de[semillas],
            "rango": rango, "i_act": i_act, "i_gris": i_gris}


def inercia(inicial, final, centrales):
    """Núcleo candidato (más cercano en gris al final), fracción del camino recorrida y núcleo vecino.

    El núcleo vecino es el adyacente en gris del lado de donde vino el píxel (si no hay, el mismo).
    """
    candidato = np.argmin(np.abs(final[..., None] - centrales[None, None, :]), axis=-1)
    objetivo = centrales[candidato]
    distancia = objetivo - inicial
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(np.abs(distancia) > 1e-4, (final - inicial) / distancia, 1.0)
    vecino = np.clip(candidato + np.where(inicial < objetivo, -1, 1), 0, len(centrales) - 1)
    return candidato, f, vecino


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


def acierto_por_fase(prediccion):
    return [np.mean(prediccion[etiquetas == i] == i) for i in range(3)]


etiquetas, rng = tres_fases_con_presencia()
print("fracciones verdaderas:", {n: f"{np.mean(etiquetas == i):.1%}" for i, n in enumerate(FASES)})
ruido = rng.normal(0, 0.005, etiquetas.shape)
partida = {"nítida": GRISES[etiquetas] + ruido,
           "con degradé (σ=1.5 px)": ski.filters.gaussian(GRISES[etiquetas], 1.5) + ruido}

datos = {}
for nombre, img in partida.items():
    ruido_est = estimar_ruido(img)
    final, cambios = choque_con_trayectoria(img, 1.0, 0.1 * ruido_est)
    act = actividad_de_partida(img)
    n = nucleos(final, act)
    candidato, f, vecino = inercia(img, final, n["centrales"])
    moviendose = np.abs(cambios) > 0.1 * ruido_est
    freno = np.where(moviendose.any(axis=0), len(cambios) - np.argmax(moviendose[::-1], axis=0), 0)
    fuera = n["en_nucleo"] == 0
    datos[nombre] = {"img": img, "final": final, "act": act, "n": n, "candidato": candidato, "f": f,
                     "vecino": vecino, "freno": freno, "fuera": fuera, "pasos": len(cambios)}
    print(f"{nombre}: {len(cambios)} pasos de choque; núcleos en {np.round(n['centrales'], 3)}; "
          f"fuera de los núcleos {fuera.mean():.0%} de los píxeles")
    for c in range(len(n["centrales"])):
        sel = fuera & (candidato == c)
        print(f"   candidato {FASES[c]:<9} {sel.mean():5.1%} de los píxeles; fase real "
              + ", ".join(f"{FASES[i]} {np.mean(etiquetas[sel] == i):.0%}" for i in range(3))
              + "; f mediana por fase real "
              + ", ".join(f"{FASES[i]} {np.median(f[sel & (etiquetas == i)]) if (sel & (etiquetas == i)).any() else np.nan:+.2f}"
                          for i in range(3)))

# 09a: histogramas de f y del momento de frenado, por núcleo candidato y fase real (degradé)
nombre_deg = "con degradé (σ=1.5 px)"
d = datos[nombre_deg]
n_nuc = len(d["n"]["centrales"])
fig, ejes = plt.subplots(2, n_nuc, figsize=(9 * n_nuc, 14))
for c in range(n_nuc):
    sel = d["fuera"] & (d["candidato"] == c)
    for i in range(3):
        m = sel & (etiquetas == i)
        if m.sum() < 50:
            continue
        ejes[0, c].hist(np.clip(d["f"][m], -1, 2), bins=80, range=(-1, 2), histtype="step", lw=3, color=COLORES[i],
                        label=f"{FASES[i]}: {m.sum()} px")
        ejes[1, c].hist(d["freno"][m], bins=np.arange(d["pasos"] + 2) - 0.5, histtype="step", lw=3,
                        color=COLORES[i], label=f"{FASES[i]}")
    ejes[0, c].axvline(0, color="k", lw=1)
    ejes[0, c].axvline(1, color="k", lw=1, ls="--")
    ejes[0, c].set_xlabel("f = fracción del camino hacia el núcleo candidato")
    ejes[0, c].set_ylabel("cantidad de píxeles")
    ejes[0, c].set_title(f"candidato: núcleo {FASES[c]} ({d['n']['centrales'][c]:.3f})\n"
                         f"píxeles fuera de los núcleos")
    ejes[0, c].legend(loc="upper left")
    ejes[1, c].set_xlabel("iteración en que se frenó")
    ejes[1, c].set_ylabel("cantidad de píxeles")
    ejes[1, c].set_title("momento de frenado")
    ejes[1, c].legend(loc="upper left")
guardar(fig, "09a_inercia_histogramas.png",
        f"{nombre_deg}: inercia de los píxeles fuera de los núcleos, separada por fase real")

# 09b: paisaje coloreado por f medio y por fase real
n = d["n"]
plano = np.ravel_multi_index((n["i_act"], n["i_gris"]), (BINS, BINS)).ravel()
cuantos = np.bincount(plano[d["fuera"].ravel()], minlength=BINS ** 2)
f_medio = np.bincount(plano[d["fuera"].ravel()], np.clip(d["f"], -1, 2).ravel()[d["fuera"].ravel()],
                      minlength=BINS ** 2) / np.maximum(cuantos, 1)
f_medio = np.where(cuantos >= 3, f_medio, np.nan).reshape(BINS, BINS)
por_fase = np.stack([np.bincount(plano[(etiquetas == i).ravel()], minlength=BINS ** 2).reshape(BINS, BINS)
                     for i in range(3)])
total = por_fase.sum(axis=0)
rgb = np.array([[int(c[k:k + 2], 16) / 255 for k in (1, 3, 5)] for c in COLORES])
mezcla = np.einsum("fyx,fc->yxc", por_fase, rgb) / np.maximum(total, 1)[..., None]
brillo = (np.log1p(total) / np.log1p(total).max())[..., None]
extent = (*n["rango"], 0, 1)
centros_g = n["rango"][0] + (np.arange(BINS) + 0.5) / BINS * (n["rango"][1] - n["rango"][0])
centros_a = (np.arange(BINS) + 0.5) / BINS
fig, ejes = plt.subplots(1, 2, figsize=(24, 10))
ax = ejes[0]
im = ax.imshow(f_medio, origin="lower", aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.5, extent=extent)
fig.colorbar(im, ax=ax, fraction=0.046, label="f medio (verde = llegó, rojo = se resistió)")
ax.set_title("f medio de los píxeles fuera de los núcleos\n(contorno negro = núcleos)")
ax = ejes[1]
ax.imshow(1 - brillo * (1 - mezcla), origin="lower", aspect="auto", extent=extent)
ax.set_title("fase real (color) con brillo = log cuentas")
for ax in ejes:
    for k in range(1, n_nuc + 1):
        ax.contour(centros_g, centros_a, n["semillas"] == k, levels=[0.5], colors="k", linewidths=2)
    ax.set_xlabel("gris tras el choque")
    ax.set_ylabel("actividad de partida (0-1)")
guardar(fig, "09b_inercia_paisaje.png", f"{nombre_deg}: inercia en el paisaje contra la fase real")

# 09c: regla de la inercia contra pozos por inundación (06e) y núcleo más cercano
fig, ejes = plt.subplots(2, 4, figsize=(26, 14))
for fila, (nombre, d) in enumerate(datos.items()):
    n = d["n"]
    dentro = n["en_nucleo"] - 1
    f_fuera = np.clip(d["f"][d["fuera"]], -1, 2)
    umbral = ski.filters.threshold_otsu(f_fuera) if f_fuera.size > 100 else 0.5
    resiste = d["f"] < umbral
    regla = np.where(d["fuera"], np.where(resiste, d["vecino"], d["candidato"]), dentro)
    cercano = np.where(d["fuera"], d["candidato"], dentro)
    for col, (titulo, pred) in enumerate((("pozos por inundación (06e)", n["pozo"]),
                                          ("núcleo más cercano", cercano),
                                          (f"regla de la inercia (umbral f = {umbral:.2f})", regla))):
        acierto = acierto_por_fase(pred)
        mapa_fases(ejes[fila, col], pred, f"{nombre}\n{titulo}")
        leyenda_acierto(ejes[fila, col], acierto)
        print(f"{nombre:<24} {titulo:<42} acierto " + ", ".join(f"{FASES[i]} {a:.0%}" for i, a in enumerate(acierto)))
    mapa_fases(ejes[fila, 3], etiquetas, "verdad")
guardar(fig, "09c_inercia_regla.png", "Asignación por inercia contra pozos y núcleo más cercano")
print("ok")
