"""Simulada de TRES fases: segmentación por REGIONES con área en el paisaje normalizado.

Misma simulada que tres_fases_paisaje_normalizado.py (matriz 56 %, patrón A 25 %, patrón B 19 %),
versiones normal e invertida, nítida y con degradé (gaussiana 1.5 px).

Procesamiento, sobre la imagen suavizada y sobre la variante c (enfoque + choque):
1. paisaje gris-actividad (actividad = gaussiana(sobel), 3 px), suavizado 1 bin y normalizado por
   el máximo de cada fila (filas con < 0.5 % de los píxeles de la más poblada, en cero);
2. regiones: bins que superan la mitad del máximo de su fila (media altura), conexos en 2D;
3. se descartan las regiones con menos del 1 % de los píxeles de la imagen (masa, no área en bins);
4. vuelta a la imagen: cada píxel toma la región de su bin; los que no caen en ninguna toman la
   etiqueta del píxel etiquetado más cercano en la imagen;
5. evaluación: cada región se asocia a la fase verdadera mayoritaria entre sus píxeles.
Figuras por caso: 03a imágenes e histogramas, 03b paisajes, 03c segmentación.
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
from sem.suavizado import estimar_ruido, suavizar

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15})

SALIDA = Path("pruebas_sem/tres_fases")
SALIDA.mkdir(parents=True, exist_ok=True)
ALTO, ANCHO = 520, 560
GRISES = {"normal": np.array([0.52, 0.60, 0.70]),       # 0 = patrón B, 1 = patrón A, 2 = matriz
          "invertida": np.array([0.70, 0.62, 0.52])}
FASES = ["patrón B", "patrón A", "matriz"]
COLORES = ["#1f77b4", "#2ca02c", "#ff7f0e"]
SIGMA_ACT, BINS, FILA_MINIMA = 3, 128, 0.005
MEDIA_ALTURA, MASA_MINIMA = 0.5, 0.01
CMAP_PAISAJE = LinearSegmentedColormap.from_list("rojo_fucsia", ["#ffffff", "#ff2a2a", "#ff00ff"])
Y0, X0, LADO = 170, 190, 170


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


def regiones_del_paisaje(img):
    """Paisaje normalizado por fila, regiones con masa y segmentación de la imagen por vuelta atrás."""
    rango = tuple(np.percentile(img, (0.1, 99.9)))
    act = ski.filters.gaussian(ski.filters.sobel(img), SIGMA_ACT)
    act = (act - act.min()) / (act.max() - act.min())
    conteos, _, bordes_g = np.histogram2d(act.ravel(), img.ravel(), bins=BINS, range=((0, 1), rango))
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
    segmentada = directa[iy, ix]
    centros_g = (bordes_g[:-1] + bordes_g[1:]) / 2
    return {"log": np.log1p(conteos), "normalizado": normalizado, "regiones": regiones, "directa": directa,
            "segmentada": segmentada, "centros_g": centros_g, "rango": rango, "masas": masas[con_masa - 1]}


def evaluar(r, etiquetas):
    """Asocia cada región a la fase verdadera mayoritaria y mide el acierto por fase."""
    n = r["regiones"].max()
    fase_de = np.array([np.bincount(etiquetas[r["segmentada"] == k], minlength=3).argmax() for k in range(1, n + 1)])
    prediccion = fase_de[r["segmentada"] - 1]
    acierto = [np.mean(prediccion[etiquetas == i] == i) for i in range(3)]
    return fase_de, prediccion, acierto


def imagen_recorte(ax, img, titulo, rango):
    ax.imshow(img[Y0:Y0 + LADO, X0:X0 + LADO], cmap="gray", vmin=rango[0], vmax=rango[1])
    ax.set_title(titulo)
    ax.set_axis_off()


def mapa_fases(ax, fases, titulo):
    ax.imshow(fases[Y0:Y0 + LADO, X0:X0 + LADO], cmap=ListedColormap(COLORES), vmin=-0.5, vmax=2.5,
              interpolation="nearest")
    ax.set_title(titulo)
    ax.set_axis_off()


etiquetas, rng = tres_fases_con_presencia()
print("fracciones verdaderas:", {n: f"{np.mean(etiquetas == i):.1%}" for i, n in enumerate(FASES)})
ruido = rng.normal(0, 0.005, etiquetas.shape)

for clave_grises, grises in GRISES.items():
    nombres = [f"{n} ({g:.2f})" for n, g in zip(FASES, grises)]
    leyenda = [Patch(color=c, label=n) for c, n in zip(COLORES, nombres)]
    casos = {"nitida": ("nítida", grises[etiquetas] + ruido),
             "degrade": ("con degradé (σ=1.5 px)", ski.filters.gaussian(grises[etiquetas], 1.5) + ruido)}
    for clave, (nombre, ficticia) in casos.items():
        suave = suavizar(ficticia)
        sigma_deg, _ = estimar_sigma_borde(suave)
        realzada = ski.filters.unsharp_mask(suave, radius=max(sigma_deg, 0.5), amount=1.0, preserve_range=True)
        procesada, pasos = choque(realzada, 1.0, 0.1 * estimar_ruido(suave))
        caso = f"{clave_grises}, {nombre}"
        procesos = {"antes (suavizada)": suave, "después (variante c)": procesada}
        resultados = {k: regiones_del_paisaje(v) for k, v in procesos.items()}

        # 03a: imágenes ficticias e histogramas de intensidad
        fig, ejes = plt.subplots(2, 4, figsize=(24, 12))
        rango_comun = tuple(np.percentile(np.concatenate([ficticia.ravel(), procesada.ravel()]), (0.1, 99.9)))
        imagenes = {"ficticia (con ruido)": ficticia, **procesos}
        for col, (titulo, v) in enumerate(imagenes.items()):
            imagen_recorte(ejes[0, col], v, titulo, rango_comun)
            ax = ejes[1, col]
            ax.hist(v.ravel(), bins=256, range=rango_comun, color="0.3")
            for g, c in zip(grises, COLORES):
                ax.axvline(g, color=c, lw=4, alpha=0.6)
            ax.set_xlabel("gris")
            ax.set_ylabel("cantidad de píxeles")
            ax.set_title(f"histograma: {titulo}")
        mapa_fases(ejes[0, 3], etiquetas, "verdad")
        ax = ejes[1, 3]
        for (titulo, v), estilo in zip(imagenes.items(), ("-", "--", "-")):
            h, b = np.histogram(v.ravel(), bins=256, range=rango_comun)
            ax.plot((b[:-1] + b[1:]) / 2, h, ls=estilo, lw=2.5, label=titulo)
        for g, c in zip(grises, COLORES):
            ax.axvline(g, color=c, lw=4, alpha=0.6)
        ax.set_xlabel("gris")
        ax.set_title("histogramas superpuestos")
        ax.legend()
        fig.suptitle(f"{caso}: imágenes (recorte) e histogramas; líneas = gris de cada fase", fontsize=21)
        fig.legend(handles=leyenda, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(SALIDA / f"03a_imagenes_{clave_grises}_{clave}.png", dpi=80)
        plt.close(fig)

        # 03b: paisajes antes y después, sin normalizar y normalizados con las regiones
        fig, ejes = plt.subplots(2, 2, figsize=(20, 15))
        for col, (titulo, r) in enumerate(resultados.items()):
            fase_de, _, _ = evaluar(r, etiquetas)
            extent = (*r["rango"], 0, 1)
            for fila, (clave_p, nombre_p) in enumerate((("log", "log(1 + cuentas)"),
                                                        ("normalizado", "fila / máximo de la fila"))):
                ax = ejes[fila, col]
                im = ax.imshow(r[clave_p], origin="lower", aspect="auto", cmap=CMAP_PAISAJE, extent=extent)
                fig.colorbar(im, ax=ax, fraction=0.046)
                if clave_p == "normalizado":
                    centros_a = (np.arange(BINS) + 0.5) / BINS
                    for k, f in enumerate(fase_de, start=1):
                        ax.contour(r["centros_g"], centros_a, r["regiones"] == k, levels=[0.5],
                                   colors=COLORES[f], linewidths=3)
                ax.set_xlabel("gris")
                ax.set_ylabel(f"actividad (σ={SIGMA_ACT} px, 0-1)")
                extra = f"\n{len(fase_de)} regiones con masa ≥ 1 % (contorno = fase asociada)" \
                    if clave_p == "normalizado" else ""
                ax.set_title(f"{titulo}: {nombre_p}{extra}")
        fig.suptitle(f"{caso}: paisaje gris-actividad", fontsize=21)
        fig.legend(handles=leyenda, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(SALIDA / f"03b_paisajes_{clave_grises}_{clave}.png", dpi=80)
        plt.close(fig)

        # 03c: segmentación por vuelta atrás desde las regiones
        fig, ejes = plt.subplots(2, 3, figsize=(21, 14))
        for fila, (titulo, r) in enumerate(resultados.items()):
            fase_de, prediccion, acierto = evaluar(r, etiquetas)
            sin_region = np.mean(r["directa"] == 0)
            directa = np.where(r["directa"] > 0, fase_de[np.maximum(r["directa"], 1) - 1], -1)
            ax = ejes[fila, 0]
            ax.imshow(directa[Y0:Y0 + LADO, X0:X0 + LADO], cmap=ListedColormap(["k", *COLORES]),
                      vmin=-1.5, vmax=2.5, interpolation="nearest")
            ax.set_title(f"{titulo}: vuelta directa\nnegro = sin región ({sin_region:.0%} de los píxeles)")
            ax.set_axis_off()
            mapa_fases(ejes[fila, 1], prediccion, f"{titulo}: resultado\n(sin región → vecino más cercano)")
            ejes[fila, 1].legend(handles=[Patch(color=c, label=f"{n}: {a:.0%} bien")
                                          for c, n, a in zip(COLORES, FASES, acierto)], loc="lower left")
            mapa_fases(ejes[fila, 2], etiquetas, "verdad (mismo recorte)")
            print(f"{caso:<38} {titulo:<22} regiones {len(fase_de)} -> fases {[FASES[f] for f in fase_de]}; "
                  f"sin región {sin_region:.0%}; acierto " + ", ".join(f"{n} {a:.0%}" for n, a in zip(FASES, acierto)))
        fig.suptitle(f"{caso}: segmentación por regiones del paisaje normalizado", fontsize=21)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(SALIDA / f"03c_segmentacion_{clave_grises}_{clave}.png", dpi=80)
        plt.close(fig)
print("ok")
