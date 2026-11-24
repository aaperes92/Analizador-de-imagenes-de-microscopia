"""Regiones de análisis combinando gris y actividad: mezcla de gaussianas con K elegido por BIC.

Cada píxel = punto (gris de la suavizada, actividad = gaussiana(sobel(suavizada), 25 px)).
Escalado robusto sin recortar: (x - mediana) / rango intercuartil.
GaussianMixture (scikit-learn) con K = 1..6 sobre una muestra de píxeles; se elige el K con menor BIC.
Se corre desde la raíz del proyecto. Guarda las figuras en pruebas_sem/segmentacion/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "src")
from sem.suavizado import suavizar

SALIDA = Path("pruebas_sem/segmentacion")
SIGMA_ACTIVIDAD = 25
K_MAXIMO = 6
MUESTRA = 60_000
SEMILLA = 0
COLORES = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377"]


def escalar(x):
    q1, mediana, q3 = np.percentile(x, (25, 50, 75))
    return (x - mediana) / (q3 - q1), (mediana, q3 - q1)


rng = np.random.default_rng(SEMILLA)
for k in (1, 3, 4):
    roi = np.load(f"data/sem/roi/BSE_1/BSE_1_{k}/roi_BSE_1_{k}.npy")
    suave = suavizar(roi)
    actividad = ski.filters.gaussian(ski.filters.sobel(suave), SIGMA_ACTIVIDAD)
    gris_e, (m_g, s_g) = escalar(suave.ravel())
    act_e, (m_a, s_a) = escalar(actividad.ravel())
    X = np.column_stack([gris_e, act_e])
    muestra = X[rng.choice(len(X), min(MUESTRA, len(X)), replace=False)]

    modelos, bics = [], []
    for kk in range(1, K_MAXIMO + 1):
        gmm = GaussianMixture(n_components=kk, covariance_type="full", random_state=SEMILLA, n_init=2)
        gmm.fit(muestra)
        modelos.append(gmm)
        bics.append(gmm.bic(muestra))
    mejor = int(np.argmin(bics))
    gmm = modelos[mejor]
    n = gmm.n_components
    # Regiones ordenadas por gris medio (de la más oscura a la más clara)
    orden = np.argsort(gmm.means_[:, 0])
    reetiqueta = np.empty(n, int)
    reetiqueta[orden] = np.arange(n)
    regiones = reetiqueta[gmm.predict(X)].reshape(suave.shape)
    fracciones = [np.mean(regiones == r) for r in range(n)]
    print(f"ROI_{k}: BIC por K = {np.round(np.array(bics) / 1e3, 1)} (miles) -> K = {n}; áreas {np.round(fracciones, 3)}")
    for r, c in enumerate(orden):
        g = gmm.means_[c, 0] * s_g + m_g
        a = gmm.means_[c, 1] * s_a + m_a
        print(f"   región {r}: gris medio {g:.3f}, actividad media {a:.4f}")

    fig, ejes = plt.subplots(1, 5, figsize=(34, 7), gridspec_kw={"width_ratios": [1.2, 0.8, 1, 1, 0.6]})
    ax = ejes[0]
    ax.hist2d(gris_e, act_e, bins=200, cmap="Greys", norm="log")
    for r, c in enumerate(orden):
        vals, vecs = np.linalg.eigh(gmm.covariances_[c])
        angulo = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(gmm.means_[c], 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=angulo,
                             fill=False, lw=2.5, color=COLORES[r]))
        ax.plot(*gmm.means_[c], "o", color=COLORES[r], ms=10, mec="k")
    ax.set_xlabel("gris (escalado)", fontsize=13)
    ax.set_ylabel("actividad (escalada)", fontsize=13)
    ax.set_title(f"ROI {k} — plano gris-actividad\n(elipses = nubes encontradas, 2 desvíos)", fontsize=14)

    ax = ejes[1]
    ax.plot(range(1, K_MAXIMO + 1), np.array(bics) / 1e3, "o-", color="k")
    ax.plot(n, bics[mejor] / 1e3, "o", color="C3", ms=14, mfc="none", mew=3)
    ax.set_xlabel("K (cantidad de nubes)", fontsize=13)
    ax.set_ylabel("BIC (miles; menor = mejor)", fontsize=13)
    ax.set_title(f"BIC según K: elegido K = {n}", fontsize=14)

    ax = ejes[2]
    ax.imshow(regiones, cmap=ListedColormap(COLORES[:n]), vmin=-0.5, vmax=n - 0.5, interpolation="nearest")
    ax.set_title(f"{n} regiones (colores sólidos)", fontsize=15)
    ax.set_axis_off()

    ax = ejes[3]
    lo, hi = np.percentile(suave, (0.5, 99.5))
    ax.imshow(suave, cmap="gray", vmin=lo, vmax=hi)
    bordes = ski.segmentation.find_boundaries(regiones, mode="inner")
    ax.imshow(np.ma.masked_where(~bordes, bordes), cmap=ListedColormap(["#ff2020"]), interpolation="nearest")
    ax.set_title("bordes sobre la ROI suavizada", fontsize=15)
    ax.set_axis_off()

    ax = ejes[4]
    ax.bar(range(n), [100 * f for f in fracciones], color=COLORES[:n], edgecolor="k")
    ax.set_xticks(range(n), [f"región {r}" for r in range(n)], fontsize=11, rotation=30)
    ax.set_ylim(0, 105)
    ax.set_title("área (%)", fontsize=15)
    for r, f in enumerate(fracciones):
        ax.text(r, 100 * f + 2, f"{100 * f:.1f}%", ha="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(SALIDA / f"13_regiones_gris_y_actividad_ROI{k}.png", dpi=42)
    plt.close(fig)
print("ok")
