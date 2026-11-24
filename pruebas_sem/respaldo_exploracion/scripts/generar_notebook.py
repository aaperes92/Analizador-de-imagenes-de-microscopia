"""Genera notebooks/sem_suavizado.ipynb (se corre desde la raíz del proyecto)."""

import json
from pathlib import Path


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


celdas = [
md(r"""
# SEM — Etapa 2: suavizado de la ROI

Reducción del ruido de las ROIs de imágenes SEM de electrones retrodispersados (BSE), antes de
corregir la iluminación y ecualizar.

**Método:** *non-local means* (`skimage.restoration.denoise_nl_means`). Para cada píxel busca
parches parecidos al suyo y los promedia; en un borde entre fases los parches de la otra fase no
se parecen, así que el borde se conserva.

**Parámetros, fijados con criterios objetivos (sin ajustar a mano):**

| Parámetro | Cómo se fija |
|---|---|
| `sigma_ruido` | Se estima de la propia imagen (`estimar_ruido`, wavelets + mediana) |
| `factor_h` (intensidad, `h = factor_h · σ`) | Criterio de discrepancia: lo quitado debe tener la desviación estándar del ruido (`buscar_factor_h`) |
| `tamano_parche` | 5×5, elegido con la prueba de parches del anexo |

Código: `src/sem/suavizado.py`. Las ROIs se generan antes con la Etapa 1 (`scripts/sem_carga.py`).
"""),
md("## Configuración"),
code(r"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Raíz del proyecto: la primera carpeta hacia arriba que tiene pyproject.toml
RAIZ = Path.cwd()
while not (RAIZ / "pyproject.toml").exists():
    RAIZ = RAIZ.parent
sys.path.insert(0, str(RAIZ / "src"))

from sem.suavizado import buscar_factor_h, estimar_ruido, suavizar

# Carpeta con las ROIs de la imagen a analizar (generadas con la Etapa 1)
CARPETA_ROIS = RAIZ / "data" / "sem" / "roi" / "BSE_1"
"""),
md(r"""
## 1. ROIs de entrada

Se cargan los `.npy` (float32 en [0, 1]) guardados por la Etapa 1. Los histogramas usan eje
vertical logarítmico para ver también las colas.
"""),
code(r"""
rutas = sorted(CARPETA_ROIS.glob("*/roi_*.npy"))
rois = {ruta.stem: np.load(ruta) for ruta in rutas}

fig, ejes = plt.subplots(2, len(rois), figsize=(4.2 * len(rois), 7.5), squeeze=False)
for i, (nombre, roi) in enumerate(rois.items()):
    ejes[0, i].imshow(roi, cmap="gray", vmin=0, vmax=1)
    ejes[0, i].set_title(f"{nombre}  {roi.shape[1]}x{roi.shape[0]} px", fontsize=10)
    ejes[0, i].set_axis_off()
    ejes[1, i].hist(roi.ravel(), bins=256, range=(0, 1), color="0.3")
    ejes[1, i].set_yscale("log")
    ejes[1, i].set_xlabel("intensidad")
ejes[1, 0].set_ylabel("cantidad de píxeles (log)")
fig.tight_layout()
plt.show()
"""),
md(r"""
## 2. Estimación del ruido

`estimar_ruido` usa `skimage.restoration.estimate_sigma`: toma los coeficientes wavelet más
finos (casi todo ruido) y calcula la desviación estándar con la mediana de su valor absoluto
(`σ = mediana(|c|) / 0.6745`). La mediana hace que los bordes entre fases, que ocupan una fracción
chica de la imagen, casi no afecten la estimación.

Si todas las ROIs salen de la misma imagen, σ debería dar parecido en todas: el ruido depende del
equipo y de la adquisición, no de la zona.
"""),
code(r"""
sigmas = {nombre: estimar_ruido(roi) for nombre, roi in rois.items()}
for nombre, sigma in sigmas.items():
    print(f"{nombre}: σ = {sigma:.4f}  (≈ {sigma * 255:.1f} niveles de gris en 8 bits)")
"""),
md(r"""
## 3. Intensidad del filtrado: criterio de discrepancia

El filtrado es adecuado cuando **lo que se quita tiene la misma desviación estándar que el
ruido**:

    desviación estándar(original − suavizada) = σ

Si se quita menos, queda ruido; si se quita más, se borra señal. `buscar_factor_h` encuentra el
`factor_h` que lo cumple por bisección (corta cuando la diferencia es menor al 2 %).
"""),
code(r"""
factores = {}
for nombre, roi in rois.items():
    t0 = time.perf_counter()
    factores[nombre] = buscar_factor_h(roi, sigmas[nombre])
    print(f"{nombre}: factor_h = {factores[nombre]:.2f}  ({time.perf_counter() - t0:.1f} s)")
"""),
md(r"""
## 4. Suavizado y comparación

Por fila: ROI original, suavizada, lo quitado (original − suavizada) e histogramas antes y
después, con la misma escala.

**Qué mirar:**
- *Lo quitado* debería verse como ruido sin forma. Si se distinguen bordes o fases, el filtro se
  está llevando señal.
- En los histogramas, al bajar el ruido las poblaciones se angostan y pueden aparecer picos o
  valles que antes estaban tapados.
"""),
code(r"""
suavizadas = {}
fig, ejes = plt.subplots(len(rois), 5, figsize=(22, 4.3 * len(rois)), squeeze=False)
for i, (nombre, roi) in enumerate(rois.items()):
    sigma, factor_h = sigmas[nombre], factores[nombre]
    suave = suavizar(roi, factor_h=factor_h, sigma_ruido=sigma)
    suavizadas[nombre] = suave
    quitado = roi - suave
    print(f"{nombre}: quitado = {quitado.std():.4f} ({quitado.std() / sigma:.0%} de σ), "
          f"σ después = {estimar_ruido(suave):.4f}")

    ejes[i, 0].imshow(roi, cmap="gray", vmin=0, vmax=1)
    ejes[i, 0].set_title(f"{nombre} original (σ={sigma:.4f})", fontsize=10)
    ejes[i, 1].imshow(suave, cmap="gray", vmin=0, vmax=1)
    ejes[i, 1].set_title(f"suavizada (5x5, h={factor_h:.2f}σ)", fontsize=10)
    ejes[i, 2].imshow(quitado, cmap="gray", vmin=-3 * sigma, vmax=3 * sigma)
    ejes[i, 2].set_title("lo quitado (±3σ)", fontsize=10)
    for ax in ejes[i, :3]:
        ax.set_axis_off()
    for j, (imagen, titulo) in enumerate(((roi, "histograma original"), (suave, "histograma suavizada"))):
        ax = ejes[i, 3 + j]
        ax.hist(imagen.ravel(), bins=256, range=(0, 1), color="0.3")
        ax.set_yscale("log")
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("intensidad")
    techo = max(ejes[i, 3].get_ylim()[1], ejes[i, 4].get_ylim()[1])
    ejes[i, 3].set_ylim(0.7, techo)
    ejes[i, 4].set_ylim(0.7, techo)
fig.tight_layout()
plt.show()
"""),
md(r"""
## Anexo: por qué parche de 5×5

La discrepancia fija **cuánto** quitar, pero no elige el tamaño del parche: para cada parche hay
un `factor_h` que la cumple. Para comparar parches se mira **qué** se quitó, con la
**autocorrelación a 1 píxel** de lo quitado (correlación entre lo quitado y lo quitado corrido un
píxel):

- cerca de 0: los vecinos no se parecen, lo quitado es ruido;
- positiva: lo quitado tiene forma, el filtro se llevó parte de la imagen.

Se elige el parche con menor autocorrelación. En las ROIs de `BSE_1` el mínimo fue 5×5 en todas;
3×3 se lleva la microestructura. La autocorrelación horizontal sale siempre algo mayor que la
vertical por el barrido del haz (el ruido de una misma fila está algo relacionado), así que el
"piso" no es exactamente 0.

Con imágenes muy distintas (otra magnificación, otro equipo) conviene repetir esta prueba. Para
que sea rápida, por defecto corre sobre la ROI más chica.
"""),
code(r'''
def autocorrelacion_1px(residuo: np.ndarray) -> tuple[float, float]:
    """Autocorrelación del residuo con su vecino a 1 píxel: (horizontal, vertical)."""
    horizontal = np.corrcoef(residuo[:, :-1].ravel(), residuo[:, 1:].ravel())[0, 1]
    vertical = np.corrcoef(residuo[:-1, :].ravel(), residuo[1:, :].ravel())[0, 1]
    return horizontal, vertical


PARCHES = (3, 5, 7, 9)
nombre = min(rois, key=lambda n: rois[n].size)   # ROI más chica
roi, sigma = rois[nombre], sigmas[nombre]

fig, ejes = plt.subplots(1, len(PARCHES), figsize=(4.2 * len(PARCHES), 4.6))
print(f"{nombre}\nparche  factor_h  autocorr H  autocorr V")
for ax, parche in zip(ejes, PARCHES):
    factor_h = buscar_factor_h(roi, sigma, tamano_parche=parche)
    residuo = roi - suavizar(roi, factor_h=factor_h, tamano_parche=parche, sigma_ruido=sigma)
    horizontal, vertical = autocorrelacion_1px(residuo)
    print(f"{parche}x{parche}    {factor_h:5.2f}     {horizontal:+.3f}      {vertical:+.3f}")
    ax.imshow(residuo, cmap="gray", vmin=-3 * sigma, vmax=3 * sigma)
    ax.set_title(f"lo quitado, parche {parche}x{parche}\nautocorr H={horizontal:+.2f} V={vertical:+.2f}", fontsize=10)
    ax.set_axis_off()
fig.tight_layout()
plt.show()
'''),
]

for i, celda in enumerate(celdas):
    celda["id"] = f"celda-{i:02d}"
nb = {
    "cells": celdas,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("notebooks/sem_suavizado.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print("ok")
