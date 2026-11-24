"""Tests de src/sem/bse (segmentación de imágenes SEM de electrones retrodispersados).

Todas las imágenes son simuladas (numpy): fases con un gris propio, con o sin degradé (gaussiana)
y ruido, así que no hace falta tener archivos SEM reales.
"""

import numpy as np
import pytest
from scipy import ndimage

from sem.bse import (
    actividad,
    aislar_region,
    cortes_por_valles,
    estimar_degrade,
    filtro_choque,
    filtro_choque_con_pozos,
    nucleos_y_pozos,
    normalizar_por_fila,
    paisaje,
    poblaciones_de_region,
    procesar_region,
    realzar,
    regiones_de_analisis,
    segmentar_roi,
)

# --------------------------------------------------------------------
# Imágenes simuladas
# --------------------------------------------------------------------


def _manchas(lado: int = 160, semilla: int = 0) -> np.ndarray:
    """Máscara de manchas alargadas (~30 % del área) de un campo aleatorio suavizado."""
    rng = np.random.default_rng(semilla)
    campo = ndimage.gaussian_filter(rng.normal(size=(lado, lado)), (3, 6))
    return campo > np.percentile(campo, 70)


def _dos_fases(lado: int = 160, degrade: float = 0.0, ruido: float = 0.005, semilla: int = 0):
    """Matriz 0.70 con manchas 0.55; devuelve (imagen, verdad: 1 = manchas)."""
    verdad = _manchas(lado, semilla)
    imagen = np.where(verdad, 0.55, 0.70)
    if degrade > 0:
        imagen = ndimage.gaussian_filter(imagen, degrade)
    imagen = imagen + np.random.default_rng(semilla + 1).normal(0, ruido, imagen.shape)
    return imagen, verdad


# --------------------------------------------------------------------
# realce
# --------------------------------------------------------------------


def test_estimar_degrade_recupera_sigma_de_un_borde_aislado():
    y, x = np.mgrid[0:120, 0:120]
    imagen = ndimage.gaussian_filter((x > 60).astype(float), 2.0)
    sigma, usados = estimar_degrade(imagen)
    assert usados >= 10
    assert sigma == pytest.approx(2.0, abs=0.3)


def test_estimar_degrade_sin_bordes_devuelve_valor_fijo():
    sigma, usados = estimar_degrade(np.full((80, 80), 0.5))
    assert (sigma, usados) == (1.0, 0)


def test_estimar_degrade_rechaza_mascara_de_otro_tamano():
    with pytest.raises(ValueError):
        estimar_degrade(np.zeros((50, 50)), mascara=np.ones((40, 40), bool))


def test_realzar_conserva_tamano_y_empina_el_borde():
    y, x = np.mgrid[0:60, 0:60]
    imagen = ndimage.gaussian_filter((x > 30).astype(float), 2.0)
    realzada = realzar(imagen, 2.0)
    assert realzada.shape == imagen.shape
    assert np.abs(np.diff(realzada[30])).max() > np.abs(np.diff(imagen[30])).max()


def test_filtro_choque_lleva_el_degrade_a_dos_mesetas():
    y, x = np.mgrid[0:60, 0:60]
    imagen = ndimage.gaussian_filter((x > 30).astype(float), 2.0)
    tras, pasos = filtro_choque(imagen, sigma_ruido=0.01)
    assert pasos >= 1
    # Lejos del salto, los píxeles quedan en el gris de su lado
    assert np.all(tras[:, :26] < 0.05) and np.all(tras[:, 35:] > 0.95)


def test_filtro_choque_no_cambia_una_imagen_nitida():
    imagen, _ = _dos_fases()
    tras, pasos = filtro_choque(imagen, sigma_ruido=0.005)
    assert pasos <= 3
    assert np.mean(np.abs(tras - imagen)) < 0.002


def test_filtro_choque_con_pozos_sin_intensidad_es_el_choque_solo():
    imagen, _ = _dos_fases(degrade=1.5)
    act = actividad(imagen)
    solo, pasos_solo = filtro_choque(imagen, 0.005)
    con, pasos, activacion = filtro_choque_con_pozos(imagen, act, 0.005, intensidad=0)
    assert activacion is None and pasos == pasos_solo
    assert np.array_equal(con, solo)


def test_filtro_choque_con_pozos_acerca_cada_fase_a_su_gris():
    imagen, verdad = _dos_fases(lado=200, degrade=1.5)
    act = actividad(imagen)
    solo, _ = filtro_choque(imagen, 0.005)
    con, pasos, activacion = filtro_choque_con_pozos(imagen, act, 0.005, intensidad=2)
    assert activacion is not None and pasos > activacion
    for gris, fase in ((0.55, verdad), (0.70, ~verdad)):
        assert np.median(np.abs(con[fase] - gris)) <= np.median(np.abs(solo[fase] - gris)) + 1e-4


# --------------------------------------------------------------------
# regiones
# --------------------------------------------------------------------


def test_cortes_por_valles_separa_dos_picos():
    rng = np.random.default_rng(0)
    valores = np.concatenate([rng.normal(0.3, 0.03, 5000), rng.normal(0.7, 0.03, 5000)])
    cortes = cortes_por_valles(valores)
    assert len(cortes) == 1 and 0.4 < cortes[0] < 0.6


def test_cortes_por_valles_un_solo_pico_no_corta():
    valores = np.random.default_rng(0).normal(0.5, 0.05, 10000)
    assert len(cortes_por_valles(valores)) == 0


@pytest.mark.xfail(strict=True, reason=(
    "Limitación conocida: si las fases tienen un valle claro en el histograma de gris, "
    "zonas_por_gris las separa como zonas y las clases de actividad se fragmentan. "
    "En las ROIs reales (sin valle entre fases) no pasa. Pendiente: criterios de las regiones."))
def test_regiones_de_analisis_separa_una_franja_oscura():
    imagen, _ = _dos_fases(lado=200)
    y, x = np.mgrid[0:200, 0:200]
    franja = np.abs(x - y) < 15
    imagen[franja] = 0.05
    regiones, descripcion, borde, _ = regiones_de_analisis(imagen)
    assert len(descripcion) >= 2
    # La franja oscura (lejos de su borde) es una sola región, distinta del resto
    interior = np.abs(x - y) < 8
    etiquetas_franja = np.unique(regiones[interior])
    assert len(etiquetas_franja) == 1 and etiquetas_franja[0] >= 0
    assert not np.isin(etiquetas_franja[0], regiones[np.abs(x - y) > 40])


def test_aislar_region_rellena_afuera_con_valores_de_la_region():
    imagen = np.zeros((50, 50))
    mascara = np.zeros((50, 50), bool)
    mascara[10:20, 10:20] = True
    imagen[mascara] = 1.0
    aislada, caja = aislar_region(imagen, mascara, margen=5)
    assert aislada.shape == imagen[caja].shape
    assert np.all(aislada == 1.0)


def test_aislar_region_rechaza_mascara_vacia():
    with pytest.raises(ValueError):
        aislar_region(np.zeros((10, 10)), np.zeros((10, 10), bool))


# --------------------------------------------------------------------
# poblaciones
# --------------------------------------------------------------------


def test_paisaje_cuenta_todos_los_pixeles():
    imagen, _ = _dos_fases()
    conteos, fila, columna, _ = paisaje(imagen, actividad(imagen))
    assert conteos.sum() == imagen.size
    assert fila.shape == columna.shape == (imagen.size,)


def test_normalizar_por_fila_deja_maximo_uno():
    imagen, _ = _dos_fases()
    conteos, *_ = paisaje(imagen, actividad(imagen))
    normalizado = normalizar_por_fila(conteos)
    con_datos = normalizado.max(axis=1) > 0
    assert np.allclose(normalizado.max(axis=1)[con_datos], 1.0)


def test_nucleos_y_pozos_cubren_todo_el_paisaje():
    imagen, _ = _dos_fases()
    conteos, *_ = paisaje(imagen, actividad(imagen))
    normalizado = normalizar_por_fila(conteos)
    nucleos, pozos = nucleos_y_pozos(normalizado, conteos)
    assert nucleos.max() == 2
    assert pozos.min() >= 1


def test_poblaciones_de_region_nitida_acierta_todo():
    imagen, verdad = _dos_fases()
    resultado = poblaciones_de_region(imagen.ravel(), actividad(imagen).ravel())
    assert len(resultado["centrales"]) == 2
    assert resultado["centrales"] == pytest.approx([0.55, 0.70], abs=0.01)
    acierto = np.mean(resultado["poblacion"].reshape(imagen.shape) == np.where(verdad, 0, 1))
    assert acierto > 0.99


def test_poblaciones_de_region_rechaza_tamanos_distintos():
    with pytest.raises(ValueError):
        poblaciones_de_region(np.zeros(10), np.zeros(9))


# --------------------------------------------------------------------
# cadena completa
# --------------------------------------------------------------------


def test_procesar_region_con_degrade_encuentra_las_dos_fases():
    imagen, verdad = _dos_fases(lado=300, degrade=1.5)
    resultado = procesar_region(imagen, np.ones(imagen.shape, bool))
    assert len(resultado["centrales"]) == 2
    acierto = np.mean(resultado["poblacion"].reshape(imagen.shape) == np.where(verdad, 0, 1))
    assert acierto > 0.9


@pytest.mark.xfail(strict=True, reason=(
    "Limitación conocida: con fases de gris bien separado, las regiones de análisis parten la "
    "imagen por fase y la cadena se fragmenta (ver test de regiones). Pendiente: criterios."))
def test_segmentar_roi_con_degrade_encuentra_las_dos_fases():
    imagen, verdad = _dos_fases(lado=200, degrade=1.5)
    s = segmentar_roi(imagen, suavizada=True)
    assert len(s["descripcion"]) == 1
    tabla = s["tabla"]
    assert len(tabla) == 2
    oscura = s["etiquetas"] == tabla[0]["etiqueta"]
    assert np.mean(oscura == verdad) > 0.9
    assert sum(f["fraccion_roi"] for f in tabla) == pytest.approx(1.0)


def test_segmentar_roi_rechaza_imagen_3d():
    with pytest.raises(ValueError):
        segmentar_roi(np.zeros((5, 5, 3)))
