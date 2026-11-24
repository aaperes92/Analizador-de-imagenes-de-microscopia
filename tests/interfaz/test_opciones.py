"""Tests de la validación de opciones de la interfaz (sin ventanas)."""

import pytest

from interfaz.opciones import (MODO_SEM_IMAGEN, MODO_SEM_ROIS, POR_DEFECTO,
                               validar_opciones)


@pytest.fixture
def carpeta_moke(tmp_path):
    (tmp_path / "background.png").write_bytes(b"")
    return tmp_path


def campos(tecnica, **cambios):
    return {**POR_DEFECTO[tecnica], **{k: str(v) for k, v in cambios.items()}}


def test_moke_valido(carpeta_moke):
    p = validar_opciones("MOKE", campos("MOKE", carpeta=carpeta_moke,
                                        fondo=carpeta_moke / "background.png",
                                        escala="0,5", unidad="um"))
    assert p["escala"] == 0.5 and p["unidad"] == "um"
    assert p["carpeta"] == carpeta_moke


def test_moke_por_defecto_escala_1_px(carpeta_moke):
    p = validar_opciones("MOKE", campos("MOKE", carpeta=carpeta_moke,
                                        fondo=carpeta_moke / "background.png"))
    assert (p["escala"], p["unidad"]) == (1.0, "px")


@pytest.mark.parametrize("cambios, mensaje", [
    ({}, "carpeta"),
    ({"carpeta": "/no/existe"}, "No existe"),
])
def test_moke_carpeta_invalida(cambios, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        validar_opciones("MOKE", campos("MOKE", **cambios))


@pytest.mark.parametrize("escala", ["", "abc", "0", "-1", "nan", "inf"])
def test_moke_escala_invalida(carpeta_moke, escala):
    with pytest.raises(ValueError, match="Escala"):
        validar_opciones("MOKE", campos("MOKE", carpeta=carpeta_moke,
                                        fondo=carpeta_moke / "background.png", escala=escala))


def test_moke_unidad_vacia(carpeta_moke):
    with pytest.raises(ValueError, match="unidad"):
        validar_opciones("MOKE", campos("MOKE", carpeta=carpeta_moke,
                                        fondo=carpeta_moke / "background.png", unidad=" "))


def test_sem_imagen(tmp_path):
    imagen = tmp_path / "BSE_1.tif"
    imagen.write_bytes(b"")
    p = validar_opciones("SEM", campos("SEM", imagen=imagen))
    assert p == {"modo": MODO_SEM_IMAGEN, "imagen": imagen, "sufijo": None,
                 "intensidad_pozos": 2.0}
    p = validar_opciones("SEM", campos("SEM", imagen=imagen, sufijo="_3", intensidad_pozos="0"))
    assert p["sufijo"] == "_3" and p["intensidad_pozos"] == 0.0


def test_sem_imagen_no_tif(tmp_path):
    imagen = tmp_path / "BSE_1.png"
    imagen.write_bytes(b"")
    with pytest.raises(ValueError, match=".tif"):
        validar_opciones("SEM", campos("SEM", imagen=imagen))


def test_sem_sufijo_con_barra(tmp_path):
    imagen = tmp_path / "BSE_1.tif"
    imagen.write_bytes(b"")
    with pytest.raises(ValueError, match="sufijo"):
        validar_opciones("SEM", campos("SEM", imagen=imagen, sufijo="a/b"))


def test_sem_rois(tmp_path):
    p = validar_opciones("SEM", campos("SEM", modo=MODO_SEM_ROIS, carpeta=tmp_path))
    assert p == {"modo": MODO_SEM_ROIS, "carpeta": tmp_path, "intensidad_pozos": 2.0}


def test_sem_intensidad_negativa(tmp_path):
    with pytest.raises(ValueError, match="pozos"):
        validar_opciones("SEM", campos("SEM", modo=MODO_SEM_ROIS, carpeta=tmp_path,
                                       intensidad_pozos="-1"))


def test_tem_por_defecto(tmp_path):
    p = validar_opciones("TEM", campos("TEM", ruta=tmp_path))
    assert p == {"ruta": tmp_path, "filtro": "auto", "umbral": "auto", "fases": None,
                 "fondo": None, "difraccion": None, "ancho_instrumental": 0.0}


def test_tem_opciones(tmp_path):
    imagen = tmp_path / "a.dm3"
    imagen.write_bytes(b"")
    p = validar_opciones("TEM", campos("TEM", ruta=imagen, filtro="median", umbral="gmm",
                                       fases="1, 2", fondo="0", ancho_instrumental="1.5",
                                       difraccion="Analizar siempre como difracción"))
    assert p["fases"] == [1, 2] and p["fondo"] == 0
    assert p["difraccion"] is True and p["ancho_instrumental"] == 1.5
    p = validar_opciones("TEM", campos("TEM", ruta=imagen, difraccion="Nunca"))
    assert p["difraccion"] is False


@pytest.mark.parametrize("cambios, mensaje", [
    ({"fases": "1,a"}, "Fases"),
    ({"fases": "-1"}, "Fases"),
    ({"fondo": "1.5"}, "Fondo"),
    ({"filtro": "otro"}, "Filtro"),
    ({"umbral": "otro"}, "Umbral"),
    ({"ancho_instrumental": "-2"}, "Ancho"),
])
def test_tem_invalidos(tmp_path, cambios, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        validar_opciones("TEM", campos("TEM", ruta=tmp_path, **cambios))


def test_tem_formato_no_soportado(tmp_path):
    archivo = tmp_path / "a.txt"
    archivo.write_text("x")
    with pytest.raises(ValueError, match="Formato"):
        validar_opciones("TEM", campos("TEM", ruta=archivo))


def test_tecnica_desconocida():
    with pytest.raises(ValueError, match="Técnica"):
        validar_opciones("XRD", {})
