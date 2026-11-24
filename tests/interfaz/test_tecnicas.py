"""Tests de la conexión de cada técnica con la interfaz (sin ventanas)."""

import numpy as np
import pytest
from skimage import io

from interfaz import tecnicas
from tem.analisis_completo import analizar_ruta
from tem.diffraction import read_camera_metadata


def guardar(ruta, imagen):
    io.imsave(ruta, (np.clip(imagen, 0, 1) * 65535).astype(np.uint16), check_contrast=False)


@pytest.fixture
def carpeta_moke(tmp_path):
    """Tres cuadros MOKE sintéticos con un dominio que crece y su fondo."""
    rng = np.random.default_rng(0)
    f, c = np.mgrid[:600, :800]
    apertura = ((f - 300) / 280) ** 2 + ((c - 400) / 380) ** 2 < 1
    fondo = np.where(apertura, 0.6 + 0.2 * np.exp(-((f - 280) ** 2 + (c - 400) ** 2) / 2e5), 0.03)
    carpeta = tmp_path / "moke"
    carpeta.mkdir()
    guardar(carpeta / "background.png", fondo + rng.normal(0, 0.01, fondo.shape))
    for i, radio in enumerate([60, 100, 140]):
        dominio = (np.hypot(f - 300, c - 420) < radio) & apertura
        guardar(carpeta / f"{i}.png",
                fondo * np.where(dominio, 0.9, 1.0) + rng.normal(0, 0.02, fondo.shape))
    return carpeta


def test_correr_moke(carpeta_moke, tmp_path, monkeypatch, capsys):
    salida = tmp_path / "resultados"
    monkeypatch.setattr(tecnicas, "SALIDA_MOKE", salida)
    resultado = tecnicas.correr_moke({"carpeta": carpeta_moke,
                                      "fondo": carpeta_moke / "background.png",
                                      "escala": 75.0, "unidad": "nm"})
    assert resultado.salida == salida and resultado.figuras == []  # carpeta: no abre figuras
    texto = capsys.readouterr().out
    assert "[1/3] 0.png" in texto and "[3/3] 2.png" in texto
    tabla = (salida / "medidas_moke.csv").read_text(encoding="utf-8").splitlines()
    assert tabla[0].split(",")[4] == "radio_nm" and len(tabla) == 4
    radios = [float(fila.split(",")[3]) for fila in tabla[1:]]
    assert radios == pytest.approx([60, 100, 140], abs=2)
    assert sorted(p.name for p in salida.glob("*.png")) == ["dominio_0.png", "dominio_1.png",
                                                             "dominio_2.png"]


def test_buscar_rois_en_subcarpetas(tmp_path):
    (tmp_path / "A" / "A_1").mkdir(parents=True)
    (tmp_path / "B").mkdir()
    for ruta in ["A/A_1/roi_A_1.npy", "B/roi_B.npy", "A/A_1/segmentacion_roi_A_1.npy",
                 "B/roi_B.tif"]:
        (tmp_path / ruta).write_bytes(b"")
    assert [p.relative_to(tmp_path).as_posix() for p in tecnicas.buscar_rois(tmp_path)] == \
        ["A/A_1/roi_A_1.npy", "B/roi_B.npy"]


def test_correr_sem_rois_sin_rois(tmp_path):
    with pytest.raises(ValueError, match="No hay ROIs"):
        tecnicas.correr_sem_rois(tmp_path, 2.0)


def test_correr_sem_rois_sigue_si_una_falla(tmp_path, capsys):
    """Una ROI mala se informa y no frena a las demás; la buena deja sus tres archivos."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[:120, :160]
    roi = np.where((y - 60) ** 2 + (x - 80) ** 2 < 35 ** 2, 0.7, 0.3)
    roi = np.clip(roi + rng.normal(0, 0.02, roi.shape), 0, 1).astype(np.float32)
    (tmp_path / "bien").mkdir()
    (tmp_path / "mal").mkdir()
    np.save(tmp_path / "bien" / "roi_bien.npy", roi)
    (tmp_path / "mal" / "roi_mal.npy").write_bytes(b"no es un npy")
    resultado = tecnicas.correr_sem_rois(tmp_path, 2.0)
    texto = capsys.readouterr().out
    assert "[1/2]" in texto and "[2/2]" in texto and "ERROR" in texto
    assert resultado.salida == tmp_path and resultado.figuras == []
    for extension in ("npy", "csv", "png"):
        assert (tmp_path / "bien" / f"segmentacion_roi_bien.{extension}").is_file()


def test_tem_carpeta_vacia_y_ruta_inexistente(tmp_path):
    with pytest.raises(ValueError, match="No hay imágenes"):
        analizar_ruta(tmp_path, mostrar=False)
    with pytest.raises(FileNotFoundError):
        analizar_ruta(tmp_path / "no_existe", mostrar=False)


def test_read_camera_metadata_pide_con_funcion(tmp_path, capsys):
    """Con pedir_dato, los metadatos que faltan se piden con esa función (no por teclado)."""
    archivo = tmp_path / "patron.png"
    guardar(archivo, np.random.default_rng(0).random((64, 64)))
    pedidos = []

    def pedir(texto, por_defecto):
        pedidos.append(texto)
        return {"Voltaje": 300.0, "Longitud": 500.0}.get(texto.split()[0], por_defecto)

    meta = read_camera_metadata(archivo, interactive=True, pedir_dato=pedir)
    assert len(pedidos) == 3
    assert meta["voltage_kV"] == 300.0 and meta["camera_length_mm"] == 500.0
    assert meta["fuente"] == "usuario"
    # Sin interactive, la función no se usa y quedan los valores por defecto
    meta = read_camera_metadata(archivo, interactive=False, pedir_dato=pedir)
    assert len(pedidos) == 3 and meta["fuente"] == "por_defecto"
