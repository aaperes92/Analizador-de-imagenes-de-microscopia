"""Tests de la ventana (se saltean si no hay pantalla)."""

import tkinter as tk

import pytest

from interfaz import Ventana


@pytest.fixture
def ventana():
    try:
        raiz = tk.Tk()
    except tk.TclError:
        pytest.skip("No hay pantalla para abrir ventanas")
    raiz.withdraw()
    v = Ventana(raiz)
    yield v
    raiz.destroy()


def test_muestra_solo_el_panel_de_la_tecnica(ventana):
    for tecnica in ("MOKE", "SEM", "TEM"):
        ventana.tecnica.set(tecnica)
        ventana._mostrar_panel()
        visibles = [t for t, panel in ventana.paneles.items() if panel.winfo_manager()]
        assert visibles == [tecnica]


def test_valores_por_defecto(ventana):
    assert ventana.campos("MOKE")["escala"] == "1"
    assert ventana.campos("MOKE")["unidad"] == "px"
    assert ventana.campos("SEM")["intensidad_pozos"] == "2"
    assert ventana.campos("TEM")["filtro"] == "auto"
    assert ventana.campos("TEM")["ancho_instrumental"] == "0"


def test_modos_sem(ventana):
    sem = ventana.variables["SEM"]
    sem["modo"].set("ROIs guardadas")
    ventana._mostrar_modo_sem()
    assert ventana.sem_rois.winfo_manager() and not ventana.sem_imagen.winfo_manager()
    sem["modo"].set("Imagen y ROI")
    ventana._mostrar_modo_sem()
    assert ventana.sem_imagen.winfo_manager() and not ventana.sem_rois.winfo_manager()


def test_opcion_invalida_va_al_registro_y_no_analiza(ventana):
    ventana.analizar()  # MOKE sin carpeta
    assert "Elegí la carpeta" in ventana.registro.get("1.0", tk.END)
    assert not ventana.ocupado
    assert "disabled" not in ventana.boton_analizar.state()
