"""Tests de la ejecución en segundo plano (captura de mensajes, errores y preguntas)."""

import queue
import threading
import time
import warnings

from interfaz.ejecucion import (AVISO, ERROR, FIN, PREGUNTA, TEXTO, Resultado,
                                correr_capturando, iniciar_en_segundo_plano, pedir_dato,
                                pngs_nuevos)


def vaciar(cola):
    mensajes = []
    while not cola.empty():
        mensajes.append(cola.get())
    return mensajes


def test_captura_print_y_warning():
    cola = queue.Queue()

    def tarea():
        print("hola")
        warnings.warn("cuidado")
        return Resultado(salida="carpeta")

    correr_capturando(tarea, cola)
    mensajes = vaciar(cola)
    textos = "".join(m[2] for m in mensajes if m[0] == TEXTO)
    assert "hola" in textos
    assert any(m[0] == TEXTO and m[1] == AVISO and "cuidado" in m[2] for m in mensajes)
    assert mensajes[-1] == (FIN, Resultado(salida="carpeta"))


def test_error_no_se_propaga():
    cola = queue.Queue()

    def tarea():
        raise ValueError("imagen mala")

    correr_capturando(tarea, cola)
    mensajes = vaciar(cola)
    assert any(m[0] == TEXTO and m[1] == ERROR and "imagen mala" in m[2] for m in mensajes)
    assert mensajes[-1][0] == FIN and mensajes[-1][1].error == "imagen mala"


def test_pedir_dato_desde_otro_hilo():
    cola = queue.Queue()
    hilo = iniciar_en_segundo_plano(
        lambda: Resultado(salida=pedir_dato(cola, "Voltaje (kV)", 200.0)), cola)
    tipo, pregunta = cola.get(timeout=5)
    assert tipo == PREGUNTA and pregunta.por_defecto == 200.0
    pregunta.respuesta = 300.0
    pregunta.listo.set()
    hilo.join(timeout=5)
    fin = [m for m in vaciar(cola) if m[0] == FIN][0]
    assert fin[1].salida == 300.0


def test_pedir_dato_sin_respuesta_usa_por_defecto():
    cola = queue.Queue()
    resultado = {}
    hilo = threading.Thread(target=lambda: resultado.update(v=pedir_dato(cola, "x", 14.0)))
    hilo.start()
    _, pregunta = cola.get(timeout=5)
    pregunta.listo.set()
    hilo.join(timeout=5)
    assert resultado["v"] == 14.0


def test_pngs_nuevos(tmp_path):
    viejo = tmp_path / "viejo.png"
    viejo.write_bytes(b"")
    desde = time.time() + 0.01
    time.sleep(0.05)
    nuevo = tmp_path / "nuevo.png"
    nuevo.write_bytes(b"")
    (tmp_path / "tabla.csv").write_text("x")
    assert pngs_nuevos(tmp_path, desde) == [nuevo]
    assert pngs_nuevos(tmp_path / "no_existe", desde) == []
