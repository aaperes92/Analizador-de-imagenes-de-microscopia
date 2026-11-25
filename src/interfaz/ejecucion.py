"""Ejecución del análisis en segundo plano.

El análisis corre en un hilo aparte para que la ventana no se congele. Todo lo que imprime
(print), los warnings y los errores se mandan por una cola; la ventana la vacía desde el hilo
principal, que es el único que toca Tkinter. Si el análisis necesita un dato del usuario (por
ejemplo, un metadato que falta en un patrón de difracción TEM), lo pide por la misma cola y
espera la respuesta.

Autor: Ángel A. Pérez
"""

import contextlib
import queue
import threading
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Tipos de mensaje de la cola
TEXTO = "texto"          # (TEXTO, estilo, texto)
PREGUNTA = "pregunta"    # (PREGUNTA, Pregunta)
FIN = "fin"              # (FIN, Resultado)

# Estilos del texto en el registro
NORMAL = "normal"
AVISO = "aviso"
ERROR = "error"
OK = "ok"


@dataclass
class Pregunta:
    """Un dato numérico que el análisis le pide al usuario desde el hilo de fondo.

    El hilo principal muestra el diálogo, guarda la respuesta en `respuesta` y avisa con `listo`.
    """

    texto: str
    por_defecto: float
    respuesta: float | None = None
    listo: threading.Event = field(default_factory=threading.Event)


@dataclass
class Resultado:
    """Lo que devuelve una tarea al terminar.

    Attributes:
        salida: Carpeta donde quedaron los resultados (se informa en el registro).
        figuras: Figuras que hay que abrir en ventanas aparte (solo con una imagen).
        error: Mensaje si la tarea terminó con una excepción; None si terminó bien.
    """

    salida: Path | None = None
    figuras: list[Path] = field(default_factory=list)
    error: str | None = None


class EscritorCola:
    """Objeto tipo archivo que manda cada `write` a la cola con un estilo fijo."""

    def __init__(self, cola: queue.Queue, estilo: str = NORMAL) -> None:
        self.cola = cola
        self.estilo = estilo

    def write(self, texto: str) -> int:
        if texto:
            self.cola.put((TEXTO, self.estilo, texto))
        return len(texto)

    def flush(self) -> None:
        pass

    def reconfigure(self, **_kwargs) -> None:
        """Los scripts SEM llaman a sys.stdout.reconfigure; acá no hace falta nada."""

    def isatty(self) -> bool:
        return False


def pedir_dato(cola: queue.Queue, texto: str, por_defecto: float) -> float:
    """Pide un número al usuario desde el hilo de fondo y espera la respuesta.

    Args:
        cola: Cola compartida con la ventana.
        texto: Qué se pide (por ejemplo 'Voltaje de aceleración (kV)').
        por_defecto: Valor propuesto en el diálogo.

    Returns:
        El número que cargó el usuario, o `por_defecto` si cerró el diálogo sin responder.
    """
    pregunta = Pregunta(texto, por_defecto)
    cola.put((PREGUNTA, pregunta))
    pregunta.listo.wait()
    return por_defecto if pregunta.respuesta is None else pregunta.respuesta


def correr_capturando(tarea: Callable[[], Resultado], cola: queue.Queue) -> None:
    """Corre una tarea mandando a la cola sus print, warnings y errores, y al final su resultado.

    Pensada para correr en el hilo de fondo. Nunca lanza excepciones: si la tarea falla, el
    error y su traza van al registro y el resultado lleva el mensaje en `error`.

    Args:
        tarea: Función sin argumentos que hace el análisis y devuelve un Resultado.
        cola: Cola compartida con la ventana.
    """
    salida = EscritorCola(cola, NORMAL)
    errores = EscritorCola(cola, AVISO)

    def mostrar_warning(mensaje, categoria, archivo, linea, file=None, line=None) -> None:
        cola.put((TEXTO, AVISO, f"⚠ {categoria.__name__}: {mensaje}\n"))

    resultado = Resultado()
    with contextlib.redirect_stdout(salida), contextlib.redirect_stderr(errores), \
            warnings.catch_warnings():
        warnings.showwarning = mostrar_warning
        try:
            resultado = tarea() or Resultado()
        except Exception as error:  # el error se informa y la interfaz sigue abierta
            cola.put((TEXTO, ERROR, f"✖ Error: {error}\n"))
            cola.put((TEXTO, AVISO, traceback.format_exc()))
            resultado = Resultado(error=str(error))
    cola.put((FIN, resultado))


def iniciar_en_segundo_plano(tarea: Callable[[], Resultado], cola: queue.Queue) -> threading.Thread:
    """Arranca `correr_capturando(tarea, cola)` en un hilo aparte y lo devuelve."""
    hilo = threading.Thread(target=correr_capturando, args=(tarea, cola), daemon=True)
    hilo.start()
    return hilo


def pngs_nuevos(carpeta: Path, desde: float, recursivo: bool = False) -> list[Path]:
    """PNG de `carpeta` creados o modificados después del instante `desde` (time.time()).

    Sirve para saber qué figuras guardó una corrida, sin cambiar los módulos.
    """
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return []
    patron = carpeta.rglob("*.png") if recursivo else carpeta.glob("*.png")
    return sorted((p for p in patron if p.stat().st_mtime >= desde),
                  key=lambda p: p.stat().st_mtime)
