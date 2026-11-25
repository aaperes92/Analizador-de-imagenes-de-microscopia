"""Ventanas auxiliares de la interfaz: figuras guardadas y diálogo para pedir un dato.

Las figuras se abren desde el PNG que guardó cada módulo, con una Figure de matplotlib
incrustada en Tkinter (sin pyplot). Así funciona aunque el análisis haya corrido en otro hilo
con el backend 'Agg'.

Autor: Ángel A. Pérez
"""

import gc
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def abrir_figura(padre: tk.Misc, ruta: Path) -> tk.Toplevel:
    """Abre un PNG guardado en una ventana aparte, con la barra de zoom de matplotlib.

    Args:
        padre: Ventana principal.
        ruta: PNG a mostrar.

    Returns:
        La ventana creada.
    """
    imagen = mpimg.imread(ruta)
    alto, ancho = imagen.shape[:2]
    escala = min(1.0, 1100 / ancho, 750 / alto)  # que entre en una pantalla común

    ventana = tk.Toplevel(padre)
    ventana.title(Path(ruta).name)
    figura = Figure(figsize=(ancho * escala / 100, alto * escala / 100), dpi=100)
    eje = figura.add_axes((0, 0, 1, 1))
    eje.imshow(imagen)
    eje.set_axis_off()

    lienzo = FigureCanvasTkAgg(figura, master=ventana)
    barra = NavigationToolbar2Tk(lienzo, ventana, pack_toolbar=False)
    barra.update()
    barra.pack(side=tk.BOTTOM, fill=tk.X)
    lienzo.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    lienzo.draw()

    def cerrar() -> None:
        # Liberar los objetos Tk ya, en el hilo principal (si los liberara el recolector desde
        # el hilo de fondo, Tkinter daría RuntimeError)
        ventana.destroy()
        gc.collect()

    ventana.protocol("WM_DELETE_WINDOW", cerrar)
    return ventana


def pedir_numero(padre: tk.Misc, titulo: str, texto: str, por_defecto: float,
                 nota: str = "") -> tuple[float | None, bool]:
    """Diálogo modal que pide un número, con la opción de repetir la respuesta.

    Args:
        padre: Ventana principal.
        titulo: Título del diálogo.
        texto: Qué se pide.
        por_defecto: Valor inicial del campo.
        nota: Texto aclaratorio debajo (por ejemplo, el archivo que se analiza).

    Returns:
        (número o None si se canceló, True si marcó "usar para las demás imágenes").
    """
    dialogo = tk.Toplevel(padre)
    dialogo.title(titulo)
    dialogo.transient(padre)
    dialogo.resizable(False, False)
    marco = ttk.Frame(dialogo, padding=12)
    marco.pack(fill=tk.BOTH, expand=True)

    ttk.Label(marco, text=texto).grid(row=0, column=0, columnspan=2, sticky="w")
    if nota:
        ttk.Label(marco, text=nota, foreground="gray").grid(row=1, column=0, columnspan=2,
                                                            sticky="w")
    valor = tk.StringVar(value=f"{por_defecto:g}")
    entrada = ttk.Entry(marco, textvariable=valor, width=14)
    entrada.grid(row=2, column=0, columnspan=2, sticky="w", pady=6)
    repetir = tk.BooleanVar(value=False)
    ttk.Checkbutton(marco, text="Usar este valor para las demás imágenes",
                    variable=repetir).grid(row=3, column=0, columnspan=2, sticky="w")
    error = ttk.Label(marco, text="", foreground="red")
    error.grid(row=4, column=0, columnspan=2, sticky="w")

    respuesta: list[float | None] = [None]

    def aceptar(_evento=None) -> None:
        try:
            numero = float(valor.get().strip().replace(",", "."))
        except ValueError:
            error.config(text="Tiene que ser un número.")
            return
        if not numero > 0 or numero == float("inf"):
            error.config(text="Tiene que ser mayor que 0.")
            return
        respuesta[0] = numero
        dialogo.destroy()

    botones = ttk.Frame(marco)
    botones.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))
    ttk.Button(botones, text="Aceptar", command=aceptar).pack(side=tk.LEFT, padx=4)
    ttk.Button(botones, text="Usar el valor propuesto",
               command=dialogo.destroy).pack(side=tk.LEFT)
    dialogo.bind("<Return>", aceptar)
    dialogo.bind("<Escape>", lambda _e: dialogo.destroy())

    entrada.focus_set()
    entrada.select_range(0, tk.END)
    dialogo.wait_visibility()
    dialogo.grab_set()
    padre.wait_window(dialogo)
    return respuesta[0], repetir.get()
