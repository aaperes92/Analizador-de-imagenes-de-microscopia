"""Ventana principal del Analizador de imágenes de microscopía.

Se elige la técnica (MOKE, SEM o TEM) y la ventana muestra solo sus opciones. "Analizar" corre
el análisis en segundo plano; los mensajes, el avance, los errores y la carpeta de resultados
aparecen en el registro. Con una sola imagen, las figuras guardadas se abren en ventanas
aparte.

Autor: Ángel A. Pérez
"""

import contextlib
import gc
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib

from . import tecnicas
from .ejecucion import (AVISO, ERROR, FIN, NORMAL, OK, PREGUNTA, TEXTO, EscritorCola,
                        Resultado, iniciar_en_segundo_plano, pedir_dato)
from .figuras import abrir_figura, pedir_numero
from .opciones import (DIFRACCION_TEM, FILTROS_TEM, FORMATOS_TEM, MODO_SEM_IMAGEN,
                       MODO_SEM_ROIS, MODOS_SEM, POR_DEFECTO, TECNICAS, UMBRALES_TEM,
                       validar_opciones)

# Cada cuánto se vacía la cola de mensajes (ms)
INTERVALO_COLA_MS = 100

COLORES_REGISTRO = {NORMAL: "black", AVISO: "#a15c00", ERROR: "#c62828", OK: "#2e7d32"}


class Ventana(ttk.Frame):
    """Ventana principal: técnica, entrada, opciones, botón Analizar y registro."""

    def __init__(self, raiz: tk.Tk) -> None:
        super().__init__(raiz, padding=10)
        self.raiz = raiz
        self.cola: queue.Queue = queue.Queue()
        self.ocupado = False
        self.respuestas_fijas: dict[str, float] = {}  # "usar para las demás imágenes"
        self.controles: list[ttk.Widget] = []         # se deshabilitan durante el análisis
        self.variables: dict[str, dict[str, tk.StringVar]] = {
            tecnica: {clave: tk.StringVar(value=valor) for clave, valor in campos.items()}
            for tecnica, campos in POR_DEFECTO.items()
        }
        self.tecnica = tk.StringVar(value=TECNICAS[0])

        raiz.title("Analizador de imágenes de microscopía")
        raiz.minsize(720, 560)
        self.pack(fill=tk.BOTH, expand=True)
        self._armar_selector_tecnica()
        self.paneles = {
            "MOKE": self._armar_moke(),
            "SEM": self._armar_sem(),
            "TEM": self._armar_tem(),
        }
        self._armar_boton_y_registro()
        self._mostrar_panel()
        self.after(INTERVALO_COLA_MS, self._revisar_cola)

    # ------------------------------------------------------------------ armado de la ventana

    def _armar_selector_tecnica(self) -> None:
        marco = ttk.Frame(self)
        marco.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(marco, text="Técnica:", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        for tecnica in TECNICAS:
            boton = ttk.Radiobutton(marco, text=tecnica, value=tecnica, variable=self.tecnica,
                                    command=self._mostrar_panel)
            boton.pack(side=tk.LEFT, padx=8)
            self.controles.append(boton)

    def _fila_ruta(self, padre: ttk.Frame, fila: int, etiqueta: str, variable: tk.StringVar,
                   botones: list[tuple[str, callable]]) -> None:
        """Una fila 'etiqueta [ruta] [botón…]'."""
        ttk.Label(padre, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=2)
        entrada = ttk.Entry(padre, textvariable=variable)
        entrada.grid(row=fila, column=1, sticky="ew", padx=6, pady=2)
        self.controles.append(entrada)
        for i, (texto, comando) in enumerate(botones):
            boton = ttk.Button(padre, text=texto, command=comando)
            boton.grid(row=fila, column=2 + i, sticky="ew", pady=2, padx=(0, 4))
            self.controles.append(boton)
        padre.columnconfigure(1, weight=1)

    def _campo(self, padre: ttk.Frame, fila: int, columna: int, etiqueta: str,
               variable: tk.StringVar, ancho: int = 10, ayuda: str = "") -> None:
        """Una fila 'etiqueta [campo] ayuda'."""
        ttk.Label(padre, text=etiqueta).grid(row=fila, column=columna, sticky="w", pady=2)
        entrada = ttk.Entry(padre, textvariable=variable, width=ancho)
        entrada.grid(row=fila, column=columna + 1, sticky="w", padx=6, pady=2)
        self.controles.append(entrada)
        if ayuda:
            ttk.Label(padre, text=ayuda, foreground="gray").grid(row=fila, column=columna + 2,
                                                                 sticky="w")

    def _lista(self, padre: ttk.Frame, fila: int, etiqueta: str, variable: tk.StringVar,
               valores: tuple[str, ...]) -> None:
        """Una fila 'etiqueta [lista desplegable]'."""
        ttk.Label(padre, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=2)
        lista = ttk.Combobox(padre, textvariable=variable, values=valores, state="readonly",
                             width=12)
        lista.grid(row=fila, column=1, sticky="w", padx=6, pady=2)
        self.controles.append(lista)

    def _armar_moke(self) -> ttk.Frame:
        v = self.variables["MOKE"]
        panel = ttk.Frame(self)
        entrada = ttk.LabelFrame(panel, text="Entrada", padding=8)
        entrada.pack(fill=tk.X)
        self._fila_ruta(entrada, 0, "Carpeta (0.png, 1.png, …):", v["carpeta"],
                        [("Elegir carpeta…", lambda: self._elegir_carpeta(v["carpeta"]))])
        self._fila_ruta(entrada, 1, "Fondo:", v["fondo"],
                        [("Elegir archivo…", self._elegir_fondo_moke)])
        opciones = ttk.LabelFrame(panel, text="Opciones MOKE", padding=8)
        opciones.pack(fill=tk.X, pady=(8, 0))
        self._campo(opciones, 0, 0, "Escala:", v["escala"],
                    ayuda="tamaño físico de un píxel")
        self._campo(opciones, 1, 0, "Unidad:", v["unidad"], ayuda="por ejemplo px, nm o um")
        ttk.Label(opciones, text=f"Resultados en: {tecnicas.SALIDA_MOKE}",
                  foreground="gray").grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        return panel

    def _armar_sem(self) -> ttk.Frame:
        v = self.variables["SEM"]
        panel = ttk.Frame(self)
        ttk.Label(panel, text="Solo imágenes BSE (electrones retrodispersados): el análisis "
                              "de imágenes SE todavía no está implementado.",
                  foreground="#a15c00").pack(anchor="w", pady=(0, 6))

        modos = ttk.Frame(panel)
        modos.pack(fill=tk.X)
        ttk.Label(modos, text="Modo:").pack(side=tk.LEFT)
        for modo in MODOS_SEM:
            boton = ttk.Radiobutton(modos, text=modo, value=modo, variable=v["modo"],
                                    command=self._mostrar_modo_sem)
            boton.pack(side=tk.LEFT, padx=8)
            self.controles.append(boton)

        self.sem_imagen = ttk.LabelFrame(panel, text="Entrada: imagen y ROI", padding=8)
        self._fila_ruta(self.sem_imagen, 0, "Imagen BSE (.tif):", v["imagen"],
                        [("Elegir archivo…", self._elegir_imagen_sem)])
        self._campo(self.sem_imagen, 1, 0, "Sufijo de la ROI:", v["sufijo"],
                    ayuda='opcional, ej. "_1"; vacío = el primer _N libre')
        ttk.Label(self.sem_imagen, text="Al analizar se abre el selector: arrastrá el "
                                        "rectángulo y presioná Enter o cerrá la ventana.",
                  foreground="gray").grid(row=2, column=0, columnspan=3, sticky="w")

        self.sem_rois = ttk.LabelFrame(panel, text="Entrada: ROIs guardadas", padding=8)
        self._fila_ruta(self.sem_rois, 0, "Carpeta:", v["carpeta"],
                        [("Elegir carpeta…", lambda: self._elegir_carpeta(
                            v["carpeta"], inicial=tecnicas.SALIDA_SEM))])
        ttk.Label(self.sem_rois, text="Se segmentan todas las roi_*.npy de la carpeta y sus "
                                      "subcarpetas.",
                  foreground="gray").grid(row=1, column=0, columnspan=3, sticky="w")

        self.sem_opciones = ttk.LabelFrame(panel, text="Opciones SEM", padding=8)
        self._campo(self.sem_opciones, 0, 0, "Intensidad de pozos:", v["intensidad_pozos"],
                    ayuda="0 = choque solo")
        ttk.Label(self.sem_opciones, text="Las ROIs se guardan en "
                                          f"{tecnicas.SALIDA_SEM}; la segmentación, junto a "
                                          "cada ROI.",
                  foreground="gray").grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._mostrar_modo_sem()
        return panel

    def _armar_tem(self) -> ttk.Frame:
        v = self.variables["TEM"]
        panel = ttk.Frame(self)
        entrada = ttk.LabelFrame(panel, text="Entrada", padding=8)
        entrada.pack(fill=tk.X)
        self._fila_ruta(entrada, 0, "Imagen o carpeta:", v["ruta"],
                        [("Elegir imagen…", self._elegir_imagen_tem),
                         ("Elegir carpeta…", lambda: self._elegir_carpeta(v["ruta"]))])
        ttk.Label(entrada, text=f"Formatos: {', '.join(FORMATOS_TEM)}",
                  foreground="gray").grid(row=1, column=0, columnspan=4, sticky="w")

        opciones = ttk.LabelFrame(panel, text="Opciones TEM", padding=8)
        opciones.pack(fill=tk.X, pady=(8, 0))
        self._lista(opciones, 0, "Filtro (Etapa 1):", v["filtro"], FILTROS_TEM)
        self._lista(opciones, 1, "Umbral (Etapa 2):", v["umbral"], UMBRALES_TEM)
        self._campo(opciones, 2, 0, "Fases (Etapa 3):", v["fases"], ancho=14,
                    ayuda="vacío = todas menos el fondo; o una lista, ej. 1,2")
        self._campo(opciones, 3, 0, "Fondo (Etapas 3 y 4):", v["fondo"], ancho=14,
                    ayuda="vacío = fase mayoritaria; o una etiqueta, ej. 0")

        ttk.Label(opciones, text="Difracción (Etapa 4.5):").grid(row=4, column=0, sticky="nw",
                                                                 pady=2)
        radios = ttk.Frame(opciones)
        radios.grid(row=4, column=1, columnspan=2, sticky="w", padx=6)
        for texto in DIFRACCION_TEM:
            boton = ttk.Radiobutton(radios, text=texto, value=texto, variable=v["difraccion"])
            boton.pack(anchor="w")
            self.controles.append(boton)
        self._campo(opciones, 5, 0, "Ancho instrumental (px):", v["ancho_instrumental"],
                    ayuda="se resta en Scherrer; 0 = no se resta")
        ttk.Label(opciones, text="Si a un patrón de difracción le faltan voltaje, longitud de "
                                 "cámara o píxel del detector, se piden en un diálogo.",
                  foreground="gray").grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))
        return panel

    def _armar_boton_y_registro(self) -> None:
        self.marco_paneles_fin = ttk.Frame(self)  # separa los paneles del botón
        self.marco_paneles_fin.pack(fill=tk.X)
        botones = ttk.Frame(self)
        botones.pack(fill=tk.X, pady=8)
        self.boton_analizar = ttk.Button(botones, text="Analizar", command=self.analizar)
        self.boton_analizar.pack(side=tk.RIGHT)
        self.controles.append(self.boton_analizar)
        self.estado = ttk.Label(botones, text="", foreground="gray")
        self.estado.pack(side=tk.RIGHT, padx=10)
        limpiar = ttk.Button(botones, text="Limpiar registro", command=self._limpiar_registro)
        limpiar.pack(side=tk.LEFT)

        marco = ttk.LabelFrame(self, text="Registro", padding=4)
        marco.pack(fill=tk.BOTH, expand=True)
        self.registro = ScrolledText(marco, height=14, wrap=tk.WORD, state=tk.DISABLED,
                                     font=("TkFixedFont", 9))
        self.registro.pack(fill=tk.BOTH, expand=True)
        for estilo, color in COLORES_REGISTRO.items():
            self.registro.tag_configure(estilo, foreground=color)

    # ------------------------------------------------------------------ paneles

    def _mostrar_panel(self) -> None:
        """Muestra solo el panel de la técnica elegida."""
        for panel in self.paneles.values():
            panel.pack_forget()
        self.paneles[self.tecnica.get()].pack(fill=tk.X, before=self.marco_paneles_fin)

    def _mostrar_modo_sem(self) -> None:
        """Muestra la entrada del modo SEM elegido."""
        for marco in (self.sem_imagen, self.sem_rois, self.sem_opciones):
            marco.pack_forget()
        if self.variables["SEM"]["modo"].get() == MODO_SEM_IMAGEN:
            self.sem_imagen.pack(fill=tk.X, pady=(6, 0))
        else:
            self.sem_rois.pack(fill=tk.X, pady=(6, 0))
        self.sem_opciones.pack(fill=tk.X, pady=(8, 0))

    # ------------------------------------------------------------------ diálogos de archivo

    @staticmethod
    def _carpeta_inicial(texto: str, por_defecto: Path | None = None) -> str:
        ruta = Path(texto.strip()).expanduser() if texto.strip() else None
        if ruta is not None:
            if ruta.is_dir():
                return str(ruta)
            if ruta.parent.is_dir():
                return str(ruta.parent)
        if por_defecto is not None and por_defecto.is_dir():
            return str(por_defecto)
        return str(tecnicas.RAIZ)

    def _elegir_carpeta(self, variable: tk.StringVar, inicial: Path | None = None) -> None:
        carpeta = filedialog.askdirectory(parent=self.raiz, mustexist=True,
                                          initialdir=self._carpeta_inicial(variable.get(),
                                                                           inicial))
        if carpeta:
            variable.set(carpeta)

    def _elegir_archivo(self, variable: tk.StringVar, tipos: list[tuple[str, str]],
                        inicial: str = "") -> None:
        archivo = filedialog.askopenfilename(
            parent=self.raiz, filetypes=tipos + [("Todos los archivos", "*")],
            initialdir=self._carpeta_inicial(inicial or variable.get()))
        if archivo:
            variable.set(archivo)

    def _elegir_fondo_moke(self) -> None:
        v = self.variables["MOKE"]
        self._elegir_archivo(v["fondo"], [("Imágenes", "*.png *.PNG *.jpg *.JPG")],
                             inicial=v["fondo"].get() or v["carpeta"].get())

    def _elegir_imagen_sem(self) -> None:
        self._elegir_archivo(self.variables["SEM"]["imagen"],
                             [("Imágenes TIFF", "*.tif *.tiff *.TIF *.TIFF")])

    def _elegir_imagen_tem(self) -> None:
        patrones = " ".join(f"*{e} *{e.upper()}" for e in FORMATOS_TEM)
        self._elegir_archivo(self.variables["TEM"]["ruta"], [("Imágenes TEM", patrones)])

    # ------------------------------------------------------------------ registro

    def escribir(self, texto: str, estilo: str = NORMAL) -> None:
        """Agrega texto al registro (solo desde el hilo principal)."""
        if estilo == NORMAL:
            estilo = _estilo_por_contenido(texto)
        self.registro.configure(state=tk.NORMAL)
        self.registro.insert(tk.END, texto, estilo)
        self.registro.see(tk.END)
        self.registro.configure(state=tk.DISABLED)

    def _limpiar_registro(self) -> None:
        self.registro.configure(state=tk.NORMAL)
        self.registro.delete("1.0", tk.END)
        self.registro.configure(state=tk.DISABLED)

    def _revisar_cola(self) -> None:
        """Pasa al registro los mensajes del hilo de fondo; atiende preguntas y el final."""
        try:
            while True:
                mensaje = self.cola.get_nowait()
                if mensaje[0] == TEXTO:
                    self.escribir(mensaje[2], mensaje[1])
                elif mensaje[0] == PREGUNTA:
                    self._responder(mensaje[1])
                elif mensaje[0] == FIN:
                    self._terminar(mensaje[1])
        except queue.Empty:
            pass
        self.after(INTERVALO_COLA_MS, self._revisar_cola)

    def _responder(self, pregunta) -> None:
        """Muestra el diálogo de una pregunta del análisis y devuelve la respuesta."""
        if pregunta.texto in self.respuestas_fijas:
            pregunta.respuesta = self.respuestas_fijas[pregunta.texto]
        else:
            valor, repetir = pedir_numero(self.raiz, "Falta un dato", pregunta.texto,
                                          pregunta.por_defecto,
                                          nota="No está en los metadatos del archivo.")
            pregunta.respuesta = pregunta.por_defecto if valor is None else valor
            if repetir:
                self.respuestas_fijas[pregunta.texto] = pregunta.respuesta
        self.escribir(f"  {pregunta.texto}: {pregunta.respuesta:g} (cargado a mano)\n", AVISO)
        pregunta.listo.set()

    # ------------------------------------------------------------------ análisis

    def _habilitar(self, habilitado: bool) -> None:
        for control in self.controles:
            control.state(["!disabled"] if habilitado else ["disabled"])

    def campos(self, tecnica: str) -> dict[str, str]:
        """Texto actual de los campos de una técnica."""
        return {clave: variable.get() for clave, variable in self.variables[tecnica].items()}

    def analizar(self) -> None:
        """Valida las opciones y arranca el análisis de la técnica elegida."""
        if self.ocupado:
            return
        tecnica = self.tecnica.get()
        try:
            parametros = validar_opciones(tecnica, self.campos(tecnica))
        except ValueError as error:
            self.escribir(f"✖ {error}\n", ERROR)
            return
        self.respuestas_fijas.clear()
        self.escribir(f"\n=== {tecnica} ===\n", OK)

        if tecnica == "MOKE":
            self._iniciar(lambda: tecnicas.correr_moke(parametros))
        elif tecnica == "TEM":
            preguntar = lambda texto, defecto: pedir_dato(self.cola, texto, defecto)  # noqa: E731
            self._iniciar(lambda: tecnicas.correr_tem(parametros, preguntar))
        elif parametros["modo"] == MODO_SEM_ROIS:
            self._iniciar(lambda: tecnicas.correr_sem_rois(parametros["carpeta"],
                                                           parametros["intensidad_pozos"]))
        else:
            self._analizar_imagen_sem(parametros)

    def _analizar_imagen_sem(self, parametros: dict) -> None:
        """Carga la imagen y abre el selector de ROI (hilo principal); segmenta en segundo plano."""
        self._habilitar(False)
        self.estado.config(text="Elegí la ROI…")
        ruta_roi = None
        with contextlib.redirect_stdout(EscritorCola(self.cola)), \
                contextlib.redirect_stderr(EscritorCola(self.cola, AVISO)):
            try:
                recortada, metadata = tecnicas.preparar_imagen_sem(parametros["imagen"])
                ruta_roi = tecnicas.elegir_y_guardar_roi(parametros["imagen"], recortada,
                                                         metadata, parametros["sufijo"])
            except Exception as error:  # imagen mala, sufijo repetido…
                print(f"✖ Error: {error}")
        if ruta_roi is None:
            self._habilitar(True)
            self.estado.config(text="")
            return
        self._iniciar(lambda: tecnicas.correr_sem_roi(ruta_roi,
                                                      parametros["intensidad_pozos"]))

    def _iniciar(self, tarea) -> None:
        gc.collect()  # que la basura de Tk se libere en el hilo principal, no en el de fondo
        self.ocupado = True
        self._habilitar(False)
        self.estado.config(text="Analizando…")
        iniciar_en_segundo_plano(tarea, self.cola)

    def _terminar(self, resultado: Resultado) -> None:
        """Al terminar: informa la salida, abre las figuras y rehabilita los controles."""
        self.ocupado = False
        self._habilitar(True)
        if resultado.error:
            self.estado.config(text="Terminó con errores")
        else:
            self.estado.config(text="Listo")
            if resultado.salida is not None:
                self.escribir(f"✔ Resultados en: {resultado.salida}\n", OK)
        for figura in resultado.figuras:
            try:
                abrir_figura(self.raiz, figura)
            except Exception as error:
                self.escribir(f"No se pudo abrir {figura}: {error}\n", AVISO)


def _estilo_por_contenido(texto: str) -> str:
    """Colorea los mensajes de los módulos según cómo empiezan o qué dicen."""
    limpio = texto.lstrip()
    if limpio.startswith(("❌", "✖")) or "ERROR" in texto:
        return ERROR
    if limpio.startswith(("⚠", "Warning", "warning")):
        return AVISO
    if limpio.startswith(("✅", "✔")):
        return OK
    return NORMAL


def abrir_interfaz() -> None:
    """Abre la ventana principal y espera hasta que se cierre."""
    # El análisis corre en otro hilo: pyplot no debe abrir ventanas ('Agg'). Las figuras se
    # abren desde el PNG guardado (figuras.py) y el selector de ROI cambia a 'TkAgg' un rato.
    matplotlib.use("Agg")
    raiz = tk.Tk()
    Ventana(raiz)
    raiz.mainloop()
