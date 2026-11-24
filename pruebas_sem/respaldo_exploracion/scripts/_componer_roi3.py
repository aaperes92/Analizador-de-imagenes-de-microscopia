"""Arma roi3_regiones_variante_c.py copiando funciones de scripts anteriores (con ast)."""

import ast
from pathlib import Path

AQUI = Path(__file__).parent


def extraer(archivo, nombres):
    fuente = (AQUI / archivo).read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    lineas = fuente.splitlines()
    partes = []
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name in nombres:
            partes.append("\n".join(lineas[nodo.lineno - 1:nodo.end_lineno]))
    return "\n\n\n".join(partes)


regiones = extraer("segmentacion_por_region.py",
                   {"escalar", "valles_entre", "masas", "cortes_por_valles", "zonas_por_gris",
                    "franja_de_borde", "actividad_sin_cruzar"})
realce = extraer("laberinto_deconv_vs_choque.py", {"estimar_sigma_borde", "_minmod"})
choque = extraer("_variantes_parte2.py", {"choque"})
cuerpo = (AQUI / "_roi3_cuerpo.py").read_text(encoding="utf-8")
cabecera, resto = cuerpo.split("# @@FUNCIONES@@")
salida = (cabecera + "# --- funciones copiadas de scripts anteriores ---\n\n" + regiones + "\n\n\n" + realce
          + "\n\n\n" + choque + "\n\n" + resto)
(AQUI / "roi3_regiones_variante_c.py").write_text(salida, encoding="utf-8")
print("armado")
