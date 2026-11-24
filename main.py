import argparse
from src.moke import moke, 
from src.tem import compositional, diffraction, fraction, loader, morphology, phase_analysis, segmentation, spatial 
# import sem
# import tem

# Tipo de medida -> función que analiza una carpeta de imágenes.
# Va el nombre de la función SIN paréntesis: se guarda la función, no se llama.
# Las tres tienen que recibir lo mismo: (carpeta, escala=..., unidad=...).
ANALIZADORES = {
    "moke": moke.analizadorMOKE,
    # "sem": sem.analizar_carpeta,
    # "tem": tem.analizar_carpeta,
}


def analizar(tecnica, name_background, carpeta, escala=1.0, unidad="px"):

    tecnica = tecnica.lower()
    if tecnica not in ANALIZADORES:
        raise ValueError(f"Técnica desconocida: '{tecnica}'. Opciones: {', '.join(ANALIZADORES)}")

    funcion = ANALIZADORES[tecnica]
    return funcion(name_background, carpeta, escala=escala, unidad=unidad)


def main():
    parser = argparse.ArgumentParser(description="Analizador de imágenes de microscopía")
    parser.add_argument("tecnica", choices=list(ANALIZADORES), help="tipo de medida")
    parser.add_argument("name_background", help="nombre del archivo de fondo")
    parser.add_argument("carpeta", help="carpeta con las imágenes")
    parser.add_argument("--escala", type=float, default=1.0, help="tamaño de un píxel (por defecto 1)")
    parser.add_argument("--unidad", default="px", help="unidad de la escala (por defecto px)")
    args = parser.parse_args()

    analizar(args.tecnica, args.name_background, args.carpeta, escala=args.escala, unidad=args.unidad)


if __name__ == "__main__":
    main()