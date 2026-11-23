# Instrucciones para Claude Code — Módulo SEM

> Este archivo aplica SOLO al módulo SEM, a cargo de Mauricio Romano
> (src/sem/, scripts/sem_*, docs/sem.md). Si estás trabajando en TEM o MOKE, ignoralo:
> esas partes son de otros integrantes del equipo.

Proyecto: Analizador de imágenes de microscopía (SEM/TEM/MOKE).
Equipo: Mauricio Romano (SEM), Daniel Romero, Angel A. Perez.

## Reglas de comportamiento

1. **Autoría en commits.** Nunca agregues tu nombre ni referencias a Claude o Anthropic en
   los commits: nada de `Co-Authored-By: Claude`, `Generated with Claude Code` ni `🤖`.
2. **Mensaje de commit.** Antes de cualquier `git commit`, mostrá `git status` y
   `git diff --stat` y preguntá: "¿Qué mensaje querés para este commit?". Si la respuesta es
   "el que quieras", proponé uno y pedí confirmación.
3. **Push.** Nunca hagas `git push` (tampoco `--force` ni `--tags`) sin un "sí" explícito.
   Antes informá la rama, el remoto y los commits (`git log origin/<rama>..HEAD --oneline`).
4. **Explicar antes de ejecutar.** Antes de crear, editar o borrar archivos, instalar paquetes,
   correr scripts o hacer cualquier operación de git, explicá qué vas a hacer, qué archivos
   se tocan y preguntá "¿Puedo proceder?". Leer archivos no requiere confirmación.
5. **Paso a paso.** Implementá una función por vez: primero la firma y el docstring, esperá
   la confirmación y después el cuerpo.
6. **Preguntas sin responder.** Si hiciste una pregunta y no fue respondida, no avances ni
   elijas por tu cuenta: esperá la respuesta.
7. **Alcance.** No toques nada fuera de SEM (TEM, MOKE, README, configuración común) sin
   consultarlo antes.

## Imágenes SEM

Hay imágenes de electrones retrodispersados (BSE) y de electrones secundarios (SE). Cada tipo
requiere un análisis distinto.

## Etapas del pipeline SEM

1. **Carga y ROI:** lectura, metadata, selección y guardado de la ROI.
2. **Mejora de la ROI:** suavizado, corrección de iluminación no uniforme y ecualización.
3. **Procesamiento:** umbralizado, segmentación, eliminación de objetos chicos y medición
   de propiedades de regiones.
4. **Resultados:** comparaciones antes/después, contornos sobre la imagen, tablas
   exportables y figuras de publicación.

## Estructura del proyecto (parte SEM)

```
src/sem/            código del módulo (una etapa por archivo)
scripts/sem_*.py    scripts de línea de comandos
docs/sem.md         documentación: qué hace y cómo funciona cada parte
notebooks/sem_*     notebooks para probar cada etapa con las ROIs reales
imagenes/SEM/BSE/   imágenes BSE originales (no se versionan)
imagenes/SEM/SE/    imágenes SE originales (no se versionan)
data/sem/roi/       ROIs generadas por la carga (no se versionan)
```

## Estilo de código

- Programación funcional: funciones con nombre descriptivo, type hints y docstring completo.
- Los parámetros se fijan con criterios objetivos a partir de la imagen, no ajustándolos a
  mano para cada caso.
- Documentación de nivel profesional: docstrings, comentarios y `docs/sem.md` actualizados.
- El procesamiento de imágenes se hace solo con `skimage`.
- Priorizar el estilo del curso de Python Científico: https://wpregliasco.github.io/python-sci/

## Referencias de skimage

- Thresholding: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html
- Multi-Otsu: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
- Random walker: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_random_walker_segmentation.html
- Watershed compacto: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_compact_watershed.html
- Segmentación entrenable: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html
- Objetos chicos: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_remove_objects.html
- Huecos y picos: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html
- Ridge filters: https://scikit-image.org/docs/stable/auto_examples/edges/plot_ridge_filter.html
- Edge filters: https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html
- Contornos: https://scikit-image.org/docs/stable/auto_examples/edges/plot_contours.html
- Regionprops: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
