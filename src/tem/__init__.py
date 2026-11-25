"""Módulo TEM del Analizador de Imágenes de Microscopía."""

# Etapa 1: carga y preprocesamiento
from .loader import (
    load_tem_image,
    pipeline_tem,
    preprocess_tem_image,
    show_preprocessing_results,
)

# Etapa 2: segmentación de fases
from .segmentation import (
    analyze_histogram,
    apply_segmentation,
    clean_mask,
    compute_phase_stats,
    detect_interfaces,
    detect_n_phases,
    pipeline_segmentation,
    select_threshold_method,
    separate_touching_regions,
    show_segmentation_results,
)

# Etapa 3: análisis morfológico
from .morphology import (
    analyze_orientation,
    analyze_shape,
    analyze_size_distribution,
    extract_region_properties,
    pipeline_morphology,
    plot_orientation_rose,
    show_morphology_results,
)

# Etapa 4: análisis de fases (fracción, composicional y espacial)
from .fraction import (
    compute_area_fractions,
    compute_interface_fraction,
    compute_number_density,
    estimate_volume_fractions,
    pipeline_fraction,
    show_fraction_results,
)
from .compositional import (
    build_phase_intensity_map,
    compute_intensity_gradients,
    compute_intensity_profiles,
    detect_inhomogeneities,
    pipeline_compositional,
    show_compositional_results,
)
from .spatial import (
    compute_density_map,
    compute_rdf,
    detect_clustering,
    detect_spatial_gradients,
    pipeline_spatial,
    show_spatial_results,
)
from .phase_analysis import pipeline_phase_analysis

# Etapa 4.5: patrones de difracción (SAED, NBED, perfil 1D)
from .diffraction import (
    analyze_1d_profile,
    analyze_spot_geometry,
    compute_radial_profile,
    correct_center_interactive,
    detect_beam_stopper,
    detect_diffraction_rings,
    detect_diffraction_spots,
    detect_pattern_type,
    electron_wavelength,
    estimate_crystallite_size,
    find_beam_center,
    is_diffraction_pattern,
    pipeline_diffraction,
    preprocess_diffraction,
    read_camera_metadata,
    show_diffraction_results,
)

# Análisis completo de una imagen o carpeta (lo usan scripts/analizar_tem.py y la interfaz)
from .analisis_completo import analizar_ruta

__all__ = [
    # Análisis completo
    "analizar_ruta",
    # Etapa 1
    "load_tem_image",
    "preprocess_tem_image",
    "show_preprocessing_results",
    "pipeline_tem",
    # Etapa 2
    "analyze_histogram",
    "detect_n_phases",
    "select_threshold_method",
    "apply_segmentation",
    "clean_mask",
    "separate_touching_regions",
    "detect_interfaces",
    "compute_phase_stats",
    "show_segmentation_results",
    "pipeline_segmentation",
    # Etapa 3
    "extract_region_properties",
    "analyze_size_distribution",
    "analyze_shape",
    "analyze_orientation",
    "plot_orientation_rose",
    "show_morphology_results",
    "pipeline_morphology",
    # Etapa 4 — Módulo A: fracción
    "compute_area_fractions",
    "estimate_volume_fractions",
    "compute_number_density",
    "compute_interface_fraction",
    "show_fraction_results",
    "pipeline_fraction",
    # Etapa 4 — Módulo B: composicional
    "compute_intensity_profiles",
    "compute_intensity_gradients",
    "build_phase_intensity_map",
    "detect_inhomogeneities",
    "show_compositional_results",
    "pipeline_compositional",
    # Etapa 4 — Módulo C: espacial
    "compute_rdf",
    "compute_density_map",
    "detect_clustering",
    "detect_spatial_gradients",
    "show_spatial_results",
    "pipeline_spatial",
    # Etapa 4 — pipeline maestro
    "pipeline_phase_analysis",
    # Etapa 4.5 — difracción
    "is_diffraction_pattern",
    "detect_pattern_type",
    "read_camera_metadata",
    "electron_wavelength",
    "find_beam_center",
    "correct_center_interactive",
    "detect_beam_stopper",
    "preprocess_diffraction",
    "compute_radial_profile",
    "detect_diffraction_rings",
    "estimate_crystallite_size",
    "detect_diffraction_spots",
    "analyze_spot_geometry",
    "analyze_1d_profile",
    "show_diffraction_results",
    "pipeline_diffraction",
]
