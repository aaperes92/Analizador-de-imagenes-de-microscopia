import runpy, sys, io, contextlib
import numpy as np
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(sys.argv[1] + "/roi3_regiones_variante_c.py")
estimar_sigma_borde, choque, dos_poblaciones = g["estimar_sigma_borde"], g["choque"], g["dos_poblaciones"]
ski, suavizar, estimar_ruido = g["ski"], g["suavizar"], g["estimar_ruido"]
# ROI 3, región del patrón: qué daría sin el criterio del valle
info, _ = dos_poblaciones(g["procesada"][g["regiones"] == 3])
c, h, (a, b) = info["centros"], info["h"], info["dos"]
v = a + int(np.argmin(h[a:b + 1]))
vals = g["procesada"][g["regiones"] == 3]
print(f"ROI 3, patrón: picos {c[a]:.3f} y {c[b]:.3f}, valle {c[v]:.3f} al {info['relacion']:.0%}; "
      f"sin criterio del valle la fase oscura sería {np.mean(vals <= c[v]):.1%} de la región")
# ROI 1, variante c con el mismo criterio de parada: relación del valle
suave1 = suavizar(np.load("data/sem/roi/BSE_1/BSE_1_1/roi_BSE_1_1.npy"))
s1, _ = estimar_sigma_borde(suave1)
r1 = ski.filters.unsharp_mask(suave1, radius=s1, amount=1.0, preserve_range=True)
p1, pasos1 = choque(r1, 1.0, tolerancia=0.1 * estimar_ruido(suave1) / np.ptp(r1))
info1, dec1 = dos_poblaciones(p1)
print(f"ROI 1 variante c ({pasos1} pasos): {dec1}")
print(f"negro: mínimo de la imagen procesada {g['procesada'].min():.3f} (la suavizada mínima {g['suave'].min():.3f})")
