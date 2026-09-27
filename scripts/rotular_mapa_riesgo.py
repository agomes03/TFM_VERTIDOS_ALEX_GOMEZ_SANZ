"""
Corrige los rotulos del mapa de riesgo sin regenerar la figura.

La figura original (generar_figuras_era5.py) la titula "Mapa de riesgo previsto
(2026)", que induce a dos errores: sugiere un pronostico, cuando el modelo estima
con covariables del mismo mes, y dice 2026 cuando la evaluacion cubre enero-agosto.
Regenerarla exigiria la linea de costa de Natural Earth, que no esta en el equipo,
asi que se reescriben solo los dos rotulos sobre el PNG.

El tamano de letra se calibra midiendo el ancho del texto original.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
FIG = BASE / "figuras"
ORIGEN = FIG / "fig_mapa_riesgo.png"
DESTINO = FIG / "fig_mapa_riesgo_v2.png"

FUENTES = Path("C:/Windows/Fonts")
INK = (18, 32, 31)
MUTED = (107, 122, 123)

TITULO_VIEJO = "Mapa de riesgo previsto (2026)"
TITULO_NUEVO = "Riesgo de detección estimado (enero–agosto de 2026)"
BARRA_VIEJA = "Probabilidad prevista de detección"
BARRA_NUEVA = "Probabilidad estimada de detección"


def caja_de_tinta(arr, filas, columnas, umbral=350):
    """Recuadro que ocupa el texto dentro de la region indicada."""
    region = arr[filas[0]:filas[1], columnas[0]:columnas[1]]
    oscuro = region.sum(axis=2) < umbral
    f = np.where(oscuro.any(axis=1))[0]
    c = np.where(oscuro.any(axis=0))[0]
    return (columnas[0] + c.min(), filas[0] + f.min(), columnas[0] + c.max(), filas[0] + f.max())


def ultima_columna_de_texto(arr, desde_x, umbral=600):
    """Rotulo de la barra: es el bloque de texto mas a la derecha, separado de los
    numeros de la escala por un hueco en blanco."""
    oscuro = arr[:, desde_x:].sum(axis=2) < umbral
    cols = np.where(oscuro.any(axis=0))[0]
    corte = np.where(np.diff(cols) > 5)[0]
    grupo = cols[corte[-1] + 1:] if len(corte) else cols
    filas = np.where(oscuro[:, grupo].any(axis=1))[0]
    return (desde_x + grupo.min(), filas.min(), desde_x + grupo.max(), filas.max())


def calibrar(texto, ancho_objetivo, fichero):
    """Tamano de letra que reproduce el ancho del texto original."""
    mejor, mejor_dif = 10, 1e9
    for tam in range(12, 60):
        f = ImageFont.truetype(str(FUENTES / fichero), tam)
        ancho = f.getbbox(texto)[2] - f.getbbox(texto)[0]
        dif = abs(ancho - ancho_objetivo)
        if dif < mejor_dif:
            mejor, mejor_dif = tam, dif
    return mejor


im = Image.open(ORIGEN).convert("RGB")
arr = np.array(im)
alto, ancho = arr.shape[:2]

x0, y0, x1, y1 = caja_de_tinta(arr, (0, 80), (0, ancho))
tam_titulo = calibrar(TITULO_VIEJO, x1 - x0, "segoeuib.ttf")
fuente_titulo = ImageFont.truetype(str(FUENTES / "segoeuib.ttf"), tam_titulo)
print("titulo: caja (%d,%d)-(%d,%d) -> %d pt" % (x0, y0, x1, y1, tam_titulo))

bx0, by0, bx1, by1 = ultima_columna_de_texto(arr, int(ancho * 0.90))
tam_barra = calibrar(BARRA_VIEJA, by1 - by0, "segoeui.ttf")
fuente_barra = ImageFont.truetype(str(FUENTES / "segoeui.ttf"), tam_barra)
print("barra:  caja (%d,%d)-(%d,%d) -> %d pt" % (bx0, by0, bx1, by1, tam_barra))

d = ImageDraw.Draw(im)
# titulo: se borra la franja y se escribe alineado a la izquierda, misma linea base
d.rectangle([x0 - 4, y0 - 8, ancho - 4, y1 + 8], fill=(255, 255, 255))
d.text((x0, y0), TITULO_NUEVO, font=fuente_titulo, fill=INK, anchor="la")

# etiqueta de la barra: texto girado, centrado donde estaba
d.rectangle([bx0 - 6, by0 - 10, bx1 + 6, by1 + 10], fill=(255, 255, 255))
caja = fuente_barra.getbbox(BARRA_NUEVA)
tira = Image.new("RGBA", (caja[2] - caja[0] + 4, caja[3] - caja[1] + 10), (255, 255, 255, 0))
ImageDraw.Draw(tira).text((2 - caja[0], 4 - caja[1]), BARRA_NUEVA, font=fuente_barra, fill=MUTED)
tira = tira.rotate(90, expand=True)
im.paste(tira, ((bx0 + bx1) // 2 - tira.width // 2, (by0 + by1) // 2 - tira.height // 2), tira)

im.save(DESTINO)
print("guardado:", DESTINO.name, im.size)
