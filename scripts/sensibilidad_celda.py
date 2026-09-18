"""
Sensibilidad al tamano de celda (Modifiable Areal Unit Problem).

La rejilla de 10 km se eligio sin justificacion empirica. Aqui se reconstruye
el panel completo a 5, 10 y 20 km y se reajusta el GLM en cada resolucion,
para comprobar si los rate ratios -- y por tanto las conclusiones -- dependen
de esa eleccion arbitraria.

Se reutilizan los datos ya descargados (detecciones, esfuerzo por escena,
viento, AIS) reasignandolos a cada rejilla; no se vuelve a consultar ninguna
API.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from shapely.geometry import box

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
BBOX = (47.5, 23.5, 56.5, 30.5)
UTM = "EPSG:32640"
TAMANOS = [5, 10, 20]

FEATURES = ["wind_speed", "dist_costa_km", "log_ais_hours", "mes_sin", "mes_cos", "anios_desde_2023"]


def build_grid(cell_km):
    minx, miny, maxx, maxy = BBOX
    lat_mid = (miny + maxy) / 2
    dlat = cell_km / 111.0
    dlon = cell_km / (111.0 * np.cos(np.radians(lat_mid)))
    xs = np.arange(minx, maxx, dlon)
    ys = np.arange(miny, maxy, dlat)
    rows = [{"cell_id": f"{i}_{j}", "geometry": box(x, y, x + dlon, y + dlat)}
            for i, x in enumerate(xs) for j, y in enumerate(ys)]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


print("Cargando insumos ya descargados...")
det = pd.read_csv(OUT_DIR / "cerulean_golfo_persico_limpio.csv",
                  usecols=["id", "slick_timestamp", "centroid_lon", "centroid_lat", "s1_scene_id"])
det["mes"] = pd.to_datetime(det["slick_timestamp"], utc=True).dt.to_period("M").astype(str)

# cell_id de 10 km + escena; en el repositorio se distribuye comprimido (.csv.gz)
ruta_esfuerzo = OUT_DIR / "esfuerzo_golfo_persico.csv"
if not ruta_esfuerzo.exists():
    ruta_esfuerzo = OUT_DIR / "esfuerzo_golfo_persico.csv.gz"
esfuerzo = pd.read_csv(ruta_esfuerzo)
# El esfuerzo se reconstruye desde las escenas: se necesita su huella. Se usa el
# panel de 10 km como referencia de que escenas hubo cada mes, y se reasigna el
# numero de pasadas proporcionalmente al area -- para 5 y 20 km se recalcula
# desde el catalogo de escenas guardado.
esfuerzo["fecha"] = pd.to_datetime(esfuerzo["timestamp_ms"], unit="ms", utc=True)
esfuerzo["mes"] = esfuerzo["fecha"].dt.to_period("M").astype(str)

viento = pd.read_csv(OUT_DIR / "viento_celda_mes_era5.csv")
ais = pd.read_csv(OUT_DIR / "trafico_ais_celda_mes.csv")
cov10 = pd.read_csv(OUT_DIR / "covariables_espaciales_celda.csv")

# Coordenadas de referencia de las celdas de 10 km, para traspasar viento/AIS
grid10 = build_grid(10)
grid10["cx"] = grid10.geometry.centroid.x
grid10["cy"] = grid10.geometry.centroid.y
ref10 = grid10[["cell_id", "cx", "cy"]].copy()

viento10 = viento.merge(ref10, on="cell_id", how="inner")
ais10 = ais.merge(ref10, on="cell_id", how="inner")
cov10 = cov10.merge(ref10, on="cell_id", how="inner")

# Catalogo de escenas con su huella, reconstruido del esfuerzo de 10 km:
# cada escena cubre las celdas de 10 km que intersecta; para otras
# resoluciones se aproxima la huella por la envolvente de esas celdas.
print("Reconstruyendo huellas de escena a partir del esfuerzo de 10 km...")
esc = esfuerzo.merge(ref10, on="cell_id", how="inner")
huellas = esc.groupby(["s1_scene_id", "mes"]).agg(
    xmin=("cx", "min"), xmax=("cx", "max"), ymin=("cy", "min"), ymax=("cy", "max")
).reset_index()
print(f"Escenas-mes: {len(huellas)}")

resultados = []
for cell_km in TAMANOS:
    print(f"\n=== Rejilla de {cell_km} km ===")
    grid = build_grid(cell_km)
    grid["cx"] = grid.geometry.centroid.x
    grid["cy"] = grid.geometry.centroid.y
    print(f"  celdas: {len(grid)}")

    # --- detecciones por celda-mes ---
    pts = gpd.GeoDataFrame(det[["mes"]].copy(),
                            geometry=gpd.points_from_xy(det["centroid_lon"], det["centroid_lat"]),
                            crs="EPSG:4326")
    j = gpd.sjoin(pts, grid[["cell_id", "geometry"]], how="inner", predicate="within")
    eventos = j.groupby(["cell_id", "mes"]).size().reset_index(name="n_events")

    # --- esfuerzo por celda-mes: celdas dentro de la envolvente de cada escena ---
    gc = grid[["cell_id", "cx", "cy"]].to_numpy()
    ids = grid["cell_id"].to_numpy()
    cx = grid["cx"].to_numpy(); cy = grid["cy"].to_numpy()
    filas = []
    for h in huellas.itertuples():
        m = (cx >= h.xmin) & (cx <= h.xmax) & (cy >= h.ymin) & (cy <= h.ymax)
        for cid in ids[m]:
            filas.append((cid, h.mes))
    pases = pd.DataFrame(filas, columns=["cell_id", "mes"])
    pases = pases.groupby(["cell_id", "mes"]).size().reset_index(name="n_pases")
    print(f"  filas celda-mes con esfuerzo: {len(pases)}")

    panel = pases.merge(eventos, on=["cell_id", "mes"], how="left")
    panel["n_events"] = panel["n_events"].fillna(0).astype(int)

    # --- covariables: vecino mas cercano desde la rejilla de 10 km ---
    gpts = gpd.GeoDataFrame(grid[["cell_id"]], geometry=gpd.points_from_xy(grid["cx"], grid["cy"]),
                            crs="EPSG:4326").to_crs(UTM)

    def traspasar(tabla, cols, por_mes):
        base = tabla.drop_duplicates("cell_id")[["cell_id", "cx", "cy"]]
        src = gpd.GeoDataFrame(base[["cell_id"]].rename(columns={"cell_id": "src_id"}),
                               geometry=gpd.points_from_xy(base["cx"], base["cy"]),
                               crs="EPSG:4326").to_crs(UTM)
        nn = gpd.sjoin_nearest(gpts, src, how="left")[["cell_id", "src_id"]]
        if por_mes:
            out = nn.merge(tabla.rename(columns={"cell_id": "src_id"})[["src_id", "mes"] + cols],
                           on="src_id", how="left")
            return out[["cell_id", "mes"] + cols]
        out = nn.merge(tabla.rename(columns={"cell_id": "src_id"})[["src_id"] + cols],
                       on="src_id", how="left")
        return out[["cell_id"] + cols]

    v = traspasar(viento10, ["wind_speed"], True)
    a = traspasar(ais10, ["ais_hours"], True)
    c = traspasar(cov10, ["dist_costa_km"], False)

    panel = panel.merge(v, on=["cell_id", "mes"], how="left")
    panel = panel.merge(a, on=["cell_id", "mes"], how="left")
    panel = panel.merge(c, on="cell_id", how="left")
    panel["ais_hours"] = panel["ais_hours"].fillna(0.0)

    # --- variables derivadas y GLM ---
    panel["anio"] = panel["mes"].str.slice(0, 4).astype(int)
    panel["mes_num"] = panel["mes"].str.slice(5, 7).astype(int)
    panel["mes_sin"] = np.sin(2 * np.pi * panel["mes_num"] / 12)
    panel["mes_cos"] = np.cos(2 * np.pi * panel["mes_num"] / 12)
    panel["anios_desde_2023"] = panel["anio"] - 2023 + (panel["mes_num"] - 1) / 12.0
    panel["log_ais_hours"] = np.log1p(panel["ais_hours"])
    panel = panel[panel["n_pases"] > 0].dropna(subset=FEATURES + ["n_events", "n_pases"])
    print(f"  filas para el modelo: {len(panel)}")

    X = panel[FEATURES].copy()
    for col in ["wind_speed", "dist_costa_km", "log_ais_hours", "anios_desde_2023"]:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    X = sm.add_constant(X)

    fit = sm.GLM(panel["n_events"].values, X, family=sm.families.Poisson(),
                 offset=np.log(panel["n_pases"].values)).fit()

    for var in FEATURES:
        resultados.append({
            "cell_km": cell_km, "variable": var,
            "rate_ratio": float(np.exp(fit.params[var])),
            "p": float(fit.pvalues[var]),
            "n_filas": len(panel), "n_celdas": panel["cell_id"].nunique(),
        })
    print(f"  rate ratios: " + ", ".join(
        f"{v}={np.exp(fit.params[v]):.2f}" for v in FEATURES))

res = pd.DataFrame(resultados)
tabla = res.pivot(index="variable", columns="cell_km", values="rate_ratio")
print("\n=== Rate ratios por tamano de celda ===")
print(tabla.round(3).to_string())
res.to_csv(OUT_DIR / "maup_resultados.csv", index=False)
tabla.to_csv(OUT_DIR / "maup_tabla.csv")

with open(OUT_DIR / "maup_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Sensibilidad al tamano de celda (MAUP) ===\n\n")
    f.write("Rate ratios del GLM Poisson con offset de esfuerzo, por resolucion:\n\n")
    f.write(tabla.round(3).to_string())
    f.write("\n\nFilas y celdas por resolucion:\n")
    f.write(res.groupby("cell_km")[["n_filas", "n_celdas"]].first().to_string())

print(f"\nGuardado en {OUT_DIR}")
