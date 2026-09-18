"""
Continua la extraccion de AIS (GFW) desde donde se quedo: el token solo
permite 1 informe concurrente, y un fallo transitorio en 2025-02 dejo
bloqueadas todas las peticiones siguientes con 429. Aqui se reintenta con
espera entre peticiones y backoff, solo para los meses que faltan, y se
fusiona con lo ya descargado.
"""
import asyncio
from pathlib import Path

import pandas as pd
import geopandas as gpd

import gfwapiclient as gfw

BBOX = (47.5, 23.5, 56.5, 30.5)
TOKEN_PATH = Path(__file__).parent / "gfw_token.txt"
OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
RAW_PATH = OUT_DIR / "trafico_ais_gfw_celda01_mes.csv"


async def fetch_month(client, geojson, start, end, max_retries=5, base_wait=20):
    for attempt in range(1, max_retries + 1):
        try:
            result = await client.fourwings.create_ais_presence_report(
                spatial_resolution="LOW",
                temporal_resolution="MONTHLY",
                start_date=start,
                end_date=end,
                geojson=geojson,
            )
            df = result.df()
            if df.empty:
                return pd.DataFrame(columns=["lat", "lon", "hours"])
            return df.groupby(["lat", "lon"], as_index=False)["hours"].sum()
        except Exception as e:
            wait = base_wait * attempt
            print(f"    intento {attempt}/{max_retries} fallo ({type(e).__name__}); espero {wait}s...")
            await asyncio.sleep(wait)
    print(f"  [ERROR] {start}: agotados los reintentos")
    return None


async def main():
    token = TOKEN_PATH.read_text().strip()
    client = gfw.Client(access_token=token)

    minx, miny, maxx, maxy = BBOX
    geojson = {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}

    existing = pd.read_csv(RAW_PATH) if RAW_PATH.exists() else pd.DataFrame(columns=["lat", "lon", "hours", "mes"])
    meses_ok = set(existing["mes"].unique())
    print(f"Meses ya descargados: {sorted(meses_ok)}")

    todos_meses = pd.period_range("2023-01", pd.Timestamp.now().to_period("M"), freq="M")
    faltan = [p for p in todos_meses if str(p) not in meses_ok]
    print(f"Meses que faltan: {[str(p) for p in faltan]}")

    nuevas = [existing] if not existing.empty else []
    for period in faltan:
        start = period.start_time.strftime("%Y-%m-%d")
        end = (period.end_time + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        agg = await fetch_month(client, geojson, start, end)
        if agg is None:
            continue
        agg["mes"] = str(period)
        nuevas.append(agg)
        print(f"  {period}: {len(agg)} celdas GFW, {agg['hours'].sum():.0f} horas-buque")
        await asyncio.sleep(8)  # margen entre peticiones para no chocar con el limite de concurrencia

    traffic_raw = pd.concat(nuevas, ignore_index=True)
    traffic_raw.to_csv(RAW_PATH, index=False)
    print(f"\nGuardado bruto actualizado: {RAW_PATH} ({len(traffic_raw)} filas, {traffic_raw['mes'].nunique()} meses)")

    # --- Reconstruir salida final con lo que haya disponible ---
    canon = traffic_raw[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    canon_pts = gpd.GeoDataFrame(canon, geometry=gpd.points_from_xy(canon["lon"], canon["lat"]), crs="EPSG:4326")

    grid = gpd.read_file(OUT_DIR / "rejilla_golfo_persico.geojson")
    grid["centroid"] = grid.geometry.centroid
    grid_pts = gpd.GeoDataFrame(grid[["cell_id"]], geometry=grid["centroid"], crs=grid.crs)

    nearest = gpd.sjoin_nearest(grid_pts, canon_pts, how="left")[["cell_id", "lat", "lon"]]
    traffic_fino = nearest.merge(traffic_raw, on=["lat", "lon"], how="left")
    traffic_fino = traffic_fino.rename(columns={"hours": "ais_hours"})[["cell_id", "mes", "ais_hours"]]
    traffic_fino["ais_hours"] = traffic_fino["ais_hours"].fillna(0.0)
    traffic_fino.to_csv(OUT_DIR / "trafico_ais_celda_mes.csv", index=False)
    print(f"Guardado: trafico_ais_celda_mes.csv ({len(traffic_fino)} filas)")

    panel = pd.read_csv(OUT_DIR / "panel_modelo_fase6.csv")
    panel["cell_id"] = panel["cell_id"].astype(str)
    traffic_fino["cell_id"] = traffic_fino["cell_id"].astype(str)
    final = panel.merge(traffic_fino, on=["cell_id", "mes"], how="left")
    final.to_csv(OUT_DIR / "panel_modelo_fase6_con_ais.csv", index=False)
    cobertura = final["ais_hours"].notna().mean() * 100
    print(f"Guardado: panel_modelo_fase6_con_ais.csv ({len(final)} filas, {cobertura:.1f}% con AIS asignado)")


asyncio.run(main())
