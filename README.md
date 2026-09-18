# Material complementario del TFM

**Patrones espacio-temporales de las detecciones de manchas de petróleo del catálogo Cerulean en el golfo Pérsico**

Alex Gómez Sanz · Máster en Data Analytics · Universidad Internacional de Valencia · Curso 2025-2026

Este paquete contiene el código, los datos intermedios y los resultados necesarios para reproducir los análisis de la memoria.

## Estructura

```
material_complementario/
├── README.md                    este documento
├── requirements.txt             dependencias de Python con las versiones empleadas
├── notebooks/                   cuadernos de Google Colab (descarga, limpieza, esfuerzo, modelos)
└── scripts/
    ├── *.py                     un script por fase o análisis (orden de ejecución más abajo)
    ├── salida_golfo_persico/    datos intermedios, panel celda-mes y resultados numéricos
    └── figuras/                 figuras de la memoria (los scripts de figuras escriben aquí)
```

Todos los scripts leen y escriben en `scripts/salida_golfo_persico/` y `scripts/figuras/` mediante rutas relativas al propio script, por lo que funcionan desde cualquier ubicación.

## Instalación

Se ha utilizado Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Credenciales

No se incluye ninguna credencial. Solo son necesarias para volver a descargar datos (fases 0 a 4). Los análisis de la fase 6 en adelante se reproducen con los datos incluidos.

| Servicio | Uso | Configuración |
|---|---|---|
| Cerulean (SkyTruth) | Detecciones de manchas | API pública, sin autenticación |
| Google Earth Engine | Catálogo Sentinel-1, Sentinel-2, batimetría | `earthengine authenticate` y variable de entorno `EE_PROJECT` con el identificador de un proyecto propio |
| Copernicus Climate Data Store | Viento ERA5 | Aceptar la licencia de `reanalysis-era5-single-levels` y crear `scripts/cdsapirc.txt` con dos líneas: `url: https://cds.climate.copernicus.eu/api` y `key: <clave personal>` |
| Global Fishing Watch | Tráfico AIS | Token personal de la API en `scripts/gfw_token.txt` |

Para las figuras de mapas se necesita además la línea de costa de Natural Earth a 10 m, en `scripts/ne_10m_coastline.geojson`. Se descarga de https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/.

## Reproducción rápida, sin credenciales

Desde la carpeta `scripts/`:

```bash
python glm_fase6_era5.py            # modelo explicativo: Poisson y binomial negativa (tabla 3)
python sobredispersion.py           # diagnóstico de sobredispersión (apartado 6.3)
python comparar_rf_xgb.py           # Random Forest frente a XGBoost (tabla 4)
python bootstrap_diferencia.py      # intervalo bootstrap de la diferencia de PR-AUC
python bootstrap_bloques.py         # sensibilidad del bootstrap a la dependencia espacial y temporal
python analisis_hitl.py             # detecciones revisadas por humanos (tabla 6)
python autocorrelacion.py           # I de Moran y errores estándar agrupados
python sensibilidad_celda.py        # rejillas de 5, 10 y 20 km (tabla 7)
python descomponer_tendencia.py     # origen de la tendencia temporal
python caracterizar_focos.py
python control_uniformidad.py
python descomposicion_kitagawa.py
python tendencia_efectos_fijos.py   # tabla 8
```

## Reproducción completa: orden de ejecución

| Fase | Script | Credenciales | Principales salidas |
|---|---|---|---|
| 0-2. Descarga y limpieza | `relanzar_golfo_persico.py` | Ninguna | `cerulean_golfo_persico_limpio.csv/.geojson`, `positivos_celda_escena.csv`, `rejilla_golfo_persico.geojson` |
| 1-3. Esfuerzo de observación | `catalogo_esfuerzo_golfo_persico.py` | Earth Engine | `esfuerzo_golfo_persico.csv`, `exposicion_mensual_celda.csv` |
| 4. Viento provisional (GFS) | `incorporar_viento.py` | Earth Engine | `viento_celda_mes.csv`, `exposicion_mensual_celda_con_viento.csv` |
| 4. Distancia a costa | `distancia_costa_rutas.py` | Ninguna | `covariables_espaciales_celda.csv`, `panel_modelo_fase6.csv` |
| 4. Tráfico AIS | `incorporar_ais.py`, y `incorporar_ais_resume.py` si la descarga se interrumpe | Global Fishing Watch | `trafico_ais_celda_mes.csv`, `panel_modelo_fase6_con_ais.csv` |
| 4. Viento ERA5 | `descargar_era5.py` y después `procesar_era5.py` | Copernicus CDS | `viento_celda_mes_era5.csv`, `panel_modelo_fase6_era5.csv` |
| 6. Modelo explicativo | `glm_fase6_era5.py`, `sobredispersion.py` | Ninguna | razones de tasas, diagnóstico de sobredispersión |
| 7. Estimación fuera de muestra | `modelo_predictivo_fase7_era5.py`, `tune_xgb.py`, `comparar_rf_xgb.py`, `bootstrap_diferencia.py`, `bootstrap_bloques.py` | Ninguna | métricas, predicciones de enero-agosto de 2026, intervalos bootstrap |
| Robustez | `analisis_hitl.py`, `autocorrelacion.py`, `sensibilidad_celda.py` | Ninguna | tablas 6 y 7, errores estándar agrupados |
| Robustez óptica | `buscar_optico.py`, `chips_optico.py`, `analisis_someras.py`, `fig_profundidad.py` | Earth Engine | emparejamiento Sentinel-1/Sentinel-2, tasa por profundidad |
| Tendencia | `descomponer_tendencia.py`, `caracterizar_focos.py`, `control_uniformidad.py`, `descomposicion_kitagawa.py`, `tendencia_efectos_fijos.py` | Ninguna | apartado 6.7.4 |
| Figuras | `generar_figuras_era5.py`, `figuras_comparativa.py`, `figuras_robustez.py`, `figuras_tendencia.py`, `figura_esquema_metodologia.py` | Ninguna | figuras 1 a 18 |

Los scripts `glm_fase6.py`, `glm_fase6_con_ais.py` y `modelo_predictivo_fase7.py` son las versiones previas con viento GFS o con el proxy de rutas de OpenStreetMap. Se conservan por trazabilidad.

## Fuentes de datos y licencias

Los datos incluidos son productos derivados de fuentes públicas. Su reutilización está sujeta a los términos de cada fuente.

| Fuente | Datos derivados incluidos | Términos |
|---|---|---|
| SkyTruth, Cerulean | Catálogo depurado de detecciones y paneles derivados | Uso con atribución a SkyTruth ([términos de servicio](https://skytruth.org/terms-of-service)) |
| Global Fishing Watch | Tráfico AIS agregado por celda y mes | CC BY-NC 4.0: solo uso no comercial y con atribución ([licencia de la API](https://globalfishingwatch.org/our-apis/documentation/docs/license-rate-limits)) |
| Copernicus Climate Change Service, ERA5 | Viento medio por celda y mes | Licencia de Copernicus. Contiene información modificada del Copernicus Climate Change Service |
| Copernicus Sentinel-1 y Sentinel-2, a través de Google Earth Engine | Catálogo de escenas y esfuerzo de observación | Datos Copernicus Sentinel de acceso libre, con atribución |
| Natural Earth, OpenStreetMap | Distancia a costa y a rutas marítimas | Natural Earth es de dominio público. OpenStreetMap: © colaboradores de OpenStreetMap, ODbL |

El catálogo de esfuerzo `scripts/salida_golfo_persico/esfuerzo_golfo_persico.csv.gz` se distribuye comprimido por su tamaño. Los scripts lo leen directamente.

## Notas

- **Periodo:** enero de 2023 a agosto de 2026. Agosto de 2026 está incompleto en el catálogo de esfuerzo.
- **Partición temporal del modelo de clasificación:** ajuste de enero de 2023 a junio de 2025, calibración de julio a diciembre de 2025 y evaluación de enero a agosto de 2026.
- **Lectura de las detecciones:** una detección de Cerulean es una mancha identificada automáticamente en una imagen radar, no un vertido confirmado.
- **Cuadernos de Colab:** reproducen las fases 0 a 7 en Google Colab. Requieren autenticarse en Earth Engine dentro del cuaderno.
