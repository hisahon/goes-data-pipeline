# -*- coding: utf-8 -*-
"""
tracker_spyder.py
=================

Interface interativa para o rastreamento de clusters convectivos (MCS)
no **Spyder IDE**.  Execute célula por célula com Ctrl+Enter (ou F9
para a linha atual).  Todos os parâmetros ficam na célula [0] — edite
lá e reexecute as células seguintes conforme necessário.

Autor: Rogério Hisashi Honda <hisashi.brase@gmail.com>
"""

# %% [0]  CONFIGURAÇÃO  ← edite aqui, depois rode o resto
# =============================================================================

from pathlib import Path

# --- Caminhos ---
INPUT_PARQUET  = Path("merged/c13_2018-2025_SouthAmerica.parquet")
OUTPUT_PARQUET = Path("merged/c13_2018-2025_SA_Tracked.parquet")
OUTPUT_SUMMARY = OUTPUT_PARQUET.with_name(OUTPUT_PARQUET.stem + "_summary.csv")
FIGURES_DIR    = Path("figures_tracks")

# --- Filtros aplicados ANTES do tracking ---
MIN_AREA_KM2 = 600_000      # None para desativar
LAT_MIN, LAT_MAX = -40, -12 # None, None para sem recorte
LON_MIN, LON_MAX = -67, -42 # None, None para sem recorte

# --- Parâmetros do tracking ---
MAX_TIME_GAP   = "20min"    # tolerância entre frames consecutivos
MAX_KM         = 35.0       # deslocamento máximo (km)
MAX_SPEED_KMH  = 120.0      # velocidade máxima (km/h) — None para desativar
MIN_POINTS     = 60         # pontos mínimos por trilha (~60 = 10h de scan)
ASSIGNMENT     = "greedy"   # "greedy" (rápido) ou "hungarian" (ótimo por frame)

# --- Plot ---
PLOT_EXTENT    = (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)  # (lon_min, lon_max, lat_min, lat_max)
PLOT_PREFIX    = "SA_tracked"
PLOT_DPI       = 300

# =============================================================================


# %% [1]  IMPORTS & SETUP DE LOGGING
# =============================================================================

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tracker import (
    TrackerOptions,
    track_centroids,
    summarize_tracks,
    plot_yearly_tracks,
    plot_all_years,
)

# Logging visível no console do Spyder
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,                 # reaplica mesmo se já estava configurado
)
log = logging.getLogger("tracker_spyder")
log.info("Imports OK.")


# %% [2]  CARREGAR DETECÇÕES
# =============================================================================

log.info("Lendo %s ...", INPUT_PARQUET)
df_raw = pd.read_parquet(INPUT_PARQUET)
df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])

log.info(
    "Carregado: %d linhas | %d–%d | colunas: %s",
    len(df_raw),
    df_raw["datetime"].dt.year.min(),
    df_raw["datetime"].dt.year.max(),
    list(df_raw.columns),
)

# Visão rápida — aparece no Variable Explorer com duplo-clique
print(df_raw.head())
print(df_raw.dtypes)


# %% [3]  RODAR O TRACKING
# =============================================================================
#
# Se OUTPUT_PARQUET já existir, carrega em vez de recalcular.
# Apague o arquivo ou mude OUTPUT_PARQUET para forçar o recálculo.

bbox = None
if None not in (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX):
    bbox = (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

opts = TrackerOptions(
    max_time_gap   = pd.Timedelta(MAX_TIME_GAP),
    max_km         = MAX_KM,
    max_speed_kmh  = MAX_SPEED_KMH,
    min_points     = MIN_POINTS,
    assignment     = ASSIGNMENT,
    min_area_km2   = MIN_AREA_KM2,
    bbox           = bbox,
)

if OUTPUT_PARQUET.exists():
    log.info("Parquet tracked encontrado — carregando (apague para recalcular).")
    df_tracked = pd.read_parquet(OUTPUT_PARQUET)
    df_tracked["datetime"] = pd.to_datetime(df_tracked["datetime"])
else:
    log.info("Rodando tracking com opções:\n  %s", opts)
    df_tracked = track_centroids(df_raw, opts)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_tracked.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("Salvo: %s", OUTPUT_PARQUET)

log.info(
    "Tracking pronto: %d detecções | %d trilhas",
    len(df_tracked),
    df_tracked["track_id"].nunique(),
)


# %% [4]  SUMÁRIO DAS TRILHAS
# =============================================================================
#
# df_summary: uma linha por trilha — bom para inspecionar no Variable Explorer.

df_summary = summarize_tracks(df_tracked)
df_summary["duration_h"] = (
    (df_summary["t1"] - df_summary["t0"]).dt.total_seconds() / 3600
).round(1)

df_summary.to_csv(OUTPUT_SUMMARY, index=False)
log.info("Sumário salvo: %s", OUTPUT_SUMMARY)

print(f"\n{'='*50}")
print(f"Total de trilhas : {len(df_summary)}")
print(f"Duração média    : {df_summary['duration_h'].mean():.1f} h")
print(f"Duração máxima   : {df_summary['duration_h'].max():.1f} h")
print(f"Detecções/trilha : {df_summary['n_points'].mean():.1f} (média)")
print(f"{'='*50}\n")
print(df_summary.describe().round(2))


# %% [5]  DISTRIBUIÇÃO DE DURAÇÃO  (diagnóstico rápido)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(df_summary["duration_h"], bins=40, edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("Duração (h)")
axes[0].set_ylabel("Contagem")
axes[0].set_title("Distribuição de duração das trilhas")

axes[1].hist(df_summary["n_points"], bins=40, edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("Número de detecções")
axes[1].set_ylabel("Contagem")
axes[1].set_title("Distribuição de tamanho das trilhas")

fig.tight_layout()
plt.show()


# %% [6]  INSPECIONAR UMA TRILHA ESPECÍFICA
# =============================================================================
#
# Mude TRACK_ID para qualquer track_id que queira examinar.
# Use df_summary no Variable Explorer para escolher um interessante.

TRACK_ID = int(df_summary.sort_values("duration_h", ascending=False).iloc[0]["track_id"])

grp = df_tracked[df_tracked["track_id"] == TRACK_ID].sort_values("datetime")
print(f"\nTrilha {TRACK_ID}:")
print(grp[["datetime", "centroid_lat", "centroid_lon", "area_km2"]].to_string(index=False))

# Plot simples da trajetória no Spyder (sem Cartopy)
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(
    grp["centroid_lon"], grp["centroid_lat"],
    c=np.arange(len(grp)), cmap="plasma", s=15, zorder=3,
)
ax.plot(grp["centroid_lon"], grp["centroid_lat"], "-", lw=0.8, alpha=0.5)
ax.scatter(*grp[["centroid_lon","centroid_lat"]].iloc[0],
           s=60, marker="^", color="blue", zorder=5, label="Início")
ax.scatter(*grp[["centroid_lon","centroid_lat"]].iloc[-1],
           s=60, marker="s", color="red",  zorder=5, label="Fim")
plt.colorbar(sc, ax=ax, label="Frame (ordem temporal)")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
t0 = grp["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M")
t1 = grp["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M")
ax.set_title(f"Trilha {TRACK_ID}  |  {t0} → {t1}")
ax.legend()
fig.tight_layout()
plt.show()


# %% [7]  MAPA ANUAL COM CARTOPY  (um ano específico)
# =============================================================================
#
# Gera o mapa para um único ano e abre inline no Spyder.
# Para todos os anos de uma vez, use a célula [8].

YEAR = int(df_tracked["datetime"].dt.year.mode()[0])   # ano mais frequente por default

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
out_png = FIGURES_DIR / f"{PLOT_PREFIX}_{YEAR}.png"

plot_yearly_tracks(
    df_tracked, YEAR, out_png,
    extent=PLOT_EXTENT, dpi=PLOT_DPI,
)
log.info("Mapa salvo: %s", out_png)

# Exibe no painel de plots do Spyder
img = plt.imread(out_png)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.axis("off")
ax.set_title(f"SA_tracked_{YEAR}")
plt.tight_layout()
plt.show()


# %% [8]  MAPAS PARA TODOS OS ANOS
# =============================================================================
#
# Pode demorar alguns minutos dependendo do volume de dados.

paths = plot_all_years(
    df_tracked,
    out_dir        = FIGURES_DIR,
    filename_prefix= PLOT_PREFIX,
    extent         = PLOT_EXTENT,
    dpi            = PLOT_DPI,
)
log.info("%d mapas gerados em %s", len(paths), FIGURES_DIR)


# %% [9]  EXPERIMENTOS  (rodar sem salvar, para comparar parâmetros)
# =============================================================================
#
# Útil para testar variações sem sobrescrever OUTPUT_PARQUET.
# Resultado fica em df_exp — inspecione no Variable Explorer.

opts_exp = TrackerOptions(
    max_time_gap  = pd.Timedelta("20min"),
    max_km        = 50.0,          # <- mude aqui
    max_speed_kmh = 80.0,          # <- e aqui
    min_points    = 30,            # <- e aqui
    assignment    = "hungarian",   # <- greedy ou hungarian
    min_area_km2  = MIN_AREA_KM2,
    bbox          = bbox,
)

df_exp = track_centroids(df_raw, opts_exp)
summary_exp = summarize_tracks(df_exp)

print(f"\nExperimento:")
print(f"  Trilhas : {df_exp['track_id'].nunique()}")
print(f"  Duração média : {summary_exp['duration_h'].mean():.1f} h")  # type: ignore[index]
print(f"\nComparação com produção:")
print(f"  Produção — trilhas: {df_tracked['track_id'].nunique()}, "
      f"duração média: {df_summary['duration_h'].mean():.1f} h")
