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
INPUT_PARQUET  = Path("merged/2962021.parquet")
TRACKED_PARQUET = Path("merged/2962021u.parquet")
OUTPUT_SUMMARY = TRACKED_PARQUET.with_name(TRACKED_PARQUET.stem + "_summary.csv")
FIGURES_DIR    = Path("figures")

# --- Filtros aplicados ANTES do tracking ---
MIN_AREA_KM2 = None      # None para desativar
LAT_MIN, LAT_MAX = -50, -15 # None, None para sem recorte
LON_MIN, LON_MAX = -65, -30 # None, None para sem recorte

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

if TRACKED_PARQUET.exists():
    log.info("Parquet tracked encontrado — carregando (apague para recalcular).")
    df_tracked = pd.read_parquet(TRACKED_PARQUET)
    df_tracked["datetime"] = pd.to_datetime(df_tracked["datetime"])
else:
    log.info("Rodando tracking com opções:\n  %s", opts)
    df_tracked = track_centroids(df_raw, opts)

    # OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    # df_tracked.to_parquet(OUTPUT_PARQUET, index=False)
    # log.info("Salvo: %s", OUTPUT_PARQUET)

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


# %% [5]  DIAGNÓSTICOS DE TRILHAS  (por track_id)
# =============================================================================
# Três painéis:
#   (a) Distribuição de duração (h)        — uma entrada por track_id
#   (b) Distribuição de nº de detecções    — uma entrada por track_id
#   (c) Concorrência temporal              — quantos tracks coexistem

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df_summary["duration_h"], bins=40, edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("Duração (h)")
axes[0].set_ylabel("Nº de trilhas")
axes[0].set_title("Duração por track_id")

axes[1].hist(df_summary["n_points"], bins=40, edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("Nº de detecções")
axes[1].set_ylabel("Nº de trilhas")
axes[1].set_title("Tamanho por track_id")

# Concorrência: nº de track_ids distintos ativos em cada timestamp
concurrent = df_tracked.groupby("datetime")["track_id"].nunique()
max_conc = int(concurrent.max())
axes[2].hist(
    concurrent.values,
    bins=np.arange(max_conc + 2) - 0.5,
    edgecolor="white", linewidth=0.4,
)
axes[2].set_xticks(range(1, max_conc + 1))
axes[2].set_xlabel("Nº de tracks simultâneos")
axes[2].set_ylabel("Nº de timestamps")
axes[2].set_title(f"Concorrência temporal (máx = {max_conc})")

fig.tight_layout()
plt.show()

# Stats rápidas
print(f"Total de trilhas          : {len(df_summary)}")
print(f"Timestamps c/ >1 sistema  : {(concurrent > 1).sum()} "
      f"({100 * (concurrent > 1).mean():.1f}%)")
print(f"Máx. sistemas simultâneos : {max_conc}")


# %% [6]  PANORAMA DAS TOP-N TRILHAS  (trajetórias + Gantt temporal)
# =============================================================================
# Plota as TOP_N trilhas mais longas todas no mesmo mapa, cada uma com uma
# cor distinta, ao lado de um Gantt mostrando quais se sobrepõem no tempo.
# Substitui a inspeção de 1 trilha única do pipeline antigo.

TOP_N = 15

top_ids = (
    df_summary.sort_values("duration_h", ascending=False)
              .head(TOP_N)["track_id"]
              .tolist()
)
print(f"Top {TOP_N} trilhas por duração: {top_ids}")

cmap = plt.get_cmap("tab10")
fig, (ax_map, ax_gantt) = plt.subplots(
    1, 2, figsize=(13, 5),
    gridspec_kw={"width_ratios": [1.2, 1]},
)

# (a) Trajetórias sobrepostas
for i, tid in enumerate(top_ids):
    grp = df_tracked[df_tracked["track_id"] == tid].sort_values("datetime")
    color = cmap(i % 10)
    dur_h = (grp["datetime"].iloc[-1] - grp["datetime"].iloc[0]).total_seconds() / 3600
    label = f"#{tid}  ({dur_h:.1f} h, {len(grp)} pts)"

    ax_map.plot(
        grp["centroid_lon"], grp["centroid_lat"],
        "-o", markersize=3, lw=0.8, alpha=0.8, color=color, label=label,
    )
    ax_map.scatter(
        grp["centroid_lon"].iloc[0], grp["centroid_lat"].iloc[0],
        s=70, marker="^", facecolor=color, edgecolor="black", zorder=5,
    )
    ax_map.scatter(
        grp["centroid_lon"].iloc[-1], grp["centroid_lat"].iloc[-1],
        s=70, marker="s", facecolor=color, edgecolor="black", zorder=5,
    )

ax_map.set_xlabel("Longitude (°)")
ax_map.set_ylabel("Latitude (°)")
ax_map.set_title(f"Top {TOP_N} trilhas — trajetórias (▲ início, ■ fim)")
ax_map.legend(fontsize=8, loc="best")
ax_map.grid(True, linestyle=":", linewidth=0.5)

# (b) Gantt: janela temporal de cada trilha (linhas horizontais grossas)
for i, tid in enumerate(top_ids):
    grp = df_tracked[df_tracked["track_id"] == tid]
    t0, t1 = grp["datetime"].min(), grp["datetime"].max()
    ax_gantt.plot(
        [t0, t1], [i, i],
        lw=10, color=cmap(i % 10), alpha=0.85, solid_capstyle="butt",
    )

ax_gantt.set_yticks(range(len(top_ids)))
ax_gantt.set_yticklabels([f"#{tid}" for tid in top_ids])
ax_gantt.invert_yaxis()
ax_gantt.set_xlabel("Tempo")
ax_gantt.set_title("Janela temporal (sobreposições = coexistência)")
ax_gantt.grid(True, axis="x", linestyle=":", linewidth=0.5)
fig.autofmt_xdate()

fig.tight_layout()
plt.show()


# %% [6b]  INSPECIONAR UMA TRILHA ESPECÍFICA  (versão original, para zoom)
# =============================================================================
# Use quando quiser examinar em detalhe uma trilha específica (trajetória
# colorida pela ordem temporal). Escolha o ID pelo df_summary no Variable
# Explorer ou use a maior do TOP_N.

TRACK_ID = top_ids[0]   # ou: int(df_summary.iloc[idx]["track_id"])

grp = df_tracked[df_tracked["track_id"] == TRACK_ID].sort_values("datetime")
print(f"\nTrilha {TRACK_ID}:")
print(grp[["datetime", "centroid_lat", "centroid_lon", "area_km2"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(
    grp["centroid_lon"], grp["centroid_lat"],
    c=np.arange(len(grp)), cmap="plasma", s=15, zorder=3,
)
ax.plot(grp["centroid_lon"], grp["centroid_lat"], "-", lw=0.8, alpha=0.5)
ax.scatter(grp["centroid_lon"].iloc[0], grp["centroid_lat"].iloc[0],
           s=60, marker="^", color="blue", zorder=5, label="Início")
ax.scatter(grp["centroid_lon"].iloc[-1], grp["centroid_lat"].iloc[-1],
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