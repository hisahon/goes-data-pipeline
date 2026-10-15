# -*- coding: utf-8 -*-
"""
aggregator_spyder.py
====================

Interface interativa para **agregação e análise exploratória** dos
relatórios de detecção (Spyder IDE).

Execute célula por célula com Ctrl+Enter (F9 para uma linha).  Todos os
parâmetros ficam na célula [0] — edite lá e reexecute o que precisar.

Autor: Rogério Hisashi Honda <hisashi.brase@gmail.com>
"""

# %% [0]  CONFIGURAÇÃO  ← edite aqui, depois rode o resto
# =============================================================================

from pathlib import Path

# --- Entrada: árvore dos reports/*.txt produzidos por visual.py ---

year,doy = 2025, 298
REPORTS_DIR     = Path(f"reports/{year:04d}/{doy:03d}")

# --- Saídas em parquet (idempotentes: reusa se já existir) ---
AGG_PARQUET     = Path(f"merged/{year}-{doy}_raw.parquet")       # todas as detecções (bruto)
AGGREGATED_PARQUET = Path(f"merged/{year}-{doy}_aggregated_clusters.parquet")      # mesmas linhas + track_id

# --- Pastas de figuras ---
FIGURES_DIR     = Path("results")

# --- Parâmetros da análise ---
SIZE_COL        = "area_km2"    # coluna usada no tamanho dos pontos
MIN_HOURS       = 4.            # duração mínima de sistemas (tracks)
GAP_MINUTES     = 30            # gap temporal máximo dentro de um track
AREA_LABEL      = "Area > 200k km² | Duration > 4h"
PLOT_PREFIX     = "doy_hour_runs"
PALETTE         = "tab20"       # 20 cores para distinguir mais tracks simultâneos
N_COLORS        = 20
MAX_SPEED_KMH   = 120           # velocidade máxima assumida do sistema
DPI             = 300

# =============================================================================


# %% [1]  IMPORTS & LOGGING
# =============================================================================

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aggregator import (
    reports2df,
    plot_tracks_latlon,
    make_unique_by_time,               # ainda disponível (desempate simples)
    track_multiple_clusters,
    filter_continuous_runs,
    plot_doy_hour,
    plot_runs_by_year,
    plot_multicluster_overlay,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("aggregator_spyder")
log.info("Imports OK.")


# %% [2]  AGREGAR REPORTS → PARQUET BRUTO
# =============================================================================
#
# Se AGG_PARQUET já existir, carrega em vez de varrer a árvore de novo.
# Apague o parquet para forçar releitura.

if AGG_PARQUET.exists():
    log.info("Carregando agregado existente: %s", AGG_PARQUET)
    df = pd.read_parquet(AGG_PARQUET)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
else:
    log.info("Varrendo %s ...", REPORTS_DIR)
    df = reports2df(REPORTS_DIR)
    if df.empty:
        raise RuntimeError(f"Nenhum relatório encontrado em {REPORTS_DIR}")

    keep = ["datetime", "area_km2", "temp_min_C", "temp_mean_C",
            "centroid_lat", "centroid_lon"]
    df = df[[c for c in keep if c in df.columns]]

    AGG_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(AGG_PARQUET, index=False)
    log.info("Salvo: %s", AGG_PARQUET)

log.info("Registros agregados: %d", len(df))
print(df.head())
print(df.describe().round(2))


# %% [3]  RASTREAR CLUSTERS NO ESPAÇO-TEMPO  →  DF COM track_id
# =============================================================================
#
# track_multiple_clusters NÃO descarta linhas (ao contrário de
# make_unique_by_time). Todas as detecções ficam preservadas; o que muda é
# a adição da coluna ``track_id`` que identifica qual sistema cada ponto
# pertence. Sistemas simultâneos viram tracks distintos.

if AGGREGATED_PARQUET.exists():
    log.info("Carregando tracked existente: %s", AGGREGATED_PARQUET)
    df_u = pd.read_parquet(AGGREGATED_PARQUET)
    df_u["datetime"] = pd.to_datetime(df_u["datetime"], errors="coerce")
else:
    df_u = track_multiple_clusters(
        df,
        max_speed_kmh=MAX_SPEED_KMH,
        max_gap_minutes=GAP_MINUTES,
    )
    log.info("Tracked: %s", AGGREGATED_PARQUET)

n_tracks = df_u["track_id"].nunique()
log.info("Rastreados %d sistemas em %d observações (nenhum registro descartado).",
         n_tracks, len(df_u))


# %% [4]  SEGMENTAR EM RUNS (TRACKS) CONTÍNUOS
# =============================================================================
#
# df_cont: registros de tracks com duração >= MIN_HOURS
# runs:    resumo por track (início, fim, duração em minutos)

df_cont, runs = filter_continuous_runs(
    df_u,
    min_hours=MIN_HOURS,
    gap_minutes=GAP_MINUTES,
    time_col='datetime'
)

AGGREGATED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_cont.to_parquet(AGGREGATED_PARQUET, index=False)

print(f"\nTotal de tracks: {len(runs)}")
print(f"Tracks válidos (≥ {MIN_HOURS} h): {(runs['duration_min'] >= MIN_HOURS*60).sum()}")
print(f"Registros retidos: {len(df_cont)}")
print("\nTop 10 tracks por duração:")
print(
    runs.sort_values("duration_min", ascending=False)
        .head(10)
        [["run_id", "run_start", "run_end", "duration_min"]]
        .to_string(index=False)
)


# %% [5]  PLOT DOY × HORA (inline)  — todos os anos
# =============================================================================
# Agora cada ponto recebe cor segundo o ``track_id`` (sistema rastreado),
# em vez de cor proporcional ao tamanho. Assim dá para ver os sistemas
# simultâneos como famílias de pontos distintas.

plot_doy_hour(
    df_u,
    out_path=None,
    size_col=SIZE_COL,
    size_scale=1e5,
    one_axes_per_year=True,
    color_by="track_id",
    palette=PALETTE,
    n_colors=N_COLORS,
)
plt.show()


# %% [6]  PLOT POR ANO COM SISTEMAS COLORIDOS  — salva PNGs
# =============================================================================
# As cores representam o ``run_id`` (= track_id de tracks válidos ≥ MIN_HOURS).

paths = plot_runs_by_year(
    df_cont,
    out_dir         = FIGURES_DIR,
    min_hours       = MIN_HOURS,
    area_label      = AREA_LABEL,
    filename_prefix = PLOT_PREFIX,
    palette         = PALETTE,
    n_colors        = N_COLORS,
    dpi             = DPI,
)
log.info("%d PNGs salvos em %s", len(paths), FIGURES_DIR)


# %% [7]  OVERLAY DE TIMESTAMPS MULTICLUSTER
# =============================================================================
# Destaca com um "×" os timestamps em que 2 ou mais sistemas
# *independentes* (track_ids distintos) estão ativos simultaneamente.
# O scatter de fundo também é colorido por track_id.

plot_multicluster_overlay(
    df_u,
    out_path=FIGURES_DIR / "doy_hour_multicluster.png",
    size_col=SIZE_COL,
    size_scale=1e5,
    color_by="track_id",
    palette=PALETTE,
    n_colors=N_COLORS,
)
plt.show()


 
# %% [8]  INSPEÇÃO LIVRE
# =============================================================================
 
# Histograma de duração dos tracks
plt.figure(figsize=(7, 4))
plt.hist(runs["duration_min"] / 60, bins=50, edgecolor="white", linewidth=0.4)
plt.axvline(MIN_HOURS, color="red", linestyle="--", label=f"threshold = {MIN_HOURS} h")
plt.xlabel("Duração do track (h)")
plt.ylabel("Contagem")
plt.title("Distribuição da duração dos sistemas convectivos rastreados")
plt.legend()
plt.tight_layout()
plt.show()
 
# # Ciclo diurno (histograma de hora UTC, ponderado por área)
# # Nota: plt.hist trata NaN em `weights` contaminando o bin inteiro →
# # filtra linhas com NaN em SIZE_COL antes de plotar.
# plt.figure(figsize=(7, 4))
# df_cont_h = df_cont[["datetime", SIZE_COL]].copy()
# df_cont_h["hour"] = df_cont_h["datetime"].dt.hour
# df_cont_h = df_cont_h.dropna(subset=["hour", SIZE_COL])
# if df_cont_h.empty:
#     log.warning("Nenhum dado válido para o ciclo diurno (todos NaN?).")
# else:
#     plt.hist(df_cont_h["hour"], bins=np.arange(25) - 0.5,
#              weights=df_cont_h[SIZE_COL], edgecolor="white", linewidth=0.4)
#     plt.xticks(range(0, 24, 3))
#     plt.xlabel("Hora UTC")
#     plt.ylabel(f"Soma de {SIZE_COL}")
#     plt.title("Ciclo diurno dos sistemas (ponderado pela área)")
#     plt.tight_layout()
# plt.show()
 
# Número de sistemas simultâneos ao longo do tempo
plt.figure(figsize=(10, 3))
simult = df_u.groupby("datetime")["track_id"].nunique()
plt.plot(simult.index, simult.values, linewidth=1.)
plt.ylabel("Nº de sistemas simultâneos")
plt.xlabel("Tempo")
plt.title("Concorrência temporal de sistemas convectivos")
plt.tight_layout()
plt.show()
 

# %% [9]  PLOT TRAJETÓRIAS (LAT x LON) 
# =============================================================================
# Mapeia o deslocamento geográfico dos sistemas rastreados.
# Usamos o `df_cont` para visualizar apenas os sistemas validados (longos),
# evitando que a tela fique poluída com nuvens pequenas isoladas.

plot_tracks_latlon(
    df_cont, 
    out_path=FIGURES_DIR / f"{year}-{doy}_SpatialTrajectories.png",
    size_col=SIZE_COL,
    size_scale=1e4 # Ajuste esse valor para calibrar o tamanho das "bolhas" no gráfico
)
plt.show()
