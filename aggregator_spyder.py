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
REPORTS_DIR     = Path("reports_S")

# --- Saídas em parquet (idempotentes: reusa se já existir) ---
AGG_PARQUET     = Path("merged/c13_2018-2025.parquet")       # todas as detecções
UNIQUE_PARQUET  = Path("merged/c13_2018-2025u.parquet")      # 1 cluster por timestamp

# --- Pastas de figuras ---
FIGURES_DIR     = Path("results")

# --- Parâmetros da análise ---
SIZE_COL        = "area_km2"    # coluna usada no desempate / no tamanho dos pontos
MIN_HOURS       = 16            # duração mínima de runs contínuos
GAP_MINUTES     = 30            # gap máximo dentro de um run
AREA_LABEL      = "Area > 200k km²"
PLOT_PREFIX     = "doy_hour_runs"
PALETTE         = "tab10"
N_COLORS        = 10
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
    make_unique_by_time,
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

    # mantém apenas as colunas essenciais (ajuste se precisar)
    keep = ["datetime", "area_km2", "temp_min_C", "temp_mean_C",
            "centroid_lat", "centroid_lon"]
    df = df[[c for c in keep if c in df.columns]]

    AGG_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(AGG_PARQUET, index=False)
    log.info("Salvo: %s", AGG_PARQUET)

log.info("Registros agregados: %d", len(df))
print(df.head())
print(df.describe().round(2))


# %% [3]  REDUZIR PARA UM CLUSTER POR TIMESTAMP
# =============================================================================
#
# Mantém em cada timestamp apenas o cluster com maior SIZE_COL.

if UNIQUE_PARQUET.exists():
    log.info("Carregando único existente: %s", UNIQUE_PARQUET)
    df_u = pd.read_parquet(UNIQUE_PARQUET)
    df_u["datetime"] = pd.to_datetime(df_u["datetime"], errors="coerce")
else:
    df_u = make_unique_by_time(df, size_col=SIZE_COL)
    UNIQUE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_u.to_parquet(UNIQUE_PARQUET, index=False)
    log.info("Salvo: %s", UNIQUE_PARQUET)

log.info("Reduziu %d → %d registros", len(df), len(df_u))


# %% [4]  SEGMENTAR EM RUNS CONTÍNUOS
# =============================================================================
#
# df_cont: registros de runs >= MIN_HOURS
# runs:    resumo por run (início, fim, duração em minutos)

df_cont, runs = filter_continuous_runs(
    df_u, min_hours=MIN_HOURS, gap_minutes=GAP_MINUTES,
)

print(f"\nTotal de runs: {len(runs)}")
print(f"Runs válidos (≥ {MIN_HOURS} h): {(runs['duration_min'] >= MIN_HOURS*60).sum()}")
print(f"Registros retidos: {len(df_cont)}")
print("\nTop 10 runs por duração:")
print(
    runs.sort_values("duration_min", ascending=False)
        .head(10)
        [["run_id", "run_start", "run_end", "duration_min"]]
        .to_string(index=False)
)


# %% [5]  PLOT DOY × HORA (inline)  — todos os anos
# =============================================================================

plot_doy_hour(
    df_u,
    out_path=None,
    size_col=SIZE_COL,
    size_scale=1e5,
    one_axes_per_year=True,
)
plt.show()


# %% [6]  PLOT POR ANO COM RUNS COLORIDOS  — salva PNGs
# =============================================================================

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
#
# Usa o parquet BRUTO (antes do unique) — é ali que aparecem timestamps
# com 2+ clusters simultâneos.

plot_multicluster_overlay(
    df,
    out_path=FIGURES_DIR / "doy_hour_multicluster.png",
    size_col=SIZE_COL,
    size_scale=1e5,
)
plt.show()


# %% [8]  INSPEÇÃO LIVRE
# =============================================================================
#
# Espaço para exploração ad-hoc. Alguns prontos úteis:

# Histograma de duração dos runs
plt.figure(figsize=(7, 4))
plt.hist(runs["duration_min"] / 60, bins=50, edgecolor="white", linewidth=0.4)
plt.axvline(MIN_HOURS, color="red", linestyle="--", label=f"threshold = {MIN_HOURS} h")
plt.xlabel("Duração do run (h)")
plt.ylabel("Contagem")
plt.title("Distribuição da duração dos runs")
plt.legend()
plt.tight_layout()
plt.show()

# Ciclo diurno (histograma de hora UTC, ponderado por área)
plt.figure(figsize=(7, 4))
df_cont_h = df_cont.copy()
df_cont_h["hour"] = df_cont_h["datetime"].dt.hour
plt.hist(df_cont_h["hour"], bins=np.arange(25) - 0.5,
         weights=df_cont_h[SIZE_COL], edgecolor="white", linewidth=0.4)
plt.xticks(range(0, 24, 3))
plt.xlabel("Hora UTC")
plt.ylabel(f"Soma de {SIZE_COL}")
plt.title("Ciclo diurno (ponderado pela área)")
plt.tight_layout()
plt.show()
