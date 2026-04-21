# -*- coding: utf-8 -*-
"""
aggregator.py
=============

Agregação dos relatórios de detecção (``reports/C13_YYYY-DOY_HHMM.txt``
gerados por ``detector.py``) e análises exploratórias da série temporal
resultante.

Posição no pipeline::

    reports/*.txt
        │ reports2df()
        ▼
    parquet bruto (todas as detecções)
        │ make_unique_by_time()
        ▼
    parquet "único" (um cluster/timestamp)
        │ filter_continuous_runs()
        ▼
    séries contínuas → plots DOY×hora → tracker.py

Interface pública
-----------------
* :func:`reports2df`               — lê uma árvore de ``reports/*.txt``.
* :func:`make_unique_by_time`      — mantém um cluster por timestamp.
* :func:`filter_continuous_runs`   — segmenta por gap temporal.
* :func:`plot_doy_hour`            — scatter DOY × hora UTC por ano.
* :func:`plot_runs_by_year`        — barras verticais coloridas por run.
* :func:`plot_multicluster_overlay`— destaca timestamps multicluster.

Criado por Rogério Hisashi Honda <hisashi.brase@gmail.com>.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# C13_2019-007_0315.txt  ->  year=2019, doy=007, hm=0315
_FNAME_RE = re.compile(r"C13_(\d{4})-(\d{3})_(\d{4})", re.IGNORECASE)

_NUMERIC_COLS = (
    "area_km2", "temp_min_C", "temp_mean_C",
    "temp_std_C", "centroid_lat", "centroid_lon", "n_clusters",
)


# ---------------------------------------------------------------------------
# 1. Leitura da árvore de reports
# ---------------------------------------------------------------------------
def reports2df(
    root_dir: str | Path,
    pattern: str = "C13_*.txt",
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Varre ``root_dir`` recursivamente e concatena os relatórios de detecção.

    Parameters
    ----------
    root_dir
        Raiz da árvore ``reports/`` produzida por ``visual.py``.
    pattern
        Glob para os arquivos (default: ``"C13_*.txt"``).
    usecols
        Se informado, mantém apenas essas colunas na saída.

    Returns
    -------
    DataFrame
        Uma linha por cluster detectado, com colunas:
        ``area_km2, temp_min_C, temp_mean_C, centroid_lat, centroid_lon,
        datetime, source_file`` (+ ``date, hour, minute`` auxiliares).
        Retorna DataFrame vazio se nenhum arquivo for encontrado.
    """
    root = Path(root_dir)
    files = sorted(root.rglob(pattern))
    logger.info("Encontrados %d arquivos em %s", len(files), root)
    if not files:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    skipped = 0
    for i, fp in enumerate(files, start=1):
        m = _FNAME_RE.search(fp.name)
        if not m:
            skipped += 1
            continue
        year, doy, hm = m.groups()
        try:
            dt = datetime.strptime(year + doy + hm, "%Y%j%H%M")
        except ValueError:
            skipped += 1
            continue

        try:
            df = pd.read_csv(fp, sep=",")
        except Exception as e:
            logger.warning("Falha ao ler %s: %s", fp, e)
            skipped += 1
            continue

        df["datetime"] = pd.Timestamp(dt)
        df["source_file"] = str(fp)
        rows.append(df)

        if i % 5000 == 0:
            logger.info("  lidos %d/%d (pulados %d)", i, len(files), skipped)

    if not rows:
        logger.warning("Nenhum relatório válido em %s", root)
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out.columns = out.columns.str.strip()

    # tipagem numérica
    for c in _NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values(
        ["datetime", "area_km2"], ascending=[True, False], ignore_index=True,
    )

    out["date"] = out["datetime"].dt.date
    out["hour"] = out["datetime"].dt.hour
    out["minute"] = out["datetime"].dt.minute

    if usecols is not None:
        cols = [c for c in usecols if c in out.columns]
        out = out[cols]

    logger.info("Agregados %d registros (pulados %d arquivos).", len(out), skipped)
    return out


# ---------------------------------------------------------------------------
# 2. Desempate por timestamp
# ---------------------------------------------------------------------------
def make_unique_by_time(
    df: pd.DataFrame,
    time_col: str = "datetime",
    size_col: str = "area_km2",
) -> pd.DataFrame:
    """Para cada timestamp, mantém apenas a linha com o maior ``size_col``.

    Vetorizado via ``groupby().idxmax()``.
    """
    if df.empty:
        return df.copy()
    if size_col not in df.columns:
        raise KeyError(f"Coluna {size_col!r} ausente.")
    idx = df.groupby(time_col)[size_col].idxmax()
    return df.loc[idx].sort_values(time_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Segmentação em runs contínuos
# ---------------------------------------------------------------------------
def filter_continuous_runs(
    df: pd.DataFrame,
    min_hours: float = 1.0,
    gap_minutes: float = 60.0,
    time_col: str = "datetime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra séries contínuas com duração mínima de ``min_hours``.

    Considera quebra de sequência quando o intervalo entre registros
    consecutivos é maior que ``gap_minutes``.

    Returns
    -------
    df_cont : DataFrame
        Subset de ``df`` com os registros pertencentes a runs válidos.
        Inclui a coluna ``run_id``.
    runs : DataFrame
        Um registro por run, com ``run_id``, ``run_start``, ``run_end`` e
        ``duration_min``.
    """
    if df.empty:
        return df.copy(), pd.DataFrame(columns=["run_id", "run_start", "run_end", "duration_min"])

    df = df.sort_values(time_col).copy()
    gap = df[time_col].diff()
    new_run = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    df["run_id"] = new_run.cumsum()

    runs = (
        df.groupby("run_id")[time_col]
          .agg(run_start="min", run_end="max")
          .reset_index()
    )
    runs["duration_min"] = (
        (runs["run_end"] - runs["run_start"]).dt.total_seconds() / 60.0
    )

    min_dur = min_hours * 60.0
    valid = runs.loc[runs["duration_min"] >= min_dur, "run_id"]
    df_cont = df[df["run_id"].isin(valid)].copy()

    logger.info(
        "Runs: %d detectados | %d válidos (>= %.1f h) | %d registros retidos",
        len(runs), len(valid), min_hours, len(df_cont),
    )
    return df_cont, runs


# ---------------------------------------------------------------------------
# 4. Visualizações DOY × hora
# ---------------------------------------------------------------------------
def _ensure_doy_hour(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    if "year" not in df.columns:
        df["year"] = df[time_col].dt.year
    if "doy" not in df.columns:
        df["doy"] = df[time_col].dt.dayofyear
    if "hour" not in df.columns:
        df["hour"] = df[time_col].dt.hour
    return df


def plot_doy_hour(
    df: pd.DataFrame,
    out_path: str | Path | None = None,
    size_col: str = "area_km2",
    size_scale: float = 1e5,
    one_axes_per_year: bool = True,
    dpi: int = 150,
):
    """Scatter DOY × hora UTC, tamanho/cor proporcional a ``size_col``.

    Com ``one_axes_per_year=True`` empilha um subplot por ano; com
    ``False`` sobrepõe todos os anos no mesmo eixo.
    """
    import matplotlib.pyplot as plt

    df = _ensure_doy_hour(df)
    years = sorted(df["year"].dropna().unique().astype(int))
    if not years:
        logger.warning("Sem anos para plotar.")
        return None

    if one_axes_per_year:
        n = len(years)
        fig, axes = plt.subplots(
            n, 1, figsize=(10, 3.2 * n), sharex=True, sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for ax, yr in zip(axes, years):
            d = df[df["year"] == yr]
            sizes = np.clip(d[size_col] / size_scale, 5, 60) if size_col in d else 10
            ax.scatter(d["doy"], d["hour"], s=sizes, c=sizes, alpha=0.6)
            ax.set_title(str(yr))
            ax.grid(True, linestyle=":", linewidth=0.8)
        for ax in axes:
            ax.set_xlim(1, 366)
            ax.set_ylim(-0.5, 23.5)
            ax.set_yticks(range(0, 24, 2))
        axes[0].set_ylabel("Hour of Day")
        axes[-1].set_xlabel("Day of Year")
    else:
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=dpi)
        sizes = df[size_col] / size_scale if size_col in df else 10
        ax.scatter(df["doy"], df["hour"], s=sizes, c=sizes, alpha=0.6)
        ax.set_xlim(1, 366)
        ax.set_ylim(-0.5, 23.5)
        ax.set_yticks(range(0, 24, 2))
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Hour of Day")
        ax.grid(True, linestyle=":", linewidth=0.8)

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info("Figura salva: %s", out_path)
    return fig


def plot_runs_by_year(
    df_cont: pd.DataFrame,
    out_dir: str | Path,
    min_hours: float = 16.0,
    area_label: str = "Area > 200k km²",
    filename_prefix: str = "doy_hour_runs",
    palette: str = "tab10",
    n_colors: int = 10,
    dpi: int = 300,
) -> list[Path]:
    """Um PNG por ano: barras verticais coloridas ciclicamente por ``run_id``.

    Reproduz o plot do ``bigpicture.py`` original (marker ``"|"``,
    colormap cíclico, grade minor/major).
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cont = _ensure_doy_hour(df_cont)
    years = sorted(df_cont["year"].dropna().unique().astype(int))
    saved: list[Path] = []

    cmap = mpl.colormaps.get_cmap(palette)
    boundaries = np.arange(-0.5, n_colors + 0.5, 1)
    norm = mpl.colors.BoundaryNorm(boundaries, n_colors)

    for yr in years:
        d = df_cont[df_cont["year"] == yr].sort_values("datetime").copy()
        if d.empty:
            continue
        codes, _ = pd.factorize(d["run_id"])
        color_idx = np.where(codes >= 0, codes % n_colors, 0)

        fig, ax = plt.subplots(figsize=(10, 2.5), dpi=dpi)
        ax.scatter(
            d["doy"], d["hour"],
            s=12, c=color_idx, marker="|",
            cmap=cmap, norm=norm, alpha=1.0,
        )
        ax.set_title("Mesoscale Convective Systems over South America",
                     loc="left", fontweight="bold")
        ax.set_title(f"[{area_label} | Duration > {min_hours:g} h]", loc="right")
        ax.set_xlim(1, 366)
        ax.set_ylim(-1, 24)
        ax.set_yticks(range(0, 23, 6))
        ax.set_xlabel(f"{yr}, day of year")
        ax.set_ylabel("Time, UT")
        ax.xaxis.set_major_locator(MultipleLocator(15))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, which="major", linestyle=":", linewidth=0.8)
        ax.grid(True, which="minor", linestyle="-", linewidth=0.2, alpha=0.5)
        fig.tight_layout()

        out_path = out_dir / f"{filename_prefix}_{yr}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        logger.info("Salvo: %s", out_path)

    return saved


def plot_multicluster_overlay(
    df_full: pd.DataFrame,
    out_path: str | Path | None = None,
    size_col: str = "area_km2",
    size_scale: float = 1e5,
    dpi: int = 150,
):
    """Subplots empilhados (um por ano); sobrepõe ``×`` nos timestamps
    com 2+ clusters simultâneos.

    Requer o DataFrame **completo** (antes de ``make_unique_by_time``) —
    caso contrário não há multicluster.
    """
    import matplotlib.pyplot as plt

    df_full = _ensure_doy_hour(df_full)
    multi_dt = (
        df_full.groupby("datetime").size()
               .loc[lambda s: s > 1].index
    )

    years = sorted(df_full["year"].dropna().unique().astype(int))
    if not years:
        return None

    n = len(years)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, yr in zip(axes, years):
        d = df_full[df_full["year"] == yr]
        sizes = d[size_col] / size_scale if size_col in d else 10
        ax.scatter(d["doy"], d["hour"], s=sizes, c=sizes, cmap="rainbow", alpha=0.3)

        d_multi = d[d["datetime"].isin(multi_dt)]
        if not d_multi.empty:
            ax.scatter(
                d_multi["doy"], d_multi["hour"],
                s=5, facecolors="black", linewidths=0.5, marker="x", zorder=3,
            )
        ax.set_title(str(yr))
        ax.grid(True, linestyle=":", linewidth=0.8)

    for ax in axes:
        ax.set_xlim(1, 366)
        ax.set_ylim(-0.5, 23.5)
        ax.set_yticks(range(0, 24, 2))
    axes[0].set_ylabel("Hour of Day")
    axes[-1].set_xlabel("Day of Year")
    axes[0].scatter([], [], s=25, c="black", marker="x",
                    label="≥2 clusters at same time")
    axes[0].legend(loc="upper right", frameon=True)

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info("Figura salva: %s", out_path)
    return fig
