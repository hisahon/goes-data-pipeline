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
        │ track_multiple_clusters()   (ou make_unique_by_time para desempate simples)
        ▼
    parquet com track_id (todos os clusters preservados)
        │ filter_continuous_runs()
        ▼
    séries contínuas → plots DOY×hora → tracker.py

Interface pública
-----------------
* :func:`reports2df`               — lê uma árvore de ``reports/*.txt``.
* :func:`make_unique_by_time`      — mantém um cluster por timestamp.
* :func:`track_multiple_clusters`  — rastreia sistemas espaço-temporais.
* :func:`filter_continuous_runs`   — segmenta por gap temporal.
* :func:`plot_doy_hour`            — scatter DOY × hora UTC por ano.
* :func:`plot_runs_by_year`        — barras verticais coloridas por track.
* :func:`plot_multicluster_overlay`— destaca timestamps com 2+ tracks.

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
    """Varre ``root_dir`` recursivamente e concatena os relatórios de detecção."""
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
# 2. Desempate por timestamp (alternativa simples ao tracking)
# ---------------------------------------------------------------------------
def make_unique_by_time(
    df: pd.DataFrame,
    time_col: str = "datetime",
    size_col: str = "area_km2",
) -> pd.DataFrame:
    """Para cada timestamp, mantém apenas a linha com o maior ``size_col``."""
    if df.empty:
        return df.copy()
    if size_col not in df.columns:
        raise KeyError(f"Coluna {size_col!r} ausente.")
    idx = df.groupby(time_col)[size_col].idxmax()
    return df.loc[idx].sort_values(time_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2.1 Rastreamento espaço-temporal de múltiplos clusters
# ---------------------------------------------------------------------------
def haversine_dist(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def track_multiple_clusters(
    df: pd.DataFrame,
    max_speed_kmh: float = 120.0,
    max_gap_minutes: float = 60.0,
    time_col: str = "datetime",
) -> pd.DataFrame:
    """Rastreia múltiplos clusters independentes no espaço-tempo.

    Atribui uma coluna ``track_id`` única para cada sistema contínuo.
    NÃO descarta linhas — preserva todos os clusters, apenas adiciona o
    ``track_id`` que conecta observações sucessivas do mesmo sistema.

    Parameters
    ----------
    df
        DataFrame com colunas ``datetime``, ``centroid_lat``, ``centroid_lon``.
    max_speed_kmh
        Velocidade máxima plausível de deslocamento do sistema. Usada como
        limite dinâmico: distância_max = max_speed_kmh × Δt_horas.
    max_gap_minutes
        Gap temporal máximo para manter um track "ativo". Acima disso o
        track é encerrado e novos clusters viram novos tracks.
    """
    if df.empty:
        return df.copy()

    logger.info("Tracking multiple clusters spatially and temporally...")
    # Reset index to guarantee uniqueness (df.at requires unique labels).
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df["track_id"] = -1
    next_track_id = 1

    # Last known state of active tracks
    active_tracks: dict[int, dict] = {}

    timestamps = df[time_col].unique()

    # Precompute index groups by timestamp to avoid O(N*M) scan
    groups = df.groupby(time_col, sort=False).indices

    for t in timestamps:
        current_idx = groups[t]
        current_lats = df.loc[current_idx, "centroid_lat"].to_numpy()
        current_lons = df.loc[current_idx, "centroid_lon"].to_numpy()

        # 1. Drop expired tracks
        current_time = pd.Timestamp(t)
        active_tracks = {
            tid: data for tid, data in active_tracks.items()
            if (current_time - data['time']).total_seconds() / 60.0 <= max_gap_minutes
        }

        # 2. No active tracks -> all current clusters get new IDs
        if not active_tracks:
            for i, idx in enumerate(current_idx):
                df.at[idx, "track_id"] = next_track_id
                active_tracks[next_track_id] = {
                    'lat': current_lats[i],
                    'lon': current_lons[i],
                    'time': current_time,
                }
                next_track_id += 1
            continue

        # 3. Greedy distance-based matching
        track_ids = list(active_tracks.keys())
        track_lats = np.array([active_tracks[tid]['lat'] for tid in track_ids])
        track_lons = np.array([active_tracks[tid]['lon'] for tid in track_ids])
        track_times = [active_tracks[tid]['time'] for tid in track_ids]

        dist_matrix = np.zeros((len(track_ids), len(current_idx)))
        time_deltas_hours = np.zeros(len(track_ids))

        for i, tid in enumerate(track_ids):
            dt_hours = (current_time - track_times[i]).total_seconds() / 3600.0
            time_deltas_hours[i] = dt_hours
            for j in range(len(current_idx)):
                dist_matrix[i, j] = haversine_dist(
                    track_lats[i], track_lons[i],
                    current_lats[j], current_lons[j],
                )

        thresholds = (time_deltas_hours * max_speed_kmh).reshape(-1, 1)
        valid_matches = dist_matrix <= thresholds

        assigned_current: set[int] = set()
        assigned_tracks: set[int] = set()

        flat_indices = np.argsort(dist_matrix, axis=None)
        for flat_idx in flat_indices:
            t_idx, c_idx = np.unravel_index(flat_idx, dist_matrix.shape)
            if c_idx in assigned_current or t_idx in assigned_tracks:
                continue
            if not valid_matches[t_idx, c_idx]:
                continue
            tid = track_ids[t_idx]
            df.at[current_idx[c_idx], "track_id"] = tid
            active_tracks[tid] = {
                'lat': current_lats[c_idx],
                'lon': current_lons[c_idx],
                'time': current_time,
            }
            assigned_current.add(int(c_idx))
            assigned_tracks.add(int(t_idx))

        # 4. Unmatched current clusters -> new tracks
        for j, idx in enumerate(current_idx):
            if j not in assigned_current:
                df.at[idx, "track_id"] = next_track_id
                active_tracks[next_track_id] = {
                    'lat': current_lats[j],
                    'lon': current_lons[j],
                    'time': current_time,
                }
                next_track_id += 1

    n_tracks = df["track_id"].nunique()
    logger.info("Rastreados %d sistemas em %d observações.", n_tracks, len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Segmentação em runs contínuos
# ---------------------------------------------------------------------------
def filter_continuous_runs(
    df: pd.DataFrame,
    min_hours: float = 1.0,
    gap_minutes: float = 60.0,  # mantido para compatibilidade de assinatura
    time_col: str = "datetime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtra tracks com duração mínima.

    Assume que o DataFrame de entrada já possui ``track_id`` gerado por
    :func:`track_multiple_clusters`. Copia ``track_id`` para ``run_id`` por
    compatibilidade com as rotinas de plot.
    """
    if df.empty:
        return (
            df.copy(),
            pd.DataFrame(columns=["run_id", "run_start", "run_end", "duration_min"]),
        )

    df = df.copy()
    if "track_id" in df.columns:
        df["run_id"] = df["track_id"]
    elif "run_id" not in df.columns:
        raise KeyError(
            "Coluna 'track_id' ausente. Execute track_multiple_clusters() primeiro."
        )

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
        "Tracks: %d detectados | %d válidos (>= %.1f h) | %d registros retidos",
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


def _track_colors(values: pd.Series, palette: str, n_colors: int):
    """Converte uma série qualquer (ex.: track_id) em índices de cor ciclícos
    mais cmap/norm prontos para scatter."""
    import matplotlib as mpl
    codes, _ = pd.factorize(values)
    color_idx = np.where(codes >= 0, codes % n_colors, 0)
    cmap = mpl.colormaps.get_cmap(palette)
    boundaries = np.arange(-0.5, n_colors + 0.5, 1)
    norm = mpl.colors.BoundaryNorm(boundaries, n_colors)
    return color_idx, cmap, norm


def plot_doy_hour(
    df: pd.DataFrame,
    out_path: str | Path | None = None,
    size_col: str = "area_km2",
    size_scale: float = 1e5,
    one_axes_per_year: bool = True,
    dpi: int = 150,
    color_by: str | None = "track_id",
    palette: str = "tab20",
    n_colors: int = 20,
):
    """Scatter DOY × hora UTC, um subplot por ano.

    Parameters
    ----------
    color_by
        Se for nome de coluna existente em ``df`` (default ``"track_id"``),
        colore cada ponto segundo essa coluna usando ``palette`` cíclica.
        Útil para visualizar como os pontos se agrupam em sistemas rastreados.
        Passe ``None`` para voltar ao esquema antigo (cor proporcional ao
        ``size_col``).
    """
    import matplotlib.pyplot as plt

    df = _ensure_doy_hour(df)
    years = sorted(df["year"].dropna().unique().astype(int))
    if not years:
        logger.warning("Sem anos para plotar.")
        return None

    use_track_color = (color_by is not None) and (color_by in df.columns)

    def _scatter(ax, d):
        if size_col in d.columns:
            sizes = np.clip(d[size_col] / size_scale, 5, 60).to_numpy()
        else:
            sizes = np.full(len(d), 10)

        if use_track_color:
            color_idx, cmap, norm = _track_colors(d[color_by], palette, n_colors)
            ax.scatter(
                d["doy"], d["hour"], s=sizes,
                c=color_idx, cmap=cmap, norm=norm, alpha=0.75,
            )
        else:
            ax.scatter(d["doy"], d["hour"], s=sizes, c=sizes, alpha=0.6)

    if one_axes_per_year:
        n = len(years)
        fig, axes = plt.subplots(
            n, 1, figsize=(10, 3.2 * n), sharex=True, sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for ax, yr in zip(axes, years):
            d = df[df["year"] == yr]
            _scatter(ax, d)
            ax.set_title(str(yr))
            ax.grid(True, linestyle=":", linewidth=0.8)
        for ax in axes:
            ax.set_xlim(1, 366)
            ax.set_ylim(-0.5, 23.5)
            ax.set_yticks(range(0, 24, 2))
        axes[0].set_ylabel("Hour of Day")
        axes[-1].set_xlabel("Day of Year")
        if use_track_color:
            n_tracks = df[color_by].nunique()
            axes[0].text(
                0.99, 0.95, f"{n_tracks} tracks (color cycles every {n_colors})",
                transform=axes[0].transAxes, ha="right", va="top",
                fontsize=8, color="gray",
            )
    else:
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=dpi)
        _scatter(ax, df)
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
    palette: str = "tab20",
    n_colors: int = 20,
    dpi: int = 300,
    marker_size: int = 14,
) -> list[Path]:
    """Um PNG por ano: marcadores verticais coloridos por ``run_id``.

    As cores distinguem os diferentes sistemas (tracks) que satisfazem o
    critério de duração mínima. Com múltiplos sistemas simultâneos agora
    possíveis no mesmo (doy, hour), o default aumentou para ``tab20``/20
    cores para melhor separação visual.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cont = _ensure_doy_hour(df_cont)
    years = sorted(df_cont["year"].dropna().unique().astype(int))
    saved: list[Path] = []

    for yr in years:
        d = df_cont[df_cont["year"] == yr].sort_values("datetime").copy()
        if d.empty:
            continue

        color_idx, cmap, norm = _track_colors(d["run_id"], palette, n_colors)

        fig, ax = plt.subplots(figsize=(10, 2.5), dpi=dpi)
        ax.scatter(
            d["doy"], d["hour"],
            s=marker_size, c=color_idx, marker="|",
            cmap=cmap, norm=norm, alpha=1.0,
        )

        n_tracks_yr = d["run_id"].nunique()
        ax.set_title(
            "Mesoscale Convective Systems over South America",
            loc="left", fontweight="bold",
        )
        ax.set_title(
            f"[{area_label} | Duration > {min_hours:g} h | {n_tracks_yr} systems]",
            loc="right",
        )
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
    color_by: str | None = "track_id",
    palette: str = "tab20",
    n_colors: int = 20,
):
    """Subplots empilhados por ano; destaca com ``×`` os timestamps em que
    2+ sistemas **independentes** (track_ids distintos) coexistem.

    O scatter de fundo agora também reflete ``track_id`` (se presente), de
    modo que cada sistema aparece com uma cor própria — assim fica visível
    não só *onde* há multicluster, mas *quais* sistemas estão envolvidos.

    Se a coluna ``track_id`` não estiver presente, cai no comportamento
    antigo: detecta multicluster por simples contagem de linhas por
    timestamp e colore por tamanho.
    """
    import matplotlib.pyplot as plt

    df_full = _ensure_doy_hour(df_full)

    # Define multicluster timestamps usando semântica de tracks, se disponível.
    if "track_id" in df_full.columns:
        g = df_full.groupby("datetime")["track_id"].nunique()
        legend_label = "≥2 independent systems"
    else:
        g = df_full.groupby("datetime").size()
        legend_label = "≥2 clusters at same time"
    multi_dt = g.loc[g > 1].index

    years = sorted(df_full["year"].dropna().unique().astype(int))
    if not years:
        return None

    use_track_color = (color_by is not None) and (color_by in df_full.columns)

    n = len(years)
    fig, axes = plt.subplots(
        n, 1, figsize=(10, 2.5 * n), sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, yr in zip(axes, years):
        d = df_full[df_full["year"] == yr]

        if size_col in d.columns:
            sizes = np.clip(d[size_col] / size_scale, 5, 60).to_numpy()
        else:
            sizes = np.full(len(d), 10)

        if use_track_color:
            color_idx, cmap, norm = _track_colors(d[color_by], palette, n_colors)
            ax.scatter(
                d["doy"], d["hour"], s=sizes,
                c=color_idx, cmap=cmap, norm=norm, alpha=0.4,
            )
        else:
            ax.scatter(
                d["doy"], d["hour"], s=sizes,
                c=sizes, cmap="rainbow", alpha=0.3,
            )

        d_multi = d[d["datetime"].isin(multi_dt)]
        if not d_multi.empty:
            ax.scatter(
                d_multi["doy"], d_multi["hour"],
                s=20, c="black", linewidths=1.0, marker="x", zorder=3,
            )
        ax.set_title(str(yr))
        ax.grid(True, linestyle=":", linewidth=0.8)

    for ax in axes:
        ax.set_xlim(1, 366)
        ax.set_ylim(-0.5, 23.5)
        ax.set_yticks(range(0, 24, 2))
    axes[0].set_ylabel("Hour of Day")
    axes[-1].set_xlabel("Day of Year")
    axes[0].scatter(
        [], [], s=30, c="black", marker="x", label=legend_label,
    )
    axes[0].legend(loc="upper right", frameon=True)

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        logger.info("Figura salva: %s", out_path)
    return fig