# -*- coding: utf-8 -*-
"""
tracker.py
==========

Rastreamento temporal de clusters convectivos (MCS) detectados em imagens
GOES-C13. Consome o parquet agregado dos relatórios do ``detector.py`` e
atribui a cada detecção um ``track_id``, formando trajetórias.

Interface pública:

* :func:`haversine_km`         — distância em km (vetorizada).
* :func:`track_centroids`      — algoritmo de tracking.
* :func:`plot_yearly_tracks`   — mapa anual de trajetórias (Cartopy).
* :data:`DEFAULT_OPTIONS`      — parâmetros default do tracking.

Criado por Rogério Hisashi Honda (hisashi.brase@gmail.com).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distância haversine em km. Aceita escalares ou arrays ``numpy``.

    Todos os argumentos estão em graus. Broadcasting do ``numpy`` vale:
    passe ``lat1[:, None]`` e ``lat2[None, :]`` para obter a matriz de
    distâncias.
    """
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Opções
# ---------------------------------------------------------------------------
@dataclass
class TrackerOptions:
    """Parâmetros do algoritmo de tracking.

    Attributes
    ----------
    time_col, lat_col, lon_col
        Nomes das colunas no DataFrame de entrada.
    max_time_gap
        Gap temporal máximo admitido entre observações consecutivas
        de uma mesma trilha. O scan nominal do GOES-C13 é 10 min;
        20 min tolera um frame perdido.
    max_km
        Deslocamento máximo admitido entre observações consecutivas.
    max_speed_kmh
        Se informado, filtra pares com velocidade acima deste limiar
        (útil para rejeitar "pulos" improváveis entre detecções).
    min_points
        Trilhas com menos que este número de observações são descartadas.
    assignment
        ``"greedy"`` (nearest-neighbor por custo crescente) ou
        ``"hungarian"`` (ótimo por timestep, via
        ``scipy.optimize.linear_sum_assignment``).
    min_area_km2
        Se informado, descarta detecções com ``area_km2`` abaixo do
        limiar antes do tracking.
    bbox
        ``(lat_min, lat_max, lon_min, lon_max)``. Se informado, recorta
        detecções fora da caixa antes do tracking.
    """

    time_col: str = "datetime"
    lat_col: str = "centroid_lat"
    lon_col: str = "centroid_lon"
    max_time_gap: pd.Timedelta = field(default_factory=lambda: pd.Timedelta("20min"))
    max_km: float = 35.0
    max_speed_kmh: Optional[float] = 120.0
    min_points: int = 60
    assignment: str = "greedy"
    min_area_km2: Optional[float] = None
    bbox: Optional[tuple[float, float, float, float]] = None


DEFAULT_OPTIONS = TrackerOptions()


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
def _prefilter(df: pd.DataFrame, opts: TrackerOptions) -> pd.DataFrame:
    """Aplica filtros geográficos e de área antes do tracking."""
    out = df
    if opts.min_area_km2 is not None and "area_km2" in out.columns:
        out = out[out["area_km2"] >= opts.min_area_km2]
    if opts.bbox is not None:
        lat_min, lat_max, lon_min, lon_max = opts.bbox
        out = out[
            (out[opts.lat_col] >= lat_min)
            & (out[opts.lat_col] <= lat_max)
            & (out[opts.lon_col] >= lon_min)
            & (out[opts.lon_col] <= lon_max)
        ]
    return out


def _assign_greedy(cost: np.ndarray) -> np.ndarray:
    """Atribuição gulosa: percorre pares em ordem crescente de custo,
    escolhendo sem reposição. Retorna vetor ``assigned[i] = j`` com a
    trilha ativa ``j`` atribuída à detecção ``i`` (ou ``-1``)."""
    n_det, n_act = cost.shape
    assigned = np.full(n_det, -1, dtype=np.int64)
    if n_act == 0:
        return assigned
    used_act = np.zeros(n_act, dtype=bool)
    order = np.argsort(cost, axis=None, kind="stable")
    for k in order:
        r, c = divmod(int(k), n_act)
        if not np.isfinite(cost[r, c]):
            break  # todos os seguintes também são inf
        if assigned[r] == -1 and not used_act[c]:
            assigned[r] = c
            used_act[c] = True
    return assigned


def _assign_hungarian(cost: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Atribuição ótima por timestep usando Hungarian."""
    from scipy.optimize import linear_sum_assignment

    n_det, n_act = cost.shape
    assigned = np.full(n_det, -1, dtype=np.int64)
    if n_act == 0:
        return assigned
    # Hungarian não lida com inf: substitui por valor "grande".
    big = float(np.nanmax(cost[np.isfinite(cost)], initial=1.0)) * 1e3 + 1.0
    c_safe = np.where(np.isfinite(cost), cost, big)
    rows, cols = linear_sum_assignment(c_safe)
    for r, c in zip(rows, cols):
        if eligible[r, c]:
            assigned[r] = c
    return assigned


def track_centroids(
    df: pd.DataFrame,
    opts: TrackerOptions | None = None,
) -> pd.DataFrame:
    """Atribui ``track_id`` a centróides (lat, lon) ao longo do tempo.

    Processa as detecções frame-a-frame (agrupadas por ``time_col``),
    vetoriza o cálculo de distâncias sobre o conjunto de trilhas ativas e
    escolhe a atribuição por modo guloso ou Hungarian.

    Parameters
    ----------
    df
        DataFrame com pelo menos as colunas ``time_col``, ``lat_col`` e
        ``lon_col``. Outras colunas são preservadas.
    opts
        Instância de :class:`TrackerOptions`. ``None`` usa defaults.

    Returns
    -------
    DataFrame
        Cópia de ``df`` com a coluna ``track_id`` adicionada e trilhas
        com menos de ``min_points`` observações removidas.
    """
    if opts is None:
        opts = DEFAULT_OPTIONS

    gdf = _prefilter(df, opts).copy()
    if gdf.empty:
        logger.warning("Nenhuma detecção após pré-filtros; retornando vazio.")
        gdf["track_id"] = np.array([], dtype=np.int64)
        return gdf

    gdf[opts.time_col] = pd.to_datetime(gdf[opts.time_col])
    gdf = gdf.sort_values(opts.time_col, kind="stable").reset_index(drop=True)

    track_ids = np.full(len(gdf), -1, dtype=np.int64)

    # Trilhas ativas: arrays paralelos (lista -> array a cada timestep)
    act_ids: list[int] = []
    act_lats: list[float] = []
    act_lons: list[float] = []
    act_times: list[pd.Timestamp] = []
    next_id = 0

    groups = list(gdf.groupby(opts.time_col, sort=True))
    logger.info(
        "Tracking: %d detecções em %d timestamps (assignment=%s)",
        len(gdf), len(groups), opts.assignment,
    )

    for t, frame in groups:
        # --- poda trilhas ativas que excederam max_time_gap ---
        if act_times:
            ages = pd.to_datetime(act_times).values - np.datetime64(t)
            alive = (-ages) <= np.timedelta64(opts.max_time_gap)
            if not alive.all():
                act_ids = [v for v, a in zip(act_ids, alive) if a]
                act_lats = [v for v, a in zip(act_lats, alive) if a]
                act_lons = [v for v, a in zip(act_lons, alive) if a]
                act_times = [v for v, a in zip(act_times, alive) if a]

        idx = frame.index.to_numpy()
        lats = frame[opts.lat_col].to_numpy(dtype=float)
        lons = frame[opts.lon_col].to_numpy(dtype=float)
        valid = np.isfinite(lats) & np.isfinite(lons)
        idx, lats, lons = idx[valid], lats[valid], lons[valid]
        if len(idx) == 0:
            continue

        if not act_ids:
            # sem trilhas ativas: todas viram novas trilhas
            for k in range(len(idx)):
                track_ids[idx[k]] = next_id
                act_ids.append(next_id)
                act_lats.append(float(lats[k]))
                act_lons.append(float(lons[k]))
                act_times.append(t)
                next_id += 1
            continue

        A_lats = np.asarray(act_lats)
        A_lons = np.asarray(act_lons)
        A_times = pd.to_datetime(act_times).values

        # matriz [n_det, n_act] de distâncias haversine
        D = haversine_km(
            lats[:, None], lons[:, None],
            A_lats[None, :], A_lons[None, :],
        )

        dt_hours = (np.datetime64(t) - A_times) / np.timedelta64(1, "h")
        eligible = D <= opts.max_km
        if opts.max_speed_kmh is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                speed = D / np.where(dt_hours > 0, dt_hours, np.nan)
            eligible &= np.isfinite(speed) & (speed <= opts.max_speed_kmh)

        cost = np.where(eligible, D, np.inf)

        if opts.assignment == "hungarian":
            assigned = _assign_hungarian(cost, eligible)
        elif opts.assignment == "greedy":
            assigned = _assign_greedy(cost)
        else:
            raise ValueError(f"assignment desconhecido: {opts.assignment!r}")

        # aplica as atribuições e cria trilhas novas para o que sobrou
        for k in range(len(idx)):
            c = int(assigned[k])
            if c >= 0:
                tid = act_ids[c]
                track_ids[idx[k]] = tid
                act_lats[c] = float(lats[k])
                act_lons[c] = float(lons[k])
                act_times[c] = t
            else:
                track_ids[idx[k]] = next_id
                act_ids.append(next_id)
                act_lats.append(float(lats[k]))
                act_lons.append(float(lons[k]))
                act_times.append(t)
                next_id += 1

    gdf["track_id"] = track_ids

    # filtra trilhas curtas
    sizes = gdf.groupby("track_id", sort=False).size()
    keep = sizes[sizes >= opts.min_points].index
    n_before = sizes.size
    gdf = gdf[gdf["track_id"].isin(keep)].reset_index(drop=True)
    logger.info(
        "Trilhas: %d criadas, %d retidas (min_points=%d)",
        n_before, keep.size, opts.min_points,
    )
    return gdf


# ---------------------------------------------------------------------------
# Sumário por trilha (útil para exports de publicação)
# ---------------------------------------------------------------------------
def summarize_tracks(
    df_tracked: pd.DataFrame,
    time_col: str = "datetime",
    lat_col: str = "centroid_lat",
    lon_col: str = "centroid_lon",
) -> pd.DataFrame:
    """Retorna uma tabela com uma linha por trilha (início/fim e tamanho)."""
    rows = []
    for tid, g in df_tracked.groupby("track_id", sort=False):
        if tid == -1:
            continue
        g = g.sort_values(time_col)
        rows.append({
            "track_id": tid,
            "t0": g[time_col].iloc[0],
            "t1": g[time_col].iloc[-1],
            "lat0": g[lat_col].iloc[0],
            "lon0": g[lon_col].iloc[0],
            "lat1": g[lat_col].iloc[-1],
            "lon1": g[lon_col].iloc[-1],
            "n_points": len(g),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotagem (importação tardia de Cartopy para não forçar dependência)
# ---------------------------------------------------------------------------
def plot_yearly_tracks(
    df_tracked: pd.DataFrame,
    year: int,
    out_path: str | Path,
    extent: tuple[float, float, float, float] = (-67.0, -42.0, -40.0, -15.0),
    time_col: str = "datetime",
    lat_col: str = "centroid_lat",
    lon_col: str = "centroid_lon",
    point_size: float = 0.1,
    dpi: int = 300,
) -> Path:
    """Gera o mapa anual de trajetórias com início (azul) e fim (vermelho).

    Parameters
    ----------
    df_tracked
        Saída de :func:`track_centroids`.
    year
        Ano a plotar (lido de ``df_tracked[time_col].dt.year``).
    out_path
        Arquivo PNG de saída.
    extent
        ``(lon_min, lon_max, lat_min, lat_max)`` para ``set_extent``.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_year = df_tracked[pd.to_datetime(df_tracked[time_col]).dt.year == year]
    if df_year.empty:
        logger.warning("Sem trilhas no ano %d; pulando plot.", year)
        return out_path

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(list(extent), crs=proj)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.6)

    gl = ax.gridlines(
        crs=proj, draw_labels=True, linewidth=0.3, linestyle=":", alpha=0.8,
    )
    gl.top_labels = False
    gl.right_labels = False

    def _ts(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%j [%H ut]")

    for tid, grp in df_year.groupby("track_id", sort=False):
        if tid == -1:
            continue
        g = grp.sort_values(time_col)
        lons = g[lon_col].to_numpy()
        lats = g[lat_col].to_numpy()
        ax.scatter(lons, lats, s=point_size, alpha=0.9, transform=proj, zorder=5)

        t0 = g[time_col].iloc[0]
        t1 = g[time_col].iloc[-1]
        ax.scatter([lons[0]], [lats[0]], s=5, c="b", transform=proj, zorder=6)
        ax.scatter([lons[-1]], [lats[-1]], s=5, c="r", transform=proj, zorder=6)
        ax.text(lons[0], lats[0], _ts(t0), transform=proj, fontsize=5,
                ha="center", va="bottom", zorder=7)
        ax.text(lons[-1], lats[-1], _ts(t1), transform=proj, fontsize=5,
                ha="center", va="top", zorder=7)

    ax.set_title(
        f"Cold Cores track | start=blue end=red | {year}",
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Mapa %d salvo em %s", year, out_path)
    return out_path


def plot_all_years(
    df_tracked: pd.DataFrame,
    out_dir: str | Path,
    years: Iterable[int] | None = None,
    **kwargs,
) -> list[Path]:
    """Gera um PNG por ano em ``out_dir/SA_tracked_<YEAR>.png``."""
    out_dir = Path(out_dir)
    if years is None:
        years = sorted(pd.to_datetime(df_tracked["datetime"]).dt.year.dropna().unique())
    paths = []
    for y in years:
        paths.append(plot_yearly_tracks(
            df_tracked, int(y), out_dir / f"SA_tracked_{int(y)}.png", **kwargs,
        ))
    return paths
