# -*- coding: utf-8 -*-
"""
tracker_terminal.py
===================

Wrapper CLI para o rastreamento de clusters convectivos. Fino, no estilo
de ``visual_terminal.py``: toda a lógica vive em ``tracker.py``.

Exemplos
--------

Pipeline completo (roda tracking + salva parquet + gera mapas anuais)::

    python tracker_terminal.py --server \\
        --input database_numpy/c13_2018-2025_SouthAmerica.parquet \\
        --output tracks/c13_2018-2025_SA_Tracked.parquet \\
        --min-area 600000 --max-km 35 --min-points 60 --assignment hungarian \\
        --bbox -40 -12 -67 -42 --plot

Apenas replotar a partir de um parquet já rastreado::

    python tracker_terminal.py --from-tracked tracks/c13_2018-2025_SA_Tracked.parquet \\
        --plot --bbox -40 -12 -67 -42
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from tracker import (
    TrackerOptions,
    track_centroids,
    summarize_tracks,
    plot_all_years,
)

logger = logging.getLogger("tracker")


# ---------------------------------------------------------------------------
# Layout de pastas por ambiente (mesmo padrão de visual.py / download.py)
# ---------------------------------------------------------------------------
def defining_folders(server: bool) -> dict[str, Path]:
    """Escolhe raízes de dados e saídas de acordo com o ambiente."""
    home = Path.home()
    if os.name == "nt":
        root = Path("D:/banco_dados/cptec")
    elif server:
        root = home
    else:
        root = home / "database" / "cptec"

    return {
        "root": root,
        "database": root / "database_numpy",
        "tracks": root / "tracks",
        "figures": root / "figures_tracks",
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    folders = defining_folders(args.server)
    for p in (folders["tracks"], folders["figures"]):
        p.mkdir(parents=True, exist_ok=True)

    # --- resolve caminho de saída com output_key opcional ---
    out_path = Path(args.output) if args.output else (
        folders["tracks"] / f"{args.output_prefix}{args.output_key}.parquet"
    )

    # --- Caminho 1: apenas replotar de um parquet tracked pronto ---
    if args.from_tracked:
        tracked_in = Path(args.from_tracked)
        logger.info("Lendo tracks já calculados: %s", tracked_in)
        df_tracked = pd.read_parquet(tracked_in)
    else:
        # --- Caminho 2: rodar tracking (ou reusar se já existir) ---
        if out_path.exists() and not args.force:
            logger.info("Já existe %s — use --force para recalcular.", out_path)
            df_tracked = pd.read_parquet(out_path)
        else:
            if not args.input:
                logger.error("--input é obrigatório (a menos que use --from-tracked).")
                return 2
            in_path = Path(args.input)
            logger.info("Lendo detecções: %s", in_path)
            df = pd.read_parquet(in_path)

            opts = TrackerOptions(
                max_time_gap=pd.Timedelta(args.max_gap),
                max_km=args.max_km,
                max_speed_kmh=args.max_speed,
                min_points=args.min_points,
                assignment=args.assignment,
                min_area_km2=args.min_area,
                bbox=tuple(args.bbox) if args.bbox else None,
            )
            logger.info("Opções: %s", opts)

            df_tracked = track_centroids(df, opts)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_tracked.to_parquet(out_path, index=False)
            logger.info("Salvo: %s (%d linhas)", out_path, len(df_tracked))

            # sumário (um CSV por trilha) ao lado do parquet
            summary = summarize_tracks(df_tracked)
            sum_path = out_path.with_name(out_path.stem + "_summary.csv")
            summary.to_csv(sum_path, index=False)
            logger.info("Sumário: %s (%d trilhas)", sum_path, len(summary))

    # --- Plot opcional ---
    if args.plot:
        fig_dir = folders["figures"] / f"tracked{args.output_key}"
        extent = (-67.0, -42.0, -40.0, -15.0)
        if args.bbox:
            lat_min, lat_max, lon_min, lon_max = args.bbox
            extent = (lon_min, lon_max, lat_min, lat_max)
        paths = plot_all_years(
            df_tracked, fig_dir, extent=extent,
            filename_prefix=args.plot_prefix,
        )
        logger.info("%d mapas gerados em %s", len(paths), fig_dir)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rastreamento temporal de clusters convectivos (MCS) "
                    "a partir do parquet agregado de detecções GOES-C13."
    )
    io = p.add_argument_group("entrada / saída")
    io.add_argument("--input", type=str,
                    help="Parquet de entrada com detecções agregadas.")
    io.add_argument("--output", type=str,
                    help="Parquet de saída (padrão: tracks/c13_tracked<KEY>.parquet).")
    io.add_argument("--from-tracked", type=str,
                    help="Pula o tracking e usa um parquet já rastreado "
                         "(útil apenas para replotar).")
    io.add_argument("-o", "--output_key", type=str, default="",
                    help="Sufixo acrescentado às pastas de saída.")
    io.add_argument("--output-prefix", dest="output_prefix", type=str,
                    default="c13_tracked",
                    help="Prefixo do parquet de saída quando --output não é informado. "
                         "Default: c13_tracked.")
    io.add_argument("--plot-prefix", dest="plot_prefix", type=str,
                    default="SA_tracked",
                    help="Prefixo dos PNGs anuais gerados com --plot. "
                         "Default: SA_tracked.")
    io.add_argument("--server", action="store_true",
                    help="Usa layout de pastas do servidor.")
    io.add_argument("--force", action="store_true",
                    help="Reexecuta mesmo se o parquet de tracks já existir.")

    trk = p.add_argument_group("parâmetros de tracking")
    trk.add_argument("--max-gap", type=str, default="20min",
                     help="Gap temporal máximo (ex.: '20min'). Default: 20min.")
    trk.add_argument("--max-km", type=float, default=35.0,
                     help="Distância máxima entre obs consecutivas (km). Default: 35.")
    trk.add_argument("--max-speed", type=float, default=120.0,
                     help="Velocidade máxima (km/h). Default: 120. Use 0 para desativar.")
    trk.add_argument("--min-points", type=int, default=60,
                     help="Mínimo de observações por trilha. Default: 60.")
    trk.add_argument("--assignment", choices=("greedy", "hungarian"),
                     default="greedy",
                     help="Estratégia de atribuição por timestep. Default: greedy.")

    flt = p.add_argument_group("pré-filtros")
    flt.add_argument("--min-area", type=float, default=None,
                     help="Filtra detecções com area_km2 >= MIN_AREA.")
    flt.add_argument("--bbox", type=float, nargs=4,
                     metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                     help="Recorta detecções à caixa geográfica.")

    vis = p.add_argument_group("visualização")
    vis.add_argument("--plot", action="store_true",
                     help="Gera um PNG por ano com as trajetórias.")

    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Aumenta verbosidade (-v, -vv).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # --max-speed 0 => desliga o filtro
    if args.max_speed is not None and args.max_speed <= 0:
        args.max_speed = None

    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
