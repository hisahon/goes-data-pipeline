# -*- coding: utf-8 -*-
"""
aggregator_terminal.py
======================

Wrapper CLI para o estágio de agregação e visualização exploratória.
Fino, no estilo de ``visual_terminal.py`` e ``tracker_terminal.py``.

Subcomandos
-----------

* ``aggregate``  — varre ``reports/*.txt`` e gera o parquet agregado.
* ``unique``     — reduz para um cluster por timestamp (maior área).
* ``plot-runs``  — gera o PNG anual DOY × hora com runs contínuos.
* ``plot-multi`` — gera o overlay de timestamps multicluster.

Exemplos
--------

Pipeline padrão (roda os três primeiros estágios)::

    python aggregator_terminal.py aggregate --server \\
        --reports-dir reports_S \\
        --output merged/c13_2018-2025.parquet

    python aggregator_terminal.py unique \\
        --input  merged/c13_2018-2025.parquet \\
        --output merged/c13_2018-2025u.parquet

    python aggregator_terminal.py plot-runs \\
        --input  merged/c13_2018-2025u.parquet \\
        --out-dir results --min-hours 16 --gap-minutes 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from aggregator import (
    reports2df,
    make_unique_by_time,
    filter_continuous_runs,
    plot_runs_by_year,
    plot_multicluster_overlay,
)

logger = logging.getLogger("aggregator")


# ---------------------------------------------------------------------------
# Layout de pastas por ambiente (mesmo padrão dos outros terminais)
# ---------------------------------------------------------------------------
def defining_folders(server: bool) -> dict[str, Path]:
    home = Path.home()
    if os.name == "nt":
        root = Path("D:/banco_dados/cptec")
    elif server:
        root = home
    else:
        root = home / "database" / "cptec"
    return {
        "root": root,
        "reports": root / "reports",
        "merged": root / "merged",
        "figures": root / "figures_runs",
    }


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------
def cmd_aggregate(args: argparse.Namespace) -> int:
    folders = defining_folders(args.server)
    reports = Path(args.reports_dir) if args.reports_dir else folders["reports"]
    out = Path(args.output) if args.output else (
        folders["merged"] / f"c13_aggregated{args.output_key}.parquet"
    )

    if out.exists() and not args.force:
        logger.info("%s já existe — use --force para reescrever.", out)
        return 0

    df = reports2df(reports)
    if df.empty:
        logger.error("Nenhum relatório válido em %s.", reports)
        return 1

    if args.keep_cols:
        keep = [c for c in args.keep_cols if c in df.columns]
        df = df[keep]

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Salvo: %s (%d linhas)", out, len(df))
    return 0


def cmd_unique(args: argparse.Namespace) -> int:
    folders = defining_folders(args.server)
    in_path = Path(args.input) if args.input else (
        folders["merged"] / "c13_aggregated.parquet"
    )
    out = Path(args.output) if args.output else (
        in_path.with_name(in_path.stem + "u.parquet")
    )

    if out.exists() and not args.force:
        logger.info("%s já existe — use --force para reescrever.", out)
        return 0

    df = pd.read_parquet(in_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    logger.info("Lido: %s (%d linhas)", in_path, len(df))

    df_u = make_unique_by_time(df, size_col=args.size_col)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_u.to_parquet(out, index=False)
    logger.info("Salvo: %s (%d → %d linhas)", out, len(df), len(df_u))
    return 0


def cmd_plot_runs(args: argparse.Namespace) -> int:
    folders = defining_folders(args.server)
    in_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else folders["figures"]

    df = pd.read_parquet(in_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df_cont, _ = filter_continuous_runs(
        df, min_hours=args.min_hours, gap_minutes=args.gap_minutes,
    )
    if df_cont.empty:
        logger.warning("Nenhum run válido; nada a plotar.")
        return 0

    paths = plot_runs_by_year(
        df_cont, out_dir,
        min_hours=args.min_hours,
        area_label=args.area_label,
        filename_prefix=args.prefix,
        palette=args.palette,
        n_colors=args.n_colors,
        dpi=args.dpi,
    )
    logger.info("%d PNGs gerados em %s", len(paths), out_dir)
    return 0


def cmd_plot_multi(args: argparse.Namespace) -> int:
    folders = defining_folders(args.server)
    in_path = Path(args.input)
    out = Path(args.output) if args.output else (
        folders["figures"] / "doy_hour_multicluster.png"
    )
    df = pd.read_parquet(in_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    plot_multicluster_overlay(df, out_path=out, dpi=args.dpi)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agregação de reports + análises exploratórias DOY×hora.",
    )
    p.add_argument("--server", action="store_true",
                   help="Usa layout de pastas do servidor.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Verbosidade (-v, -vv).")

    sub = p.add_subparsers(dest="cmd", required=True)

    # -- aggregate --
    a = sub.add_parser("aggregate",
                       help="Varre reports/*.txt e gera o parquet agregado.")
    a.add_argument("--reports-dir", type=str,
                   help="Raiz da árvore de reports. Default: <root>/reports.")
    a.add_argument("--output", type=str,
                   help="Parquet de saída. Default: merged/c13_aggregated.parquet")
    a.add_argument("-o", "--output_key", type=str, default="",
                   help="Sufixo para o nome default do parquet.")
    a.add_argument("--keep-cols", nargs="*", default=None,
                   help="Colunas a preservar (default: todas).")
    a.add_argument("--force", action="store_true",
                   help="Sobrescreve mesmo se o parquet já existir.")
    a.set_defaults(func=cmd_aggregate)

    # -- unique --
    u = sub.add_parser("unique",
                       help="Reduz para um cluster por timestamp (maior área).")
    u.add_argument("--input",  type=str, help="Parquet de entrada.")
    u.add_argument("--output", type=str, help="Parquet de saída.")
    u.add_argument("--size-col", type=str, default="area_km2",
                   help="Coluna usada para escolher o 'vencedor'. Default: area_km2.")
    u.add_argument("--force", action="store_true")
    u.set_defaults(func=cmd_unique)

    # -- plot-runs --
    pr = sub.add_parser("plot-runs",
                        help="PNG anual DOY × hora com runs contínuos.")
    pr.add_argument("--input", type=str, required=True)
    pr.add_argument("--out-dir", type=str)
    pr.add_argument("--min-hours", type=float, default=16.0,
                    help="Duração mínima da sequência (h). Default: 16.")
    pr.add_argument("--gap-minutes", type=float, default=30.0,
                    help="Gap máximo entre observações (min). Default: 30.")
    pr.add_argument("--area-label", type=str, default="Area > 200k km²",
                    help="Texto à direita do título.")
    pr.add_argument("--prefix", type=str, default="doy_hour_runs",
                    help="Prefixo do nome do PNG. Default: doy_hour_runs.")
    pr.add_argument("--palette", type=str, default="tab10",
                    help="Colormap cíclico. Default: tab10.")
    pr.add_argument("--n-colors", type=int, default=10,
                    help="Nº de cores no colormap cíclico. Default: 10.")
    pr.add_argument("--dpi", type=int, default=300)
    pr.set_defaults(func=cmd_plot_runs)

    # -- plot-multi --
    pm = sub.add_parser("plot-multi",
                        help="Overlay de timestamps multicluster.")
    pm.add_argument("--input", type=str, required=True,
                    help="Parquet BRUTO (antes de 'unique'); multicluster "
                         "só existe no parquet agregado completo.")
    pm.add_argument("--output", type=str)
    pm.add_argument("--dpi", type=int, default=150)
    pm.set_defaults(func=cmd_plot_multi)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
