# -*- coding: utf-8 -*-
"""
03_aggregate_terminal.py
========================

Etapa 3 do pipeline (download -> detect -> **aggregate** -> track).
Junta os relatórios por frame (``reports/YYYY/DDD/C13_YYYY-DDD_HHMM.txt``,
um .txt por imagem) em tabelas, e gera figuras exploratórias.
Wrapper fino sobre ``aggregator.py`` (que precisa estar na mesma pasta).

Subcomandos
-----------
aggregate   (PADRÃO: um arquivo POR DIA)
    Varre ``reports/YYYY/DDD/`` e grava, para cada dia que tenha células:
        tracked-thunderstorms/YYYY/YYYY_DOY_DDMES_C13.parquet
    ex.: 13/12/2023 (DOY 347) -> tracked-thunderstorms/2023/2023_347_13DEZ_C13.parquet
    (MES em português: JAN FEV MAR ABR MAI JUN JUL AGO SET OUT NOV DEZ)
    * dias já gerados são PULADOS (rodar de novo só faz o que falta);
    * dias sem nenhuma célula detectada não geram arquivo;
    * ``--single`` volta ao comportamento antigo: UM parquet com tudo.
unique      reduz para uma célula por horário (a de maior área).
plot-runs   PNG anual DOY x hora com sequências contínuas.
plot-multi  overlay dos horários com mais de uma célula.

Opções do ``aggregate``
-----------------------
--reports-dir DIR   raiz dos reports (contém YYYY/DDD/*.txt).
                    Default: <root>/reports  (NÃO é a pasta do visual.py;
                    no Windows passe D:/goes-data-pipeline/reports<chave>).
--out-dir DIR       raiz de saída. Default: <pasta deste script>/tracked-thunderstorms.
--year N            só este ano.            --doy N   só este dia do ano.
--format parquet|csv   formato dos arquivos diários (default parquet).
--keep-cols A B ..  só estas colunas (default: todas).
--force             refaz mesmo se o arquivo já existir.
--single            um único parquet (usa --output / -o em vez de --out-dir).
--server            layout de pastas do servidor (<root> = home).
-v / -vv            mais mensagens de log.

<root> = D:/banco_dados/cptec (Windows) | ~ (--server) | ~/database/cptec (Linux).

Como executar
-------------
Windows (Anaconda Prompt / PowerShell), a partir da pasta dos scripts::

    # tudo (todos os anos e dias):
    python 03_aggregate_terminal.py aggregate --reports-dir D:/goes-data-pipeline/reports_xxx

    # um dia (para testar primeiro):
    python 03_aggregate_terminal.py aggregate --reports-dir D:/goes-data-pipeline/reports_xxx ^
        --year 2023 --doy 347

    # um ano, saída em outra pasta, só algumas colunas:
    python 03_aggregate_terminal.py aggregate --reports-dir D:/goes-data-pipeline/reports_xxx ^
        --year 2023 --out-dir D:/goes-data-pipeline/tracked-thunderstorms ^
        --keep-cols datetime centroid_lat centroid_lon area_km2 temp_min_C temp_mean_C temp_std_C

    # refazer um dia que já existe:
    python 03_aggregate_terminal.py aggregate --reports-dir ... --year 2023 --doy 347 --force

Servidor (Linux)::

    python 03_aggregate_terminal.py --server aggregate --reports-dir reports_S

Parquet único (para unique / plot-runs / plot-multi, que leem UM arquivo)::

    python 03_aggregate_terminal.py aggregate --reports-dir ... --single \\
        --output merged/c13_2018-2025.parquet
    python 03_aggregate_terminal.py unique \\
        --input  merged/c13_2018-2025.parquet --output merged/c13_2018-2025u.parquet
    python 03_aggregate_terminal.py plot-runs \\
        --input  merged/c13_2018-2025u.parquet --out-dir results --min-hours 16 --gap-minutes 30

Ler os arquivos diários em Python (Spyder)::

    from pathlib import Path
    import pandas as pd
    df = pd.concat(pd.read_parquet(f) for f in
                   sorted(Path("D:/goes-data-pipeline/tracked-thunderstorms").rglob("*.parquet")))

Observações
-----------
* Cada linha do .txt é uma célula; ``n_clusters`` é o índice da célula no frame.
* ``datetime`` vem do nome do .txt (feito por ``reports2df``).
* ``unique``, ``plot-runs`` e ``plot-multi`` ainda esperam UM parquet em ``--input``.
"""

from __future__ import annotations

import argparse
import datetime as dt
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

MESES = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN",
         "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]


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
        "daily": Path(__file__).resolve().parent / "tracked-thunderstorms",
    }


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------
def _aggregate_single(args: argparse.Namespace) -> int:
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


def _aggregate_daily(args: argparse.Namespace) -> int:
    """Um arquivo por dia: reports/YYYY/DDD/*.txt -> <out>/YYYY/YYYY_DOY_DDMES_C13.<ext>."""
    folders = defining_folders(args.server)
    reports = Path(args.reports_dir) if args.reports_dir else folders["reports"]
    out_root = Path(args.out_dir) if args.out_dir else folders["daily"]
    ext = "csv" if args.format == "csv" else "parquet"

    day_dirs = sorted(d for d in reports.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9]")
                      if d.is_dir())
    if args.year is not None:
        day_dirs = [d for d in day_dirs if int(d.parent.name) == args.year]
    if args.doy is not None:
        day_dirs = [d for d in day_dirs if int(d.name) == args.doy]
    if not day_dirs:
        logger.error("Nenhuma pasta YYYY/DDD em %s.", reports)
        return 1

    n_ok = n_skip = n_empty = 0
    for d in day_dirs:
        year, doy = d.parent.name, d.name
        day = dt.date(int(year), 1, 1) + dt.timedelta(days=int(doy) - 1)
        out = out_root / year / f"{year}_{doy}_{day:%d}{MESES[day.month - 1]}_C13.{ext}"
        if out.exists() and not args.force:
            n_skip += 1
            continue
        df = reports2df(d)
        if df.empty:                      # dia sem nenhuma célula detectada
            n_empty += 1
            continue
        if args.keep_cols:
            df = df[[c for c in args.keep_cols if c in df.columns]]
        out.parent.mkdir(parents=True, exist_ok=True)
        if ext == "csv":
            df.to_csv(out, index=False)
        else:
            df.to_parquet(out, index=False)
        n_ok += 1
        print(f"{year}-{doy} | {len(df)} linhas -> {out.name}")

    print(f"Done: {n_ok} gerados | {n_skip} já existiam | {n_empty} sem células "
          f"(de {len(day_dirs)} dias) em {out_root}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Padrão: um arquivo por dia. ``--single``: um único parquet."""
    return _aggregate_single(args) if args.single else _aggregate_daily(args)


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
        prog="03_aggregate_terminal.py",
        description="Agregação de reports + análises exploratórias DOY×hora.",
    )
    p.add_argument("--server", action="store_true",
                   help="Usa layout de pastas do servidor.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Verbosidade (-v, -vv).")

    sub = p.add_subparsers(dest="cmd", required=True)

    # -- aggregate --
    a = sub.add_parser("aggregate",
                       help="Varre reports/*.txt e gera um parquet por dia "
                            "(--single: um parquet único).")
    a.add_argument("--reports-dir", type=str,
                   help="Raiz da árvore de reports. Default: <root>/reports.")
    a.add_argument("--single", action="store_true",
                   help="Um único parquet com tudo (em vez de um por dia).")
    a.add_argument("--out-dir", type=str,
                   help="[por dia] Raiz de saída. Default: <pasta deste script>/tracked-thunderstorms.")
    a.add_argument("--year", type=int, help="[por dia] Só este ano.")
    a.add_argument("--doy", type=int, help="[por dia] Só este dia do ano.")
    a.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                   help="[por dia] Formato dos arquivos. Default: parquet.")
    a.add_argument("--output", type=str,
                   help="[--single] Parquet de saída. Default: merged/c13_aggregated.parquet")
    a.add_argument("-o", "--output_key", type=str, default="",
                   help="[--single] Sufixo para o nome default do parquet.")
    a.add_argument("--keep-cols", nargs="*", default=None,
                   help="Colunas a preservar (default: todas).")
    a.add_argument("--force", action="store_true",
                   help="Refaz mesmo se o arquivo já existir.")
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