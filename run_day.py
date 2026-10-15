# -*- coding: utf-8 -*-
"""
run_day.py
==========

Executa o pipeline completo para UM dia:

    download -> detect (visual) -> aggregate -> track -> runs -> figuras

Uso (a partir de qualquer pasta; os dados ficam em --workdir):

    python run_day.py --date 2025-10-25 --workdir D:/banco_dados/cptec
    python run_day.py --date 2025-10-25 --workdir D:/banco_dados/cptec --steps aggregate track
    python run_day.py --year 2025 --doy 298 --workdir D:/banco_dados/cptec --skip-download

Etapas: download, detect, aggregate, track
(--steps escolhe quais rodar; padrão = todas. --skip-download remove o download.)

Cada etapa deixa um arquivo intermediário e é pulada se ele já existir
(use --force para refazer).

Observações
-----------
* O ``workdir`` precisa conter ``mask_SouthAmerica.npz`` (o download.py o
  carrega com caminho relativo). Todos os caminhos relativos (reports/,
  merged/, results/) são resolvidos a partir dele.
* O tracking grava o resultado COMPLETO em ``*_tracked.parquet`` e os runs
  filtrados (>= min-hours) em ``*_runs.parquet`` — arquivos separados, para
  o cache do tracking não ser sobrescrito pelo filtro.
* A etapa ``detect`` chama ``visual_terminal.py`` sem argumentos, porque ele
  não aceita data. Ele processa o que estiver faltando.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

log = logging.getLogger("run_day")
ALL_STEPS = ["download", "detect", "aggregate", "track"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline MCS para um único dia.")
    d = p.add_argument_group("data (use --date OU --year e --doy)")
    d.add_argument("--date", type=str, help="AAAA-MM-DD")
    d.add_argument("--year", type=int)
    d.add_argument("--doy", type=int)

    p.add_argument("--workdir", type=str, default=".",
                   help="Pasta de trabalho/dados (contém mask_SouthAmerica.npz).")
    p.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Refaz as etapas aggregate/track mesmo se já existirem.")
    p.add_argument("-w", "--workers", type=int, default=4,
                   help="Workers do download. Default: 4.")

    t = p.add_argument_group("parâmetros do tracking")
    t.add_argument("--max-speed", type=float, default=120.0,
                   help="Velocidade máxima (km/h). Default: 120.")
    t.add_argument("--gap-minutes", type=float, default=30.0,
                   help="Gap temporal máximo dentro de um track (min). Default: 30.")
    t.add_argument("--min-hours", type=float, default=4.0,
                   help="Duração mínima de um run (h). Default: 4.")
    t.add_argument("--size-col", type=str, default="area_km2")

    p.add_argument("--plot", action="store_true", help="Salva figuras em results/.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.date:
        day = dt.date.fromisoformat(args.date)
    elif args.year and args.doy:
        day = dt.date(args.year, 1, 1) + dt.timedelta(days=args.doy - 1)
    else:
        p.error("informe --date AAAA-MM-DD ou --year e --doy")
    args.day = day
    args.year = day.year
    args.doy = day.timetuple().tm_yday
    return args


# ---------------------------------------------------------------------------
def step_download(args) -> None:
    cmd = [sys.executable, str(PROJECT / "download_terminal.py"),
           "-y", str(args.year), "-m", str(args.day.month),
           "-d", str(args.day.day), "-w", str(args.workers)]
    log.info("download: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # cwd = workdir (já feito com os.chdir)


def step_detect(args) -> None:
    cmd = [sys.executable, str(PROJECT / "visual_terminal.py")]
    log.warning("detect: visual_terminal.py não filtra por data; "
                "processa o que estiver faltando.")
    subprocess.run(cmd, check=True)


def step_aggregate_track(args, do_aggregate: bool, do_track: bool) -> None:
    import pandas as pd
    from aggregator import (
        reports2df, track_multiple_clusters, filter_continuous_runs,
        plot_doy_hour, plot_tracks_latlon,
    )

    tag = f"{args.year}-{args.doy:03d}"
    reports_dir = Path(f"reports/{args.year:04d}/{args.doy:03d}")
    raw_pq = Path(f"merged/{tag}_raw.parquet")
    tracked_pq = Path(f"merged/{tag}_tracked.parquet")
    runs_pq = Path(f"merged/{tag}_runs.parquet")
    fig_dir = Path("results")
    for d in (raw_pq.parent, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- aggregate -------------------------------------------------------
    if raw_pq.exists() and not args.force:
        log.info("aggregate: usando %s", raw_pq)
        df = pd.read_parquet(raw_pq)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        if not do_aggregate:
            raise SystemExit(f"{raw_pq} não existe; rode a etapa 'aggregate'.")
        log.info("aggregate: lendo %s", reports_dir)
        df = reports2df(reports_dir)
        if df.empty:
            raise SystemExit(f"Nenhum relatório em {reports_dir.resolve()}")
        keep = ["datetime", "area_km2", "temp_min_C", "temp_mean_C",
                "centroid_lat", "centroid_lon"]
        df = df[[c for c in keep if c in df.columns]]
        df.to_parquet(raw_pq, index=False)
        log.info("aggregate: %d registros -> %s", len(df), raw_pq)

    if not do_track:
        return

    # --- track -----------------------------------------------------------
    if tracked_pq.exists() and not args.force:
        log.info("track: usando %s", tracked_pq)
        df_u = pd.read_parquet(tracked_pq)
        df_u["datetime"] = pd.to_datetime(df_u["datetime"], errors="coerce")
    else:
        df_u = track_multiple_clusters(
            df, max_speed_kmh=args.max_speed, max_gap_minutes=args.gap_minutes)
        df_u.to_parquet(tracked_pq, index=False)
    log.info("track: %d sistemas em %d observações",
             df_u["track_id"].nunique(), len(df_u))

    df_cont, runs = filter_continuous_runs(
        df_u, min_hours=args.min_hours, gap_minutes=args.gap_minutes,
        time_col="datetime")
    df_cont.to_parquet(runs_pq, index=False)
    log.info("runs >= %.1f h: %d de %d tracks -> %s",
             args.min_hours, len(runs), df_u["track_id"].nunique(), runs_pq)
    if not runs.empty:
        top = runs.sort_values("duration_min", ascending=False).head(5)
        print(top[["run_id", "run_start", "run_end", "duration_min"]]
              .to_string(index=False))

    # --- figuras ---------------------------------------------------------
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        plot_doy_hour(df_u, out_path=fig_dir / f"{tag}_doy_hour.png",
                      size_col=args.size_col, size_scale=1e5,
                      one_axes_per_year=True, color_by="track_id",
                      palette="tab20", n_colors=20)
        if not df_cont.empty:
            plot_tracks_latlon(df_cont,
                               out_path=fig_dir / f"{tag}_trajectories.png",
                               size_col=args.size_col, size_scale=1e4)
        log.info("figuras salvas em %s", fig_dir.resolve())


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    import os
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    log.info("Dia %s (DOY %03d) | workdir=%s", args.day, args.doy, workdir)

    steps = [s for s in args.steps if not (s == "download" and args.skip_download)]
    if ("download" in steps or "detect" in steps) and not Path(
            "mask_SouthAmerica.npz").exists():
        log.error("mask_SouthAmerica.npz não encontrado em %s", workdir)
        return 2

    if "download" in steps:
        step_download(args)
    if "detect" in steps:
        step_detect(args)
    if "aggregate" in steps or "track" in steps:
        step_aggregate_track(args, "aggregate" in steps, "track" in steps)

    log.info("Concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())