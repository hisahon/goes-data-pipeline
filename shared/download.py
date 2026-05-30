#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOES-16 / GOES-19 downloader from CPTEC/INPE.

The script discovers .nc files for a given channel / year / month(s) / day(s),
downloads them in parallel through a process pool, converts each file to a
.npz (numpy) snapshot with an optional regional crop, and removes the raw .nc
afterwards.

The correct satellite (GOES-16 or GOES-19) is chosen automatically from the
(year, month) pair — GOES-16 up to March 2025, GOES-19 from April 2025 on.

Created on Wed Oct 22 17:32:47 2025
@author: hisashi
"""

import os
import re
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from shared.detector import read_goes_file
from shared.converter import save_netcdf2numpy


# =============================================================================
# Environment setup
# =============================================================================
# Prevent BLAS/OpenMP oversubscription inside worker processes. Each worker
# does I/O + light numeric work, so a single thread per process is ideal.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")


# =============================================================================
# Satellite / URL selection
# =============================================================================
# CPTEC switched the primary GOES feed on April 2025. Everything on/after
# 2025-04 is served under the "goes19" path; everything before that lives
# under "goes16".
_SATELLITE_CUTOFF_YYYYMM = 202504  # First year-month that uses GOES-19.


def satellite_for_date(year: int, month: int) -> str:
    """Return 'goes16' or 'goes19' based on the observation year/month."""
    yyyymm = year * 100 + month
    return "goes19" if yyyymm >= _SATELLITE_CUTOFF_YYYYMM else "goes16"


def build_month_url(year: int, month: int, channel: int) -> str:
    """Build the CPTEC directory URL that lists .nc files for one month."""
    satellite = satellite_for_date(year, month)
    return (
        f"https://ftp.cptec.inpe.br/goes/{satellite}"
        f"/retangular/ch{channel}/{year:04d}/{month:02d}/"
    )


# =============================================================================
# HTTP session (one per process, reused for all requests)
# =============================================================================
def _build_session() -> requests.Session:
    """Create a requests Session with retry/backoff for transient failures."""
    s = requests.Session()
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=5,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET", "HEAD"]),
            )
        ),
    )
    s.headers.update({"User-Agent": "goes-dl/mp-1xN"})
    return s


# Persistent per-worker session, initialized by the pool's initializer.
_SESS: requests.Session | None = None


def _init_worker() -> None:
    """ProcessPoolExecutor initializer: create one Session per worker."""
    global _SESS
    _SESS = _build_session()


def _get_sess() -> requests.Session:
    """Return the worker session, lazily creating it if needed."""
    global _SESS
    if _SESS is None:
        _SESS = _build_session()
    return _SESS


# =============================================================================
# Small utilities: dates, filenames, worker counts
# =============================================================================
def _normalize_days(year: int, month: int, days):
    """Validate/expand the list of days within a given year/month."""
    import calendar as _cal
    _, last = _cal.monthrange(year, month)
    if days is None:
        # None means "every day in the month".
        return list(range(1, last + 1))
    out = sorted({int(d) for d in days if 1 <= int(d) <= last})
    if not out:
        raise ValueError(f"Nenhum dia válido para {year}-{month:02d}.")
    return out


def _list_nc_files(sess: requests.Session, url: str, timeout: float):
    """Scrape a CPTEC directory index and return all .nc filenames linked."""
    try:
        r = sess.get(url, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return [
            a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].endswith(".nc") and not a["href"].startswith("?")
        ]
    except Exception as e:
        print(f"[WARN] Falha ao listar {url}: {e}")
        return []


def npz_path_from_nc(nc_like_name: str, out_root: Path) -> Path:
    """
    Map a raw .nc filename to its destination .npz path.

    Example:
        'S10635346_YYYYMMDDHHMM.nc'
        -> out_root / 'YYYY' / 'MM' / 'S10635346_YYYYMMDDHHMM.npz'
    """
    m = re.search(r"_(\d{4})(\d{2})", nc_like_name)
    if not m:
        raise ValueError(f"Não consigo extrair YYYYMM do nome: {nc_like_name}")
    year, month = m.group(1), m.group(2)
    # NOTE: the original code used out_root.parent/"database_numpy" here.
    # We now receive the numpy database root directly in `out_root`, so we
    # just append YYYY/MM to it.
    npz_dir = out_root / year / month
    npz_dir.mkdir(parents=True, exist_ok=True)
    return npz_dir / (Path(nc_like_name).with_suffix(".npz").name)


def _stamp_from_name(name: str) -> str:
    """Pull a human-readable timestamp 'YYYY-MM-DD HHMM' out of a filename."""
    m = re.search(r"_(\d{4})(\d{2})(\d{2})(\d{4})", name)
    if not m:
        return ""
    y, mo, d, hm = m.groups()
    return f"{y}-{mo}-{d} {hm}"


def _auto_workers() -> int:
    """Default worker count: leave one CPU free for the OS / controller."""
    total = os.cpu_count() or 2
    return max(1, total - 1)


def effective_workers(requested: int | None) -> int:
    """Clamp the user-requested worker count to [1, CPU-1]."""
    cap = _auto_workers()
    if requested is None:
        return cap
    return max(1, min(int(requested), cap))


# =============================================================================
# Worker job (runs inside each subprocess)
# =============================================================================
def _worker_job(item: dict):
    """
    Download one .nc file, convert it to .npz, and clean up the raw file.

    `item` carries every piece of state the worker needs so the function is
    self-contained and picklable.
    """
    from time import perf_counter

    url       = item["url"]
    temp_nc   = Path(item["temp_nc_path"])
    npz_path  = Path(item["npz_path"])
    extent    = item["extent"]
    timeout   = float(item["timeout"])
    chunk_mb  = int(item.get("chunk_mb", 8))  # Larger chunks reduce overhead.

    # Race-safe double-check: another worker may have just finished this file.
    if npz_path.exists() and npz_path.stat().st_size > 0:
        return ("skipped", npz_path.name, "already exists")

    sess = _get_sess()
    CHUNK = max(1, chunk_mb) * 1024 * 1024

    t0 = perf_counter()
    tmp_part = temp_nc.with_suffix(temp_nc.suffix + ".part")
    temp_nc.parent.mkdir(parents=True, exist_ok=True)

    try:
        # --- Download the .nc stream to a temporary .part file ---
        t_net0 = perf_counter()
        with sess.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp_part, "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
        # Atomic rename once the download completes.
        os.replace(tmp_part, temp_nc)
        t_net1 = perf_counter()

        # --- Read + regional crop ---
        temp, lat, lon = read_goes_file(temp_nc, extent=extent)
        t_proc1 = perf_counter()

        # --- Save the processed data as .npz ---
        save_netcdf2numpy(npz_path, temp, lat, lon)
        t_save1 = perf_counter()

        timing = (
            f"net={t_net1 - t_net0:.2f}s "
            f"proc={t_proc1 - t_net1:.2f}s "
            f"save={t_save1 - t_proc1:.2f}s "
            f"tot={t_save1 - t0:.2f}s"
        )
        status = ("ok", npz_path.name, timing)

    except Exception as e:
        status = ("err", Path(url).name, str(e))

    finally:
        # Best-effort cleanup of the temporary raw file(s).
        for p in (tmp_part, temp_nc):
            try:
                if p.exists():
                    p.unlink(missing_ok=True)
            except Exception:
                pass

    return status


# =============================================================================
# Multiprocess controller
# =============================================================================
def download_goes_multiproc(
    channel: int,
    year: int,
    folders: dict,
    months=None,
    days=None,
    extent=None,
    timeout: float = 30.0,
    max_workers: int | None = None,
    chunk_mb: int = 8,  # 4–8 MB is a sweet spot.
):
    """
    Orchestrate a parallel download for one year / channel.

    The satellite path (GOES-16 vs GOES-19) is selected automatically for each
    month via `build_month_url`.
    """
    if not (1 <= channel <= 16):
        raise ValueError("Channel deve estar entre 1 e 16.")

    months = months or list(range(1, 13))
    max_workers = effective_workers(max_workers)

    # Session used only by the controller for directory listings.
    sess = _build_session()

    # Root for the raw .nc temporary files.
    tmp_root = folders["folder_temporary_goes"]
    tmp_root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Build the task list. Pre-filter files whose .npz already exists so we
    # don't pay network cost for work that is already done.
    # -------------------------------------------------------------------------
    tasks: list[dict] = []
    pre_skipped = 0

    for m in months:
        month_url = build_month_url(year, m, channel)  # Auto GOES-16/19 pick.
        names     = _list_nc_files(sess, month_url, timeout)
        valid     = _normalize_days(year, m, days)

        for name in names:
            # Filename embeds YYYYMMDDHHMM; pull out the day and filter.
            m_dt = re.search(r"_(\d{8})(\d{4})", name)
            if not m_dt:
                continue
            if int(m_dt.group(1)[6:8]) not in valid:
                continue

            url     = urljoin(month_url, name)
            temp_nc = tmp_root / name
            npz_out = npz_path_from_nc(name, folders["folder_numpy_database"])

            # Skip if the destination .npz already exists and is non-empty.
            if npz_out.exists() and npz_out.stat().st_size > 0:
                pre_skipped += 1
                continue

            tasks.append(dict(
                url=url,
                temp_nc_path=str(temp_nc),
                npz_path=str(npz_out),
                extent=extent,
                timeout=timeout,
                chunk_mb=chunk_mb,
            ))

    if not tasks:
        print(f"Nada a fazer (pre-skipped={pre_skipped}).")
        return dict(
            year=year, channel=channel,
            total=0, processed=0, failed=0, skipped=pre_skipped,
        )

    print(
        f"Iniciando multiprocessamento com {max_workers} workers… "
        f"(tarefas: {len(tasks)}, pre-skipped: {pre_skipped})"
    )

    # -------------------------------------------------------------------------
    # Run the pool and collect results with a tqdm progress bar.
    # -------------------------------------------------------------------------
    ok = err = skipped_runtime = 0
    last_label = "—"

    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker
    ) as ex:
        # chunksize 2–8 keeps workers "warm" without starving the scheduler.
        results = ex.map(_worker_job, tasks, chunksize=2)

        with tqdm(total=len(tasks), desc=f"last: {last_label}", unit="file") as pbar:
            for status, name, info in results:
                if status == "ok":
                    ok += 1
                    if ok % 10 == 0:
                        stamp = _stamp_from_name(name)
                        if stamp:
                            pbar.set_description(f"last: {stamp}")
                elif status == "skipped":
                    skipped_runtime += 1
                else:
                    err += 1
                    if err <= 5:                          # mostra os 5 primeiros erros
                        print(f"\n[ERRO] {name}: {info}")
                    pbar.set_postfix_str(f"errors: {err}")
                pbar.update(1)                            # avança para TODO arquivo

    print(
        f"\nResumo MP: total_enfileirado={len(tasks)} "
        f"ok={ok} err={err} skipped_rt={skipped_runtime} "
        f"pre_skipped={pre_skipped}"
    )
    return dict(
        year=year,
        channel=channel,
        total=len(tasks),
        processed=ok,
        failed=err,
        skipped=pre_skipped + skipped_runtime,
    )


# =============================================================================
# Settings summary + confirmation prompt
# =============================================================================
def print_settings_and_confirm(
    channel, year, months, days, extent, folders, timeout, req_workers
) -> bool:
    """Pretty-print the run settings and ask the user to confirm."""
    months_str  = "TODOS" if months is None else " ".join(map(str, months))
    days_str    = "TODOS" if days is None else " ".join(map(str, days))
    extent_str  = "None" if extent is None else f"[{extent[0]}, {extent[1]}, {extent[2]}, {extent[3]}]"
    folder_root = str(folders["folder_numpy_database"])
    folder_temp = str(folders["folder_temporary_goes"])

    eff_workers = effective_workers(req_workers)
    req_label   = "auto" if req_workers is None else str(req_workers)

    # Preview which satellite(s) will be used so the user can sanity-check.
    preview_months = months if months is not None else list(range(1, 13))
    satellites = sorted({satellite_for_date(year, m) for m in preview_months})
    sats_str = ", ".join(s.upper().replace("GOES", "GOES-") for s in satellites)

    print("\n===== CONFIGURAÇÕES =====")
    print(f"Canal (channel):       {channel}")
    print(f"Ano (year):            {year}")
    print(f"Meses (months):        {months_str}")
    print(f"Dias (days):           {days_str}")
    print(f"Satélite(s):           {sats_str}")
    print(f"Recorte (extent):      {extent_str}")
    print(f"Pasta numpy (folder):  {folder_root}")
    print(f"Pasta temp (folder):   {folder_temp}")
    print(f"Timeout (s):           {timeout}")
    print(f"Workers (requested):   {req_label}")
    print(f"Workers (effective):   {eff_workers}  (capped to CPU-1)")
    print("=========================\n")

    ans = input("Prosseguir com essas configurações? [s/N]: ").strip().lower()
    return ans in ("s", "sim", "y", "yes")


# =============================================================================
# Folder layout per platform / environment
# =============================================================================
def defining_folders(server_ON: bool) -> dict:
    """
    Resolve the output and temp directories for the current host.

    Windows always uses the local D: drive layout; Linux/macOS pick between
    a 'server' layout (home-relative) and a developer layout.
    """
    if os.name == "nt":
        # Windows workstation.
        FOLDER_ROOT = Path("database_numpy")
        FOLDER_GOES_TEMPORARY = Path("goes_temporary_raw")
    elif os.name == "posix":
        if server_ON:
            # Remote/headless server: write under $HOME.
            FOLDER_ROOT = Path("~/database_numpy").expanduser()
            FOLDER_GOES_TEMPORARY = Path("~/download_goes_temporary").expanduser()
        else:
            # Local POSIX workstation (sshfs-mounted CPTEC layout).
            FOLDER_ROOT = Path("~/database/cptec/database_numpy").expanduser()
            FOLDER_GOES_TEMPORARY = Path("~/download_goes_temporary").expanduser()
    else:
        raise RuntimeError(f"Unsupported OS: {os.name}")

    return {
        "folder_numpy_database": FOLDER_ROOT,
        "folder_temporary_goes": FOLDER_GOES_TEMPORARY,
    }


# =============================================================================
# Entry points
# =============================================================================
def main_manual():
    """Run with hard-coded parameters (no CLI)."""
    # Load the South America mask; `extent` is [lon_min, lon_max, lat_min, lat_max].
    dat = np.load("mask_SouthAmerica.npz")

    # --- CONFIG ---
    channel     = 13
    year        = 2021
    months      = [10]
    days        = [23]
    extent      = dat["extent"]
    timeout     = 30.0
    req_workers = 16     # None => auto; higher values are capped at CPU-1.
    server_ON   = True

    folders = defining_folders(server_ON)

    if not print_settings_and_confirm(
        channel, year, months, days, extent, folders, timeout, req_workers
    ):
        print("Cancelado pelo usuário.")
        return

    max_workers = effective_workers(req_workers)

    summary = download_goes_multiproc(
        channel=channel,
        year=year,
        months=months,
        days=days,
        folders=folders,
        extent=extent,
        timeout=timeout,
        max_workers=max_workers,
    )

    print("\nResumo final:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def main_terminal():
    """Run with parameters parsed from the command line (argparse)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download GOES-16/19 data from CPTEC by year/month/day.",
    )
    parser.add_argument("-c", "--channel", type=int, default=13,
                        help="GOES channel (1–16)")
    parser.add_argument("-y", "--year", type=int, required=True,
                        help="Year (e.g., 2021)")
    parser.add_argument("-m", "--months", nargs="*", type=int, default=None,
                        help="Months (e.g., 10 11 12). Default: full year")
    parser.add_argument("-d", "--days", nargs="*", type=int, default=None,
                        help="Days of month (e.g., 20 21 22). Default: full month")
    parser.add_argument("-t", "--timeout", type=float, default=30.0,
                        help="Request timeout in seconds (default: 30)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Parallel workers (capped to CPU-1; default: auto)")
    parser.add_argument("--server", action="store_true",
                        help="Use the server-side folder layout.")
    args = parser.parse_args()

    # Project defaults — edit here if a different crop is needed.
    dat = np.load("mask_SouthAmerica.npz")
    extent = dat["extent"]  # [lon_min, lon_max, lat_min, lat_max]

    folders = defining_folders(args.server)
    req_workers = args.workers

    if not print_settings_and_confirm(
        channel=args.channel,
        year=args.year,
        months=args.months,
        days=args.days,
        extent=extent,
        folders=folders,
        timeout=args.timeout,
        req_workers=req_workers,
    ):
        print("Cancelado pelo usuário.")
        return

    max_workers = effective_workers(req_workers)

    summary = download_goes_multiproc(
        channel=args.channel,
        year=args.year,
        months=args.months,
        days=args.days,
        folders=folders,
        extent=extent,
        timeout=args.timeout,
        max_workers=max_workers,
    )

    print("\nResumo final:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()             # Hard-coded parameters.
    # main_terminal()      # CLI mode (argparse).


# -----------------------------------------------------------------------------
# Usage notes
# -----------------------------------------------------------------------------
# 1) Folder paths are platform-dependent — check `defining_folders` before
#    running on a new machine.
# 2) The satellite is chosen automatically: GOES-16 for dates up to 2025-03
#    and GOES-19 from 2025-04 onward. Adjust `_SATELLITE_CUTOFF_YYYYMM`
#    above if CPTEC changes the cutoff.
# 3) Regional cropping happens inside this pipeline — see `extent`.
#
# --- Terminal examples ---
# sshfs hisashi@150.163.46.4:/home/hisashi/cptec_codes ~/cptec_codes \
#       -o reconnect,uid=$(id -u),gid=$(id -g),default_permissions,umask=0002
# sshfs hisashi@150.163.46.4:/home/hisashi/database/cptec ~/cptec \
#       -o reconnect,uid=$(id -u),gid=$(id -g),default_permissions,umask=0002
#
# source mpi-env/bin/activate
# python ~/cptec_codes/download.py -y 2025 -m 3 4 5 6 7 8 9 10 --server -w 10