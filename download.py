#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:32:47 2025

@author: hisashi
"""

# -*- coding: utf-8 -*-
import os, re#, calendar, sys, traceback
from pathlib import Path
from tqdm import tqdm
import numpy as np

from concurrent.futures import ProcessPoolExecutor#, as_completed

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from detector import read_goes_file
from converter import save_netcdf2numpy

# # tqdm (safe fallback)
# try:
#     from tqdm import tqdm
# except Exception:
#     class tqdm:  # no-op fallback
#         def __init__(self, *a, **k): pass
#         def update(self, *a, **k): pass
#         def set_postfix_str(self, *a, **k): pass
#         def set_description(self, *a, **k): pass
#         def close(self): pass
#         def __enter__(self): return self
#         def __exit__(self, *a): pass

# Avoid BLAS oversubscription inside workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

# ---------------- HTTP/session ----------------
def _session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"])
    )))
    s.headers.update({"User-Agent": "goes16-dl/mp-1xN"})
    return s

# one persistent Session per worker process
_SESS = None
def _init_worker():
    global _SESS
    _SESS = _session()

def _get_sess() -> requests.Session:
    global _SESS
    if _SESS is None:
        _SESS = _session()
    return _SESS

# ---------------- helpers ----------------
def _normalize_days(year: int, month: int, days):
    import calendar as _cal
    _, last = _cal.monthrange(year, month)
    if days is None:
        return list(range(1, last + 1))
    out = sorted({int(d) for d in days if 1 <= int(d) <= last})
    if not out:
        raise ValueError(f"Nenhum dia válido para {year}-{month:02d}.")
    return out

def _list_nc_files(sess: requests.Session, url: str, timeout: float):
    try:
        r = sess.get(url, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return [a["href"] for a in soup.find_all("a", href=True)
                if a["href"].endswith(".nc") and not a["href"].startswith("?")]
    except Exception as e:
        print(f"[WARN] Falha ao listar {url}: {e}")
        return []

def npz_path_from_nc(nc_like_name: str, out_root: Path) -> Path:
    """
    'S10635346_YYYYMMDDHHMM.nc' -> out_root.parent/'database_numpy'/YYYY/MM/'...'.npz
    """
    m = re.search(r'_(\d{4})(\d{2})', nc_like_name)
    if not m:
        raise ValueError(f"Não consigo extrair YYYYMM do nome: {nc_like_name}")
    year, month = m.group(1), m.group(2)
    npz_dir = out_root.parent / "database_numpy" / year / month
    npz_dir.mkdir(parents=True, exist_ok=True)
    return npz_dir / (Path(nc_like_name).with_suffix(".npz").name)

def _stamp_from_name(name: str) -> str:
    m = re.search(r'_(\d{4})(\d{2})(\d{2})(\d{4})', name)
    if not m: return ""
    y, mo, d, hm = m.groups()
    return f"{y}-{mo}-{d} {hm}"

def _cap_workers(n: int | None) -> int:
    total = os.cpu_count() or 2
    cap = max(1, total - 1)
    return cap if n is None else max(1, min(int(n), cap))

# ---------------- worker (runs in each process) ----------------
def _worker_job(item):
    """
    item: dict with url, temp_nc_path, npz_path, extent, timeout, chunk_mb
    """
    from time import perf_counter

    url         = item["url"]
    temp_nc     = Path(item["temp_nc_path"])
    npz_path    = Path(item["npz_path"])
    extent      = item["extent"]
    timeout     = float(item["timeout"])
    chunk_mb    = int(item.get("chunk_mb", 8))  # bigger chunks reduce overhead

    # double-check skip in-process (race-safe)
    if npz_path.exists() and npz_path.stat().st_size > 0:
        return ("skipped", npz_path.name, "already exists")

    sess = _get_sess()
    CHUNK = max(1, chunk_mb) * 1024 * 1024

    t0 = perf_counter()
    tmp_part = temp_nc.with_suffix(temp_nc.suffix + ".part")
    temp_nc.parent.mkdir(parents=True, exist_ok=True)

    try:
        # download .nc to temp
        t_net0 = perf_counter()
        with sess.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp_part, "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_part, temp_nc)
        t_net1 = perf_counter()

        # read/process
        temp, lat, lon = read_goes_file(temp_nc, extent=extent)
        t_proc1 = perf_counter()

        # save npz (your saver expects destination path)
        save_netcdf2numpy(npz_path, temp, lat, lon)
        t_save1 = perf_counter()

        timing = f"net={t_net1-t_net0:.2f}s proc={t_proc1-t_net1:.2f}s save={t_save1-t_proc1:.2f}s tot={t_save1-t0:.2f}s"
        status = ("ok", npz_path.name, timing)

    except Exception as e:
        status = ("err", Path(url).name, str(e))

    finally:
        # cleanup temp files
        try:
            if tmp_part.exists():
                tmp_part.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if temp_nc.exists():
                temp_nc.unlink(missing_ok=True)
        except Exception:
            pass

    return status

# ---------------- controlador multiprocessos ----------------
def download_goes16_multiproc(
    channel: int,
    year: int,
    folders: dict,
    months=None,
    days=None,
    extent=None,
    timeout: float = 30.0,
    max_workers: int | None = None,
    chunk_mb: int = 8,                  # 4–8 MB is a good range
):
    if not (1 <= channel <= 16):
        raise ValueError("Channel deve estar entre 1 e 16.")

    months = months or list(range(1, 13))
    max_workers = _cap_workers(max_workers)

    # controller session for listing pages only
    sess = _session()
    # base_https = f"https://ftp.cptec.inpe.br/goes/goes16/retangular/ch{channel}/{year:04d}"
    base_https = f"https://ftp.cptec.inpe.br/goes/goes19/retangular/ch{channel}/{year:04d}"

    # temp dir: one level above 'database' (adjust if needed)
    # folder_root = Path(folder_root).resolve()
    # tmp_root = folder_root.parent.parent / "download_goes_temporary"
    tmp_root = folders['folder_temporary_goes']
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Build task list (skip NPZs that already exist up-front)
    tasks = []
    pre_skipped = 0
    for m in months:
        month_url = f"{base_https}/{m:02d}/"
        names     = _list_nc_files(sess, month_url, timeout)
        valid     = _normalize_days(year, m, days)

        for name in names:
            m_dt = re.search(r"_(\d{8})(\d{4})", name)
            if not m_dt:
                continue
            if int(m_dt.group(1)[6:8]) not in valid:
                continue

            url     = urljoin(month_url, name)
            temp_nc = tmp_root / name
            # npz_out = npz_path_from_nc(name, folder_root)
            npz_out = npz_path_from_nc(name, folders['folder_numpy_database'])

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
        return dict(year=year, channel=channel, total=0, processed=0, failed=0, skipped=pre_skipped)

    print(f"Iniciando multiprocessamento com {max_workers} workers… (tarefas: {len(tasks)}, pre-skipped: {pre_skipped})")

    ok = err = skipped_runtime = 0
    last_label = "—"
    # Use initializer to create one Session per worker
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as ex:
        results = ex.map(_worker_job, tasks, chunksize=2)  # chunksize 2–8 keeps workers “warm”

        with tqdm(total=len(tasks), desc=f"last: {last_label}", unit="file") as pbar:
            for status, name, info in results:
                if status == "ok":
                    ok += 1
                    pbar.update(1)
                    if ok % 10 == 0:  # update occasionally to keep UI light
                        stamp = _stamp_from_name(name)
                        if stamp:
                            pbar.set_description(f"last: {stamp}")
                    # Optional: print timing sometimes
                    # if ok % 20 == 0: print(f"[timing] {name} {info}")
                elif status == "skipped":
                    skipped_runtime += 1
                else:
                    err += 1
                    if err % 5 == 0:
                        pbar.set_postfix_str(f"errors: {err}")

    print(f"\nResumo MP: total_enfileirado={len(tasks)} ok={ok} err={err} skipped_rt={skipped_runtime} pre_skipped={pre_skipped}")
    return dict(
        year=year,
        channel=channel,
        total=len(tasks),
        processed=ok,
        failed=err,
        skipped=pre_skipped + skipped_runtime
    )


# def download_goes16_multiproc_we(
#     channel: int,
#     year: int,
#     folders: dict,
#     months=None,
#     days=None,
#     extent=None,
#     timeout: float = 30.0,
#     max_workers: int | None = None,
#     chunk_mb: int = 8,                  # 4–8 MB is a good range
# ):
#     if not (1 <= channel <= 16):
#         raise ValueError("Channel deve estar entre 1 e 16.")

#     months = months or list(range(1, 13))
#     max_workers = _cap_workers(max_workers)

#     # controller session for listing pages only
#     sess = _session()
#     base_https = f"https://ftp.cptec.inpe.br/goes/goes16/retangular/ch{channel}/{year:04d}"

#     # temp dir (you already pass it in folders)
#     tmp_root = folders['folder_temporary_goes']
#     tmp_root.mkdir(parents=True, exist_ok=True)

#     # Build task list (skip NPZs that already exist up-front)
#     tasks = []
#     pre_skipped = 0
#     for m in months:
#         month_url = f"{base_https}/{m:02d}/"
#         names     = _list_nc_files(sess, month_url, timeout)
#         valid     = _normalize_days(year, m, days)

#         for name in names:
#             m_dt = re.search(r"_(\d{8})(\d{4})", name)
#             if not m_dt:
#                 continue
#             if int(m_dt.group(1)[6:8]) not in valid:
#                 continue

#             url     = urljoin(month_url, name)
#             temp_nc = tmp_root / name
#             npz_out = npz_path_from_nc(name, folders['folder_numpy_database'])

#             if npz_out.exists() and npz_out.stat().st_size > 0:
#                 pre_skipped += 1
#                 continue

#             tasks.append(dict(
#                 url=url,
#                 temp_nc_path=str(temp_nc),
#                 npz_path=str(npz_out),
#                 extent=extent,
#                 timeout=timeout,
#                 chunk_mb=chunk_mb,
#             ))

#     if not tasks:
#         print(f"Nada a fazer (pre-skipped={pre_skipped}).")
#         return dict(year=year, channel=channel, total=0, processed=0, failed=0, skipped=pre_skipped)

#     print(f"Iniciando multiprocessamento com {max_workers} workers… (tarefas: {len(tasks)}, pre-skipped: {pre_skipped})")

#     ok = err = skipped_runtime = 0

#     # Run workers; print any errors
#     with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as ex:
#         future_to_task = {ex.submit(_worker_job, t): t for t in tasks}

#         for fut in as_completed(future_to_task):
#             t = future_to_task[fut]
#             try:
#                 status, name, info = fut.result()
#             except Exception as e:
#                 err += 1
#                 # Crash in worker: print full traceback
#                 where = t.get("npz_path") or t.get("temp_nc_path") or t.get("url") or "<desconhecido>"
#                 print(f"[ERROR] Worker exception em: {where}", file=sys.stderr)
#                 traceback.print_exc()
#                 continue

#             if status == "ok":
#                 ok += 1
#             elif status == "skipped":
#                 skipped_runtime += 1
#             else:
#                 err += 1
#                 # _worker_job returned an error tuple -> print concise message
#                 print(f"[ERROR] {name}: {info}", file=sys.stderr)

#     print(f"\nResumo MP: total_enfileirado={len(tasks)} ok={ok} err={err} skipped_rt={skipped_runtime} pre_skipped={pre_skipped}")
#     return dict(
#         year=year,
#         channel=channel,
#         total=len(tasks),
#         processed=ok,
#         failed=err,
#         skipped=pre_skipped + skipped_runtime
#     )


# ---------------- workers helpers ----------------
def _auto_workers() -> int:
    total = os.cpu_count() or 2
    return max(1, total - 1)

def effective_workers(requested: int | None) -> int:
    """Cap workers to CPU-1 even if user passed a higher value; never below 1."""
    cap = _auto_workers()
    if requested is None:
        return cap
    return max(1, min(int(requested), cap))

# ---------------- settings print + confirm ----------------
def print_settings_and_confirm(
    channel, year, months, days, extent, folders, timeout, req_workers
) -> bool:
    """Pretty-print settings and ask user to confirm."""
    months_str = "TODOS" if months is None else " ".join(map(str, months))
    days_str   = "TODOS" if days is None else " ".join(map(str, days))
    extent_str = "None" if extent is None else f"[{extent[0]}, {extent[1]}, {extent[2]}, {extent[3]}]"
    folder_root = str(folders['folder_numpy_database'])
    folder_temp = str(folders['folder_temporary_goes'])
    
    eff_workers = effective_workers(req_workers)
    req_label = "auto" if req_workers is None else str(req_workers)

    print("\n===== CONFIGURAÇÕES =====")
    print(f"Canal (channel):       {channel}")
    print(f"Ano (year):            {year}")
    print(f"Meses (months):        {months_str}")
    print(f"Dias (days):           {days_str}")
    print(f"Recorte (extent):      {extent_str}")
    print(f"Pasta numpy (folder):   {folder_root}")
    print(f"Pasta temp (folder):   {folder_temp}")
    print(f"Timeout (s):           {timeout}")
    print(f"Workers (requested):   {req_label}")
    print(f"Workers (effective):   {eff_workers}  (capped to CPU-1)")
    print("=========================\n")

    ans = input("Prosseguir com essas configurações? [s/N]: ").strip().lower()
    return ans in ("s", "sim", "y", "yes")

def defining_folders(server_ON):
    if os.name == "nt":
        FOLDER_ROOT = Path("D:/banco_dados/cptec/database_numpy") 
        FOLDER_GOES_TEMPORARY = Path("D:/banco_dados/cptec/goes_temporary_raw") 
        # FOLDER_NUMPY = Path("D:/banco_dados/cptec/database_numpy") 

    elif os.name == "posix":
        if server_ON:
            FOLDER_ROOT = Path("~/database_numpy").expanduser()
            FOLDER_GOES_TEMPORARY = Path("~/download_goes_temporary").expanduser()
        else:
            # FOLDER_ROOT = Path("~/database/cptec/database_numpy").expanduser()
            # FOLDER_GOES_TEMPORARY = Path("~/download_goes_temporary").expanduser()
            
            FOLDER_ROOT = Path("~/database/cptec/database_numpy").expanduser()
            FOLDER_GOES_TEMPORARY = Path("~/download_goes_temporary").expanduser()

   
    folders = {'folder_numpy_database':FOLDER_ROOT,
               'folder_temporary_goes':FOLDER_GOES_TEMPORARY}
    
    return folders

# ---------------- main (no terminal) ----------------
def main():
    
    dat = np.load("mask_SouthAmerica.npz")

    
    
    
    # --- CONFIG ---
    channel     = 13
    year        = 2021
    months      = [10]
    days        = [23]
    # months = None
    # days = None
    # folder_root = "database"
    # extent      = [-77, -34, -43, -2]   # ou None
    extent = dat["extent"]  # [lon_min, lon_max, lat_min, lat_max]
    timeout     = 30.0
    req_workers = 16                  # None => auto; higher values will be capped
    server_ON = True
    
    
    folders = defining_folders(server_ON)
    
    if not print_settings_and_confirm(
        channel, year, months, days, extent, folders, timeout, req_workers
    ):
        print("Cancelado pelo usuário.")
        return

    max_workers = effective_workers(req_workers)
    

    
    # folder_root = Path(folder_root).resolve()

    summary = download_goes16_multiproc(
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

# ---------------- main via terminal (argparse) ----------------
def main_terminal():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download GOES-19 data from CPTEC by year/month/day."
    )
    parser.add_argument("-c", "--channel", type=int, default=13, help="GOES channel (1–16)")
    parser.add_argument("-y", "--year", type=int, required=True, help="Year (e.g., 2021)")
    parser.add_argument("-m", "--months", nargs="*", type=int, default=None,
                        help="Months (e.g., 10 11 12). Default: full year")
    parser.add_argument("-d", "--days", nargs="*", type=int, default=None,
                        help="Days of month (e.g., 20 21 22). Default: full month")
    parser.add_argument("-o", "--folder", type=str, default="database",
                        help="Output base folder (default: ./database)")
    parser.add_argument("-t", "--timeout", type=float, default=30.0,
                        help="Request timeout in seconds (default: 30)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Parallel workers (will be capped to CPU-1; default: auto)")
    parser.add_argument("--server", action="store_true")

    args = parser.parse_args()

    # fixed/project defaults (edit if needed)
    dat = np.load("mask_SouthAmerica.npz")
    extent = dat["extent"]  # [lon_min, lon_max, lat_min, lat_max]
    # extent      = [-77, -34, -43, -2]   # or None
    
    # folder_root = args.folder
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
    

    
    summary = download_goes16_multiproc(
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



    
    # main()              # sem terminal
    main_terminal()       # com terminal (argparse)

'''    
    Example of usage:
        1-Atention to the folders names, it changes depending of computer
        2-Atention to the GOES 16 for dates <= 03/2025 or 19 for dates >= 04/2025 
          on lines 196 and 197 maybe deploy an automatic change in the future
        3-Regional cut is made in this routine, pay attention to extent parameter
        
        ---Terminal commands        
        sshfs hisashi@150.163.46.4:/home/hisashi/cptec_codes ~/cptec_codes -o reconnect, uid=$(id -u), gid=$(id -g), default_permissions, umask=0002
        sshfs hisashi@150.163.46.4:/home/hisashi/database/cptec ~/cptec -o reconnect,uid=$(id -u),gid=$(id -g),default_permissions,umask=0002

        source mpi-env/bin/activate
        python ~/cptec_codes/download_v4.py -y 2025 -m 3 4 5 6 7 8 9 10 --server -w 10
'''    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
