# -*- coding: utf-8 -*-
"""
GOES IR Mesoscale Convective System (MCS) Detector.

Reads GOES .npz snapshots (produced by `converter.py`), runs the cluster
detection pipeline from `detector.py` on each frame, writes a per-frame
report CSV and optionally a figure, and optionally prunes frames that
contain no cold clusters.

Three execution modes are supported:
  - serial      : one file at a time (`process_goes_folder`).
  - parallel    : multiprocessing Pool (`process_goes_parallel`).
  - MPI dynamic : master/worker over mpi4py (`process_goes_mpi_dynamic`).

The parallel backend is the primary path; the MPI and serial paths are
retained for large-cluster / debugging use.

Use `visual_terminal.py` as the CLI entry point, or edit the `__main__`
block at the bottom of this file.
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Iterable, List, Union

import numpy as np

# Module renames applied in this refactor:
#   dbscan     -> detector
#   goes2numpy -> converter
from shared.detector import detect_clusters_area6 as dca
from shared.detector import read_npz_goes
from shared.converter import extract_datetime_from_filename
from shared.download import _auto_workers


# =============================================================================
# File discovery
# =============================================================================
def list_files(root_folder: Union[str, Path],
               extensions: Iterable[str] = (".nc", ".nc4")) -> List[Path]:
    """
    Return every file under `root_folder` (recursively) whose name ends with
    any of the given extensions (case-insensitive). Result is sorted by name.

    Uses Path.rglob — convenient, but slower than `list_files_fast` on remote
    mounts because rglob issues more stat() calls under the hood.
    """
    root = Path(root_folder)
    if not root.is_dir():
        print(f"⚠️ Pasta não encontrada: {root}")
        return []

    # Normalize extensions: lowercase, guaranteed to start with a dot.
    exts = []
    for e in extensions:
        e = str(e).lower()
        if not e.startswith("."):
            e = "." + e
        exts.append(e)

    files = []
    for p in root.rglob("*"):
        if p.is_file():
            name = p.name.lower()
            if any(name.endswith(ext) for ext in exts):
                files.append(p)

    files.sort(key=lambda f: f.name)
    if not files:
        print(f"⚠️ Nenhum arquivo encontrado em {root}")
    return files


def is_mounted(path: Union[str, Path]) -> bool:
    """Best-effort check whether `path` is an actual mount point (Linux)."""
    p = Path(path)
    try:
        return os.path.ismount(p)
    except Exception:
        return False


def list_files_fast(root_folder: Union[str, Path],
                    extensions: Iterable[str] = (".nc", ".nc4"),
                    ignore_hidden: bool = True) -> List[Path]:
    """
    Faster alternative to `list_files` for large or remote trees (sshfs).

    Uses os.scandir + an explicit stack instead of Path.rglob, which avoids
    the extra stat() calls rglob triggers per entry. Intended for server
    runs where the database root is mounted remotely.
    """
    if not root_folder.is_dir():
        print(f"⚠️ Pasta não encontrada: {root_folder}")
        return []

    # Build the extension set once; comparison below is O(1) per file.
    exts = {("." + e.lower().lstrip(".")) for e in extensions}

    out: List[Path] = []
    stack = [root_folder]

    # Iterative DFS — flat memory profile and fewer syscalls than rglob.
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    name = entry.name
                    if ignore_hidden and name.startswith("."):
                        continue
                    full = cur / name
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(full)
                    else:
                        # Check suffix directly from the filename — no extra stat.
                        if Path(name).suffix.lower() in exts:
                            out.append(full)
        except PermissionError:
            print(f"⚠️ Sem permissão: {cur}")
        except FileNotFoundError:
            # Over sshfs an entry can disappear mid-walk if the link drops.
            continue

    out.sort(key=lambda p: p.name.lower())
    if not out:
        print(f"⚠️ Nenhum arquivo encontrado em {root_folder}")
    return out


# =============================================================================
# Path derivation
# =============================================================================
def figure_path(file_path, OPTIONS):
    """
    Map a GOES .npz filename to its figure + report destination paths.

    Example
    -------
    database_numpy/2021/10/S10635346_202110152020.npz
        -> figures/2021/288/C13_2021-288_2020.png
        -> reports/2021/288/C13_2021-288_2020.txt

    Returns
    -------
    dict with keys:
        fig_folder   : directory where the PNG lives
        fig_path     : full PNG path
        save_path    : full report (.txt/.csv) path
        time_string  : human-readable timestamp used in plot titles
    """
    file_path = Path(file_path)
    # Filename embeds YYYYMMDDHHMM after the underscore.
    m = re.search(r"_(\d{12})", file_path.name)
    if not m:
        return None, None, None

    ts = datetime.strptime(m.group(1), "%Y%m%d%H%M")
    year = ts.strftime("%Y")
    doy = ts.timetuple().tm_yday        # Day-of-year (1–366), used as folder name.
    hhmm = ts.strftime("%H%M")
    # Plot-title format: "YYYY [DD/MM][DOY] HH:MM"  e.g. "2021 [15/10][288] 20:20"
    ts_title = f"{ts:%Y} [{ts:%d/%m}][{doy:03d}] {ts:%H:%M}"

    fig_folder = OPTIONS["root_folder_fig"] / year / f"{doy:03d}"
    fig_name = f"C13_{year}-{doy:03d}_{hhmm}.png"
    fig_path = fig_folder / fig_name

    save_folder = OPTIONS["root_folder_reports"] / year / f"{doy:03d}"
    save_name = f"C13_{year}-{doy:03d}_{hhmm}.txt"
    save_path = save_folder / save_name

    return {
        "fig_folder": fig_folder,
        "fig_path": fig_path,
        "save_path": save_path,
        "time_string": ts_title,
    }


# =============================================================================
# Parallel backend (multiprocessing.Pool) — primary execution path
# =============================================================================
# def process_file_parallel(f, OPTIONS):
#     """
#     Multiprocessing-pool worker: processes one GOES frame.

#     Returns a short status string (logged by the caller). The possible outcomes:
#         1. Report already exists        -> skip.
#         2. NPZ fails to load            -> delete file and skip (likely corrupt).
#         3. Cold cluster found + plot    -> figure saved inside `dca`.
#         4. Cold cluster found, no plot  -> report only.
#         5. No cluster at all            -> optionally delete the NPZ.
#     """
#     try:
#         folders = figure_path(f, OPTIONS)

#         # (1) Skip frames that already have a report — idempotent re-runs.
#         if folders['save_path'].exists():
#             return f"{folders['time_string']} | Report exists [skipped] {f}"

#         # (2) Load the .npz and run detection. A broken .npz is treated as
#         #     unrecoverable and removed so later runs don't hit it again.
#         try:
#             temp, lats, lons = read_npz_goes(f)
#             dca(f, temp, OPTIONS, folders, plot=True)
#         except Exception:
#             f.unlink(missing_ok=True)
#             return f"{folders['time_string']} | Numpy file broken [deleted] {f}"

#         # (3–5) Branch on the detection outcome.
#         if cold_spot and plot_ON:
#             return f"{folders['time_string']} | Clusters found [figure]  {folders['fig_path']}"
#         elif cold_spot and not plot_ON:
#             return f"{folders['time_string']} | Clusters found [no figure]"
#         else:
#             # No cluster: optionally prune the .npz to keep the database lean.
#             if OPTIONS['delete_noCloud_file']:
#                 f.unlink(missing_ok=True)
#                 return f"{folders['time_string']} | No clusters found [deleted] {f}"
#             return f"{folders['time_string']} | No clusters found [keeping] {f}"

#     except Exception as e:
#         # Last-resort catch: anything that escapes the inner blocks ends up here.
#         # Soft-failing by design so one bad frame does not kill the entire pool.
#         print(f"Error processing {f.name}: {e}")

def process_file_parallel(f, OPTIONS):
    """Pool worker: same job as the serial `process_file`. Never deletes files."""
    try:
        folders = figure_path(f, OPTIONS)
        ts = folders["time_string"]

        if folders["fig_path"].exists():                 # already done -> skip
            return f"{ts} | skipped"

        temp, lats, lons = read_npz_goes(f)
        dca(f, temp, OPTIONS, folders, plot_ON=True)
        return f"{ts} | ok"

    except Exception as e:                               # returned, so the parent prints it
        return f"{f.name} | ERROR {type(e).__name__}: {e}"



# def process_goes_parallel(files, OPTIONS):
#     """Fan out `process_file_parallel` across a multiprocessing Pool."""
#     # Fall back to CPU-1 workers when the caller did not specify a count.
#     n_workers = OPTIONS['n_workers'] or _auto_workers()
#     print(f"Starting parallel processing with {n_workers} workers...")

#     # imap_unordered prints results as they arrive (rather than waiting for the
#     # original input order) and keeps all workers busy.
#     with Pool(processes=n_workers) as pool:
#         for msg in pool.imap_unordered(
#             partial(process_file_parallel, OPTIONS=OPTIONS), files
#         ):
#             print(msg)

#     print(f"All {len(files)} files processed.")

def process_goes_parallel(files, OPTIONS):
    """Fan out `process_file_parallel` across a multiprocessing Pool."""
    n_workers = OPTIONS['n_workers'] or _auto_workers()
    n = len(files)
    w = len(str(n))
    print(f"Starting parallel processing with {n_workers} workers...")

    counts = {"ok": 0, "skipped": 0, "ERROR": 0}
    # maxtasksperchild=1: every frame runs in a brand-new process, so the OS
    # frees ALL its memory before the next frame (slower start, no leaks).
    # `imap` keeps the input (chronological) order, so the log reads in time order.
    with Pool(processes=n_workers, maxtasksperchild=1) as pool:
        for i, msg in enumerate(
            pool.imap(partial(process_file_parallel, OPTIONS=OPTIONS), files), 1):
            print(f"[{i:>{w}}/{n}] {msg}")
            for k in counts:
                if f"| {k}" in msg:
                    counts[k] += 1

    print(f"Done: {counts['ok']} ok | {counts['skipped']} skipped | {counts['ERROR']} errors "
          f"(of {n} files).")



# =============================================================================
# MPI backend (mpi4py) — dynamic master/worker + static round-robin variants
# =============================================================================
# NOTE: functions in this section reference `MPI` at module scope. The current
# entry point imports mpi4py inside `main()` (lazy import), so these helpers
# assume they're only ever reached after `main()` ran through the `mpi_ON`
# branch. Keeping this as-is to match the historical behavior.

def filter_files_without_figures(files):
    """Return only the files whose figure does not yet exist on disk."""
    keep = []
    skipped = 0
    for f in files:
        f = Path(f).resolve()
        try:
            figpaths = figure_path(f)        # helper: returns (.., FIG_PATH, ..)
            if figpaths[1].exists():         # figure already present?
                skipped += 1
            else:
                keep.append(f)
        except Exception:
            # On a check error, prefer to reprocess rather than silently drop.
            keep.append(f)
    return keep, skipped


def process_file_mpi(f, opt):
    """Process one GOES file inside an MPI worker; tag output with host+rank."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    host = MPI.Get_processor_name()          # physical node name

    try:
        fpaths = figure_path(f)
        if fpaths[1].exists():
            return f"[{host} | rank {rank}] {f.name}: Skipping (figure exists)"

        date_report = extract_datetime_from_filename(f)
        temp, lats, lons = read_npz_goes(f)
        labels, cold_spot = dca(f, temp, opt, fpaths, lats, lons, plot=True)

        if cold_spot:
            return f"[{host[0:4]} | rank {rank}] {date_report}: Clusters found -> {fpaths[1]}"
        else:
            return f"[{host[0:4]} | rank {rank}] {date_report}: No clusters"

    except Exception as e:
        return f"[{host} | rank {rank}] {getattr(f, 'name', f)}: ERROR ({e})"


# MPI message tags used in the master/worker dance below.
TAG_TASK = 1   # Master sending a file path to a worker.
TAG_DONE = 2   # Worker reporting a completed file.
TAG_STOP = 3   # Master telling a worker to shut down.


def process_goes_mpi_dynamic(files, options):
    """
    Dynamic master/worker scheduling over MPI.

    Rank 0 acts as master: it seeds each worker with one task, then hands
    the next file to whichever worker finishes first. This keeps a slow
    node from becoming the critical path. Workers loop on recv() until
    they receive TAG_STOP.
    """
    comm = MPI.COMM_WORLD                    # global communicator (all ranks)
    rank = comm.Get_rank()                   # current process id (0 = master)
    # Guard against the typo'd capitalized form present in some mpi4py builds.
    size = comm.Get_Size() if hasattr(comm, "Get_Size") else comm.Get_size()

    # ---- Special case: a single process means no real MPI parallelism. ----
    if size == 1:
        files = [Path(p).resolve() for p in files]
        files.sort(key=lambda p: p.name)
        files, skipped = filter_files_without_figures(files)
        print(f"[Solo] {len(files)} a processar | {skipped} ignorados (figure exists)")
        for f in files:
            print(process_file_mpi(f, options), flush=True)
        print(f"\n✅ All {len(files)} files processed (no MPI workers).")
        return

    # ---- Only the master touches the file list up front. ----
    if rank == 0:
        files = [Path(p).resolve() for p in files]
        files.sort(key=lambda p: p.name)
        files, skipped = filter_files_without_figures(files)
        print(f"[Master] {len(files)} a processar | {skipped} ignorados (figure exists)")

    # Broadcast options so every rank sees the same config dict.
    options = comm.bcast(options if rank == 0 else None, root=0)

    if rank == 0:
        # ======================== MASTER ========================
        n_files = len(files)

        # No work? Release every worker right away.
        if n_files == 0:
            for worker in range(1, size):
                comm.send(None, dest=worker, tag=TAG_STOP)
            print("\n✅ No files to process. All workers stopped.")
            return

        next_idx = 0          # Index of the next file to hand out.
        completed = 0         # Number of DONE messages received.
        stopped = set()       # Workers that already received TAG_STOP.
        active = set()        # Workers currently processing a file.

        # Seed every worker with its first task. If there are fewer files
        # than workers, the leftover workers get STOP immediately.
        for worker in range(1, size):
            if next_idx < n_files:
                comm.send(files[next_idx], dest=worker, tag=TAG_TASK)
                active.add(worker)
                next_idx += 1
            else:
                comm.send(None, dest=worker, tag=TAG_STOP)
                stopped.add(worker)

        status = MPI.Status()                # used to inspect sender + tag
        while completed < n_files:
            # Block until any worker reports back; status tells us which one.
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_DONE, status=status)
            worker = status.Get_source()
            print(msg, flush=True)
            completed += 1

            # Either feed that worker another file, or retire it.
            if next_idx < n_files:
                comm.send(files[next_idx], dest=worker, tag=TAG_TASK)
                next_idx += 1
            else:
                comm.send(None, dest=worker, tag=TAG_STOP)
                stopped.add(worker)
                if worker in active:
                    active.remove(worker)

        # Defensive sweep: stop anyone still marked active (shouldn't happen).
        for worker in list(active):
            if worker not in stopped:
                comm.send(None, dest=worker, tag=TAG_STOP)
                stopped.add(worker)

        print(f"\n✅ All {n_files} files processed dynamically across {size} ranks.")

    else:
        # ======================== WORKERS ========================
        status = MPI.Status()
        while True:
            # Block on any message from master and inspect its tag.
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == TAG_STOP:
                break
            # task is a Path to a single .npz frame.
            msg = process_file_mpi(task, options)
            comm.send(msg, dest=0, tag=TAG_DONE)


def process_goes_mpi(files, options):
    """
    Static round-robin MPI scheduler — simpler alternative to the dynamic one.

    Each rank takes a contiguous slice `files[start:end]` and processes it
    independently. No master coordination, but load balance is fixed at
    start. Good enough when per-file cost is uniform.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Only the master walks + filters; workers receive the final list.
        files = [Path(p).resolve() for p in files]
        files.sort(key=lambda p: p.name)
        files_to_process, skipped = filter_files_without_figures(files)
        print(f"[Master] {len(files_to_process)} a processar | {skipped} ignorados (figure exists)")
        options = dict(options)              # defensive copy
    else:
        files_to_process = None
        options = None

    # Every rank needs the same list + options.
    files_to_process = comm.bcast(files_to_process, root=0)
    options = comm.bcast(options, root=0)

    # Static block split: the first `extra` ranks get one bonus file each.
    n = len(files_to_process)
    base, extra = divmod(n, size)
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    local_files = files_to_process[start:end]

    for f in local_files:
        msg = process_file_mpi(f, options)
        print(msg, flush=True)


# =============================================================================
# Serial backend (no parallelism, fallback/debug)
# =============================================================================
def process_file(f, OPTIONS, folders):
    """Single-file worker used by the serial path."""
    temp, lats, lons = read_npz_goes(f)
    dca(f, temp, OPTIONS, folders, plot_ON=True)

    # if cold_spot:
    #     ...
    # else:
    #     print(f"No clusters -> deleting: {f.name}")


def process_goes_folder(files, OPTIONS):
    """Run `process_file` for each frame, with inline progress reporting."""
    for i, f in enumerate(files):
        folders = figure_path(f, OPTIONS)

        # Skip anything whose figure is already present.
        if folders["fig_path"].exists():
            print(f"Skipping (figure exists): {f.name}")
            continue

        try:
            process_file(f, OPTIONS, folders)
        except Exception as e:
            print("  Skipping:", e)

        print(f"{i+1}/{len(files)} Processing: {f.name}")


# =============================================================================
# User prompt helper
# =============================================================================
def ask_yes_no(prompt="Prosseguir? [s/N]: ", default=False):
    """
    Interactive yes/no prompt that accepts PT or EN answers.

    Returns True for yes, False for no. Recognized:
        yes -> s / si / sim / y / yes
        no  -> n / nao / não / no
    An empty response returns `default`. When stdin is not a TTY (piped,
    nohup, cron), also returns `default` so the script runs unattended.
    """
    if not sys.stdin.isatty():
        return default

    yes = {"s", "si", "sim", "y", "yes"}
    no = {"n", "nao", "não", "no"}

    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[INFO] Confirmação cancelada pelo usuário.")
        return False

    if ans == "":
        return default
    if ans in yes:
        return True
    if ans in no:
        return False
    # Anything unrecognized: follow the documented default instead of looping.
    return default


# =============================================================================
# Main entry point
# =============================================================================
def main(server_ON=False, parallel_ON=False, mpi_ON=False,
         n_workers=None, output_key=""):
    """
    Set up paths + options, then dispatch to the chosen execution backend.

    Parameters
    ----------
    server_ON   : use the home-relative server layout (posix only).
    parallel_ON : run the multiprocessing-pool backend.
    mpi_ON      : run the dynamic MPI backend (requires mpi4py).
    n_workers   : worker count for the parallel backend (None = auto).
    output_key  : suffix appended to the figures/reports folder names,
                  useful when running several parameter sets side by side.
    """
    # -------------------------------------------------------------------------
    # Folder layout per platform / environment.
    # -------------------------------------------------------------------------
    if os.name == "nt":
        # Windows workstation.
        ROOT_FOLDER = Path("D:/goes-data-pipeline/database_numpy")
        ROOT_FOLDER_FIG = Path(f"D:/goes-data-pipeline/figures{output_key}")
        ROOT_FOLDER_REPORTs = Path(f"D:/goes-data-pipeline/reports{output_key}")
    elif os.name == "posix":
        if server_ON:
            # Remote / headless server: everything under $HOME.
            ROOT_FOLDER = Path("~/database_numpy").expanduser()
            ROOT_FOLDER_FIG = Path(f"~/cptec/figures{output_key}").expanduser()
            ROOT_FOLDER_REPORTs = Path(f"~/cptec/reports{output_key}").expanduser()
        else:
            # Local POSIX workstation (sshfs-mounted CPTEC layout).
            ROOT_FOLDER = Path("~/database/cptec/database_numpy").expanduser()
            ROOT_FOLDER_FIG = Path(f"~/database/cptec/figures{output_key}").expanduser()
            ROOT_FOLDER_REPORTs = Path(f"~/database/cptec/reports{output_key}").expanduser()
    else:
        raise RuntimeError(f"Sistema operacional não suportado: {os.name}")

    # Must be a tuple so the rglob / scandir helpers iterate it correctly.
    EXTENSIONS = (".npz",)

    # South America domain (coords + boolean mask), built offline once.
    dat = np.load("mask_SouthAmerica.npz")

    # -------------------------------------------------------------------------
    # Detection tuning. These are the knobs worth tweaking per run.
    # -------------------------------------------------------------------------
    OPTIONS = {
        "region_temperature_threshold": -30,    # Initial pixel filter (°C).
        "min_temperature_threshold": -30,       # Cluster must reach this min (°C).
        "min_area_km2": 1.e3,                   # Minimum cluster area.
        "smooth_sigma": 10.,                   # Gaussian smoothing radius (px).
        "delete_noCloud_file": True,            # Prune frames with no cluster.
        # Alternative parameter set kept for reference:
        # "region_temperature_threshold": -45,
        # "min_temperature_threshold": -70,
        # "min_area_km2": 200e3,
        # "smooth_sigma": 20.0,
        # "delete_noCloud_file": False,
        "pixel_km": 3.3423826253593094,         # Nominal GOES retangular pixel size.
        "extent": dat["extent"],                # [lon_min, lon_max, lat_min, lat_max]
        "root_folder_fig": ROOT_FOLDER_FIG,
        "root_folder_reports": ROOT_FOLDER_REPORTs,
        "n_workers": n_workers,                 # None => auto (CPU-1).
    }

    ROOT_FOLDER_FIG.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Config summary printed before any I/O-heavy work starts.
    # -------------------------------------------------------------------------
    print("--------------------------setup------------------------------------")
    print(f"[INFO] ROOT_FOLDER: {ROOT_FOLDER}")
    print(f"[INFO] ROOT_FOLDER_FIG: {ROOT_FOLDER_FIG}")
    print(f"[INFO] ROOT_FOLDER_REPORTs: {ROOT_FOLDER_REPORTs}")
    print(f"[INFO] MPI={mpi_ON} | PARALLEL={parallel_ON} | SERVER={server_ON}")
    print(f"[INFO] Deleting no Cluster Found Files: {OPTIONS['delete_noCloud_file']}")
    print(f"[INFO] Region Tmax: {OPTIONS['region_temperature_threshold']} C, "
          f"Cluster Tmax: {OPTIONS['min_temperature_threshold']} C")
    print(f"[INFO] Area min: {OPTIONS['min_area_km2']/1000}k km2, "
          f"Sigma Smooth: {OPTIONS['smooth_sigma']}")

    # -------------------------------------------------------------------------
    # Dispatch — three mutually exclusive backends.
    # -------------------------------------------------------------------------
    if mpi_ON:
        # Lazy import so the rest of the script works without mpi4py installed.
        from mpi4py import MPI
        if MPI is None:
            raise RuntimeError("mpi4py não encontrado. Instale ou remova o --mpi.")
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Only rank 0 walks the filesystem; workers get the list via bcast.
        if rank == 0:
            files = list_files(ROOT_FOLDER, extensions=EXTENSIONS)
            print(f"{len(files)} arquivos encontrados.")
        else:
            files = None
        process_goes_mpi_dynamic(files, OPTIONS)

    elif parallel_ON:
        # Use the fast scanner when on a remote mount; rglob otherwise.
        if server_ON:
            files = list_files_fast(ROOT_FOLDER, extensions=EXTENSIONS)
        else:
            files = list_files(ROOT_FOLDER, extensions=EXTENSIONS)

        print(f"[INFO] Found {len(files)} files, {n_workers} workers.")
        print("--------------------------setup------------------------------------")
        # Confirmation gate — auto-skipped when stdin is not a TTY.
        if ask_yes_no("Prosseguir com essas configurações? [s/N]: ", default=True):
            process_goes_parallel(files, OPTIONS)

    else:
        files = list_files(ROOT_FOLDER, extensions=EXTENSIONS)
        print(f"{len(files)} arquivos encontrados.")
        process_goes_folder(files, OPTIONS)


if __name__ == "__main__":
    # Quick in-file run. Use visual_terminal.py for the CLI-argument version.
    mpi_ON = False
    parallel_ON = True
    server_ON = False
    n_workers = 2
    output_key = ''

    main(server_ON, parallel_ON, mpi_ON, n_workers, output_key)