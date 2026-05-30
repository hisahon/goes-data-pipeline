# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 15:58:31 2025

@author: hisas
"""

# -*- coding: utf-8 -*-
"""
GOES -> NumPy (.npz)
Lê arquivos GOES com seu read_goes_file() e salva arrays NumPy.

Saídas (.npz) contêm:
  - temp  : ndarray (mesma unidade que seu leitor retorna)
  - lats  : ndarray
  - lons  : ndarray
  - meta  : dict com info básica (arquivo, dt, extent)

Uso (terminal):
  python goes_to_numpy.py --in "/dados/goes/C13_*.nc" --out npy_out
  python goes_to_numpy.py --in "/dados/goes/**/*.nc" --out npy_out --recursive
  python goes_to_numpy.py --in "/dados/goes/C13_*.nc" --out npy_out --extent -60 -30 -40 -10
"""

# from __future__ import annotations
# import os
# import re
# import json
from pathlib import Path
# from datetime import datetime
import numpy as np
from tqdm import tqdm
from shared.detector import read_goes_file
from datetime import datetime
import re
# from typing import Iterable, Optional, Tuple

# ---- importe seu leitor aqui ----
# from seu_modulo import read_goes_file
# para este script, vou assumir que a função existe no mesmo diretório:
# from read_goes_file import read_goes_file   # adapte se necessário
# -*- coding: utf-8 -*-
"""
GOES -> NumPy (.npz)
Parallel version using multiprocessing with 4 workers.
"""

from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count

def extract_datetime_from_filename(f):
    """
    Extract datetime from full GOES file path.

    Example:
      /path/to/S10635346_202110010010.nc
        → '2021/10/01 00:10'

    Returns
    -------
    formatted : str or None
        Datetime string in 'YYYY/MM/DD HH:MM' format, or None if not found.
    """
    name = Path(f).stem  # get 'S10635346_202110010010'
    match = re.search(r'_(\d{12})', name)
    if not match:
        return None

    dt_str = match.group(1)
    dt = datetime.strptime(dt_str, "%Y%m%d%H%M")
    return dt.strftime("%Y/%m/%d %H:%M")



def make_output_path(f, src_root="database", dst_root="database_numpy", ext=".npz"):

    fpath = Path(f)
    parts = list(fpath.parts)

    try:
        # Replace root directory
        idx = parts.index(src_root)
        parts[idx] = dst_root
        out_path = Path(*parts).with_suffix(ext)
    except ValueError:
        # Fallback if src_root not found
        out_path = Path(str(fpath).replace(src_root, dst_root, 1)).with_suffix(ext)

    # Ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    skip = out_path.exists()

    return out_path, skip

def save_netcdf2numpy(out_path, temp, lat, lon):
    """Convert and save GOES data as compressed .npz."""
   
    temp = np.where(temp < 0., temp, 0.)
    temp16 = np.asarray(np.round(temp * 100, 1), dtype=np.int16)
    

    np.savez_compressed(out_path, temp=temp16, lats=lat, lons=lon)
    # return f"✓ Saved: {out_path.name} | shape={temp16.shape}"

#----------------------------------------------------------------


def _downsample2_2d(a, how="mean"):
    """Reduz 2x a resolução de um array 2D por média 2x2 (ignora NaN) ou por stride."""
    if a.ndim != 2:
        raise ValueError("Array 2D esperado.")
    H, W = a.shape
    H2, W2 = H - (H % 2), W - (W % 2)  # garante blocos 2x2
    a = a[:H2, :W2]
    if how == "mean":
        a = a.reshape(H2//2, 2, W2//2, 2)
        # média 2x2 preservando NaN como NaN se o bloco todo for NaN
        m = np.nanmean(np.nanmean(a, axis=3), axis=1)
        # onde todos eram NaN, nanmean vira NaN mesmo — ok.
        return m
    elif how == "stride":
        return a[::2, ::2]
    else:
        raise ValueError("how deve ser 'mean' ou 'stride'")

def _downsample1_1d(v, how="mean"):
    """Reduz 2x um vetor 1D por média de pares ou stride."""
    n2 = v.size - (v.size % 2)
    v = v[:n2]
    if how == "mean":
        return v.reshape(-1, 2).mean(axis=1)
    elif how == "stride":
        return v[::2]
    else:
        raise ValueError("how deve ser 'mean' ou 'stride'")

def save_netcdf2numpy_v2(out_path, temp, lat, lon, mode="mean", scale=100):
    """
    Converte e salva em .npz.
    mode:
      - 'mean'   -> downsample 2× por média 2×2
      - 'stride' -> downsample 2× por amostragem 1 a cada 2
      - 'none'   -> mantém tamanho original (sem downsample)
    scale:
      - fator de quantização (ex.: 100 => centésimos de °C em int16)
    """
    if temp.ndim != 2:
        raise ValueError("temp deve ser 2D")
    if (lat.ndim, lon.ndim) not in [(1,1), (2,2)]:
        raise ValueError("lat/lon devem ser (1D,1D) ou (2D,2D)")

    # 1) limitar positivos a 0 °C, como no seu fluxo
    temp_work = np.where(temp < 0.0, temp, 0.0)

    # 2) aplicar (ou não) o downsample
    if mode in ("mean", "stride"):
        temp_ds = _downsample2_2d(temp_work, how=mode)
        if lat.ndim == 1:
            lat_ds = _downsample1_1d(lat, how=mode)
            lon_ds = _downsample1_1d(lon, how=mode)
        else:
            lat_ds = _downsample2_2d(lat, how=mode)
            lon_ds = _downsample2_2d(lon, how=mode)
    elif mode == "none":
        temp_ds = temp_work
        lat_ds, lon_ds = lat, lon
    else:
        raise ValueError("mode deve ser 'mean', 'stride' ou 'none'")

    # 3) quantização para int16 (ex.: °C * 100)
    temp16 = np.rint(temp_ds * scale).astype(np.int16)

    # 4) salvar (inclui metadados simples)
    np.savez_compressed(
        out_path,
        temp=temp16,
        lats=lat_ds,
        lons=lon_ds,
        scale=np.int16(scale),
        units="celsius_x{:.0f}".format(scale),
        mode=mode
    )



























# ---- wrapper function for parallel processing ----
def process_one(f):
      # import inside worker to avoid issues
    extent = [-77, -34, -43, -2]
    sdate = extract_datetime_from_filename(f)
    out_path, skip = make_output_path(f, src_root="database", dst_root="database_numpy")
    if skip:
        return f"{sdate} [nc->npz] Success-Disk"
    try:
        temp, lat, lon = read_goes_file(f, extent=extent)
        save_netcdf2numpy(out_path, temp, lat, lon)
        #Path(f).unlink()  # delete original file
        return f"{sdate} [nc->npz] Success-Done"
    except:
        Path(f).unlink()  # delete original file
        return f"{sdate} [nc->npz] Broke-Deleting"


if __name__ == "__main__":
    from visual import list_files

    files_list = list_files("database")
    n_workers = 7  # or min(4, cpu_count())

    print(f"Starting parallel conversion with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        with tqdm(total=len(files_list),
                  desc="Processing files",
                  unit="file",
                  dynamic_ncols=True) as pbar:
            for msg in pool.imap_unordered(process_one, files_list):
                pbar.set_description(f"{msg[:40]}")
                pbar.update()

    print("✅ All files processed.")

