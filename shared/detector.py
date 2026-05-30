# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 12:14:48 2025

@author: hisas
"""

# -*- coding: utf-8 -*-
"""
GOES IR Convective Cluster Detection using DBSCAN
-------------------------------------------------
Detects cold cloud clusters (potential MCSs) in a single GOES-16 IR (Ch13) file.
Requires: scikit-learn, matplotlib, numpy, xarray
"""
# import re
import os
import numpy as np
from skimage.measure import regionprops, label, find_contours
from scipy.ndimage import gaussian_filter
from pathlib import Path




def area_calc_km2(lats, lons, altitude_km=12, ddeg_fallback=0.029064196742254873):
    """
    Compute pixel area (km²) varying with latitude.
    Works for 1D or 2D latitude arrays.
    """
    if lats is not None:
        # handle 1D or 2D lats safely
        if lats.ndim == 1:
            dlat_deg = np.abs(np.nanmean(np.diff(lats)))
        else:
            dlat_deg = np.abs(np.nanmean(np.diff(lats, axis=0)))
    else:
        dlat_deg = ddeg_fallback

    # same spacing assumed for lon
    R = 6371.0 + altitude_km
    km_per_deg = 2 * np.pi * R / 360.0
    dlat_km = km_per_deg * dlat_deg
    dlon_km = dlat_km  # if grid is quasi-square

    # compute area map
    if lats is not None:
        if lats.ndim == 1:
            # broadcast to 2D using outer product
            area_km2_1d = dlat_km * dlon_km * np.cos(np.deg2rad(lats))
            area_km2 = np.repeat(area_km2_1d[:, np.newaxis], len(lons), axis=1)
        else:
            # 2D lat grid
            area_km2 = dlat_km * dlon_km * np.cos(np.deg2rad(lats))
    else:
        # fallback: constant area
        area_km2 = np.full((len(lats), len(lons)), dlat_km * dlon_km)

    return area_km2

def pix2ll(y_idx, x_idx, temp, lats, lons, use_geo):
    y_idx = np.clip(np.asarray(y_idx), 0, temp.shape[0]-1).astype(int)
    x_idx = np.clip(np.asarray(x_idx), 0, temp.shape[1]-1).astype(int)
    if not use_geo:
        return x_idx, y_idx
    if np.ndim(lats) == 2 and np.ndim(lons) == 2:
        return lons[y_idx, x_idx], lats[y_idx, x_idx]
    return lons[x_idx], lats[y_idx]



# def temperature_plot(temp, lats, lons, t_threshold, min_area_km2,
#                      props, labels_filtered, fp):
#     fig, ax = plt.subplots(figsize=(10, 8))
#     use_geo = (lats is not None) and (lons is not None)
    
#     im = ax.pcolormesh(lons, lats, temp, cmap="gist_ncar",
#                            shading="auto", vmin=-80, vmax=0)

#     title = f"Cold Clusters (T<{t_threshold}°C, area>{min_area_km2:.0f} km², n={len(props)}), {fp[2]}"
#     ax.set_title(title)
#     fig.colorbar(im, ax=ax, label="Brightness Temp (°C)", shrink=1., aspect=45, pad=0.001)
#     pe = [patheffects.withStroke(linewidth=2.2, foreground="black")]
    
    
    
#     for i, p in enumerate(props, 1):
#         mask_i = labels_filtered == i
#         cluster_min = np.nanmin(temp[mask_i])
        
#         for c in find_contours(mask_i.astype(float), 0.5):
#             lon_line, lat_line = pix2ll(c[:, 0], c[:, 1], temp, lats, lons, use_geo)
#             ax.plot(lon_line, lat_line, color='black', linewidth=0.7, path_effects=pe)
    
#         cy, cx = p.centroid
#         lon_c, lat_c = pix2ll(cy, cx, temp, lats, lons, use_geo)
#         area_str = f"{int(p.real_area_km2/1000):,}k km² \n{cluster_min:.1f} °C"
        
#         if not cluster_min <= -65:
#             ax.text(lon_c, lat_c, area_str, color="white", ha="center", va="center", fontsize=8, path_effects=pe)
#         else:
#             ax.text(lon_c, lat_c, area_str, color="white", ha="center", va="center", fontsize=9, fontweight='bold', path_effects=pe)
    
#     ax.set_xlabel("Longitude (°)")
#     ax.set_ylabel("Latitude (°)")
#     ax.set_aspect("equal", adjustable="box")
#     ax.grid(True, linestyle="--", alpha=0.4)
    
#     Path(fp[0]).mkdir(parents=True, exist_ok=True)
#     # fig.tight_layout()
#     fig.savefig(fp[1], dpi=65, bbox_inches='tight')
    
    
#     plt.close('all')
    



def temperature_plot_cartopy(temp, lats, lons, t_threshold, min_area_km2,
                     props, labels_filtered, folders):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.patheffects as patheffects   # <-- ADD THIS IMPORT

    mpl.use("Agg", force=True)   # backend sem GUI
    plt.ioff()
    
    """
    Cartopy-based plot of GOES temperature with cluster contours and annotations.
    """
    if lats is None or lons is None:
        raise ValueError("Cartopy plot requires lats and lons arrays in degrees.")

    # --- Figure & geo axes ---
    fig = plt.figure(figsize=(7, 7))
    proj = ccrs.PlateCarree()

    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # --- Core image ---
    im = ax.pcolormesh(
        lons, lats, temp,
        cmap="gist_ncar",
        shading="auto",
        vmin=-80, vmax=0,
        rasterized=True,
        transform=ccrs.PlateCarree()  # data are in lat/lon
    )

    # --- Extent (with a small pad) ---
    try:
        xmin, xmax = float(np.nanmin(lons)), float(np.nanmax(lons))
        ymin, ymax = float(np.nanmin(lats)), float(np.nanmax(lats))
        pad_x = (xmax - xmin) * 0.02
        pad_y = (ymax - ymin) * 0.02

        ax.set_extent([xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y], crs=ccrs.PlateCarree())

    except Exception:
        pass  # keep default if bounds fail

    # --- Coastlines, borders, etc. ---
    ax.coastlines(resolution="110m", color="black", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.3)
    # Optional extras:
    # ax.add_feature(cfeature.LAKES, edgecolor="white", facecolor="none", linewidth=0.3)
    # ax.add_feature(cfeature.RIVERS, edgecolor="lightblue", linewidth=0.2)

    # --- Title & slim colorbar ---
    title = f"(T<{t_threshold}°C, area>{int(min_area_km2/1000)}k km², n={len(props)}), {folders['time_string']}"
    ax.set_title(title)

    cbar = fig.colorbar(
        im, ax=ax,
        label="Brightness Temp (°C)",
        fraction=0.026, pad=0.01, aspect=35
    )
    cbar.ax.tick_params(labelsize=8)

    # --- Gridlines with labels ---
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # --- Contours & labels ---
    pe = [patheffects.withStroke(linewidth=2.2, foreground="black")]

    for i, p in enumerate(props, 1):
        mask_i = labels_filtered == i
        cluster_min = np.nanmin(temp[mask_i])

        # Outlines for every cluster
        for c in find_contours(mask_i.astype(float), 0.5):
            lon_line, lat_line = pix2ll(c[:, 0], c[:, 1], temp, lats, lons, use_geo=True)
            ax.plot(
                lon_line, lat_line,
                color="black", linewidth=0.7, path_effects=pe,
                transform=ccrs.PlateCarree()
            )

        # Centroid + annotation
        cy, cx = p.centroid
        lon_c, lat_c = pix2ll(cy, cx, temp, lats, lons, use_geo=True)
        area_str = f"{int(p.real_area_km2/1000):,}k km²\n{cluster_min:.1f} °C"

        # Keep your original highlight logic (bold for ≤ -65 °C)
        if cluster_min <= -65:
            fsize, fweight, txt_color = 9, "bold", "white"
        else:
            fsize, fweight, txt_color = 8, "normal", "white"

        ax.text(
            lon_c, lat_c, area_str,
            color=txt_color, fontsize=fsize, fontweight=fweight,
            ha="center", va="center",
            path_effects=pe,
            transform=ccrs.PlateCarree()
        )

    # --- Save (cropped & lightweight) ---
    folders['fig_folder'].mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0)
    fig.savefig(folders['fig_path'], dpi=70, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def yline(lons, lon0, lat0, ang):
    m = np.tan(np.deg2rad(ang))
    return m * (lons - lon0) + lat0  # (nx,)

def region_mask_top_bottom(lats, lons, inclusive=True):
    """
    Retorna uma máscara 2D True entre o envelope:
      topo  = min(linhas_topo)   (acima é False)
      base  = max(linhas_base)   (abaixo é False)
    """
    top_lines = [
        (-49.000, -10.000,  40),
        (-38.370,  -3.227, -28),
        (-74.294, -37.175,  78),
    ]
    bottom_lines = [
        (-53.765, -35.276,  43),
        (-37.072, -15.273,  66),
    ]

    Ytop = np.min(np.stack([yline(lons, *p) for p in top_lines]), axis=0)     # (nx,)
    Ybot = np.max(np.stack([yline(lons, *p) for p in bottom_lines]), axis=0)  # (nx,)
    L = lats[:, None]  # (ny,1)
    return ((L <= Ytop) & (L >= Ybot)) if inclusive else ((L < Ytop) & (L > Ybot))


def detect_clusters_area(f, temp, OPTIONS, fp, lats=None, lons=None, plot=True):

    t_threshold = OPTIONS['region_temperature_threshold']
    tmin_threshold = OPTIONS['min_temperature_threshold']
    min_area_km2 = OPTIONS['min_area_km2']
    smooth_sigma = OPTIONS['smooth_sigma']

    area_km2 = area_calc_km2(lats, lons)
    
    temp_smooth = np.where((temp > -150.), temp, 0)
    temp_smooth = gaussian_filter(temp_smooth, sigma=smooth_sigma)

    mask = region_mask_top_bottom(lats, lons)
    mask &= temp_smooth < t_threshold
    
    

    labeled = label(mask)

            
    props = []
    # cold_clusters = []
    cold_spot = False
    labels_filtered = np.zeros_like(labeled, dtype=np.int32)
    i_new = 1
    
    # --- Loop through all regions ---
    for p in regionprops(labeled, intensity_image=temp_smooth):
        mask_i = (labeled == p.label)
        
        area_real_km2 = np.nansum(area_km2[mask_i])
        cluster_min = np.nanmin(temp[mask_i])
    
        # --- Apply combined area/temp criteria ---
        if not (area_real_km2 >= min_area_km2 and cluster_min <= tmin_threshold):
            continue
    
        # Store custom attributes
        p.real_area_km2 = area_real_km2
        p.cluster_min = cluster_min
    
        # Save region
        labels_filtered[mask_i] = i_new
        props.append(p)
    
        # If sufficiently cold, mark as cold cluster
        # cold_clusters.append(p)
        cold_spot = True
        i_new += 1

    # --- Plotting section ---
    if plot and cold_spot:
        
        # temperature_plot(temp, lats, lons, t_threshold, min_area_km2, props, labels_filtered, fp)
        temperature_plot_cartopy(temp, lats, lons, t_threshold, min_area_km2, props, labels_filtered, fp)
        

    return labels_filtered, cold_spot



def detect_clusters_area6(f, temp, OPTIONS, folders, plot=True):
# def detect_clusters_area6(f, temp, OPTIONS, folders, lats=None, lons=None, plot=True):
    t_threshold    = OPTIONS['region_temperature_threshold']
    tmin_threshold = OPTIONS['min_temperature_threshold']
    min_area_km2   = OPTIONS['min_area_km2']
    smooth_sigma   = OPTIONS['smooth_sigma']
    
    
    dat = np.load("mask_SouthAmerica.npz")
    lats   = dat["lats"]
    lons   = dat["lons"]
    # extent = dat["extent"]  # [lon_min, lon_max, lat_min, lat_max]
    mask   = dat["mask"].astype(bool)
    
    # --- pré-processamento ---
    area_km2 = area_calc_km2(lats, lons)
    temp_smooth = np.where((temp > -150.), temp, 0)
    temp_smooth = gaussian_filter(temp_smooth, sigma=smooth_sigma)

    # mask = region_mask_top_bottom(lats, lons)
    mask &= temp_smooth < t_threshold
    labeled = label(mask)

    # --- preparar saída: usa EXATAMENTE output_file_path ---
    # 
    header = "centroid_lat,centroid_lon,area_km2,temp_min_C,temp_mean_C,temp_std_C,n_clusters\n"
    linhas = []  # acumulador


    props = []
    # cold_spot = False
    labels_filtered = np.zeros_like(labeled, dtype=np.int32)
    i_new = 1
    plot_ON = False
    for p in regionprops(labeled, intensity_image=temp_smooth):
        mask_i = (labeled == p.label)

        area_real_km2 = float(np.nansum(area_km2[mask_i]))
        cluster_min   = float(np.nanmin(temp[mask_i]))

        # critério combinado
        if not (area_real_km2 >= min_area_km2 and cluster_min <= tmin_threshold):
            continue
        

            
        cluster_mean = float(np.nanmean(temp[mask_i]))
        cluster_std = float(np.nanstd(temp[mask_i]))

        #media ponderada com temperatura
        rows, cols = np.nonzero(mask_i)

        # pesos: mais frio => maior peso
        w = -temp[mask_i]                      # inverte sinal para privilegiar frio
        w = np.where(np.isfinite(w), w, 0.0)   # remove NaN/Inf
        w = np.clip(w, 0.0, None)              # garante não-negativo
        if w.sum() == 0:
            # fallback: média simples dos índices (caso todas as temperaturas sejam iguais/NaN)
            ri = int(np.round(rows.mean()))
            ci = int(np.round(cols.mean()))
            lat_c = float(lats[ri])            # lats e lons são 1D
            lon_c = float(lons[ci])
        else:
            # médias ponderadas em coordenadas geográficas (1D)
            lat_c = float(np.average(lats[rows], weights=w))
            lon_c = float(np.average(lons[cols], weights=w))
        
        
        # atualizar estruturas existentes
        p.real_area_km2 = area_real_km2
        p.cluster_min   = cluster_min
        labels_filtered[mask_i] = i_new
        props.append(p)
        # cold_spot = True
        linhas.append(f"{lat_c:.3f},{lon_c:.3f},{area_real_km2:.2f},{cluster_min:.3f},{cluster_mean:.3f},{cluster_std:.3f},{i_new}\n")
        
        #if (area_real_km2 >= 700e3 and cluster_min <= -70):
        #    plot_ON = True
        
        i_new += 1
        

        

    # --- só cria/escreve o arquivo se houver pelo menos 1 cluster aprovado ---
    if linhas:

        folders['save_path'].parent.mkdir(parents=True, exist_ok=True)
        with folders['save_path'].open("w", encoding="utf-8", newline="") as fh:
            fh.write(header)
            fh.writelines(linhas)
        
    if plot_ON:
        temperature_plot_cartopy(temp, lats, lons, t_threshold, min_area_km2, props, labels_filtered, folders)
        
    return labels_filtered, i_new>1, plot_ON



def read_goes_file(file_path, extent=None):
    import xarray as xr
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with xr.open_dataset(file_path) as ds:
        if "Band1" not in ds:
            raise KeyError(f"'Band1' variable not found in {file_path}")

        band = ds["Band1"]
        lats = ds["lat"]
        lons = ds["lon"]

        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            ilon_min = int(abs(lons.values - lon_min).argmin())
            ilon_max = int(abs(lons.values - lon_max).argmin())
            ilat_min = int(abs(lats.values - lat_min).argmin())
            ilat_max = int(abs(lats.values - lat_max).argmin())
            band = band.isel(lat=slice(ilat_min, ilat_max), lon=slice(ilon_min, ilon_max))
            lats = lats.isel(lat=slice(ilat_min, ilat_max))
            lons = lons.isel(lon=slice(ilon_min, ilon_max))

        # >>> simples e leve: float32 <<<
        temp = band.astype("float32").values / 100.0 - np.float32(273.13)
        lats = lats.values.astype("float32", copy=False)
        lons = lons.values.astype("float32", copy=False)

    return temp, lats, lons

def read_npz_goes(file_path):
    """
    Load a saved GOES .npz and return (temp, lats, lons) exactly as stored.

    Expected keys in the archive: 'temp', 'lats', 'lons'.
    """
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(fp)

    with np.load(fp, allow_pickle=False) as npz:
        try:
            temp = npz["temp"]/100.
            lats = npz["lats"]
            lons = npz["lons"]
        except KeyError as e:
            raise KeyError(f"Missing key in {fp.name}: {e}")

    return temp, lats, lons

if __name__ == "__main__":
    file_path = Path("paper/ch13/2021/10/S10635346_202110231700.nc") #CPTEC file
    
    temp, lats, lons = read_goes_file(file_path)
    
    labels, props = detect_clusters_area(
        temp, lats, lons,
        t_threshold=-45,
        min_area_km2=150000,  # adjust depending on your MCS definition
        pixel_km=3.3423826253593094,        # 2 km per pixel for CONUS domain
        smooth_sigma=20.0,
        plot=True
    )
