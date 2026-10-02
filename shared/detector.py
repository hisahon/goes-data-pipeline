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
import matplotlib as mpl


# =============================================================================
# Escala de cores do canal 13 (10,35 µm)
# =============================================================================
# Faixas coloridas de 10 °C entre -80 e -30 °C e degradê de cinza claro -> preto
# de -30 a 40 °C. Temperaturas repetidas criam os "saltos" de cor entre faixas.
VMIN_TEMP = -80
VMAX_TEMP = 40

_TB_PONTOS = [
    (-80, (224, 131, 224)), (-70, (255, 209, 255)),  # magenta -> rosa claro
    (-70, (0, 0, 255)),     (-60, (82, 69, 255)),    # azul
    (-60, (47, 218, 255)),  (-50, (156, 227, 245)),  # ciano
    (-50, (255, 255, 97)),  (-40, (235, 255, 122)),  # amarelo
    (-40, (255, 101, 1)),   (-30, (255, 158, 70)),   # laranja
    (-30, (176, 176, 176)), (40, (0, 0, 0)),         # cinza claro -> preto
]
CMAP_GOES = mpl.colors.LinearSegmentedColormap.from_list(
    "tb_realcada",
    [((t - VMIN_TEMP) / (VMAX_TEMP - VMIN_TEMP), [v / 255 for v in c])
     for t, c in _TB_PONTOS],
    N=1024,
)




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


def temperature_plot_cartopy(temp, lats, lons, t_threshold, min_area_km2,
                             props, labels_filtered, folders,
                             deep_conv_threshold=-65):

    import datetime
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.patheffects as patheffects
    mpl.use("Agg", force=True)
    plt.ioff()

    """
    Cartopy-based plot of GOES brightness temperature with cluster contours.
    Detection: contiguous pixels colder than `t_threshold` whose area exceeds
    `min_area_km2`. Clusters colder than `deep_conv_threshold` flagged as deep convection.
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
        cmap=CMAP_GOES,
        shading="auto",
        vmin=VMIN_TEMP, vmax=VMAX_TEMP,
        rasterized=True,
        transform=ccrs.PlateCarree()
    )

    # --- Extent (with a small pad) ---
    try:
        xmin, xmax = float(np.nanmin(lons)), float(np.nanmax(lons))
        ymin, ymax = float(np.nanmin(lats)), float(np.nanmax(lats))
        pad_x = (xmax - xmin) * 0.02
        pad_y = (ymax - ymin) * 0.02
        ax.set_extent([xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y],
                      crs=ccrs.PlateCarree())
    except Exception:
        pass

    # --- Coastlines, borders ---
    ax.coastlines(resolution="110m", color="black", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.3)

    # --- Summary stats (for title, caption & embedded metadata) ---
    n_clusters = len(props)
    total_area_k = sum(getattr(p, "real_area_km2", 0) for p in props) / 1000.0
    try:
        coldest = float(np.nanmin(temp[labels_filtered > 0])) if n_clusters else float("nan")
    except Exception:
        coldest = float("nan")

    # --- Compact title (centered over the MAP, not the whole figure) ---
    ax.set_title(
        f"{folders['time_string']} UTC | "
        f"Isotherms = {t_threshold} °C | "
        f"Clusters = {len(props)}"
        ,
        fontsize=9, fontweight="bold", pad=6
    )

    # --- Colorbar with threshold reference lines ---
    cbar = fig.colorbar(
        im, ax=ax,
        label="Brightness temperature (°C)",
        fraction=0.026, pad=0.01, aspect=45,
        ticks=list(range(VMIN_TEMP, VMAX_TEMP + 1, 10)),
    )
    cbar.ax.tick_params(labelsize=8)
    # detection threshold (solid) and deep-convection threshold (dotted)
    # cbar.ax.axhline(t_threshold, color="white", linewidth=1.2)
    cbar.ax.axhline(t_threshold, color="lime", linewidth=2.)
    # cbar.ax.axhline(deep_conv_threshold, color="black", linewidth=0.8, linestyle=":")

    # --- Gridlines with labels ---
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # --- Contours & labels ---
    pe = [patheffects.withStroke(linewidth=1.5, foreground="black")]
    for i, p in enumerate(props, 1):
        mask_i = labels_filtered == i
        cluster_min = np.nanmin(temp[mask_i])

        for c in find_contours(mask_i.astype(float), 0.5):
            lon_line, lat_line = pix2ll(c[:, 0], c[:, 1], temp, lats, lons, use_geo=True)
            ax.plot(
                lon_line, lat_line,
                color="lime", linewidth=0.7, path_effects=pe,
                transform=ccrs.PlateCarree()
            )
        

        if False:
            cy, cx = p.centroid
            lon_c, lat_c = pix2ll(cy, cx, temp, lats, lons, use_geo=True)
            area_str = f"{int(p.real_area_km2/1000):,}k km²\n{cluster_min:.1f} °C"
    
            if cluster_min <= deep_conv_threshold:
                fsize, fweight = 7, "bold"
            else:
                fsize, fweight = 6, "normal"
            
            # fsize, fweight = 6, "bold"
            ax.text(
                lon_c, lat_c, area_str,
                color="white", fontsize=fsize, fontweight=fweight,
                ha="center", va="center",
                path_effects=pe,
                transform=ccrs.PlateCarree()
            )   
        
    # # --- Reference marker (red star) ---
    # ax.scatter(
    #     -47.81, -21.18,
    #     marker="*", s=180,
    #     color="red", edgecolor="black", linewidth=0.6,
    #     transform=ccrs.PlateCarree(),
    #     zorder=5
    # )
    # # --- Parameter caption below the map (what the numbers mean) ---
    # caption = (
    #     f"Detection: contiguous pixels with brightness temperature "
    #     f"< {t_threshold} °C (contour line),\n"
    #     f"minimum cluster area {min_area_km2/1000:.0f}k km².  "
    #     f"Bold labels: deep convection (≤ {deep_conv_threshold} °C)."
    # )
    # ax.text(0.5, -0.16, caption, transform=ax.transAxes,
    #         ha="center", va="top", fontsize=7, color="0.30")

    # --- Embedded PNG metadata (all the verbose provenance lives here) ---
    png_meta = {
        "Title": f"GOES deep convective cloud detection {folders['time_string']}",
        "Description": (f"detection T<{t_threshold}C, min area {min_area_km2/1000:.0f}k km2, "
                        f"deep-convection <={deep_conv_threshold}C, n={n_clusters}, "
                        f"total {total_area_k:.0f}k km2, coldest {coldest:.1f}C"),
        "Source": "GOES ABI",                    # ajuste p/ GOES-16/19 + banda (ex.: Band 13 ~10.3 um)
        "Software": "matplotlib + cartopy",
        "Creation Time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # --- Save ---
    folders['fig_folder'].mkdir(parents=True, exist_ok=True)
    fig.savefig(folders['fig_path'], dpi=150, bbox_inches="tight",
                pad_inches=0.05, metadata=png_meta)
    plt.close(fig)



# def temperature_plot_cartopy(temp, lats, lons, t_threshold, min_area_km2,
#                      props, labels_filtered, folders):
    
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#     import cartopy.crs as ccrs
#     import cartopy.feature as cfeature
#     import matplotlib.patheffects as patheffects   # <-- ADD THIS IMPORT

#     mpl.use("Agg", force=True)   # backend sem GUI
#     plt.ioff()
    
#     """
#     Cartopy-based plot of GOES temperature with cluster contours and annotations.
#     """
#     if lats is None or lons is None:
#         raise ValueError("Cartopy plot requires lats and lons arrays in degrees.")

#     # --- Figure & geo axes ---
#     fig = plt.figure(figsize=(7, 7))
#     proj = ccrs.PlateCarree()

#     ax = fig.add_subplot(1, 1, 1, projection=proj)

#     # --- Core image ---
#     im = ax.pcolormesh(
#         lons, lats, temp,
#         cmap="gist_ncar",
#         shading="auto",
#         vmin=-80, vmax=0,
#         rasterized=True,
#         transform=ccrs.PlateCarree()  # data are in lat/lon
#     )

#     # --- Extent (with a small pad) ---
#     try:
#         xmin, xmax = float(np.nanmin(lons)), float(np.nanmax(lons))
#         ymin, ymax = float(np.nanmin(lats)), float(np.nanmax(lats))
#         pad_x = (xmax - xmin) * 0.02
#         pad_y = (ymax - ymin) * 0.02

#         ax.set_extent([xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y], crs=ccrs.PlateCarree())

#     except Exception:
#         pass  # keep default if bounds fail

#     # --- Coastlines, borders, etc. ---
#     ax.coastlines(resolution="110m", color="black", linewidth=0.5)
#     ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.3)
#     # Optional extras:
#     # ax.add_feature(cfeature.LAKES, edgecolor="white", facecolor="none", linewidth=0.3)
#     # ax.add_feature(cfeature.RIVERS, edgecolor="lightblue", linewidth=0.2)

#     # --- Title & slim colorbar ---
#     title = f"(T<{t_threshold}°C, area>{int(min_area_km2/1000)}k km², n={len(props)}), {folders['time_string']}"
#     ax.set_title(title)

#     cbar = fig.colorbar(
#         im, ax=ax,
#         label="Brightness Temp (°C)",
#         fraction=0.026, pad=0.01, aspect=35
#     )
#     cbar.ax.tick_params(labelsize=8)

#     # --- Gridlines with labels ---
#     gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
#                       alpha=0.5, linestyle="--")
#     gl.top_labels = False
#     gl.right_labels = False

#     # --- Contours & labels ---
#     pe = [patheffects.withStroke(linewidth=2.2, foreground="black")]

#     for i, p in enumerate(props, 1):
#         mask_i = labels_filtered == i
#         cluster_min = np.nanmin(temp[mask_i])

#         # Outlines for every cluster
#         for c in find_contours(mask_i.astype(float), 0.5):
#             lon_line, lat_line = pix2ll(c[:, 0], c[:, 1], temp, lats, lons, use_geo=True)
#             ax.plot(
#                 lon_line, lat_line,
#                 color="black", linewidth=0.7, path_effects=pe,
#                 transform=ccrs.PlateCarree()
#             )

#         # Centroid + annotation
#         cy, cx = p.centroid
#         lon_c, lat_c = pix2ll(cy, cx, temp, lats, lons, use_geo=True)
#         area_str = f"{int(p.real_area_km2/1000):,}k km²\n{cluster_min:.1f} °C"

#         # Keep your original highlight logic (bold for ≤ -65 °C)
#         if cluster_min <= -65:
#             fsize, fweight, txt_color = 9, "bold", "white"
#         else:
#             fsize, fweight, txt_color = 8, "normal", "white"

#         ax.text(
#             lon_c, lat_c, area_str,
#             color=txt_color, fontsize=fsize, fontweight=fweight,
#             ha="center", va="center",
#             path_effects=pe,
#             transform=ccrs.PlateCarree()
#         )

#     # --- Save (cropped & lightweight) ---
#     folders['fig_folder'].mkdir(parents=True, exist_ok=True)
#     fig.tight_layout(pad=0)
#     fig.savefig(folders['fig_path'], dpi=70, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)

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


def edge_correction(sigma, sigma_ref=25.0, delta_ref=20.0):
    """Quanto abaixar (°C) o campo suavizado: linear, 0 em sigma=0 e delta_ref em sigma=sigma_ref."""
    return delta_ref * float(sigma) / sigma_ref


# =============================================================================
# Máscaras de clusters (para filtrar dados de raios depois)
# =============================================================================
MASK_DIR = Path("grid_masks")   # pasta das máscaras (relativa ao diretório de execução)


def mask_path_from_csv(csv_path, mask_dir=None):
    """
    Caminho da máscara: <mask_dir>/<nome_do_csv>_labels.npz
    (um arquivo por timestamp; o nome do CSV já é único por timestamp).
    mask_dir=None usa MASK_DIR ('grid_masks').
    """
    mask_dir = Path(mask_dir) if mask_dir is not None else MASK_DIR
    return mask_dir / (Path(csv_path).stem + "_labels.npz")


def save_labels_npz(path, labels, lats, lons, time_string="", **params):
    """
    Salva a máscara de labels de UM timestamp do GOES, com os eixos de lat/lon.

    Chaves do arquivo:
      labels      int16 (ny, nx)  0 = fora; k = cluster k (mesmo n_clusters do CSV)
      lats, lons  float32 (1D)    centros das células da grade
      time_string str             timestamp da imagem
      + parâmetros da detecção passados em **params (ex.: t_threshold)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.asarray(labels).astype(np.int16),
        lats=np.asarray(lats, dtype=np.float32),
        lons=np.asarray(lons, dtype=np.float32),
        time_string=np.array(str(time_string)),
        **{k: np.float32(v) for k, v in params.items()},
    )


def load_labels_npz(path):
    """Lê um *_labels.npz e devolve (labels, lats, lons, time_string)."""
    with np.load(Path(path), allow_pickle=False) as d:
        return d["labels"], d["lats"], d["lons"], str(d["time_string"])


def filtrar_raios(lat_r, lon_r, labels, lats, lons):
    """
    Marca quais raios caem dentro dos clusters de UMA máscara (um timestamp).

    lat_r, lon_r : posições dos raios (graus)
    labels, lats, lons : saída de load_labels_npz (grade regular, 1D)

    Retorna (dentro, cluster_id):
      dentro     bool (n,)   True se o raio está dentro de algum cluster
      cluster_id int16 (n,)  número do cluster (0 = fora ou fora da grade)
    """
    lat_r = np.asarray(lat_r, dtype=float)
    lon_r = np.asarray(lon_r, dtype=float)

    dlat = (float(lats[-1]) - float(lats[0])) / (len(lats) - 1)   # funciona com lats crescente ou decrescente
    dlon = (float(lons[-1]) - float(lons[0])) / (len(lons) - 1)
    iy = np.rint((lat_r - float(lats[0])) / dlat).astype(int)
    ix = np.rint((lon_r - float(lons[0])) / dlon).astype(int)

    ok = (iy >= 0) & (iy < len(lats)) & (ix >= 0) & (ix < len(lons))   # raios fora da grade
    cluster_id = np.zeros(lat_r.shape, dtype=np.int16)
    cluster_id[ok] = labels[iy[ok], ix[ok]]
    return cluster_id > 0, cluster_id


def detect_clusters_area6(f, temp, OPTIONS, folders, plot_ON=True):
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
    temp_smooth = temp_smooth - edge_correction(smooth_sigma)

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
    # plot_ON = True
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

        # --- máscara de labels (um arquivo por timestamp do GOES) ---
        save_labels_npz(
            mask_path_from_csv(folders['save_path'], folders.get('mask_dir')),
            labels_filtered, lats, lons,
            time_string=folders['time_string'],
            t_threshold=t_threshold,
            smooth_sigma=smooth_sigma,
            min_area_km2=min_area_km2,
            tmin_threshold=tmin_threshold,
        )

    if plot_ON:
        temperature_plot_cartopy(temp, lats, lons, t_threshold, min_area_km2, props, labels_filtered, folders, tmin_threshold)

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
        t_threshold=-40,
        min_area_km2=100000,  # adjust depending on your MCS definition
        pixel_km=3.3423826253593094,        # 2 km per pixel for CONUS domain
        smooth_sigma=20.0,
        plot=True
    )