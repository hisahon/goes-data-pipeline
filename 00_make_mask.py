# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------
Created by Rogério Hisashi Honda on Thu Oct 30 09:34:46 2025
hisashi.brase@gmail.com

----------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from dbscan import read_goes_file

# --- helper: reta definida por (lon0, lat0, angulo) ---
def yline(lons, lon0, lat0, angle_deg):
    """
    Retorna latitudes da reta y = m*(x - lon0) + lat0, onde m=tan(angulo).
    lons pode ser 1D ou 2D; a saída terá o mesmo shape de lons.
    """
    m = np.tan(np.deg2rad(angle_deg))
    return m*(lons - lon0) + lat0

# --- sua máscara (idêntica à lógica original) ---
def region_mask_top_bottom(lats, lons, inclusive=True):
    """
    Retorna uma máscara 2D True entre o envelope:
      topo  = min(linhas_topo)   (acima é False)
      base  = max(linhas_base)   (abaixo é False)
    """
    top_lines = [
        # (-49.000, -10.000,  40),
        (-38.370,  -1.227, -30),
        
        (-80.294, 5,  50),
        ]
    bottom_lines = [
        (-48.765, -35.276,  53),
        (-78.294, -37.175,  97),
        (-36.072, -15.273,  66),
        (-70.072, -57,  160),
        # (-61, -50,  88),
    ]

    # Aceita lats/lons 1D ou 2D
    lats1d = lats[:, 0] if lats.ndim == 2 else lats
    lons1d = lons[0, :] if lons.ndim == 2 else lons

    # Garante que lats cresce (linha -> linha)
    if lats1d[0] > lats1d[-1]:
        lats1d = lats1d[::-1]

    # Cria grade 2D (ny, nx)
    X, Y = np.meshgrid(lons1d, lats1d)

    # Envelopes topo/base (arrays 1D por coluna)
    Ytop = np.min(np.stack([yline(lons1d, *p) for p in top_lines], axis=0), axis=0)  # (nx,)
    Ybot = np.max(np.stack([yline(lons1d, *p) for p in bottom_lines], axis=0), axis=0)  # (nx,)

    # Broadcast para (ny, nx)
    Ytop2 = np.broadcast_to(Ytop, X.shape)
    Ybot2 = np.broadcast_to(Ybot, X.shape)

    mask = (Y <= Ytop2) & (Y >= Ybot2) if inclusive else (Y < Ytop2) & (Y > Ybot2)
    return mask, (lats1d, lons1d, Ytop, Ybot), (top_lines, bottom_lines)




# --- circulo simples em graus (para desenhar) ---
def plot_deg_circle(ax, lonc, latc, rdeg, n=256, **kw):
    t = np.linspace(0, 2*np.pi, n)
    xs = lonc + rdeg*np.cos(t)
    ys = latc + rdeg*np.sin(t)
    ax.plot(xs, ys, transform=ccrs.PlateCarree(), **kw)

# --- excluir regiões circulares (raio em graus) ---
def exclude_circles_from_mask_deg(mask, lats, lons, wrap_longitude=True):
    """
    circles_deg: lista de (lon_centro, lat_centro, raio_em_graus)
    lats/lons podem ser 1D ou 2D (graus). Se wrap_longitude=True, trata dateline.
    """
    
    # exclude_circles_deg = [
    # (-104, -30.0, 30),   # raio em GRAUS
    # (-29, 26.0, 30),   # raio em GRAUS
    # (-22, -54.0, 35),   # raio em GRAUS
    # (-50, -54.0, 16),   # raio em GRAUS
    # ]
    
    exclude_circles_deg = [
    (-104, -30.0, 30),   # raio em GRAUS
    (-87, -26.0, 15),   # raio em GRAUS
    (-29, 26.0, 30),   # raio em GRAUS
    (-22, -54.0, 35),   # raio em GRAUS
    (-50, -54.0, 16),   # raio em GRAUS
    (-44, -49.0, 16),   # raio em GRAUS
    ]
    
    # normaliza grade para 2D
    if lats.ndim == 1 and lons.ndim == 1:
        LON, LAT = np.meshgrid(lons, lats)
    else:
        LAT, LON = lats, lons

    exclude_union = np.zeros_like(mask, dtype=bool)

    for lonc, latc, rdeg in exclude_circles_deg:
        dlon = LON - lonc
        if wrap_longitude:
            # menor diferença levando em conta -180/180
            dlon = (dlon + 180.0) % 360.0 - 180.0
        dlat = LAT - latc
        d2 = dlon*dlon + dlat*dlat   # distância "euclidiana" em graus
        exclude_union |= (d2 <= rdeg*rdeg)

    out = mask.copy()
    out[exclude_union] = False
    return out, exclude_union




# --- plot com Cartopy ---
def plot_region_mask_top_bottom(lats, lons, inclusive=True,
                                show_individual_lines=True,
                                alpha_mask=0.35, coastline_res="50m"):
    """
    Plota a máscara regional com Cartopy (PlateCarree).
    - lats, lons: 1D ou 2D (em graus)
    - inclusive: passa para a função de máscara
    - show_individual_lines: mostra as linhas que compõem topo/base
    """
    mask, (lats1d, lons1d, Ytop, Ybot), (top_lines, bottom_lines) = region_mask_top_bottom(lats, lons, inclusive)
    mask, _ = exclude_circles_from_mask_deg(mask, lats, lons, wrap_longitude=True)

    # Grade 2D para pcolormesh
    X, Y = np.meshgrid(lons1d, lats1d)

    # Figura/cartopy
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=proj)
    ax.set_title("Região entre topo e base (mask)")

    # Extent com margem
    lon_min, lon_max = np.nanmin(lons1d), np.nanmax(lons1d)
    lat_min, lat_max = np.nanmin(lats1d), np.nanmax(lats1d)
    dlon = (lon_max - lon_min) * 0.05
    dlat = (lat_max - lat_min) * 0.05
    ax.set_extent([lon_min - dlon, lon_max + dlon, lat_min - dlat, lat_max + dlat], crs=proj)

    # Fundo
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.COASTLINE.with_scale(coastline_res), linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle=":")
    gl.right_labels = False
    gl.top_labels = False

    # Máscara: usa pcolormesh com alpha
    # Cria uma matriz float com NaN fora e 1.0 dentro
    z = np.where(mask, 1.0, np.nan)
    ax.pcolormesh(X, Y, z, transform=proj, alpha=alpha_mask)

    # Envelopes (curvas topo/base)
    ax.plot(lons1d, Ytop, transform=proj, linewidth=2, label="Topo (min das linhas topo)")
    ax.plot(lons1d, Ybot, transform=proj, linewidth=2, linestyle="--", label="Base (max das linhas base)")

    # Linhas individuais (opcional)
    if show_individual_lines:
        for (lon0, lat0, ang) in top_lines:
            ax.plot(lons1d, yline(lons1d, lon0, lat0, ang), transform=proj, linewidth=1, alpha=0.7)
        for (lon0, lat0, ang) in bottom_lines:
            ax.plot(lons1d, yline(lons1d, lon0, lat0, ang), transform=proj, linewidth=1, alpha=0.7)

    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig, ax, mask

# -------------------------------
# EXEMPLO DE USO:
# lats e lons 1D (ou 2D) em graus
# -------------------------------
if __name__ == "__main__":
    # exemplo simples de grade regular
    
    
    
    extent=[-82, -34, -56,  13] #lnn,lnx,ltn,ltx
    
    _, lats, lons = read_goes_file('S10635346_202110230000.nc', extent=extent)
    

    fig, ax, mask = plot_region_mask_top_bottom(lats, lons, inclusive=True, show_individual_lines=True)
    # np.savez(f'mask_SouthAmerica', mask, allow_pickle=False)
    
    np.savez(
    "mask_SouthAmerica.npz",
    lats=np.asarray(lats),
    lons=np.asarray(lons),
    extent=extent,
    mask=np.asarray(mask, dtype=np.bool_)
    )
    
    plt.show()
