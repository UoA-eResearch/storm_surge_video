import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import colorcet
from bokeh import palettes

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from multiprocessing import Pool

from tqdm import tqdm
from dateutil.relativedelta import relativedelta


engine = create_engine('mysql+pymysql://storm_ro:storm@localhost/storm')


# Non linear colormap
#class nlcmap:


# Min and Max date
with engine.connect() as conn:
    start = next(conn.execute(text('select MIN(z) from ACCESS10')))[0]
    stop = next(conn.execute(text('select MAX(z) from ACCESS10')))[0]


def tuple_to_dict(x):
    return {v[0]: v[1] for v in x}


with engine.connect() as conn:
    dt_list = list(conn.execute(text("select id, datetime from date")))
    dt_map = tuple_to_dict(dt_list)
    dt_rev = {v: k for k, v in dt_map.items()}


def to_dt(z_index):
    # z to datetime
    with engine.connect() as conn:
        return next(conn.execute(text(f'select datetime from date where id={z_index}')))[0]


# x/y to lat/on
def x_lon(future=False):
    with engine.connect() as conn:
        if future:
            return list(conn.execute(text(f'SELECT x, st_x(latlng) from f_latlng where y=0')))
        else:
            return list(conn.execute(text(f'SELECT x, st_x(latlng) from latlng where y=0')))
        
def y_lat(future=False):
    with engine.connect() as conn:
        if future:
            return list(conn.execute(text(f'SELECT y, st_y(latlng) from f_latlng where x=0')))
        else:
            return list(conn.execute(text(f'SELECT y, st_y(latlng) from latlng where x=0')))


# lat/lon lookup dicts
f_lon = tuple_to_dict(x_lon(True))
p_lon = tuple_to_dict(x_lon(False))
f_lat = tuple_to_dict(y_lat(True))
p_lat = tuple_to_dict(y_lat(False))


def to_poly(x, y):
    return Polygon([(x - .125, y - .125), (x - .125, y + .125), (x + .125, y + .125), (x + .125, y - .125)])

def get_data(z_index):
    with engine.connect() as conn:
        df = pd.read_sql(f'SELECT * FROM ACCESS10 WHERE z={z_index} AND model in (0, 1)', conn)

    # if df.loc[0, 'model'] == 0:
    #     df['lat'] = df['y'].map(f_lat)
    #     df['lon'] = df['x'].map(f_lon)
    # else:
    df['lat'] = df['y'].map(f_lat)
    df['lon'] = df['x'].map(f_lon)

    # print(z_index)
    # print(df.head())

    df['geom'] = df.apply(lambda row: to_poly(row['lon'], row['lat']), axis=1)

    return gpd.GeoDataFrame(df, geometry='geom')


def z_ind_next_month(z_index, offset=None):
    if offset is None:
        next_month = dt_map[z_index] + relativedelta(months=1)
    else:
        next_month = dt_map[z_index] + offset

    next_z = dt_rev[next_month]

    return next_z


def get_monthly_data(z_index, offset=None):
    next_z = z_ind_next_month(z_index, offset)

    with engine.connect() as conn:
        df = pd.read_sql(f'SELECT * FROM ACCESS10 WHERE z>={z_index} AND z<{next_z} AND model in (0, 1)', conn)

    # model = df.loc[0, 'model']

    df = df.groupby(['x', 'y'])['height'].mean().reset_index()

    df['lat'] = df['y'].map(f_lat)
    df['lon'] = df['x'].map(f_lon)

    df['geom'] = df.apply(lambda row: to_poly(row['lon'], row['lat']), axis=1)

    return gpd.GeoDataFrame(df, geometry='geom')


# Min/max height for colourbar limits
#with engine.connect() as conn:
#    min_height, max_height = next(conn.execute(text('SELECT MIN(height), MAX(height) FROM ACCESS10')))
#min_height = -0.37
#max_height = 0.73
min_height = -0.4
max_height = 0.5


# Coastal polygons
coast = gpd.read_file('nz-coastlines-and-islands-polygons-topo-150k.gpkg')
coast = coast[coast.area > 1e-4]


def plot_ax(gdf, ax, dn):
    tick_locations = [min_height] + [-(2.0**x) for x in (-2,-3,-4,-5)] + [0.0] + [(2.0**x) for x in (-5,-4,-3,-2,-1)] + [max_height]

    cm1 = colorcet.b_diverging_linear_protanopic_deuteranopic_bjy_57_89_c34
    cm2 = palettes.Inferno256[64:-22]
    cmap = palettes.diverging_palette(cm1, cm2, n=256, midpoint=0.9)
    lcmap = colors.LinearSegmentedColormap.from_list('custom', cmap)

    gdf.plot(column='height', ax=ax, legend=True,
            cmap=lcmap,
            #vmin=min_height, vmax=max_height,
            legend_kwds={"label": "Height (m)", 'ticks': tick_locations, 'format': '{x:.2f}'},
            #norm=colors.PowerNorm(gamma=0.5))
            norm=colors.SymLogNorm(linthresh=0.05, linscale=0.5, vmin=min_height, vmax=max_height, base=2),
            )
            #ticks=tick_locations)

#    print(ax.colorbar)
#    cbar = ax.collections[-1].colorbar
#    cbar.ax.set_yticklabels(tick_locations)

    coast.plot(color='white', ax=ax)

#    plt.colorbar(ticks=tick_locations, ax=ax)

    ax.set_xlim((165, 180))
    ax.set_ylim((-48.5, -33))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.set_title(str(dt_map[dn].date()))


def plot_map(gdf, dn, fname):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    plot_ax(gdf, ax, dn)

    fig.savefig(fname)
    plt.close()


def thread_step(i):
    fn = start + i
    fname = f'/mnt/frames/frame_{i:05d}.png'
    if os.path.exists(fname):
        return

    gdf = get_data(fn)

    plot_map(gdf, fn, fname)


if __name__ == "__main__":
    # engine.dispose()

    # with Pool(4) as p:
    #     p.map(thread_step, range(stop-start))

    model_start = 50769

    for i, fn in tqdm(enumerate(range(model_start, stop)), total=stop-model_start):
        fname = f'/mnt/frames/frame_fut_{i:05d}.png'
        if os.path.exists(fname):
            continue

        gdf = get_data(fn)

        plot_map(gdf, fn, fname)
