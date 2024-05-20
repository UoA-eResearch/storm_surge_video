import os
from lib import start, stop, get_monthly_data, plot_map, z_ind_next_month
from dateutil.relativedelta import relativedelta


if __name__ == "__main__":
    folder = '/mnt/weekly_frames'


    # dn = start
    dn = 50769

    ii = 0
    i0 = dn

    while True:
        i1 = z_ind_next_month(i0, relativedelta(weeks=1))
        print(f'{i0} - {i1}')

        if i0 >= stop - 32:
            print('Done')
            break

        fname = os.path.join(folder, f'frame_model_{ii:05d}.png')
        if os.path.exists(fname):
            i0 = i1
            ii += 1
            print(f'Skipping {fname}')
            continue

        print(f'Generating {fname}')
        gdf = get_monthly_data(i0, relativedelta(weeks=1))
        plot_map(gdf, i0, fname)
        i0 = i1
        ii += 1
