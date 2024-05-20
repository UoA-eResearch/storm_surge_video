import os
from lib import start, stop, get_monthly_data, plot_map, z_ind_next_month, plot_ax, dt_rev, get_data, dt_map
from dateutil.relativedelta import relativedelta
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == "__main__":
    folder = '/mnt/month_frames'


    # dn = start
    dnl0 = dt_rev[datetime(2010, 1, 1, 12)]
    dnl1 = dt_rev[datetime(2020, 1, 1, 12)]

    dnr0 = dt_rev[datetime(2080, 1, 1, 12)]
    dnr1 = dt_rev[datetime(2090, 1, 1, 12)]


    surges_l, surges_r = 0, 0

    for i, (l, r) in enumerate(zip(range(dnl0, dnl1), range(dnr0, dnr1))):
        fname = f'/mnt/dual_frames/frame_{i:05d}.png'

        print(l, r, fname)

        if os.path.exists(fname):
            continue

        gdf_l = get_data(l)
        gdf_r = get_data(r)

        surges_l += any(gdf_l['height'] > 0.4)
        surges_r += any(gdf_r['height'] > 0.4)

        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        plot_ax(gdf_l, axs[0], l)
        plot_ax(gdf_r, axs[1], r)

        axs[0].set_title(f"{str(dt_map[l].date())} - {surges_l} surge days")
        axs[1].set_title(f"{str(dt_map[r].date())} - {surges_r} surge days")

        fig.savefig(fname)
        plt.close()
        

    # while True:
    #     i1 = z_ind_next_month(i0)
    #     print(f'{i0} - {i1}')

    #     if i0 >= stop - 32:
    #         print('Done')
    #         break

    #     fname = os.path.join(folder, f'frame_{ii:05d}.png')
    #     if os.path.exists(fname):
    #         i0 = i1
    #         ii += 1
    #         print(f'Skipping {fname}')
    #         continue

    #     print(f'Generating {fname}')
    #     gdf = get_monthly_data(i0)
    #     plot_map(gdf, i0, fname)
    #     i0 = i1
    #     ii += 1
