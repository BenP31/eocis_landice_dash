import os
import re
import sys
from datetime import datetime
from glob import glob

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from scipy.ndimage import uniform_filter1d
from shapely.geometry import Point

file_years_re = re.compile("[0-9]{6}-[0-9]{6}")

csv_columns = ["code", "period", "midpoint", "basin", "Raw SEC"]

skip_basins = {0}


def get_data_files(folder: str):
    for f in sorted(glob(os.path.join(folder, "*.nc"))):
        yield f


def plot_antarc(
    gp_dataframe: gpd.GeoDataFrame,
    shp_dataframe: gpd.GeoDataFrame,
    plot_val,
    x_values,
    y_values,
):
    crs_3031 = ccrs.Stereographic(central_latitude=-90, true_scale_latitude=-71)

    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(facecolor="whitesmoke", projection=crs_3031)

    gp_dataframe.plot(
        column=plot_val,
        ax=ax,
        legend=True,
        vmax=2,
        vmin=-2,
        marker=".",
        linewidths=0,
        markersize=(fig.get_figwidth() * fig.get_dpi()) / len(x_values),
        cmap="bwr_r",
    )

    shp_dataframe.plot(color="none", edgecolor="black", ax=ax, alpha=0.5, lw=0.7)
    gl = ax.gridlines(draw_labels=True, color="black", alpha=0.25)
    gl.ylabel_style = {
        "color": "black",
        "alpha": 0.5,
    }

    x_max, x_min = np.max(x_values.flatten()), np.min(x_values.flatten())
    y_max, y_min = np.max(y_values.flatten()), np.min(y_values.flatten())
    ax.set_xlim(x_min * 1.05, x_max * 1.05)
    ax.set_ylim(y_min * 1.05, y_max * 1.05)

    return fig


def plot_greenl(
    gp_dataframe: gpd.GeoDataFrame,
    shp_dataframe: gpd.GeoDataFrame,
    plot_val,
    x_values,
    y_values,
):
    crs_new = ccrs.NorthPolarStereo(central_longitude=-40)

    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(facecolor="whitesmoke", projection=crs_new)

    gl = ax.gridlines(draw_labels=True, color="black", alpha=0.25)
    gl.ylabel_style = {
        "color": "black",
    }

    gp_dataframe.to_crs(crs_new).plot(
        column=plot_val,
        ax=ax,
        legend=True,
        vmax=2,
        vmin=-2,
        marker="s",
        markersize=1,
        cmap="bwr_r",
    )

    shp_dataframe.to_crs(crs_new).plot(color="none", edgecolor="black", ax=ax, alpha=0.5, lw=0.7)

    return fig


def get_mean_data(gp_dataframe: gpd.GeoDataFrame, column_name: str) -> dict:
    basin_means = (
        gp_dataframe[gp_dataframe.notna()][[column_name, "basin_id"]]
        .groupby("basin_id")
        .mean()
        .to_dict()[column_name]
    )
    basin_means["all"] = gp_dataframe[column_name].mean()
    return basin_means


def time_to_ts(time):
    start_of_year = datetime(year=int(time // 1), month=1, day=1)
    end_of_year = datetime(year=int(time // 1) + 1, month=1, day=1)
    seconds_in_year = (end_of_year - start_of_year).total_seconds()

    return datetime(year=int(time // 1), month=1, day=1).timestamp() + time % 1 * seconds_in_year


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("EOCIS data processing script")
    parser.add_argument(
        "-a",
        "--area",
        choices=["GrIS", "AIS"],
    )
    parser.add_argument("-d", "--data_file_dir", help="Directory of input .nc files")
    parser.add_argument(
        "-p", "--processed_file_dir", help="Directory for processed files made here"
    )
    parser.add_argument("-x", "--aux_file_dir")
    argv = parser.parse_args()

    arg_area = argv.area
    image_dir = os.path.join(argv.processed_file_dir, "images", argv.area)
    data_file_dir = argv.data_file_dir
    aux_file_dir = argv.aux_file_dir

    image_files = glob("*.png", root_dir=image_dir)

    landice_files = get_data_files(data_file_dir)

    # load shapefile
    if arg_area == "AIS":
        basin_df = gpd.read_file(
            os.path.join(aux_file_dir, "IMBIE_AIS_Basins", "ANT_Basins_IMBIE2_v1.6.shp")
        )
        crs = "epsg:3031"
        time_series_file = os.path.join(argv.processed_file_dir, "time_series_data_AIS.csv")
    elif arg_area == "GrIS":
        basin_df = gpd.read_file(
            os.path.join(aux_file_dir, "IMBIE_GIS_Basins", "Greenland_Basins_PS_v1.4.2.shp")
        )
        crs = "epsg:3413"
        time_series_file = os.path.join(argv.processed_file_dir, "time_series_data_GIS.csv")
    else:
        sys.exit("Invalid area selection")

    all_data = []

    try:
        for file_name in landice_files:
            print("Processing", file_name, flush=True)

            file_dates_search = file_years_re.findall(file_name)
            if len(file_dates_search) == 0:
                continue
            file_year: str = file_dates_search[0]
            # 199107-199607
            fmt_year = f"{file_year[0:4]}/{file_year[4:6]} - {file_year[7:11]}/{file_year[11:13]}"
            code_year = f"{file_year[0:4]}-{file_year[7:11]}-{file_year[4:6]}"

            # load nc
            nc = Dataset(file_name)
            surf_type = nc["surface_type"][:].data
            sec = nc["sec"][:].data[0, :, :]
            basin_id = nc["basin_id"][:].data

            # Error in basin number for AIS
            if arg_area == "AIS":
                basin_id[basin_id == -25] = 0


            # make dataframe here
            if file_year + ".png" not in image_files:
                x_values = nc["x"][:].data
                y_values = nc["y"][:].data
                x_coords, y_coords = np.meshgrid(x_values, y_values, indexing="xy")
                coords_arr = [Point(x, y) for x, y in zip(x_coords.flatten(), y_coords.flatten())]
                my_data = gpd.GeoDataFrame(
                    data={
                        "Raw SEC": sec.flatten(),
                        "basin_id": basin_id.flatten(),
                        "surface_type": surf_type.flatten(),
                        "geometry": coords_arr,
                    },
                    crs=crs,
                )
            else:
                my_data = pd.DataFrame(
                    data={
                        "Raw SEC": sec.flatten(),
                        "basin_id": basin_id.flatten(),
                    }
                )

            ts_data = get_mean_data(my_data, "Raw SEC")
            midpoint = datetime.fromtimestamp(
                (time_to_ts(nc["start_time"][:].data[0]) + time_to_ts(nc["end_time"][:].data[0]))
                / 2
            )

            for k, v in ts_data.items():
                if k in skip_basins:
                    continue

                all_data.append(
                    {
                        "code": code_year,
                        "period": fmt_year,
                        "midpoint": midpoint,
                        "basin": k,
                        "Raw SEC": v,
                    }
                )

            if file_year + ".png" not in image_files:
                # generate image and save in images folder
                if arg_area == "AIS":
                    fig = plot_antarc(
                        my_data[my_data.notna()], basin_df, "Raw SEC", x_values, y_values
                    )
                elif arg_area == "GrIS":
                    fig = plot_greenl(
                        my_data[my_data.notna()], basin_df, "Raw SEC", x_values, y_values
                    )
                else:
                    sys.exit("Invalid area selection")
                fig_file_path = os.path.join(image_dir, file_year + ".png")
                print(f"Saving figure to {fig_file_path}")
                fig.savefig(fig_file_path)
                plt.close()

    except KeyboardInterrupt as e:
        print("\nRecieved KeyboardInterrupt" + str(e))
    finally:
        # print all data to csv file
        # keyboard interrupt still writes all collected data
        df = pd.DataFrame(all_data)
        df["Smooth SEC"] = np.zeros(len(df))
        df["dH"] = np.zeros(len(df))

        for basin_no in df["basin"].unique():
            basin_sec = df[df["basin"] == basin_no]["Raw SEC"]
            smooth_basins = uniform_filter1d(basin_sec, size=5)
            df.loc[df["basin"] == basin_no, "Smooth SEC"] = smooth_basins

            dh_values = np.zeros(len(basin_sec))

            for i in range(len(basin_sec)):
                dh_values[i] = np.trapz(basin_sec[: i + 1])

            df.loc[df["basin"] == basin_no, "dH"] = dh_values

        df["Raw SEC"] = df["Raw SEC"].round(3)
        df["Smooth SEC"] = df["Smooth SEC"].round(3)

        # save dataframe
        print("Saving data to", time_series_file)
        df.to_csv(time_series_file, sep=",", index=False)
