import csv
import os
import re
from glob import glob

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset  # pylint:disable=no-name-in-module
from shapely.geometry import Point

file_years_re = re.compile("[0-9]{6}-[0-9]{6}")

# Using IMBIE definitions, basin ids are 0-18
csv_columns = ["code", "period", "basin", "SEC"]


def get_data_files(folder: str):
    for f in sorted(glob(os.path.join(folder, "*.nc"))):
        yield f


def plot_val(
    gp_dataframe: gpd.GeoDataFrame,
    shp_dataframe: gpd.GeoDataFrame,
    plot_val,
    x_values,
    y_values,
    extra=False,
):
    crs_3031 = ccrs.Stereographic(central_latitude=-90, true_scale_latitude=-71)

    fig, ax = plt.subplots(
        figsize=(9, 6),
        facecolor="white",
        subplot_kw=dict(projection=crs_3031),
    )  # Create our plot

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


def get_mean_data(gp_dataframe: gpd.GeoDataFrame, column_name: str) -> dict:
    basin_means = (
        gp_dataframe[gp_dataframe.notna()][[column_name, "basin_id"]]
        .groupby("basin_id")
        .mean()
        .to_dict()[column_name]
    )
    basin_means["all"] = gp_dataframe[column_name].mean()
    return basin_means


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("EOCIS data processing script")
    parser.add_argument(
        "-x", "--aux_file_dir", help="Directory for auxillary files (shapefiles, etc.)"
    )
    parser.add_argument("-d", "--data_file_dir", help="Directory of input .nc files")
    parser.add_argument(
        "-p", "--processed_file_dir", help="Directory for processed files made here"
    )
    parser.add_argument(
        "-t", "--time_series_file", help="Path to the csv file containing the time series data"
    )

    argv = parser.parse_args()

    image_dir = argv.processed_file_dir
    data_file_dir = argv.data_file_dir
    time_series_file = argv.time_series_file
    aux_file_dir = argv.aux_file_dir

    ts_existing_entries = []
    if os.path.exists(time_series_file):
        with open(time_series_file, "r") as time_series_file_fp:
            ts_reader = csv.reader(time_series_file_fp, delimiter=",")
            for row in ts_reader:
                ts_existing_entries.append(row[0])
    else:
        with open(time_series_file, "w+") as time_series_file_fp:
            ts_writer = csv.DictWriter(time_series_file_fp, fieldnames=csv_columns, delimiter=",")
            ts_writer.writeheader()

    image_files = glob("*.png", root_dir=image_dir)

    landice_files = get_data_files(data_file_dir)

    for file_name in landice_files:
        print("Processing ", file_name, end="", flush=True)
        years = file_years_re.findall(file_name)
        if len(years) == 0:
            continue
        file_year: str = years[0]
        # 199107-199607
        fmt_year = f"{file_year[0:4]}/{file_year[4:6]} - {file_year[7:11]}/{file_year[11:13]}"

        if file_year in ts_existing_entries and file_year + ".png" in image_files:
            print(" O O (Skipped)")
            continue

        # load shapefile
        ais_basins = gpd.read_file(
            os.path.join(aux_file_dir, "IMBIE_AIS_Basins", "ANT_Basins_IMBIE2_v1.6.shp")
        )

        # load nc
        nc = Dataset(os.path.join(data_file_dir, file_name))
        x_values = nc["x"][:].data
        y_values = nc["y"][:].data
        surf_type = nc["surface_type"][:].data
        sec = nc["sec"][:].data[0, :, :]
        basin_id = nc["basin_id"][:].data
        basin_id[basin_id == -25] = 0

        coords = []
        for y in y_values:
            for x in x_values:
                coords.append(Point(x, y))

        # make dataframe here
        my_data = gpd.GeoDataFrame(
            data={
                "SEC": sec.flatten(),
                "basin_id": basin_id.flatten(),
                "surface_type": surf_type.flatten(),
                "geometry": coords,
            },
            crs="epsg:3031",
        )

        if fmt_year not in ts_existing_entries:
            # get mean data and add to csv
            ts_data = get_mean_data(my_data, "SEC")
            with open(time_series_file, "a") as time_series_file_fp:
                ts_writer = csv.DictWriter(
                    time_series_file_fp, fieldnames=csv_columns, delimiter=","
                )
                for k, v in ts_data.items():
                    out_dict = {
                        "code": file_year,
                        "period": fmt_year,
                        "basin": k,
                        "SEC": v,
                    }
                    ts_writer.writerow(out_dict)

            # str_dict = {str(k): v for k, v in ts_data.items()}
            # ts_data = str_dict
            # ts_data.update({"code": file_year, "period": fmt_year})
            # with open(time_series_file, "a") as time_series_file_fp:
            #     ts_writer = csv.DictWriter(
            #         time_series_file_fp, fieldnames=csv_columns, delimiter=","
            #     )
            #     ts_writer.writerow(ts_data)

        print(" O", end="", flush=True)

        if file_year + ".png" not in image_files:
            # generate image and save in images folder
            fig = plot_val(my_data[my_data.notna()], ais_basins, "SEC", x_values, y_values)
            fig.savefig(os.path.join(image_dir, file_year + ".png"))
            plt.close()

        print(" O", end="", flush=True)

        print("")
