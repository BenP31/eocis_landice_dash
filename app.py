import os

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config("EOCIS Land Ice dashboard", layout="wide")
alt.data_transformers.enable("vegafusion")


@st.cache_data
def load_time_data(filename):
    return pd.read_csv(filename)


if "dash_dir" not in st.session_state:
    st.session_state["dash_dir"] = os.environ.get("DASH_DIR", os.getcwd())

ais_basins = gpd.read_file(
    "https://raw.githubusercontent.com/BenP31/eocis_landice_dash/master/aux_files/IMBIE_AIS_Basins/ANT_Basins_IMBIE2_v1.6.shp"
)
ais_basins = ais_basins.reset_index().rename(columns={"index": "basin_id"}).to_crs("epsg:4326")
ais_basins["basin_id"] = ais_basins["basin_id"].astype(str)
ais_basins.head()

timedata_df = load_time_data(
    os.path.join(st.session_state["dash_dir"], "processed_files/time_series_data_3.csv")
).sort_values("code")
timedata_df = timedata_df.reset_index(drop=True).reset_index()
timedata_df["basin"] = timedata_df["basin"].astype(str)
timedata_df = pd.merge(
    timedata_df,
    ais_basins[["basin_id", "Subregion"]],
    how="outer",
    left_on="basin",
    right_on="basin_id",
).drop(columns="basin_id")

with st.sidebar:
    periods = timedata_df["period"].to_list()
    codes = timedata_df["code"].to_list()
    time_period = st.selectbox(
        "Time period",
        options=range(len(periods)),
        format_func=lambda x: periods[x],
    )

st.title(f"Surface elevation change for the period {periods[time_period]}")
st.image(
    os.path.join(
        st.session_state["dash_dir"], f"processed_files/images/run052024/{codes[time_period]}.png"
    )
)

max_v = np.round(timedata_df["SEC"].abs().max(axis=None) + 0.05, 1)

click_state = alt.selection_point(fields=["Subregion"])

map = (
    alt.Chart(
        ais_basins,
    )
    .mark_geoshape()
    .encode(
        color=alt.Color("Subregion:N").legend(None),
        opacity=alt.condition(click_state, alt.value(1), alt.value(0.2)),
        tooltip=["basin_id:N", "Subregion:N", "Regions:N"],
    )
    .project(type="stereographic")
    .properties(width=300, height=300)
)

line = (
    alt.Chart(timedata_df, title="Mean SEC per Basin")
    .mark_line()
    .encode(
        alt.X("midpoint:Q", axis=alt.Axis(labels=False), title="Time Period"),
        alt.Y(
            "SEC:Q",
            scale=alt.Scale(domain=(-max_v, max_v)),
            title="Mean elevation change (m/year)",
        ),
        opacity=alt.condition(click_state, alt.value(1), alt.value(0.05)),
        color=alt.Color("basin:N").legend(None),
        tooltip=["SEC", "period", "Subregion"],
    )
    .properties(width=900, height=300)
)

# # Draw a rule at the location of the selection
# rules = base.mark_rule(color="gray").encode(
#     x="index:Q",
#     opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
#     tooltip=[
#         alt.Tooltip("period", type="nominal", title="Time period"),
#     ],
# )

# points = line.mark_circle().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

zero = pd.DataFrame([{"zero": 0.0}])
zero_rule = (
    alt.Chart(zero).mark_rule(color="black", strokeDash=[1]).encode(y="zero:Q", size=alt.value(1))
)

line_chart = line + zero_rule

chart = (map | line_chart).add_params(click_state)

st.altair_chart(chart.interactive(), use_container_width=True)
