import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config("EOCIS Land Ice dashboard", layout="wide")


@st.cache_data
def load_time_data(filename):
    return pd.read_csv(filename)


if "dash_dir" not in st.session_state:
    st.session_state["dash_dir"] = os.environ.get("DASH_DIR", os.getcwd())


timedata_df = load_time_data(
    os.path.join(st.session_state["dash_dir"], "processed_files/time_series_data_3.csv")
).sort_values("code")

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

basin_selection = st.multiselect(
    "Basins",
    options=["all", *[str(i) for i in range(1, 28)]],
    default=["all"],
    max_selections=5,
)

max_v = np.round(timedata_df[basin_selection].abs().max(axis=None) + 0.05, 1)

if len(basin_selection) != 0:
    base = alt.Chart(timedata_df.reset_index(drop=True).reset_index())

    nearest = alt.selection_point(nearest=True, on="mouseover", fields=["index"], empty=False)

    line = base.mark_line().encode(
        alt.X("index:Q", axis=alt.Axis(labels=False), title="Time Period"),
        alt.Y(
            alt.repeat("layer"),
            type="quantitative",
            scale=alt.Scale(domain=(-max_v, max_v)),
            title="Mean elevation change (m/year)",
        ),
        color=alt.datum(alt.repeat("layer")),
    )

    # Draw a rule at the location of the selection
    rules = (
        base.mark_rule(color="gray")
        .encode(
            x="index:Q",
            opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("period", type="nominal", title="Time period"),
            ]
            + [alt.Tooltip(c, type="quantitative", format=".4f") for c in basin_selection],
        )
        .add_params(nearest)
    )

    points = line.mark_circle().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

    zero = pd.DataFrame([{"zero": 0.0}])
    zero_rule = (
        alt.Chart(zero)
        .mark_rule(color="black", strokeDash=[1])
        .encode(y="zero:Q", size=alt.value(1))
    )

    bar_chart = line + zero_rule + rules + points

    bar_chart = bar_chart.repeat(layer=basin_selection)

    map = (
        alt.Chart(
            ais_basins.reset_index().to_crs("epsg:4326"),
        )
        .mark_geoshape()
        .encode(
            color="basin_id:N", opacity=alt.condition(click_state, alt.value(1), alt.value(0.2))
        )
        .project(type="stereographic")
    )

    st.altair_chart(bar_chart.interactive(), use_container_width=True)
else:
    st.text("No data selected to plot")
