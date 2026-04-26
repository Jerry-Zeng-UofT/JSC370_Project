from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import branca.colormap as bcm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import shape

# -----------------------------#
# Since alot of code used by the visualizations is shared, this file contains functions for those visualizations to avoid code duplication.
# -----------------------------#


# Geometry & slope helpers

def parse_geometry(geom_str: str):
    """Parse a GeoJSON-like string with doubled quotes into a shapely geometry."""
    if pd.isna(geom_str):
        return None
    geom_str = str(geom_str).replace('""', '"')
    geom_dict = json.loads(geom_str)
    return shape(geom_dict)



def build_base_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame with one row per neighbourhood from a cleaned wide dataframe.

    Expects columns: hood_id, area_name, geometry.
    """
    required = {"hood_id", "area_name", "geometry"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for geometry: {sorted(missing)}")

    map_df = df[["hood_id", "area_name", "geometry"]].drop_duplicates(subset=["hood_id"]).copy()
    map_df["geometry"] = map_df["geometry"].apply(parse_geometry)
    return gpd.GeoDataFrame(map_df, geometry="geometry", crs="EPSG:4326")



def compute_slope(group: pd.DataFrame) -> float:
    """Compute slope of rate ~ year for a neighbourhood-offence series."""
    g = group.dropna(subset=["year", "rate"]).sort_values("year")
    if len(g) < 2:
        return np.nan
    return np.polyfit(g["year"], g["rate"], 1)[0]



def build_slope_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """Return slope summary by neighbourhood x offence.

    Expects df_long to contain: hood_id, area_name, offence, year, rate.
    """
    required = {"hood_id", "area_name", "offence", "year", "rate"}
    missing = required - set(df_long.columns)
    if missing:
        raise ValueError(f"Missing required columns for slope table: {sorted(missing)}")

    slope_df = (
        df_long
        .groupby(["hood_id", "area_name", "offence"], as_index=False)
        .apply(lambda g: pd.Series({
            "slope": compute_slope(g),
            "mean_rate": g["rate"].mean(),
            "rate_2014": g.loc[g["year"] == 2014, "rate"].mean(),
            "rate_2025": g.loc[g["year"] == 2025, "rate"].mean(),
        }))
        .reset_index(drop=True)
    )

    for col in ["slope", "mean_rate", "rate_2014", "rate_2025"]:
        slope_df[col] = slope_df[col].round(3)

    return slope_df



def make_slope_map(
    df: pd.DataFrame,
    df_long: pd.DataFrame,
    center: tuple[float, float] = (43.70, -79.38),
    zoom_start: int = 10,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Create a Folium map with one selectable layer per offence, colored by slope."""
    gdf_base = build_base_geodataframe(df)
    slope_df = build_slope_table(df_long)

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles=tiles)
    offences = sorted(slope_df["offence"].dropna().unique())

    for offence in offences:
        offence_df = slope_df[slope_df["offence"] == offence].copy()
        gdf_offence = gdf_base.merge(
            offence_df[["hood_id", "slope", "mean_rate", "rate_2014", "rate_2025"]],
            on="hood_id",
            how="left",
        )

        if gdf_offence["slope"].notna().sum() == 0:
            continue

        max_abs = np.nanmax(np.abs(gdf_offence["slope"]))
        if pd.isna(max_abs) or max_abs == 0:
            max_abs = 1

        colormap = bcm.LinearColormap(
            colors=["blue", "white", "red"],
            vmin=-max_abs,
            vmax=max_abs,
            caption=f"{offence.title()} slope (rate change per year)",
        )

        feature_group = folium.FeatureGroup(name=offence.title(), show=(offence == offences[0]))

        def style_function(feature, cmap=colormap):
            slope = feature["properties"].get("slope")
            if slope is None:
                fill = "#999999"
            else:
                fill = cmap(slope)
            return {
                "fillColor": fill,
                "color": "black",
                "weight": 0.6,
                "fillOpacity": 0.7,
            }

        tooltip = folium.GeoJsonTooltip(
            fields=["area_name", "slope", "mean_rate", "rate_2014", "rate_2025"],
            aliases=["Neighbourhood:", "Slope:", "Mean rate:", "2014 rate:", "2025 rate:"],
            localize=True,
            sticky=False,
            labels=True,
        )

        folium.GeoJson(
            gdf_offence,
            style_function=style_function,
            tooltip=tooltip,
        ).add_to(feature_group)

        feature_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# Cluster map helpers


def make_cluster_map(
    df: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
    center: tuple[float, float] = (43.70, -79.38),
    zoom_start: int = 10,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """Create a Folium map colored by cluster assignment.

    Parameters
    ----------
    df : cleaned wide dataframe containing hood_id, area_name, geometry
    cluster_assignments : dataframe containing hood_id and cluster columns; area_name optional
    """
    gdf = build_base_geodataframe(df)

    needed = {"hood_id", "cluster"}
    missing = needed - set(cluster_assignments.columns)
    if missing:
        raise ValueError(f"Missing required columns for cluster map: {sorted(missing)}")

    cluster_map_df = cluster_assignments[["hood_id", "cluster"]].drop_duplicates().copy()
    gdf = gdf.merge(cluster_map_df, on="hood_id", how="left")

    cluster_name_map = {}
    for c in sorted(gdf["cluster"].dropna().astype(int).unique()):
        cluster_name_map[c] = f"Cluster {c}"
    gdf["cluster_label"] = gdf["cluster"].map(cluster_name_map)

    m = folium.Map(location=list(center), zoom_start=zoom_start, tiles=tiles)

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    cluster_colors = {c: palette[i % len(palette)] for i, c in enumerate(sorted(cluster_name_map))}

    def style_function(feature):
        cluster = feature["properties"].get("cluster")
        return {
            "fillColor": cluster_colors.get(cluster, "#999999"),
            "color": "black",
            "weight": 0.7,
            "fillOpacity": 0.6,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["area_name", "cluster_label"],
        aliases=["Neighbourhood:", "Cluster:"],
        localize=True,
        sticky=False,
        labels=True,
    )

    popup = folium.GeoJsonPopup(
        fields=["area_name", "hood_id", "cluster_label"],
        aliases=["Neighbourhood:", "Hood ID:", "Cluster:"],
        localize=True,
        labels=True,
    )

    folium.GeoJson(
        gdf,
        name="Neighbourhood Clusters",
        style_function=style_function,
        tooltip=tooltip,
        popup=popup,
    ).add_to(m)

    legend_rows = ''.join(
        [f'<i style="background:{color};width:12px;height:12px;display:inline-block;"></i> Cluster {c}<br>' for c, color in cluster_colors.items()]
    )
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        width: 190px;
        z-index:9999;
        background-color:white;
        border:2px solid grey;
        padding:10px;
        font-size:14px;
    ">
    <b>Crime Profile Clusters</b><br>
    {legend_rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m

# Model comparison chart helpers

def build_model_compare_df(
    results_df: pd.DataFrame,
    tree_results_df: pd.DataFrame,
    rf_test_results_df: pd.DataFrame,
    xgb_test_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine metrics from all predictive models into one tidy dataframe."""
    basic_results = results_df.copy().rename(columns={
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R2",
        "offence": "Offence",
        "model": "Model",
    })
    tree_results = tree_results_df.copy().rename(columns={
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R2",
        "offence": "Offence",
        "model": "Model",
    })
    rf_results = rf_test_results_df.copy().rename(columns={
        "test_rmse": "RMSE",
        "test_mae": "MAE",
        "test_r2": "R2",
        "offence": "Offence",
        "model": "Model",
    })
    xgb_results = xgb_test_results_df.copy().rename(columns={
        "test_rmse": "RMSE",
        "test_mae": "MAE",
        "test_r2": "R2",
        "offence": "Offence",
        "model": "Model",
    })

    keep = ["Offence", "Model", "RMSE", "MAE", "R2"]
    model_compare_df = pd.concat(
        [basic_results[keep], tree_results[keep], rf_results[keep], xgb_results[keep]],
        ignore_index=True,
    )

    model_compare_df["Offence"] = model_compare_df["Offence"].str.title()
    model_compare_df["Model"] = model_compare_df["Model"].replace({
        "Naive baseline": "Naive Baseline",
        "Linear regression": "Linear Regression",
        "Regression tree": "Regression Tree",
        "Random forest (tuned)": "Random Forest",
        "XGBoost (tuned)": "XGBoost",
    })

    model_order = ["Naive Baseline", "Linear Regression", "Regression Tree", "Random Forest", "XGBoost"]
    model_compare_df["Model"] = pd.Categorical(model_compare_df["Model"], categories=model_order, ordered=True)
    return model_compare_df.sort_values(["Offence", "Model"]).reset_index(drop=True)



def make_model_comparison_chart(model_compare_df: pd.DataFrame) -> go.Figure:
    """Interactive grouped bar chart with dropdown for RMSE / MAE / R2."""
    metrics = ["RMSE", "MAE", "R2"]
    model_order = ["Naive Baseline", "Linear Regression", "Regression Tree", "Random Forest", "XGBoost"]
    model_order = [m for m in model_order if m in model_compare_df["Model"].astype(str).unique()]
    offence_order = model_compare_df["Offence"].drop_duplicates().tolist()

    fig = go.Figure()
    for model in model_order:
        sub = model_compare_df[model_compare_df["Model"].astype(str) == model].copy()
        sub = sub.set_index("Offence").reindex(offence_order).reset_index()
        fig.add_trace(
            go.Bar(
                x=sub["Offence"],
                y=sub["RMSE"],
                name=model,
                customdata=sub[["MAE", "R2"]].values,
                hovertemplate=(
                    "Offence: %{x}<br>"
                    f"Model: {model}<br>"
                    "RMSE: %{y:.2f}<br>"
                    "MAE: %{customdata[0]:.2f}<br>"
                    "R²: %{customdata[1]:.3f}<extra></extra>"
                ),
            )
        )

    buttons = []
    for metric in metrics:
        y_values = []
        for model in model_order:
            sub = model_compare_df[model_compare_df["Model"].astype(str) == model].copy()
            sub = sub.set_index("Offence").reindex(offence_order).reset_index()
            y_values.append(sub[metric].values)

        buttons.append(
            dict(
                label=metric,
                method="update",
                args=[
                    {"y": y_values},
                    {"title": f"Model Performance by Offence Type ({metric})", "yaxis": {"title": metric}},
                ],
            )
        )

    fig.update_layout(
        title="Model Performance by Offence Type (RMSE)",
        xaxis_title="Offence Type",
        yaxis_title="RMSE",
        barmode="group",
        legend_title="Model",
        updatemenus=[
            dict(buttons=buttons, direction="down", showactive=True, x=1.02, xanchor="left", y=1.15, yanchor="top")
        ],
    )
    return fig

# Feature importance chart helpers

def build_importance_df(rf_importance_df: pd.DataFrame, xgb_importance_df: pd.DataFrame) -> pd.DataFrame:
    """Combine RF and XGB importance outputs into one tidy dataframe."""
    rf_imp = rf_importance_df.copy()
    rf_imp["model"] = "Random Forest"
    rf_imp = rf_imp.rename(columns={"offence": "Offence", "predictor": "Predictor", "importance": "Importance"})

    xgb_imp = xgb_importance_df.copy()
    xgb_imp["model"] = "XGBoost"
    xgb_imp = xgb_imp.rename(columns={"offence": "Offence", "predictor": "Predictor", "importance": "Importance"})

    importance_df = pd.concat([rf_imp, xgb_imp], ignore_index=True)
    importance_df["Offence"] = importance_df["Offence"].str.title()
    importance_df["Predictor"] = importance_df["Predictor"].str.replace("_lag1", " (t-1)", regex=False)
    importance_df["Importance"] = importance_df["Importance"].round(4)
    return importance_df



def make_feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Interactive horizontal grouped bar chart with dropdown for offence."""
    offence_list = sorted(importance_df["Offence"].unique())
    model_list = [m for m in ["Random Forest", "XGBoost"] if m in importance_df["model"].unique()]
    default_offence = offence_list[0]

    def _trace_data(offence: str):
        sub = importance_df[importance_df["Offence"] == offence].copy()
        traces = []
        for model in model_list:
            sub_model = sub[sub["model"] == model].copy().sort_values("Importance", ascending=True)
            traces.append(
                go.Bar(
                    x=sub_model["Importance"],
                    y=sub_model["Predictor"],
                    name=model,
                    orientation="h",
                    hovertemplate=(
                        f"Offence: {offence}<br>"
                        f"Model: {model}<br>"
                        "Predictor: %{y}<br>"
                        "Importance: %{x:.4f}<extra></extra>"
                    ),
                )
            )
        return traces

    fig = go.Figure(data=_trace_data(default_offence))

    buttons = []
    for offence in offence_list:
        sub = importance_df[importance_df["Offence"] == offence].copy()
        x_vals, y_vals = [], []
        for model in model_list:
            sub_model = sub[sub["model"] == model].copy().sort_values("Importance", ascending=True)
            x_vals.append(sub_model["Importance"].values)
            y_vals.append(sub_model["Predictor"].values)
        buttons.append(
            dict(
                label=offence,
                method="update",
                args=[{"x": x_vals, "y": y_vals}, {"title": f"Feature Importance by Model: {offence}"}],
            )
        )

    fig.update_layout(
        title=f"Feature Importance by Model: {default_offence}",
        barmode="group",
        xaxis_title="Importance",
        yaxis_title="Predictor",
        legend_title="Model",
        height=500,
        updatemenus=[
            dict(buttons=buttons, direction="down", showactive=True, x=1.02, xanchor="left", y=1.15, yanchor="top")
        ],
    )
    return fig


# Save helpers (not necessarily used in the final visualizations, mainly for testing and development)

def save_plotly_html(fig: go.Figure, path: str | Path, include_plotlyjs: str = "cdn") -> None:
    """Save a Plotly figure as a standalone HTML widget."""
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs)



def save_folium_html(m: folium.Map, path: str | Path) -> None:
    """Save a Folium map as HTML."""
    m.save(str(path))
