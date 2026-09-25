"""Streamlit Executive Dashboard for EV Demand Forecasting & Capacity Planning."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ev_forecasting.audit.prediction_log import AuditLogger
from ev_forecasting.config.settings import load_config
from ev_forecasting.forecasting.forecast import ForecastService

# Page configuration
st.set_page_config(
    page_title="EV Charging Demand & Capacity Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .status-badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_service_and_config():
    config = load_config("configs/development.yaml")
    service = ForecastService(config)
    audit = AuditLogger(config)
    return config, service, audit


def load_dataset(config):
    p = Path(config.paths.processed_data_dir) / "hourly_demand.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def load_leaderboard(config):
    p = Path(config.paths.metrics_dir) / "model_comparison_leaderboard.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def load_manifest(config):
    p = Path(config.paths.manifest_dir) / "dataset_manifest.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_quality_report(config):
    p = Path(config.paths.reports_dir) / "data_quality_report.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_drift_report(config):
    p = Path(config.paths.reports_dir) / "monitoring_drift_report.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    config, forecast_service, audit_logger = get_service_and_config()
    df_hourly = load_dataset(config)
    load_manifest(config)
    leaderboard_df = load_leaderboard(config)

    st.sidebar.markdown("## ⚡ EV Forecasting Platform")
    st.sidebar.markdown(f"**Environment:** `{config.environment.upper()}`")

    prod_meta = forecast_service.metadata
    if prod_meta:
        st.sidebar.success(
            f"**Prod Model:** {prod_meta.get('model_name')}\n\n**Version:** `{prod_meta.get('model_version')}`"
        )
    else:
        st.sidebar.warning("No production model registered yet.")

    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Executive Overview",
            "2. Station Forecast & Risk",
            "3. Model Leaderboard & Benchmark",
            "4. Walk-Forward Backtesting",
            "5. Capacity & What-If Scenarios",
            "6. Data Quality & Drift Monitor",
            "7. Prediction Audit Trail & Lineage",
        ],
    )

    # -------------------------------------------------------------
    # 1. Executive Overview
    # -------------------------------------------------------------
    if page == "1. Executive Overview":
        st.markdown(
            '<div class="main-header">Executive Overview: EV Charging Demand</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Real-time system health, capacity risks, and multi-station aggregate demand</div>',
            unsafe_allow_html=True,
        )

        if df_hourly.empty:
            st.info("No processed data found. Run the ingestion and preprocessing pipelines first.")
            return

        total_stations = df_hourly["station_id"].nunique()
        total_sites = df_hourly["site_id"].nunique()
        total_energy = df_hourly["energy_demand_kwh"].sum()
        peak_demand = df_hourly["energy_demand_kwh"].max()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Active Stations", f"{total_stations}", "Caltech / JPL / Office")
        col2.metric("Sites Monitored", f"{total_sites}", "3 Environments")
        col3.metric("Total Historical Energy", f"{total_energy:,.1f} kWh", "ACN-Data")
        col4.metric("Peak Observed Demand", f"{peak_demand:.1f} kWh/h", "Single Station")

        st.markdown("---")
        st.subheader("Aggregated Hourly EV Demand Profile Across All Sites")

        # Aggregate across all stations by timestamp
        agg_time = (
            df_hourly.groupby(["timestamp", "site_id"])["energy_demand_kwh"].sum().reset_index()
        )
        fig = px.line(
            agg_time,
            x="timestamp",
            y="energy_demand_kwh",
            color="site_id",
            title="Hourly Aggregate Energy Demand by Charging Site (kWh)",
            color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B"],
        )
        fig.update_layout(
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Diurnal and weekly patterns
        col_a, col_b = st.columns(2)
        with col_a:
            df_hourly["hour"] = pd.to_datetime(df_hourly["timestamp"]).dt.hour
            hourly_profile = (
                df_hourly.groupby(["hour", "site_id"])["energy_demand_kwh"].mean().reset_index()
            )
            fig_h = px.bar(
                hourly_profile,
                x="hour",
                y="energy_demand_kwh",
                color="site_id",
                title="Average Daily Demand Profile by Hour of Day (kWh)",
                barmode="group",
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with col_b:
            df_hourly["dayofweek"] = pd.to_datetime(df_hourly["timestamp"]).dt.day_name()
            day_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            weekly_profile = (
                df_hourly.groupby(["dayofweek", "site_id"])["energy_demand_kwh"]
                .mean()
                .reindex(day_order, level=0)
                .reset_index()
            )
            fig_w = px.bar(
                weekly_profile,
                x="dayofweek",
                y="energy_demand_kwh",
                color="site_id",
                title="Weekday vs Weekend Charging Demand Profile (kWh)",
                barmode="group",
            )
            st.plotly_chart(fig_w, use_container_width=True)

    # -------------------------------------------------------------
    # 2. Station Forecast & Risk
    # -------------------------------------------------------------
    elif page == "2. Station Forecast & Risk":
        st.markdown(
            '<div class="main-header">Station Demand Forecast & Capacity Risk</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Multi-horizon hourly forecasts with deterministic risk tiering and capacity recommendations</div>',
            unsafe_allow_html=True,
        )

        if df_hourly.empty:
            st.info("No processed dataset available.")
            return

        all_stations = sorted(df_hourly["station_id"].unique().tolist())
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            selected_station = st.selectbox("Select EV Charging Station", all_stations)
        with col_ctrl2:
            horizon = st.selectbox(
                "Forecast Horizon",
                [24, 48, 168],
                index=2,
                format_func=lambda x: f"{x} Hours ({x // 24} Days)",
            )

        if st.button("⚡ Generate Forecast", type="primary"):
            with st.spinner(f"Generating {horizon}-hour forecast for {selected_station}..."):
                try:
                    res = forecast_service.predict_station_demand(
                        selected_station, horizon_hours=horizon, df_history=df_hourly
                    )
                    forecast_items = res["forecast"]
                    f_df = pd.DataFrame(forecast_items)
                    f_df["timestamp"] = pd.to_datetime(f_df["timestamp"])

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Peak Demand", f"{res['peak_forecast_kwh']:.2f} kWh", f"Horizon: {horizon}h"
                    )
                    c2.metric(
                        "Peak Utilization",
                        f"{res['peak_utilization_pct']:.1f}%",
                        "Max Threshold: 95%",
                    )
                    c3.metric(
                        "Critical Hours", f"{res['critical_risk_hours']} hrs", "Exceeds 95% Cap"
                    )
                    c4.metric(
                        "Audit Prediction ID", f"{res['prediction_id'][:8]}...", "Logged in SQL"
                    )

                    # Forecast Plot
                    fig_fc = go.Figure()
                    fig_fc.add_trace(
                        go.Scatter(
                            x=f_df["timestamp"],
                            y=f_df["forecast_kwh"],
                            name="Forecast Demand (kWh)",
                            line=dict(color="#2563EB", width=3),
                            mode="lines+markers",
                        )
                    )
                    fig_fc.add_trace(
                        go.Scatter(
                            x=f_df["timestamp"],
                            y=f_df["capacity_kwh"],
                            name="Station Physical Capacity (kWh/h)",
                            line=dict(color="#DC2626", width=2, dash="dash"),
                        )
                    )
                    fig_fc.update_layout(
                        title=f"Forecast Demand & Physical Capacity Limit ({selected_station})",
                        xaxis_title="Forecast Timestamp (UTC)",
                        yaxis_title="Energy (kWh)",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)

                    # Hourly Forecast Table
                    st.subheader("Hourly Forecast Breakdown & Actionable Guidance")
                    st.dataframe(
                        f_df[
                            [
                                "timestamp",
                                "forecast_kwh",
                                "capacity_kwh",
                                "utilization_pct",
                                "risk_level",
                                "recommendation",
                            ]
                        ],
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Error generating forecast: {e}")

    # -------------------------------------------------------------
    # 3. Model Leaderboard & Benchmark
    # -------------------------------------------------------------
    elif page == "3. Model Leaderboard & Benchmark":
        st.markdown(
            '<div class="main-header">Model Benchmark & Leaderboard</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="sub-header">Expanding-window backtested evaluation across Statistical & Feature-Based ML models</div>',
            unsafe_allow_html=True,
        )

        if leaderboard_df.empty:
            st.info("No leaderboard found. Run the training pipeline first.")
            return

        st.dataframe(
            leaderboard_df.style.highlight_min(
                subset=["rmse_mean", "mae_mean", "mape_mean"], color="#D1FAE5"
            ),
            use_container_width=True,
        )

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            fig_rmse = px.bar(
                leaderboard_df,
                x="model_name",
                y="rmse_mean",
                color="model_type",
                title="RMSE Comparison Across Models (Lower is Better)",
                text_auto=".2f",
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col_l2:
            fig_mae = px.bar(
                leaderboard_df,
                x="model_name",
                y="mae_mean",
                color="model_type",
                title="MAE Comparison Across Models (Lower is Better)",
                text_auto=".2f",
            )
            st.plotly_chart(fig_mae, use_container_width=True)

    # -------------------------------------------------------------
    # 4. Walk-Forward Backtesting
    # -------------------------------------------------------------
    elif page == "4. Walk-Forward Backtesting":
        st.markdown(
            '<div class="main-header">Walk-Forward Backtesting Diagnostics</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Inspection of fold boundaries, residual distributions, and test predictions</div>',
            unsafe_allow_html=True,
        )

        figures_dir = Path(config.paths.figures_dir)
        fig_files = list(figures_dir.glob("*.png"))

        if not fig_files:
            st.info(
                "No backtest visual artifacts found. Run the training pipeline to generate figures."
            )
        else:
            for fig_p in fig_files:
                st.subheader(fig_p.stem.replace("_", " ").title())
                st.image(str(fig_p), use_container_width=True)

    # -------------------------------------------------------------
    # 5. Capacity & What-If Scenarios
    # -------------------------------------------------------------
    elif page == "5. Capacity & What-If Scenarios":
        st.markdown(
            '<div class="main-header">Capacity Planning & Scenario Simulator</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Interactive what-if capacity upgrade modeling to reduce bottleneck hours</div>',
            unsafe_allow_html=True,
        )

        if df_hourly.empty:
            st.info("Dataset not loaded.")
            return

        all_stations = sorted(df_hourly["station_id"].unique().tolist())
        target_station = st.selectbox("Target Station for Capacity Planning", all_stations)
        base_cap = config.capacity.default_station_capacity_kwh

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            custom_cap = st.slider(
                "Current Station Base Capacity (kWh/h)",
                min_value=10.0,
                max_value=150.0,
                value=float(base_cap),
                step=5.0,
            )
        with col_s2:
            upgrade_pct = st.slider(
                "Simulated Capacity Expansion (+%)", min_value=0, max_value=100, value=20, step=5
            )

        expanded_cap = custom_cap * (1.0 + upgrade_pct / 100.0)

        # Compute simulated scenario on recent demand
        st_data = df_hourly[df_hourly["station_id"] == target_station].copy()
        if not st_data.empty:
            sample_recent = st_data.iloc[-168:].copy()
            sample_recent["base_util_pct"] = (
                sample_recent["energy_demand_kwh"] / custom_cap
            ) * 100.0
            sample_recent["expanded_util_pct"] = (
                sample_recent["energy_demand_kwh"] / expanded_cap
            ) * 100.0

            crit_before = int((sample_recent["base_util_pct"] >= 95.0).sum())
            crit_after = int((sample_recent["expanded_util_pct"] >= 95.0).sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Base Capacity", f"{custom_cap:.1f} kWh/h", f"{crit_before} Critical Hours")
            m2.metric(
                "Upgraded Capacity", f"{expanded_cap:.1f} kWh/h", f"{crit_after} Critical Hours"
            )
            m3.metric(
                "Bottleneck Reduction",
                f"{crit_before - crit_after} hrs saved",
                f"{(crit_before - crit_after) / max(crit_before, 1) * 100:.1f}% reduction",
            )

            fig_sc = go.Figure()
            fig_sc.add_trace(
                go.Scatter(
                    x=sample_recent["timestamp"],
                    y=sample_recent["base_util_pct"],
                    name="Base Utilization %",
                    line=dict(color="#EF4444"),
                )
            )
            fig_sc.add_trace(
                go.Scatter(
                    x=sample_recent["timestamp"],
                    y=sample_recent["expanded_util_pct"],
                    name="Upgraded Utilization %",
                    line=dict(color="#10B981"),
                )
            )
            fig_sc.add_hline(
                y=95.0,
                line_dash="dash",
                line_color="red",
                annotation_text="Critical Threshold (95%)",
            )
            fig_sc.add_hline(
                y=85.0, line_dash="dot", line_color="orange", annotation_text="High Threshold (85%)"
            )
            fig_sc.update_layout(
                title="Utilization Profile Before vs After Capacity Expansion",
                yaxis_title="Utilization (%)",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    # -------------------------------------------------------------
    # 6. Data Quality & Drift Monitor
    # -------------------------------------------------------------
    elif page == "6. Data Quality & Drift Monitor":
        st.markdown(
            '<div class="main-header">Data Quality & Statistical Drift Monitor</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Automated data validation gates, KS-Test p-values, and Population Stability Index (PSI)</div>',
            unsafe_allow_html=True,
        )

        q_report = load_quality_report(config)
        d_report = load_drift_report(config)

        st.subheader("1. Data Ingestion & Quality Gates")
        if q_report:
            h_rep = q_report.get("hourly_report", {})
            st.json(h_rep)
        else:
            st.info("No quality report available. Run preprocessing pipeline.")

        st.subheader("2. Distribution Drift Monitoring (KS-Test & PSI)")
        if d_report:
            st.write(
                f"**Overall Monitoring Status:** `{d_report.get('status')}` | **Retraining Recommended:** `{d_report.get('retraining_recommended')}`"
            )
            drift_items = d_report.get("drift_report", [])
            if drift_items:
                st.dataframe(pd.DataFrame(drift_items), use_container_width=True)
        else:
            st.info("No drift report available. Run monitoring pipeline.")

    # -------------------------------------------------------------
    # 7. Prediction Audit Trail & Lineage
    # -------------------------------------------------------------
    elif page == "7. Prediction Audit Trail & Lineage":
        st.markdown(
            '<div class="main-header">Audit Trail & Lineage Provenance</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sub-header">Full audit log of served predictions with SHA-256 hashes and MLflow links</div>',
            unsafe_allow_html=True,
        )

        logs_df = audit_logger.get_recent_audit_logs(limit=100)
        if logs_df.empty:
            st.info(
                "No predictions recorded in audit database yet. Make forecast requests via UI or API."
            )
        else:
            st.dataframe(logs_df, use_container_width=True)


if __name__ == "__main__":
    main()
