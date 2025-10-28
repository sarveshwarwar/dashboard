
import io
import warnings
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Optional libraries (try import, set flags)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# For saving static images from Plotly (kaleido)
try:
    import plotly.io as pio
    pio.kaleido.scope.default_format = "png"
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

# General page config
st.set_page_config(page_title="Advanced Data Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Advanced Data Analytics Dashboard")
st.markdown("Upload a CSV and explore dashboard, forecasting, ML prediction, map view and exports.")

# -----------------------------
# File upload & cached loader
# -----------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv", "txt", "xlsx"])
sample_data_info = st.sidebar.checkbox("Use sample dataset (Iris + synthetic time + geo)")

df = None
if sample_data_info:
    # Create a combined sample data set for demo (Iris + synthetic time series + geo)
    iris = px.data.iris()
    rng = pd.date_range(end=pd.Timestamp.today(), periods=200, freq="D")
    ts = pd.DataFrame({
        "date": rng,
        "sales": (np.linspace(100, 500, len(rng)) + np.random.normal(0, 30, len(rng))).astype(int),
        "region": np.random.choice(["North", "South", "East", "West"], size=len(rng)),
        "lat": np.random.uniform(8.0, 37.0, size=len(rng)),
        "lon": np.random.uniform(68.0, 97.0, size=len(rng)),
    })
    # join produce a big df for demo
    iris["date"] = pd.Timestamp.today() - pd.to_timedelta(np.random.randint(0, 365, size=len(iris)), unit='D')
    iris = iris.rename(columns={"sepal_length": "feature_1", "sepal_width": "feature_2", "petal_length": "feature_3", "petal_width": "feature_4", "species": "category"})
    df = pd.concat([iris, ts.sample(len(iris), replace=False)], ignore_index=True, sort=False)
    st.sidebar.info("Using built-in sample dataset (synthetic) for demo.")
elif uploaded:
    try:
        if str(uploaded.type).startswith("text/") or uploaded.name.lower().endswith(".csv"):
            df = load_csv(uploaded)
        elif uploaded.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            df = load_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
        st.stop()
else:
    st.info("Upload a CSV (or select sample dataset) to start. Use the sidebar to configure options.")
    st.stop()

# Basic cleaning options
st.sidebar.header("Data Cleaning")
drop_na = st.sidebar.checkbox("Drop rows with NA", value=False)
drop_duplicates = st.sidebar.checkbox("Drop duplicate rows", value=False)
fill_method = st.sidebar.selectbox("Fill missing numeric with", ["None", "Mean", "Median"], index=0)

if drop_na:
    df = df.dropna()
if drop_duplicates:
    df = df.drop_duplicates()
if fill_method != "None":
    num_cols = df.select_dtypes(include=[np.number]).columns
    if fill_method == "Mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    else:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Basic overview
st.sidebar.markdown("### Dataset Overview")
st.sidebar.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

# -----------------------------
# Sidebar: Global filters (applied across tabs)
# -----------------------------
st.sidebar.header("Global Filters")
all_columns = df.columns.tolist()

# choose up to 3 columns to filter
filter_cols = st.sidebar.multiselect("Choose columns to filter (global)", all_columns, max_selections=3)
global_filtered_df = df.copy()
for col in filter_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        min_v, max_v = st.sidebar.slider(f"Range for {col}", float(df[col].min()), float(df[col].max()), (float(df[col].min()), float(df[col].max())))
        global_filtered_df = global_filtered_df[(global_filtered_df[col] >= min_v) & (global_filtered_df[col] <= max_v)]
    else:
        vals = st.sidebar.multiselect(f"Values for {col}", options=df[col].dropna().unique().tolist(), default=df[col].dropna().unique().tolist())
        global_filtered_df = global_filtered_df[global_filtered_df[col].isin(vals)]

# -----------------------------
# Tabs layout
# -----------------------------
tab_dashboard, tab_forecast, tab_ml, tab_map, tab_export = st.tabs(["📊 Dashboard", "📈 Forecasting", "🤖 ML Predict", "🗺️ Map View", "📤 Export & Insights"])

# -----------------------------
# Helper functions
# -----------------------------
def suggest_chart(df_local, x, y):
    """Very small heuristic for chart suggestion"""
    if pd.api.types.is_numeric_dtype(df_local[x]) and pd.api.types.is_numeric_dtype(df_local[y]):
        return "Scatter"
    if pd.api.types.is_numeric_dtype(df_local[y]) and not pd.api.types.is_numeric_dtype(df_local[x]):
        return "Bar"
    return "Table"

def download_plotly_fig(fig, filename="chart.png"):
    """Return bytes for download; requires kaleido."""
    if not KALEIDO_AVAILABLE:
        st.warning("Kaleido not installed — PNG export unavailable. Install `kaleido` to enable.")
        return None
    buf = io.BytesIO()
    try:
        fig.write_image(buf, format="png", engine="kaleido")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Failed to create image: {e}")
        return None

# -----------------------------
# Tab: Dashboard
# -----------------------------
with tab_dashboard:
    st.header("Interactive Dashboard")
    st.write("Use the controls to generate interactive charts and a correlation heatmap.")

    cols = st.columns([3, 1])
    with cols[1]:
        st.markdown("### Controls")
        x_axis = st.selectbox("X axis", all_columns, index=0)
        y_axis = st.selectbox("Y axis", all_columns, index=min(1, len(all_columns)-1))
        color = st.selectbox("Color (optional)", [None] + all_columns)
        chart_type = st.selectbox("Chart type", ["Auto (recommend)", "Bar", "Line", "Scatter", "Histogram", "Box", "Treemap"])
        sample_n = st.slider("Sample rows for plotting (0 = all)", 0, int(min(20000, len(global_filtered_df))), 0)

    # choose df sample
    plot_df = global_filtered_df.copy()
    if sample_n and sample_n > 0 and len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)

    # recommend chart if auto
    if chart_type == "Auto (recommend)":
        rec = suggest_chart(plot_df, x_axis, y_axis)
        st.info(f"Recommended chart: **{rec}**")
        chart_type = rec

    # Build chart
    try:
        if chart_type == "Bar":
            fig = px.bar(plot_df, x=x_axis, y=y_axis, color=color, title=f"{y_axis} by {x_axis}", height=600)
        elif chart_type == "Line":
            fig = px.line(plot_df, x=x_axis, y=y_axis, color=color, title=f"{y_axis} over {x_axis}", height=600)
        elif chart_type == "Scatter":
            fig = px.scatter(plot_df, x=x_axis, y=y_axis, color=color, title=f"{y_axis} vs {x_axis}", height=600)
        elif chart_type == "Histogram":
            fig = px.histogram(plot_df, x=x_axis, color=color, title=f"Distribution of {x_axis}", height=600)
        elif chart_type == "Box":
            fig = px.box(plot_df, x=x_axis, y=y_axis, color=color, title=f"Box plot of {y_axis} by {x_axis}", height=600)
        elif chart_type == "Treemap":
            # require categorical path - use up to 3 levels if available
            cat_cols = [c for c in plot_df.columns if not pd.api.types.is_numeric_dtype(plot_df[c])]
            path = cat_cols[:3] if cat_cols else [x_axis]
            fig = px.treemap(plot_df, path=path, values=y_axis if y_axis in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[y_axis]) else None, color=y_axis if y_axis in plot_df.columns else None, height=600)
        else:
            fig = px.scatter(plot_df, x=x_axis, y=y_axis, color=color, title=f"{y_axis} vs {x_axis}", height=600)

        st.plotly_chart(fig, use_container_width=True)
        # allow downloading the chart as PNG
        if st.button("Download Chart as PNG"):
            buf = download_plotly_fig(fig)
            if buf:
                st.download_button("Click to download PNG", data=buf, file_name="chart.png", mime="image/png")
    except Exception as e:
        st.error(f"Failed to generate chart: {e}")

    # Correlation heatmap (numeric)
    st.markdown("---")
    st.subheader("Correlation Heatmap (numeric columns)")
    num_df = global_filtered_df.select_dtypes(include=[np.number])
    if not num_df.empty:
        corr = num_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        # Auto-insights: top correlated pairs
        corr_pairs = (
            corr.abs()
            .unstack()
            .sort_values(ascending=False)
            .drop_duplicates()
            .reset_index()
        )
        corr_pairs.columns = ["feature_1", "feature_2", "corr"]
        corr_pairs = corr_pairs[corr_pairs["feature_1"] != corr_pairs["feature_2"]]
        top_pairs = corr_pairs.drop_duplicates(subset=["corr"]).head(5)
        st.markdown("**Top correlated feature pairs (abs):**")
        for _, row in top_pairs.iterrows():
            st.write(f"- {row.feature_1} ↔ {row.feature_2}  :  {row.corr:.2f}")
    else:
        st.info("No numeric columns to compute correlations.")

# -----------------------------
# Tab: Forecasting (time-series)
# -----------------------------
with tab_forecast:
    st.header("Time-series Forecasting")
    st.write("Choose a date column and a numeric target. Uses ARIMA (statsmodels) if available; otherwise shows a simple rolling forecast.")
    date_col = st.selectbox("Date column (must be datetime)", [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])], index=0 if any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns) else None)
    if date_col is None:
        # allow parsing if there's a column that looks like dates (try to coerce)
        candidate_dates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        date_col = st.selectbox("Or pick a candidate date-like column to parse", options=[None] + candidate_dates)
        if date_col:
            try:
                global_filtered_df[date_col] = pd.to_datetime(global_filtered_df[date_col])
            except Exception:
                st.error("Failed to parse column as datetime. Please ensure a proper date column.")
                date_col = None

    target_col = st.selectbox("Target (numeric)", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])], index=0 if any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns) else None)
    periods = st.number_input("Forecast periods (future steps)", min_value=1, max_value=365, value=30, step=1)

    if date_col and target_col:
        ts_df = global_filtered_df[[date_col, target_col]].dropna().sort_values(by=date_col)
        ts_df = ts_df.rename(columns={date_col: "ds", target_col: "y"})
        ts_df.ds = pd.to_datetime(ts_df.ds)

        st.write("Showing latest time-series")
        fig_ts = px.line(ts_df, x="ds", y="y", title=f"Time-series of {target_col}", height=400)
        st.plotly_chart(fig_ts, use_container_width=True)

        if STATSMODELS_AVAILABLE:
            st.write("Fitting ARIMA model (statsmodels)... this can take a few seconds for large series.")
            # simple ARIMA order selection fallback: (1,1,1)
            try:
                with st.spinner("Training ARIMA(1,1,1)..."):
                    model = ARIMA(ts_df.y, order=(1, 1, 1))
                    fitted = model.fit()
                    # make future index
                    last_date = ts_df.ds.max()
                    freq = pd.infer_freq(ts_df.ds) or "D"
                    future_index = pd.date_range(start=last_date + pd.Timedelta(1, unit=freq[0]) if len(freq) else last_date + timedelta(days=1), periods=periods, freq=freq)
                    # Predict
                    preds = fitted.get_forecast(steps=periods)
                    pred_mean = preds.predicted_mean
                    pred_ci = preds.conf_int()
                    future_df = pd.DataFrame({
                        "ds": future_index,
                        "yhat": pred_mean.values,
                        "yhat_lower": pred_ci.iloc[:, 0].values,
                        "yhat_upper": pred_ci.iloc[:, 1].values
                    })
                    plot_df_forecast = pd.concat([ts_df.rename(columns={"y": "y"}), future_df.rename(columns={"yhat": "y"})], ignore_index=True, sort=False)
                    fig_fore = px.line(plot_df_forecast, x="ds", y="y", title=f"Forecast for {target_col}", height=500)
                    # overlay CI as filled area
                    fig_fore.add_traces(px.line(future_df, x="ds", y="yhat").data)
                    st.plotly_chart(fig_fore, use_container_width=True)
            except Exception as e:
                st.error(f"ARIMA model failed: {e}. Showing simple rolling-mean forecast instead.")
                STATSMODELS_AVAILABLE = False  # fallback
        if not STATSMODELS_AVAILABLE:
            st.info("statsmodels not available — using simple moving average forecast.")
            rolling = ts_df.y.rolling(window=7, min_periods=1).mean()
            last_mean = rolling.iloc[-1]
            future_vals = np.full(shape=periods, fill_value=last_mean)
            last_date = ts_df.ds.max()
            future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq="D")
            future_df = pd.DataFrame({"ds": future_index, "y": future_vals})
            concat = pd.concat([ts_df.rename(columns={"y": "y"}), future_df], ignore_index=True, sort=False)
            fig = px.line(concat, x="ds", y="y", title=f"Moving-average forecast for {target_col}", height=500)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please pick a date column and a numeric target column to forecast.")

# -----------------------------
# Tab: ML Predict
# -----------------------------
with tab_ml:
    st.header("Machine Learning: Train & Predict")
    st.write("Train a simple model on selected features and predict target values. Sklearn required for ML.")

    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn not installed. Install scikit-learn to use ML features (pip install scikit-learn).")
    else:
        numeric_cols = global_filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.info("Not enough numeric columns for training a model.")
        else:
            target = st.selectbox("Select target (numeric)", numeric_cols, index=0)
            features = st.multiselect("Select features (numeric)", [c for c in numeric_cols if c != target], default=[c for c in numeric_cols if c != target][:3])
            test_size = st.slider("Test set fraction", 0.05, 0.5, 0.2)
            model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest"])
            if st.button("Train & Evaluate"):
                X = global_filtered_df[features].dropna()
                y = global_filtered_df.loc[X.index, target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                st.write("Sample predictions:")
                st.dataframe(results_df.head())
                # metrics
                mae = np.mean(np.abs(y_test - y_pred))
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                st.metric("MAE", f"{mae:.3f}")
                st.metric("RMSE", f"{rmse:.3f}")
                # plot actual vs predicted
                fig = px.scatter(results_df, x="Actual", y="Predicted", trendline="ols", title="Actual vs Predicted")
                st.plotly_chart(fig, use_container_width=True)

                # Save model to memory for quick prediction (very small demo)
                st.session_state["last_model"] = model
                st.session_state["last_features"] = features
                st.success("Model trained and saved to session (temporary).")

        # Single-row prediction UI
        if "last_model" in st.session_state:
            st.markdown("---")
            st.subheader("Make predictions on new data (single row)")
            feature_inputs = {}
            for f in st.session_state["last_features"]:
                val = st.number_input(f"Value for {f}", value=float(global_filtered_df[f].median()))
                feature_inputs[f] = val
            if st.button("Predict single row"):
                X_new = pd.DataFrame([feature_inputs])
                pred = st.session_state["last_model"].predict(X_new)[0]
                st.success(f"Predicted {target}: {pred:.3f}")

# -----------------------------
# Tab: Map View
# -----------------------------
with tab_map:
    st.header("Geospatial Map View")
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]
    if not lat_cols or not lon_cols:
        st.info("No lat/lon columns detected. If you have latitude/longitude columns name them 'lat' and 'lon' or include 'lat'/'lon' in their names.")
    else:
        lat_col = st.selectbox("Latitude column", lat_cols)
        lon_col = st.selectbox("Longitude column", lon_cols)
        map_color = st.selectbox("Color by (optional)", [None] + all_columns)
        size_col = st.selectbox("Size by (optional numeric)", [None] + [c for c in all_columns if pd.api.types.is_numeric_dtype(df[c])])

        # show sampled points if huge
        sample_n_map = st.slider("Sample rows for map (0=all)", 0, min(20000, len(global_filtered_df)), 1000)
        map_df = global_filtered_df.dropna(subset=[lat_col, lon_col])
        if sample_n_map and len(map_df) > sample_n_map:
            map_df = map_df.sample(sample_n_map, random_state=1)

        fig_map = px.scatter_mapbox(map_df, lat=lat_col, lon=lon_col, color=map_color, size=size_col if size_col else None,
                                    hover_data=list(map_df.columns), zoom=3, height=700, mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------
# Tab: Export & AI Insights
# -----------------------------
with tab_export:
    st.header("Export & Auto Insights")

    st.subheader("Download filtered dataset")
    csv = global_filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "filtered_dataset.csv", "text/csv")

    st.markdown("---")
    st.subheader("AI-style Auto Insights (heuristic)")
    # Basic heuristics
    insights = []
    try:
        n_rows, n_cols = global_filtered_df.shape
        insights.append(f"- Dataset has **{n_rows} rows** and **{n_cols} columns**.")
        # top missing columns
        missing = global_filtered_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            top = missing.head(3)
            insights.append(f"- Columns with most missing values: " + ", ".join([f"{c} ({int(v)})" for c, v in top.items()]))

        # numeric summary highlights
        num = global_filtered_df.select_dtypes(include=[np.number])
        if not num.empty:
            means = num.mean().sort_values(ascending=False)
            top_mean = means.head(1)
            if not top_mean.empty:
                insights.append(f"- Highest mean among numeric columns: **{top_mean.index[0]}** = {top_mean.iloc[0]:.2f}")
            stds = num.std().sort_values(ascending=False)
            insights.append(f"- Most variable column (std): **{stds.index[0]}** = {stds.iloc[0]:.2f}")

            # correlation top pair
            corr = num.corr().abs()
            if corr.size > 1:
                # flatten and remove self correlations
                corr_vals = corr.unstack().reset_index()
                corr_vals.columns = ["a", "b", "corr"]
                corr_vals = corr_vals[corr_vals["a"] != corr_vals["b"]]
                top_corr = corr_vals.sort_values("corr", ascending=False).iloc[0]
                insights.append(f"- Strong correlation detected between **{top_corr['a']}** and **{top_corr['b']}**: {top_corr['corr']:.2f}")
        else:
            insights.append("- No numeric columns available for numeric insights.")

        # date/time trends
        date_cols = [c for c in global_filtered_df.columns if pd.api.types.is_datetime64_any_dtype(global_filtered_df[c])]
        if date_cols:
            insights.append(f"- Found date column(s): {', '.join(date_cols)}. Consider using Forecasting tab.")
    except Exception as e:
        insights.append(f"- Could not generate some insights: {e}")

    for line in insights:
        st.markdown(line)

    st.markdown("---")
    st.subheader("Save current chart (if shown) as PNG")
    st.info("If you generated a chart in Dashboard or Forecasting, you can re-generate it and click 'Download Chart as PNG' where available.")
    if not KALEIDO_AVAILABLE:
        st.warning("PNG export requires the `kaleido` package. Install with `pip install kaleido` to enable chart PNG downloads.")

# End of app
st.markdown("---")
st.caption("Built with ❤️ — Advanced dashboard. Extend/modify as needed for your dataset.")
