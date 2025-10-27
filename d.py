import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="Interactive Data Analysis Dashboard", layout="wide")
st.title("📊 Interactive Data Analysis Dashboard")
st.write("Upload a CSV file to automatically explore, summarize, and visualize your dataset.")
st.markdown("---")

# ------------------- Sidebar -------------------
st.sidebar.title("⚙️ Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

show_overview = st.sidebar.checkbox("Show Dataset Overview", value=True)
show_summary = st.sidebar.checkbox("Show Descriptive Statistics", value=True)
show_visuals = st.sidebar.checkbox("Show Visualizations", value=True)
show_correlation = st.sidebar.checkbox("Show Correlation Heatmap", value=True)
show_outliers = st.sidebar.checkbox("Highlight Outliers", value=True)
show_insights = st.sidebar.checkbox("Generate Automatic Insights", value=True)
show_download = st.sidebar.checkbox("Show Download Option", value=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # ------------------- Dataset Overview -------------------
    if show_overview:
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.write("### Column Info")
        st.dataframe(info_df)

    # ------------------- Data Filtering -------------------
    st.subheader("🔍 Data Filtering")
    filter_cols = st.multiselect("Select columns to filter", df.columns)
    filtered_df = df.copy()
    for col in filter_cols:
        if df[col].dtype in ['int64', 'float64']:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            range_vals = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[(filtered_df[col] >= range_vals[0]) & (filtered_df[col] <= range_vals[1])]
        else:
            unique_vals = df[col].unique().tolist()
            selected_vals = st.multiselect(f"Filter {col}", unique_vals, default=unique_vals)
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    st.write("### Filtered Data Preview")
    st.dataframe(filtered_df.head())

    # ------------------- Descriptive Statistics -------------------
    if show_summary:
        st.subheader("📈 D
