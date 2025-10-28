import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ---------------------------
# 🎨 Page Configuration
# ---------------------------
st.set_page_config(page_title="Advanced Data Visualization Dashboard", layout="wide")
st.title("📊 Advanced Interactive Data Visualization Dashboard")
st.markdown("Upload your dataset (CSV) and explore advanced visual insights with interactive charts and filters!")

# ---------------------------
# 📂 File Upload
# ---------------------------
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Dataset Preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # 🧹 Data Cleaning Options
    # ---------------------------
    st.sidebar.header("🧹 Data Cleaning Tools")

    if st.sidebar.checkbox("Remove Missing Values"):
        df.dropna(inplace=True)
        st.sidebar.write("Removed rows with missing values.")

    if st.sidebar.checkbox("Remove Duplicates"):
        df.drop_duplicates(inplace=True)
        st.sidebar.write("Removed duplicate rows.")

    if st.sidebar.checkbox("Fill Missing with Mean"):
        df = df.fillna(df.mean(numeric_only=True))

    # ---------------------------
    # 📊 Dataset Overview
    # ---------------------------
    st.subheader("📊 Dataset Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Shape of Dataset:**")
        st.write(df.shape)
        st.markdown("**Data Types:**")
        st.write(df.dtypes)

    with col2:
        st.markdown("**Null Values:**")
        st.write(df.isnull().sum())

    st.markdown("**Statistical Summary:**")
    st.write(df.describe())

    # ---------------------------
    # 📈 Correlation Heatmap
    # ---------------------------
    st.subheader("📈 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("⚠️ No numeric columns found for correlation heatmap.")

    # ---------------------------
    # 🎛️ Sidebar Controls
    # ---------------------------
    st.sidebar.header("⚙️ Visualization Controls")

    all_columns = df.columns.tolist()
    chart_type = st.sidebar.selectbox("Select Chart Type", 
                                      ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"])

    x_axis = st.sidebar.selectbox("Select X-axis", all_columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", all_columns)
    color_option = st.sidebar.selectbox("Color (optional)", [None] + all_columns)
    trendline_option = st.sidebar.checkbox("Add Trendline (for scatter plots only)")

    # ---------------------------
    # 🎨 Theme Toggle
    # ---------------------------
    dark_mode = st.sidebar.radio("Theme", ["Light", "Dark"])
    template = "plotly_dark" if dark_mode == "Dark" else "plotly_white"

    # ---------------------------
    # 📊 Chart Generation
    # ---------------------------
    st.subheader("📈 Interactive Chart")

    if chart_type == "Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis, color=color_option, title=f"{y_axis} vs {x_axis}", template=template)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_axis, y=y_axis, color=color_option, title=f"{y_axis} over {x_axis}", template=template)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option,
                         trendline="ols" if trendline_option else None,
                         title=f"{y_axis} vs {x_axis}", template=template)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        fig = px.pie(df, names=x_axis, values=y_axis, color=color_option, title=f"{x_axis} Distribution", template=template)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis, color=color_option, title=f"Distribution of {x_axis}", template=template)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        fig = px.box(df, x=x_axis, y=y_axis, color=color_option, title=f"Box Plot of {y_axis} by {x_axis}", template=template)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 🔍 Advanced Data Filtering
    # ---------------------------
    st.subheader("🔍 Advanced Data Filter")

    filter_columns = st.multiselect("Select columns to filter", all_columns)
    filtered_df = df.copy()

    for col in filter_columns:
        unique_vals = df[col].unique()
        selected_vals = st.multiselect(f"Filter values for {col}", unique_vals, default=unique_vals)
        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    st.write(f"Filtered data preview ({filtered_df.shape[0]} rows):")
    st.dataframe(filtered_df.head())

    # ---------------------------
    # 📤 Download Processed Data
    # ---------------------------
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Dataset", csv, "filtered_data.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to begin your analysis.")
