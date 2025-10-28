import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

# ---------------------------------------------
# 🎨 Page Configuration
# ---------------------------------------------
st.set_page_config(page_title="Advanced Data Visualization Dashboard", layout="wide")
st.title("📊 Advanced Interactive Data Visualization Dashboard")
st.markdown("""
Upload your CSV dataset and explore **interactive charts, filters, correlation analysis, and AI-generated insights!**
""")

# ---------------------------------------------
# 📂 File Upload
# ---------------------------------------------
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Preview Data
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------------------
    # 🧮 Data Overview
    # ---------------------------------------------
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("### 🧾 Summary Statistics")
    st.dataframe(df.describe())

    # ---------------------------------------------
    # 🔍 Sidebar Controls
    # ---------------------------------------------
    st.sidebar.header("⚙️ Visualization Controls")
    all_columns = df.columns.tolist()
    
    chart_type = st.sidebar.selectbox(
        "Select Chart Type", 
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Heatmap"]
    )

    x_axis = st.sidebar.selectbox("Select X-axis", all_columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", all_columns)
    color_option = st.sidebar.selectbox("Color (optional)", [None] + all_columns)

    # ---------------------------------------------
    # 🎨 Chart Generation
    # ---------------------------------------------
    st.subheader("📈 Interactive Chart")

    if chart_type == "Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis, color=color_option, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_axis, y=y_axis, color=color_option, markers=True, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option, size_max=12, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        fig = px.pie(df, names=x_axis, values=y_axis, color=color_option, title=f"{x_axis} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis, color=color_option, nbins=30, title=f"Distribution of {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        fig = px.box(df, x=x_axis, y=y_axis, color=color_option, title=f"Box Plot of {y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap":
        st.write("🔢 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------
    # 🔍 Dynamic Filtering
    # ---------------------------------------------
    st.subheader("🔎 Filter Data Dynamically")

    filter_column = st.selectbox("Select a column to filter", all_columns)
    unique_values = df[filter_column].dropna().unique().tolist()
    selected_values = st.multiselect("Select values to include", unique_values)

    if selected_values:
        filtered_df = df[df[filter_column].isin(selected_values)]
    else:
        filtered_df = df

    st.dataframe(filtered_df.head())

    # ---------------------------------------------
    # 📊 Correlation Analysis (Advanced)
    # ---------------------------------------------
    st.subheader("📉 Correlation & Insights")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found for correlation.")

    # ---------------------------------------------
    # 💡 AI-Style Data Summary
    # ---------------------------------------------
    st.subheader("🧠 Auto Insights Summary")

    st.write("Generating insights from dataset...")
    try:
        # Simple AI-like insight generation
        insight = f"""
        ✅ **Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.**  
        🧩 The most correlated features are likely related to `{numeric_df.corr().abs().unstack().sort_values(ascending=False).index[1][0]}` and `{numeric_df.corr().abs().unstack().sort_values(ascending=False).index[1][1]}`.  
        📈 The highest value in `{y_axis}` is **{df[y_axis].max()}**, while the lowest is **{df[y_axis].min()}**.  
        📊 The dataset seems suitable for regression or trend analysis on `{y_axis}` over `{x_axis}`.  
        """
        st.markdown(insight)
    except:
        st.info("Unable to generate detailed insights — check your numeric columns.")

    # ---------------------------------------------
    # 💾 Download Filtered Data
    # ---------------------------------------------
    st.subheader("📥 Download Filtered Dataset")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv", key='download-csv')

else:
    st.info("👆 Please upload a CSV file to begin interactive analysis.")
