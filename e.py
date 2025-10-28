import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------
# 🎨 Page Configuration
# ---------------------------
st.set_page_config(page_title="Interactive Data Visualization Dashboard", layout="wide")
st.title("📊 Interactive Data Visualization Dashboard")
st.markdown("Upload your dataset (CSV) and explore interactive visualizations!")

# ---------------------------
# 📂 File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Display dataframe preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # 🧮 Data Overview
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
    # 🔍 Sidebar Controls
    # ---------------------------
    st.sidebar.header("⚙️ Visualization Controls")
    all_columns = df.columns.tolist()
    
    chart_type = st.sidebar.selectbox("Select Chart Type", 
                                      ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"])

    x_axis = st.sidebar.selectbox("Select X-axis", all_columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", all_columns)
    color_option = st.sidebar.selectbox("Color (optional)", [None] + all_columns)

    # ---------------------------
    # 📈 Generate Visualization
    # ---------------------------
    st.subheader("📈 Interactive Chart")

    if chart_type == "Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis, color=color_option, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_axis, y=y_axis, color=color_option, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option, size_max=10, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        fig = px.pie(df, names=x_axis, values=y_axis, color=color_option, title=f"{x_axis} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis, color=color_option, title=f"Distribution of {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        fig = px.box(df, x=x_axis, y=y_axis, color=color_option, title=f"Box Plot of {y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 🔢 Data Filter Section
    # ---------------------------
    st.subheader("🔎 Filter Data")
    selected_column = st.selectbox("Select column to filter", all_columns)
    unique_values = df[selected_column].unique()

    selected_value = st.selectbox("Select value", unique_values)
    filtered_df = df[df[selected_column] == selected_value]

    st.write(f"Filtered data where **{selected_column} = {selected_value}**")
    st.dataframe(filtered_df.head())

else:
    st.info("👆 Please upload a CSV file to start analysis.")
