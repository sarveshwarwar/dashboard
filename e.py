import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------
# 🎨 Page Configuration
# ---------------------------
st.set_page_config(page_title="AI-Powered Data Analytics Dashboard", layout="wide")
st.title("📊 AI-Powered Interactive Data Analytics Dashboard")
st.markdown("Upload your dataset (CSV) to explore ML insights, charts, forecasts, and more!")

# ---------------------------
# 📂 File Upload
# ---------------------------
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # ---------------------------
    # 🧹 Data Cleaning
    # ---------------------------
    st.sidebar.header("🧹 Data Cleaning Tools")

    if st.sidebar.checkbox("Remove Missing Values"):
        df.dropna(inplace=True)

    if st.sidebar.checkbox("Remove Duplicates"):
        df.drop_duplicates(inplace=True)

    if st.sidebar.checkbox("Fill Missing with Mean"):
        df = df.fillna(df.mean(numeric_only=True))

    # ---------------------------
    # 📊 Tabs
    # ---------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Dataset", "📈 Visualizations", "🧮 Correlation", "🤖 ML Prediction", "📤 Export"])

    # ---------------------------
    # 📄 Tab 1: Dataset Overview
    # ---------------------------
    with tab1:
        st.subheader("📊 Dataset Overview")
        st.write(df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape:**", df.shape)
            st.write("**Data Types:**")
            st.write(df.dtypes)
        with col2:
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())

        st.subheader("📋 Statistical Summary")
        st.write(df.describe())

    # ---------------------------
    # 📈 Tab 2: Visualizations
    # ---------------------------
    with tab2:
        st.subheader("🎨 Data Visualization")
        all_columns = df.columns.tolist()

        chart_type = st.selectbox("Select Chart Type", 
                                  ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Map"])
        x_axis = st.selectbox("Select X-axis", all_columns)
        y_axis = st.selectbox("Select Y-axis", all_columns)
        color_option = st.selectbox("Color (optional)", [None] + all_columns)

        dark_mode = st.radio("Theme", ["Light", "Dark"])
        template = "plotly_dark" if dark_mode == "Dark" else "plotly_white"

        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_option, template=template)
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, color=color_option, template=template)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_option, trendline="ols", template=template)
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_axis, values=y_axis, color=color_option, template=template)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis, color=color_option, template=template)
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_axis, y=y_axis, color=color_option, template=template)
        elif chart_type == "Map":
            if "lat" in df.columns and "lon" in df.columns:
                fig = px.scatter_mapbox(df, lat="lat", lon="lon", color=color_option, zoom=3, mapbox_style="carto-positron")
            else:
                st.warning("⚠️ Please include 'lat' and 'lon' columns for map visualization.")
                fig = None

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 🧮 Tab 3: Correlation
    # ---------------------------
    with tab3:
        st.subheader("📊 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("### 🔍 Strongest Correlations")
            corr_pairs = corr.unstack().reset_index()
            corr_pairs.columns = ["feature_1", "feature_2", "corr"]
            corr_pairs = corr_pairs[(corr_pairs["feature_1"] != corr_pairs["feature_2"])]
            corr_pairs["abs_corr"] = abs(corr_pairs["corr"])
            top_corr = corr_pairs.sort_values("abs_corr", ascending=False).drop_duplicates(subset=["corr"]).head(5)

            for _, row in top_corr.iterrows():
                st.write(f"- {row['feature_1']} ↔ {row['feature_2']} : **{row['corr']:.2f}**")
        else:
            st.warning("⚠️ No numeric columns found for correlation analysis.")

    # ---------------------------
    # 🤖 Tab 4: Machine Learning Prediction
    # ---------------------------
    with tab4:
        st.subheader("🤖 Simple ML Prediction (Linear Regression)")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            feature_col = st.selectbox("Select Feature (X)", numeric_cols)
            target_col = st.selectbox("Select Target (Y)", numeric_cols)

            X = df[[feature_col]]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            st.write(results.head())

            st.line_chart(results)
            st.write(f"📈 Model R² Score: **{model.score(X_test, y_test):.2f}**")
        else:
            st.warning("⚠️ Need at least 2 numeric columns for regression analysis.")

    # ---------------------------
    # 📤 Tab 5: Export
    # ---------------------------
    with tab5:
        st.subheader("📤 Download Processed Dataset")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "processed_data.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to start exploring your dataset.")
