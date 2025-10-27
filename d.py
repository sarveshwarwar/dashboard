import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="Interactive Data Analysis Dashboard", layout="wide")
st.title("📊 Interactive Data Analysis Dashboard")
st.write("Upload a CSV file to automatically explore, summarize, and visualize your dataset.")

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ------------------- Basic Info -------------------
    st.subheader("📊 Basic Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    # Data types and missing values
    st.write("### Column Info")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(info_df)

    # ------------------- Summary Statistics -------------------
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include='all').T)

    # ------------------- Data Visualization -------------------
    st.subheader("📊 Visualizations")

    # Choose column for histogram
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(numeric_cols) > 0:
        col = st.selectbox("Select numeric column for distribution plot", numeric_cols)
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Select X-axis", numeric_cols)
        y_axis = st.selectbox("Select Y-axis", numeric_cols, index=1)
        fig2 = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig3 = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale='Viridis',
            showscale=True
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Box plot
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat_col = st.selectbox("Select categorical column for box plot", categorical_cols)
        num_col = st.selectbox("Select numeric column for box plot", numeric_cols)
        fig4 = px.box(df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}", color=cat_col)
        st.plotly_chart(fig4, use_container_width=True)

    # ------------------- Download Section -------------------
    st.subheader("📥 Download Cleaned Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name="cleaned_data.csv", mime="text/csv")

else:
    st.info("👆 Upload a CSV file to begin your analysis.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Pandas, and Plotly.")
