import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

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
        st.subheader("📈 Descriptive Statistics")
        st.write(filtered_df.describe(include='all').T)

    # ------------------- Data Visualizations -------------------
    numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = filtered_df.select_dtypes(include=['object']).columns

    if show_visuals:
        st.subheader("📊 Visualizations")

        # Histogram selector
        if len(numeric_cols) > 0:
            col_hist = st.selectbox("Select numeric column for histogram", numeric_cols)
            fig_hist = px.histogram(filtered_df, x=col_hist, nbins=30, title=f"Distribution of {col_hist}")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Scatter plot selector
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("Select X-axis for scatter plot", numeric_cols)
            y_axis = st.selectbox("Select Y-axis for scatter plot", numeric_cols, index=1)
            fig_scatter = px.scatter(filtered_df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Box plot selector
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = st.selectbox("Select categorical column for box plot", categorical_cols)
            num_col = st.selectbox("Select numeric column for box plot", numeric_cols)
            fig_box = px.box(filtered_df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}", color=cat_col)
            st.plotly_chart(fig_box, use_container_width=True)

    # ------------------- Correlation Heatmap -------------------
    if show_correlation and len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Heatmap")
        corr = filtered_df[numeric_cols].corr()
        fig_corr = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale='Viridis',
            showscale=True
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------- Download Option -------------------
    if show_download:
        st.subheader("📥 Download Filtered Data")
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, file_name="filtered_data.csv", mime="text/csv")

else:
    st.sidebar.info("👆 Upload a CSV file to begin your analysis.")
