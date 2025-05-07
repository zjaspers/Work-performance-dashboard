import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Task Performance Dashboard", layout="wide")

st.title("Task Performance Dashboard")

uploaded_file = st.file_uploader("Upload your WorkJam task CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
    df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days

    st.markdown("### Overall Metrics")
    col1, col2 = st.columns(2)
    with col1:
        overdue_count = df[(df['End date'] < pd.Timestamp.now()) & (df['Task status'] != 'Completed')].shape[0]
        st.metric("Overdue Tasks", overdue_count)
    with col2:
        avg_speed = df['Days Before Due'].dropna().mean()
        st.metric("Avg Days Before Due", f"{avg_speed:.1f}" if pd.notnull(avg_speed) else "N/A")

    st.markdown("### Speed to Execution - Top Stores")
    valid_df = df.dropna(subset=['End date', 'Date completed'])
    top_stores = valid_df.groupby('Location name')['Days Before Due'].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top_stores.values, y=top_stores.index, palette='Blues_d', ax=ax)
    ax.set_xlabel("Avg Days Before Due")
    ax.set_ylabel("Store")
    ax.set_title("Most Proactive Stores")
    st.pyplot(fig)

    st.markdown("### Raw Data Preview")
    st.dataframe(df.head(50))
else:
    st.info("Upload a WorkJam task export CSV to begin.")
