import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")
st.markdown(
    '''
    <style>
        .block-container { padding-top: 2rem; }
        .stMetric { text-align: center !important; }
        .stButton>button { background-color: #0F172A; color: white; border-radius: 5px; }
        .st-emotion-cache-1v0mbdj, .st-emotion-cache-1fcbxyh { padding: 1rem 2rem; }
    </style>
    ''',
    unsafe_allow_html=True
)

# Sidebar for uploads
st.sidebar.title("Upload Files")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv", key="task_csv")
kpi_file = st.sidebar.file_uploader("📊 Store KPI CSV", type="csv", key="kpi_csv")

st.title("Task Performance Dashboard")

if task_file:
    task_df = pd.read_csv(task_file)
    task_df['End date'] = pd.to_datetime(task_df['End date'], errors='coerce')
    task_df['Date completed'] = pd.to_datetime(task_df['Date completed'], errors='coerce')
    task_df['Days Before Due'] = (task_df['End date'] - task_df['Date completed']).dt.days
    task_df['Overdue'] = (task_df['End date'] < pd.Timestamp.now()) & (task_df['Task status'] != 'Completed')

    # Region & Store Cleanup
    task_df['Region'] = task_df['Level 1'].fillna('Unknown')
    task_df['Store'] = task_df['Location name']
    excluded_stores = ['JameTrade', 'Midwest']
    task_df = task_df[~task_df['Store'].isin(excluded_stores)]

    # Base metrics
    total_tasks = len(task_df)
    overdue_tasks = task_df['Overdue'].sum()
    completed_on_time = task_df[task_df['Days Before Due'] >= 0].shape[0]
    ad_hoc_tasks = task_df['Store'].value_counts()[task_df['Store'].value_counts() == 1].shape[0]
    avg_days_before_due = task_df['Days Before Due'].dropna().mean()
    common_category = task_df['Task category'].mode()[0] if not task_df['Task category'].mode().empty else 'N/A'

    # KPIs
    st.markdown("### High-Level KPIs")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Tasks", total_tasks)
    kpi2.metric("Completed On Time", f"{completed_on_time / total_tasks:.0%}")
    kpi3.metric("Avg Days Before Due", f"{avg_days_before_due:.1f}")

    kpi4, kpi5, kpi6 = st.columns(3)
    kpi4.metric("Overdue Tasks", overdue_tasks)
    kpi5.metric("Ad Hoc Tasks", ad_hoc_tasks)
    kpi6.metric("Top Task Category", common_category)

    # Quick List of Proactive Stores
    st.markdown("### Most Proactive Stores")
    proactive = task_df.dropna(subset=['Days Before Due'])
    proactive = proactive.groupby('Store')['Days Before Due'].mean().sort_values(ascending=False).head(10)
    for store, days in proactive.items():
        st.markdown(f"- **{store}** — Avg {days:.1f} days early")

    # Overdue store chart
    st.markdown("### Stores with Most Overdue Tasks")
    overdue_counts = task_df[task_df['Overdue']].groupby('Store').size().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=overdue_counts.values, y=overdue_counts.index, palette='Reds_d', ax=ax2)
    ax2.set_title("Top 10 Stores with Overdue Tasks")
    ax2.set_xlabel("Overdue Task Count")
    st.pyplot(fig2)

    # Integrate Store KPIs
    if kpi_file:
        kpi_df = pd.read_csv(kpi_file)
        kpi_df.rename(columns={"Location ID": "Location external ID"}, inplace=True)
        merged_df = pd.merge(task_df, kpi_df, on=["Location external ID", "Store"], how="left")

        st.markdown("### Store Health Score")
        score_df = merged_df.groupby('Store').agg({
            'Overdue': 'mean',
            'Days Before Due': 'mean',
            'CSAT Score': 'mean',
            'Cleanliness Score': 'mean',
            'Sales vs Target (%)': 'mean'
        }).dropna().head(10)

        score_df['Health Score'] = (
            (1 - score_df['Overdue']) * 0.3 +
            (score_df['Days Before Due'] / score_df['Days Before Due'].max()) * 0.2 +
            (score_df['CSAT Score'] / 100) * 0.2 +
            (score_df['Cleanliness Score'] / 100) * 0.2 +
            ((score_df['Sales vs Target (%)'] + 15) / 35) * 0.1
        ) * 100

        st.dataframe(score_df[['Health Score', 'CSAT Score', 'Cleanliness Score', 'Sales vs Target (%)']].sort_values(by='Health Score', ascending=False).style.format({
            'Health Score': '{:.1f}',
            'CSAT Score': '{:.1f}',
            'Cleanliness Score': '{:.1f}',
            'Sales vs Target (%)': '{:.1f}%'
        }), use_container_width=True)

else:
    st.info("Use the ➕ in the sidebar to upload your WorkJam task CSV.")
