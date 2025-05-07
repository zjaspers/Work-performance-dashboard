import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")
st.markdown(
    """
    <style>
        .block-container { padding: 2rem; }
        .stMetric { text-align: center !important; }
        .stSidebar { background-color: #F7F7F7; }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar uploads and week selector
st.sidebar.title("Upload & Parameters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv", key="task_csv")
kpi_file = st.sidebar.file_uploader("📊 Store KPI CSV", type="csv", key="kpi_csv")

# Once task data is uploaded, process for week selection
if task_file:
    df = pd.read_csv(task_file)
    # Parse dates
    df['End date'] = pd.to_datetime(df.get('End date'), errors='coerce')
    df['Date completed'] = pd.to_datetime(df.get('Date completed'), errors='coerce')
    # Core metrics
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days

    # Handle tasks with missing completion: assume not done and record Days Before Due as negative absolute days from today
    now = pd.Timestamp.now().normalize()
    missing_comp = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing_comp, 'Days Before Due'] = -((df.loc[missing_comp, 'End date'] - now).abs().dt.days)

    df['Overdue'] = (df['End date'] < pd.Timestamp.now()) & (df['Task status'] != 'Completed')
    # Exclude company and non-stores
    df['Region'] = df['Level 1'].fillna('Unknown')
    df['Store'] = df['Location name']
    df = df[~df['Store'].isin(['JameTrade', 'Midwest'])]

    # Determine week start (Monday) for each task
    df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)
    # Week options sorted descending
    weeks = sorted(df['Week Start'].dropna().unique(), reverse=True)
    week_labels = [f"{w.date()} to {(w + timedelta(days=6)).date()}" for w in weeks]
    week_choice = st.sidebar.selectbox("Select Week (Mon–Sun)", week_labels)

    # Filter data for the selected week
    selected_start = weeks[week_labels.index(week_choice)]
    week_df = df[df['Week Start'] == selected_start]

    # Merge with KPI data if available
    if kpi_file:
        kpi_df = pd.read_csv(kpi_file)
        # Ensure column matching
        kpi_df.rename(columns={'Location ID': 'Location external ID'}, inplace=True)
        week_df = week_df.merge(kpi_df, on=['Location external ID', 'Store'], how='left')

    # === Smart Insights ===
    insights = []
    # Region change in overdue rate vs previous week
    prev_start = selected_start - timedelta(weeks=1)
    prev_df = df[df['Week Start'] == prev_start]
    if not prev_df.empty:
        curr_region = week_df.groupby('Region')['Overdue'].mean()
        prev_region = prev_df.groupby('Region')['Overdue'].mean()
        for region in curr_region.index:
            curr_rate = curr_region.get(region, 0)
            prev_rate = prev_region.get(region, 0)
            delta = (curr_rate - prev_rate) * 100
            if abs(delta) >= 1:
                trend = "risen" if delta > 0 else "fallen"
                insights.append(f"Region {region} overdue rate has {trend} by {abs(delta):.1f}% since last week.")
    # Store speed change
    curr_speed = week_df.groupby('Store')['Days Before Due'].mean()
    prev_speed = prev_df.groupby('Store')['Days Before Due'].mean()
    for store in curr_speed.index:
        curr_s = curr_speed.get(store, np.nan)
        prev_s = prev_speed.get(store, np.nan)
        if not np.isnan(prev_s):
            delta = curr_s - prev_s
            if abs(delta) >= 0.5:
                trend = "slower" if delta < 0 else "faster"
                insights.append(f"{store} completed tasks {abs(delta):.1f} days {trend} than last week.")

    # CSAT insight
    if 'CSAT Score' in week_df.columns:
        low_csat = week_df.groupby('Store')['CSAT Score'].mean()
        for store, csat in low_csat.items():
            if csat < 70:
                insights.append(f"{store} has low CSAT ({csat:.1f}), consider reviewing customer tasks.")

    # === Header & Insights ===
    st.title("Task Performance Dashboard")
    st.markdown("#### Insights")
    for ins in insights[:5]:
        st.markdown(f"- {ins}")

    # === Weekly KPIs ===
    total_tasks = len(week_df)
    overdue_tasks = int(week_df['Overdue'].sum())
    on_time = int((week_df['Days Before Due'] >= 0).sum())
    avg_speed = week_df['Days Before Due'].mean()
    ad_hoc = int(week_df['Store'].value_counts()[week_df['Store'].value_counts() == 1].sum())
    st.markdown("### This Week's Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tasks", total_tasks)
    c2.metric("% On Time", f"{on_time/total_tasks:.0%}" if total_tasks else "N/A")
    c3.metric("Avg Days Before Due", f"{avg_speed:.1f}" if not np.isnan(avg_speed) else "N/A")
    c4, c5, c6 = st.columns(3)
    c4.metric("Overdue Tasks", overdue_tasks)
    c5.metric("Ad Hoc Tasks", ad_hoc)
    c6.metric("Completed Early", f"{(week_df['Days Before Due']>0).mean():.0%}")

    # === Regional Summary ===
    st.markdown("### Regional Summary")
    region_summary = week_df.groupby('Region').agg(
        Total_Tasks=('Overdue','size'),
        Overdue_Rate=('Overdue','mean'),
        Avg_Speed=('Days Before Due','mean')
    ).sort_values(by='Overdue_Rate', ascending=False)
    st.dataframe(region_summary.style.format({
        'Overdue_Rate':'{:.0%}',
        'Avg_Speed':'{:.1f}'
    }), use_container_width=True)

    # === Proactive Stores List ===
    st.markdown("### Top Proactive Stores")
    top_proactive = curr_speed.sort_values(ascending=False).head(5)
    for store, days in top_proactive.items():
        st.markdown(f"- {store}: {days:.1f} days early")

    # === Health Score & Alerts ===
    if 'CSAT Score' in week_df.columns:
        st.markdown("### Store Health Scores")
        health = week_df.groupby('Store').agg(
            Overdue_Rate=('Overdue','mean'),
            Speed=('Days Before Due','mean'),
            CSAT=('CSAT Score','mean'),
            Clean=('Cleanliness Score','mean'),
            Sales=('Sales vs Target (%)','mean')
        )
        # Compute composite
        health['Health Score'] = (
            (1 - health['Overdue_Rate'])*0.3 +
            (health['Speed']/health['Speed'].max())*0.2 +
            (health['CSAT']/100)*0.2 +
            (health['Clean']/100)*0.2 +
            ((health['Sales']+15)/35)*0.1
        )*100
        st.dataframe(health.sort_values('Health Score', ascending=False).style.format({
            'Health Score':'{:.1f}',
            'Overdue_Rate':'{:.0%}',
            'Speed':'{:.1f}',
            'CSAT':'{:.1f}',
            'Clean':'{:.1f}',
            'Sales':'{:.1f}%'
        }), use_container_width=True)

        # Alerts: export underperformers
        under = health[health['Health Score'] < 65].reset_index()
        if not under.empty:
            st.markdown("#### Underperforming Stores (Health < 65)")
            st.dataframe(under[['Store','Health Score']].style.format({'Health Score':'{:.1f}'}))
            csv = under.to_csv(index=False).encode('utf-8')
            st.download_button("Export Underperformers CSV", data=csv,
                               file_name="underperforming_stores.csv")
else:
    st.info("Upload the task CSV to get started.")
