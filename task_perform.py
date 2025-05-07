import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# Page configuration
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")
st.markdown(
    '''
    <style>
        .block-container { padding: 2rem; }
        .stMetric { text-align: center !important; }
        .performance-early { color: #155724; font-weight: bold; }
        .performance-on-time { color: #856404; font-weight: bold; }
        .performance-late { color: #721c24; font-weight: bold; }
        .trending { background-color: #e2e3e5; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem; }
    </style>
    ''', unsafe_allow_html=True
)

# Sidebar uploads and parameters
st.sidebar.title("Upload & Query")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv", key="task_csv")
kpi_file = st.sidebar.file_uploader("📊 Store KPI CSV", type="csv", key="kpi_csv")
store_query = st.sidebar.text_input("🔍 Store Details", help="Enter a store name for details")

if task_file:
    df = pd.read_csv(task_file)
    # Parse dates
    df['End date'] = pd.to_datetime(df.get('End date'), errors='coerce')
    df['Date completed'] = pd.to_datetime(df.get('Date completed'), errors='coerce')
    # Handle missing completions
    today = pd.Timestamp.now().normalize()
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    missing = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing, 'Days Before Due'] = -((df.loc[missing, 'End date'] - today).abs().dt.days)
    df['Overdue'] = (df['End date'] < pd.Timestamp.now()) & (df['Task status'] != 'Completed')

    # Clean store list
    df['Region'] = df['Level 1'].fillna('Unknown')
    df['Store'] = df['Location name']
    df = df[~df['Store'].isin(['JameTrade', 'Midwest'])]

    # Determine weekly periods
    df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)
    weeks = sorted(df['Week Start'].dropna().unique(), reverse=True)
    week_labels = [f"{w.date()} to {(w + timedelta(days=6)).date()}" for w in weeks]
    week_choice = st.sidebar.selectbox("Select Week", week_labels)
    sel_start = weeks[week_labels.index(week_choice)]
    week_df = df[df['Week Start'] == sel_start]

    # Merge KPI data if provided
    if kpi_file:
        kpi_df = pd.read_csv(kpi_file).rename(columns={'Location ID': 'Location external ID'})
        week_df = week_df.merge(kpi_df, on=['Location external ID', 'Store'], how='left')

    # === Smart Insights & Trending Stores ===
    insights = []
    trending_stores = []
    prev_start = sel_start - timedelta(weeks=1)
    prev_df = df[df['Week Start'] == prev_start]
    # Region overdue trend
    if not prev_df.empty:
        curr_reg = week_df.groupby('Region')['Overdue'].mean()
        prev_reg = prev_df.groupby('Region')['Overdue'].mean()
        for region in curr_reg.index:
            delta = (curr_reg[region] - prev_reg.get(region, 0)) * 100
            if abs(delta) >= 1:
                trend = "risen" if delta > 0 else "fallen"
                insights.append(f"Region {region} overdue rate has {trend} by {abs(delta):.1f}% since last week.")
    # Store speed trend
    if not prev_df.empty:
        curr_spd = week_df.groupby('Store')['Days Before Due'].mean()
        prev_spd = prev_df.groupby('Store')['Days Before Due'].mean()
        for store in curr_spd.index:
            delta = curr_spd[store] - prev_spd.get(store, np.nan)
            if abs(delta) >= 0.5:
                trend = "slower" if delta < 0 else "faster"
                msg = f"{store} completed tasks {abs(delta):.1f} days {trend} than last week."
                insights.append(msg)
                trending_stores.append(store)

    # Low CSAT insights
    if 'CSAT Score' in week_df.columns:
        for store, csat in week_df.groupby('Store')['CSAT Score'].mean().items():
            if csat < 70:
                insights.append(f"{store} has low CSAT ({csat:.1f}).")

    # === Display Header & Insights ===
    st.title("Task Performance Dashboard")
    if insights:
        st.markdown("#### Insights")
        for ins in insights[:5]:
            st.markdown(f"- {ins}", unsafe_allow_html=True)
    if trending_stores:
        st.markdown("#### Trending Stores")
        for store in trending_stores:
            st.markdown(f"<div class='trending'>{store}</div>", unsafe_allow_html=True)

    # === Weekly KPIs ===
    total = len(week_df)
    on_time = (week_df['Days Before Due'] >= 0).sum()
    avg_speed = week_df['Days Before Due'].mean()
    overdue_count = int(week_df['Overdue'].sum())
    adhoc = (week_df['Store'].value_counts() == 1).sum()
    st.markdown("### This Week's Key Metrics")
    cols = st.columns(6)
    cols[0].metric("Total Tasks", total)
    cols[1].metric("% On Time", f"{on_time/total:.0%}" if total else "N/A")
    cols[2].metric("Avg Early/Late (days)", f"{avg_speed:.1f}")
    cols[3].metric("Overdue Tasks", overdue_count)
    cols[4].metric("Ad Hoc Tasks", adhoc)
    cols[5].metric("Completed Early %", f"{(week_df['Days Before Due']>0).mean():.0%}")

    # === Regional Summary with Performance Category ===
    st.markdown("### Regional Performance")
    region_summary = week_df.groupby('Region').agg(
        Total_Tasks=('Store','size'),
        Overdue_Rate=('Overdue','mean'),
        Days_Relative=('Days Before Due','mean')
    )
    # Categorize performance
    def categorize(x):
        if x > 0: return 'Early'
        if x == 0: return 'On Time'
        return 'Late'
    region_summary['Performance'] = region_summary['Days_Relative'].apply(categorize)
    region_display = region_summary[['Total_Tasks','Overdue_Rate','Performance']]
    st.dataframe(region_display.style.format({'Overdue_Rate':'{:.0%}'}).applymap(
        lambda v: 'color: #155724; font-weight:bold;' if v=='Early' else 
                  ('color: #856404; font-weight:bold;' if v=='On Time' else 
                   'color: #721c24; font-weight:bold;'), subset=['Performance']),
                 use_container_width=True)

    # === Store Detail Lookup ===
    if store_query:
        sd = week_df[week_df['Store'].str.contains(store_query, case=False)]
        if not sd.empty:
            perf = sd['Days Before Due'].mean()
            perf_cat = categorize(perf)
            st.markdown(f"### Details for {store_query}")
            st.write(f"- Performance: **{perf_cat}**")
            st.write(f"- Avg Days Rel. to Due: {perf:.1f}")
            st.write(f"- Total Tasks: {len(sd)}")
            st.write(f"- Overdue: {sd['Overdue'].sum()}")
            if 'CSAT Score' in sd.columns:
                st.write(f"- Avg CSAT: {sd['CSAT Score'].mean():.1f}")
        else:
            st.markdown(f"No data for '{store_query}'.")
else:
    st.info("Upload the task CSV to get started.")

