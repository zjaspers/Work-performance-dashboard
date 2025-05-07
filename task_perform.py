import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# Page configuration
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")

# Cal.com AI inspired styling
st.markdown("""<style>
    .block-container { padding: 2rem; font-family: 'Inter', sans-serif; }
    .stMetric { text-align: center !important; }
    .performance-early { color: #155724; font-weight: bold; }
    .performance-on-time { color: #856404; font-weight: bold; }
    .performance-late { color: #721c24; font-weight: bold; }
    .insight { padding: 0.5rem; margin-bottom: 0.25rem; background-color: #f7f7f7; border-radius: 5px; }
    .streamlit-expanderHeader { font-weight: bold; }
</style>""", unsafe_allow_html=True)

# Sidebar: uploads and filters
st.sidebar.title("Upload & Filters")
task_file = st.sidebar.file_uploader("➕ Upload Task CSV", type="csv")
kpi_file = st.sidebar.file_uploader("📊 Upload Store KPI CSV", type="csv")

if task_file:
    # Load and preprocess
    df = pd.read_csv(task_file)
    df['End date'] = pd.to_datetime(df.get('End date'), errors='coerce')
    df['Date completed'] = pd.to_datetime(df.get('Date completed'), errors='coerce')
    today = pd.Timestamp.now().normalize()
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    missing = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing, 'Days Before Due'] = -((df.loc[missing, 'End date'] - today).abs().dt.days)
    df['Overdue'] = (df['End date'] < pd.Timestamp.now()) & (df['Task status'] != 'Completed')
    df['Region'] = df['Level 1'].fillna('Unknown')
    df['Store'] = df['Location name']
    df = df[~df['Store'].isin(['JameTrade', 'Midwest'])]
    df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)
    
    # Week selector
    weeks = sorted(df['Week Start'].dropna().unique(), reverse=True)
    week_labels = [f"{w.date()} to {(w + timedelta(days=6)).date()}" for w in weeks]
    week_choice = st.sidebar.selectbox("Select Week", week_labels)
    sel_start = weeks[week_labels.index(week_choice)]
    week_df = df[df['Week Start'] == sel_start]
    
    # Merge KPI data if provided
    if kpi_file:
        kpi_df = pd.read_csv(kpi_file).rename(columns={'Location ID': 'Location external ID'})
        week_df = week_df.merge(kpi_df, on=['Location external ID', 'Store'], how='left')
    
    # Filters
    task_options = sorted(week_df['Task name'].unique())
    selected_tasks = st.sidebar.multiselect("Filter by Task", task_options, default=task_options)
    store_options = sorted(week_df['Store'].unique())
    selected_stores = st.sidebar.multiselect("Filter by Store", store_options, default=[])
    
    # Apply filters: tasks mandatory, stores optional
    filtered = week_df[week_df['Task name'].isin(selected_tasks)]
    if selected_stores:
        filtered = filtered[filtered['Store'].isin(selected_stores)]
    
    # Top Metrics
    total = len(filtered)
    on_time = (filtered['Days Before Due'] >= 0).sum()
    avg_speed = filtered['Days Before Due'].mean()
    overdue_count = int(filtered['Overdue'].sum())
    adhoc = (filtered['Store'].value_counts() == 1).sum()
    avg_csat = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered.columns else None
    
    st.title("Task Performance Dashboard")
    st.markdown("### Key Metrics")
    cols = st.columns(6)
    cols[0].metric("Total Tasks", total)
    cols[1].metric("% On Time", f"{on_time/total:.0%}" if total else "N/A")
    cols[2].metric("Avg Days Early/Late", f"{avg_speed:.1f}" if not np.isnan(avg_speed) else "N/A")
    cols[3].metric("Overdue Tasks", overdue_count)
    cols[4].metric("Ad Hoc Tasks", adhoc)
    if avg_csat is not None:
        cols[5].metric("Avg CSAT", f"{avg_csat:.1f}")
    
    # Smart Insights
    insights = []
    prev_start = sel_start - timedelta(weeks=1)
    prev_df = df[df['Week Start'] == prev_start]
    # Regional overdue change
    if not prev_df.empty:
        curr = week_df.groupby('Region')['Overdue'].mean()
        prev = prev_df.groupby('Region')['Overdue'].mean()
        for region in curr.index:
            delta = (curr[region] - prev.get(region, 0)) * 100
            if abs(delta) >= 1:
                trend = "risen" if delta > 0 else "fallen"
                insights.append(f"Region {region} overdue rate has {trend} by {abs(delta):.1f}% from last week.")
    # Store speed change
    if not prev_df.empty:
        curr_spd = week_df.groupby('Store')['Days Before Due'].mean()
        prev_spd = prev_df.groupby('Store')['Days Before Due'].mean()
        for store in curr_spd.index:
            delta = curr_spd[store] - prev_spd.get(store, np.nan)
            if abs(delta) >= 0.5:
                trend = "slower" if delta < 0 else "faster"
                insights.append(f"{store} completed tasks {abs(delta):.1f} days {trend}.")
    # Low CSAT
    if 'CSAT Score' in week_df.columns:
        for store, csat in week_df.groupby('Store')['CSAT Score'].mean().items():
            if csat < 70:
                insights.append(f"{store} has low CSAT ({csat:.1f}).")
    
    if insights:
        st.markdown("### Insights")
        for ins in insights:
            st.markdown(f"<div class='insight'>{ins}</div>", unsafe_allow_html=True)
    
    # Store Performance Snapshot Table
    st.markdown("### Store Performance Snapshot")
    summary = filtered.groupby('Store').agg(
        Total_Tasks=('Store','size'),
        Overdue_Rate=('Overdue','mean'),
        Avg_Days_Relative=('Days Before Due','mean'),
        CSAT=('CSAT Score','mean'),
        Cleanliness=('Cleanliness Score','mean'),
        Sales_vs_Target=('Sales vs Target (%)','mean')
    ).reset_index().sort_values('Overdue_Rate', ascending=False)
    def categorize(x):
        return 'Early' if x > 0 else ('On Time' if x == 0 else 'Late')
    summary['Performance'] = summary['Avg_Days_Relative'].apply(categorize)
    styled = summary.style.format({
        'Overdue_Rate':'{:.0%}',
        'Avg_Days_Relative':'{:.1f}',
        'CSAT':'{:.1f}',
        'Cleanliness':'{:.1f}',
        'Sales_vs_Target':'{:.1f}%'
    }).applymap(
        lambda v: 'color: #155724;' if v=='Early' else (
            'color: #856404;' if v=='On Time' else 'color: #721c24;'
        ) if v in ['Early','On Time','Late'] else ''
    , subset=['Performance'])
    st.dataframe(styled, use_container_width=True)

    # Store detail lookup
    store_query = st.sidebar.text_input("🔍 Store Details")
    if store_query:
        detail = filtered[filtered['Store'].str.contains(store_query, case=False)]
        if not detail.empty:
            avg = detail['Days Before Due'].mean()
            perf = categorize(avg)
            st.markdown(f"### Details for {store_query}")
            st.write(f"- Performance: **{perf}**")
            st.write(f"- Avg Days Rel. to Due: {avg:.1f}")
            st.write(f"- Total Tasks: {len(detail)}")
            st.write(f"- Overdue Tasks: {detail['Overdue'].sum()}")
            if 'CSAT Score' in detail.columns:
                st.write(f"- Avg CSAT: {detail['CSAT Score'].mean():.1f}")
        else:
            st.write(f"No data for '{store_query}'.")
else:
    st.info("Upload Task CSV to begin.")
