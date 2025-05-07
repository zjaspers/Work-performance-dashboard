import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# Page config
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")

# CSS for performance labels
st.markdown("""<style>
    .performance-early { color: #155724; font-weight: bold; }
    .performance-on-time { color: #856404; font-weight: bold; }
    .performance-late { color: #721c24; font-weight: bold; }
    .dataframe tbody tr:hover { background-color: #f1f1f1; }
</style>""", unsafe_allow_html=True)

# Sidebar: uploads and filters
st.sidebar.title("Data & Filters")
task_file = st.sidebar.file_uploader("➕ Upload Task CSV", type="csv")
kpi_file = st.sidebar.file_uploader("📊 Upload Store KPI CSV", type="csv")

# Placeholder for filters until data is loaded
selected_tasks = []
selected_stores = []

if task_file:
    df = pd.read_csv(task_file)
    # Parse dates
    df['End date'] = pd.to_datetime(df.get('End date'), errors='coerce')
    df['Date completed'] = pd.to_datetime(df.get('Date completed'), errors='coerce')
    # Handle missing completions as late
    today = pd.Timestamp.now().normalize()
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    missing = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing, 'Days Before Due'] = -((df.loc[missing, 'End date'] - today).abs().dt.days)
    df['Overdue'] = (df['End date'] < pd.Timestamp.now()) & (df['Task status'] != 'Completed')
    
    # Clean and filter
    df['Region'] = df['Level 1'].fillna('Unknown')
    df['Store'] = df['Location name']
    df = df[~df['Store'].isin(['JameTrade', 'Midwest'])]
    
    # Determine weekly periods
    df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)
    weeks = sorted(df['Week Start'].dropna().unique(), reverse=True)
    week_labels = [f"{w.date()} to {(w + timedelta(days=6)).date()}" for w in weeks]
    week_choice = st.sidebar.selectbox("Select Week", week_labels)
    sel_start = weeks[week_labels.index(week_choice)]
    
    # Filter to selected week
    week_df = df[df['Week Start'] == sel_start]
    
    # Merge KPI data if provided
    if kpi_file:
        kpi_df = pd.read_csv(kpi_file).rename(columns={'Location ID':'Location external ID'})
        week_df = week_df.merge(kpi_df, on=['Location external ID','Store'], how='left')
    
    # Setup filters for tasks and stores
    task_options = sorted(week_df['Task name'].unique())
    store_options = sorted(week_df['Store'].unique())
    selected_tasks = st.sidebar.multiselect("Filter by Task", task_options, default=task_options)
    selected_stores = st.sidebar.multiselect("Filter by Store", store_options, default=store_options)
    
    # Apply filters
    filtered = week_df[
        week_df['Task name'].isin(selected_tasks) &
        week_df['Store'].isin(selected_stores)
    ]
    
    # Dashboard title
    st.title("Task Performance Dashboard")
    st.markdown("### Store Performance Snapshot (Filtered)")
    
    # Aggregate by store
    summary = filtered.groupby('Store').agg(
        Total_Tasks=('Store','size'),
        Overdue_Rate=('Overdue','mean'),
        Avg_Days_Relative=('Days Before Due','mean'),
        CSAT=('CSAT Score','mean'),
        Cleanliness=('Cleanliness Score','mean'),
        Sales_vs_Target=('Sales vs Target (%)','mean')
    ).reset_index()
    
    # Performance category
    def categorize(x):
        return 'Early' if x>0 else ('On Time' if x==0 else 'Late')
    summary['Performance'] = summary['Avg_Days_Relative'].apply(categorize)
    
    # Sort by Overdue_Rate descending
    summary = summary.sort_values('Overdue_Rate', ascending=False)
    
    # Display table
    st.dataframe(
        summary.style.format({
            'Overdue_Rate':'{:.0%}',
            'Avg_Days_Relative':'{:.1f}',
            'CSAT':'{:.1f}',
            'Cleanliness':'{:.1f}',
            'Sales_vs_Target':'{:.1f}%'
        }).applymap(
            lambda v: 'color: #155724;' if v=='Early' else (
                'color: #856404;' if v=='On Time' else 'color: #721c24;'
            ) if v in ['Early','On Time','Late'] else ''
        , subset=['Performance']),
        use_container_width=True
    )
    
    # Keep the existing store detail lookup
    store_query = st.sidebar.text_input("🔍 Store Details")
    if store_query:
        detail = filtered[filtered['Store'].str.contains(store_query, case=False)]
        if not detail.empty:
            st.markdown(f"### Details for {store_query}")
            tcount = len(detail)
            tod = detail['Overdue'].sum()
            avgd = detail['Days Before Due'].mean()
            cs = detail['CSAT Score'].mean() if 'CSAT Score' in detail.columns else None
            st.write(f"- Total Tasks: {tcount}")
            st.write(f"- Overdue Tasks: {tod}")
            st.write(f"- Avg Days Relative to Due: {avgd:.1f}")
            if cs is not None:
                st.write(f"- Avg CSAT Score: {cs:.1f}")
        else:
            st.write(f"No data for store matching '{store_query}'")
else:
    st.info("Upload the Task CSV to begin.")
