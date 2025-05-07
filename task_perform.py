
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# Page configuration
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")

# Styling
st.markdown("""<style>
    .block-container { padding: 2rem; font-family: 'Inter', sans-serif; }
    .stMetric { text-align: center !important; }
    .performance-early { color: #155724; font-weight: bold; }
    .performance-on-time { color: #856404; font-weight: bold; }
    .performance-late { color: #721c24; font-weight: bold; }
    .dataframe tbody tr:hover { background-color: #f1f1f1; }
</style>""", unsafe_allow_html=True)

# Sidebar: uploads and filters
st.sidebar.title("Upload & Filters")
task_file = st.sidebar.file_uploader("➕ Upload Task CSV", type="csv")
kpi_file = st.sidebar.file_uploader("📊 Upload Store KPI CSV", type="csv")

if task_file:
    # Load task data
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

    # Apply filters
    filtered = week_df[week_df['Task name'].isin(selected_tasks)]
    if selected_stores:
        filtered = filtered[filtered['Store'].isin(selected_stores)]

    # Key Metrics
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

    # Store Status Summary
    summary_base = filtered.groupby('Store').agg(
        Total_Tasks=('Store','size'),
        Overdue_Rate=('Overdue','mean'),
        Avg_Days_Relative=('Days Before Due','mean')
    )
    if 'CSAT Score' in filtered.columns:
        summary_base['CSAT'] = filtered.groupby('Store')['CSAT Score'].mean()
    if 'Cleanliness Score' in filtered.columns:
        summary_base['Cleanliness'] = filtered.groupby('Store')['Cleanliness Score'].mean()
    if 'Sales vs Target (%)' in filtered.columns:
        summary_base['Sales_vs_Target'] = filtered.groupby('Store')['Sales vs Target (%)'].mean()

    summary = summary_base.reset_index()
    def categorize(x):
        return 'Early' if x > 0 else ('On Time' if x == 0 else 'Late')
    summary['Performance'] = summary['Avg_Days_Relative'].apply(categorize)

    # Count by category
    counts = summary['Performance'].value_counts().to_dict()
    st.markdown("### Store Status Summary")
    scols = st.columns(3)
    scols[0].metric("Stores Early", counts.get('Early', 0))
    scols[1].metric("Stores On Time", counts.get('On Time', 0))
    scols[2].metric("Stores Late", counts.get('Late', 0))

    # Store Performance Snapshot Table
    display_cols = ['Store','Total_Tasks','Overdue_Rate','Avg_Days_Relative','Performance']
    if 'CSAT' in summary.columns:
        display_cols.append('CSAT')
    if 'Cleanliness' in summary.columns:
        display_cols.append('Cleanliness')
    if 'Sales_vs_Target' in summary.columns:
        display_cols.append('Sales_vs_Target')

    display_df = summary[display_cols]
    styled = display_df.style.format({
        'Overdue_Rate':'{:.0%}',
        'Avg_Days_Relative':'{:.1f}',
        'CSAT':'{:.1f}',
        'Cleanliness':'{:.1f}',
        'Sales_vs_Target':'{:.1f}%'
    }).applymap(
        lambda v: 'color: #155724;' if v=='Early' else (
            'color: #856404;' if v=='On Time' else 'color: #721c24;'
        ), subset=['Performance']
    )
    st.dataframe(styled, use_container_width=True)

    # Store Detail Lookup
    store_query = st.sidebar.text_input("🔍 Store Details")
    if store_query:
        detail = filtered[filtered['Store'].str.contains(store_query, case=False)]
        if not detail.empty:
            avg = detail['Days Before Due'].mean()
            perf = categorize(avg)
            st.markdown(f"### Details for {store_query}")
            st.write(f"- Performance: **{perf}**")
            st.write(f"- Avg Days Relative to Due: {avg:.1f}")
            st.write(f"- Total Tasks: {len(detail)}")
            st.write(f"- Overdue Tasks: {detail['Overdue'].sum()}")
            if 'CSAT Score' in detail.columns:
                st.write(f"- Avg CSAT Score: {detail['CSAT Score'].mean():.1f}")
        else:
            st.write(f"No data for '{store_query}'.")
else:
    st.info("Upload Task CSV to begin.")
