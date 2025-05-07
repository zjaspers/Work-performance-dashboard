# Here's the full updated Streamlit app code with a cleaned-up "Smart, Non‑Obvious Insights" section 
# and a new "Task Effort & Performance by Task" table.

streamlit_code = '''
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# Page configuration
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")

# Styling
st.markdown("""
<style>
    .block-container { padding: 2rem; font-family: 'Inter', sans-serif; }
    .stMetric { text-align: center !important; }
    .performance-early { color: #155724; font-weight: bold; }
    .performance-on-time { color: #856404; font-weight: bold; }
    .performance-late { color: #721c24; font-weight: bold; }
    .insight-item { padding: 0.5rem; background-color: #f7f7f7; border-radius: 4px; margin-bottom: 0.5rem; }
    .dataframe tbody tr:hover { background-color: #f1f1f1; }
    </style>
""", unsafe_allow_html=True)

# Sidebar: uploads and filters
st.sidebar.title("Upload & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file = st.sidebar.file_uploader("📊 Store KPI CSV (optional)", type="csv")

if not task_file:
    st.info("Upload a Task CSV to begin.")
else:
    # Load and preprocess
    df = pd.read_csv(task_file)
    df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
    df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')
    today = pd.Timestamp.now().normalize()
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    missing = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing, 'Days Before Due'] = -((df.loc[missing, 'End date'] - today).abs().dt.days)
    df['Overdue'] = (df['End date'] < pd.Timestamp.now()) & (df['Task status'] != 'Completed')
    df['Region'] = df['Level 1'].fillna('Unknown')
    df['Store'] = df['Location name']
    df = df[~df['Store'].isin(['JameTrade','Midwest'])]
    df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)

    # Week selector
    weeks = sorted(df['Week Start'].dropna().unique(), reverse=True)
    labels = [f"{w.date()} to {(w + timedelta(days=6)).date()}" for w in weeks]
    choice = st.sidebar.selectbox("Select Week", labels)
    start = weeks[labels.index(choice)]
    week_df = df[df['Week Start']==start]

    # Merge KPI data if provided
    if kpi_file:
        kpi_df = pd.read_csv(kpi_file).rename(columns={'Location ID':'Location external ID'})
        week_df = week_df.merge(kpi_df, on=['Location external ID','Store'], how='left')

    # Filters
    tasks = sorted(week_df['Task name'].unique())
    stores = sorted(week_df['Store'].unique())
    sel_tasks = st.sidebar.multiselect("Filter by Task", tasks, default=tasks)
    sel_stores = st.sidebar.multiselect("Filter by Store", stores, default=[])

    filtered = week_df[week_df['Task name'].isin(sel_tasks)]
    if sel_stores:
        filtered = filtered[filtered['Store'].isin(sel_stores)]

    # Key Metrics
    total = len(filtered)
    on_time = (filtered['Days Before Due']>=0).sum()
    avg_speed = filtered['Days Before Due'].mean()
    overdue = int(filtered['Overdue'].sum())
    adhoc = (filtered['Store'].value_counts()==1).sum()
    avg_csat = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered else None

    st.title("Task Performance Dashboard")
    st.markdown("### Key Metrics")
    cols = st.columns(6)
    cols[0].metric("Total Tasks", total)
    cols[1].metric("% On Time", f"{on_time/total:.0%}" if total else "N/A")
    cols[2].metric("Avg Early/Late (days)", f"{avg_speed:.1f}" if not np.isnan(avg_speed) else "N/A")
    cols[3].metric("Overdue Tasks", overdue)
    cols[4].metric("Ad Hoc Tasks", adhoc)
    if avg_csat is not None:
        cols[5].metric("Avg CSAT", f"{avg_csat:.1f}")

    # Store Status Summary
    sb = filtered.groupby('Store').agg(
        Total_Tasks=('Store','size'),
        Overdue_Rate=('Overdue','mean'),
        Avg_Days=('Days Before Due','mean')
    )
    if 'CSAT Score' in filtered: sb['CSAT'] = filtered.groupby('Store')['CSAT Score'].mean()
    if 'Cleanliness Score' in filtered: sb['Cleanliness'] = filtered.groupby('Store')['Cleanliness Score'].mean()
    if 'Sales vs Target (%)' in filtered: sb['Sales_vs_Target'] = filtered.groupby('Store')['Sales vs Target (%)'].mean()

    summary = sb.reset_index()
    summary['Performance'] = summary['Avg_Days'].apply(lambda x: 'Early' if x>0 else ('On Time' if x==0 else 'Late'))
    counts = summary['Performance'].value_counts().to_dict()

    st.markdown("### Store Status Summary")
    s1,s2,s3 = st.columns(3)
    s1.metric("Stores Early", counts.get('Early',0))
    s2.metric("Stores On Time", counts.get('On Time',0))
    s3.metric("Stores Late", counts.get('Late',0))

    # Smart, Non-Obvious Insights as clean bullet list
    st.markdown("### Smart, Non-Obvious Insights")
    insights = []
    # Effort vs Results
    if 'Expected duration' in filtered and 'CSAT Score' in filtered:
        effort = filtered.groupby('Store')['Expected duration'].sum()
        csat = filtered.groupby('Store')['CSAT Score'].mean()
        high_eff = effort[effort > effort.quantile(0.8)].index.tolist()
        insights.append(f"**Effort vs Results:** Stores {', '.join(high_eff)} are in the top 20% of effort but not highest CSAT.")
    # Task Saturation
    sat = filtered.set_index('End date').groupby('Store')['Store'].rolling('3D').count()
    high_sat = sat[sat > 10].reset_index()['Store'].unique().tolist()
    insights.append(f"**Task Saturation:** {', '.join(high_sat)} had >10 tasks in any 3-day span.")
    # Planned vs Reactive
    task_counts = filtered.groupby(['Store','Task name']).size().reset_index(name='Count')
    reactive = task_counts[task_counts['Count']==1].groupby('Store')['Count'].count()
    planned = task_counts[task_counts['Count']>1].groupby('Store')['Count'].count().add(0)
    unbal = reactive[reactive > planned].index.tolist()
    insights.append(f"**Planned vs Reactive:** {', '.join(unbal)} have more reactive than planned tasks.")
    # Focus Drift
    categories = filtered.groupby('Store')['Task category'].nunique()
    drift = categories[categories > 5].index.tolist()
    insights.append(f"**Store Focus Drift:** {', '.join(drift)} handle >5 task categories.")

    for itm in insights:
        st.markdown(f"- {itm}")

    # Task Effort & Performance by Task
    st.markdown("### Task Effort & Performance by Task")
    task_summary = filtered.groupby('Task name').agg(
        Task_Count=('Task name','size'),
        Total_Effort=('Expected duration','sum'),
        Overdue_Rate=('Overdue','mean'),
        Avg_Days=('Days Before Due','mean')
    ).reset_index().sort_values('Task_Count', ascending=False)
    task_summary['Overdue_Rate'] = task_summary['Overdue_Rate'].apply(lambda x: f"{x:.0%}")
    task_summary['Avg_Days'] = task_summary['Avg_Days'].round(1)
    st.dataframe(task_summary, use_container_width=True)

    # Spacer before table
    st.markdown("&nbsp;")

    # Store Performance Snapshot Table
    st.markdown("### Store Performance Snapshot")
    cols = ['Store','Total_Tasks','Overdue_Rate','Avg_Days','Performance']
    for k in ['CSAT','Cleanliness','Sales_vs_Target']:
        if k in summary: cols.append(k)
    df_display = summary[cols].rename(columns={'Avg_Days':'Avg Days Relative'})
    st.dataframe(df_display, use_container_width=True)

    # Store Detail Lookup
    query = st.sidebar.text_input("🔍 Store Details")
    if query:
        try:
            d = filtered[filtered['Store'].str.contains(query, case=False)]
            if d.empty:
                st.warning(f"No data found for '{query}'")
            else:
                a = d['Days Before Due'].mean(); p = 'Early' if a>0 else ('On Time' if a==0 else 'Late')
                st.markdown(f"### Details for {query}")
                st.write(f"- Performance: **{p}**")
                st.write(f"- Avg Days Relative: {a:.1f}")
                st.write(f"- Total Tasks: {len(d)}")
                st.write(f"- Overdue Tasks: {int(d['Overdue'].sum())}")
                if 'CSAT Score' in d:
                    st.write(f"- Avg CSAT: {d['CSAT Score'].mean():.1f}")
        except Exception as e:
            st.error(f"Error loading details: {e}")
'''

# Display the code for the user to copy
import textwrap
print(textwrap.dedent(streamlit_code))
