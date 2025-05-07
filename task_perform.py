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
    .insight-item {
        padding: 0.5rem;
        background-color: #f7f7f7;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .dataframe tbody tr:hover { background-color: #f1f1f1; }
</style>
""", unsafe_allow_html=True)

# Sidebar: file uploads and filters
st.sidebar.title("Upload & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file  = st.sidebar.file_uploader("📊 Store KPI CSV (optional)", type="csv")

if not task_file:
    st.info("Upload a Task CSV to begin.")
    st.stop()

# Load and preprocess task data
df = pd.read_csv(task_file)
df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')
today = pd.Timestamp.now().normalize()

# Compute Days Before Due and Overdue
df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
missing = df['Date completed'].isna() & df['End date'].notna()
df.loc[missing, 'Days Before Due'] = -((df.loc[missing, 'End date'] - today).abs().dt.days)
df['Overdue'] = (df['End date'] < pd.Timestamp.now()) & (df['Task status'] != 'Completed')

# Clean hierarchy and filter out company/region placeholders
df['Region'] = df['Level 1'].fillna('Unknown')
df['Store'] = df['Location name']
df = df[~df['Store'].isin(['JameTrade','Midwest'])]

# Assign each task to its week
df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)

# Week selector
weeks  = sorted(df['Week Start'].dropna().unique(), reverse=True)
labels = [f"{w.date()} to {(w + timedelta(days=6)).date()}" for w in weeks]
choice = st.sidebar.selectbox("Select Week", labels)
start  = weeks[labels.index(choice)]
week_df = df[df['Week Start'] == start]

# Merge in KPI data if uploaded
if kpi_file:
    kpi_df = pd.read_csv(kpi_file).rename(columns={'Location ID':'Location external ID'})
    week_df = week_df.merge(kpi_df, on=['Location external ID','Store'], how='left')

# Filters: by Task and optionally by Store
task_list  = sorted(week_df['Task name'].unique())
store_list = sorted(week_df['Store'].unique())
sel_tasks  = st.sidebar.multiselect("Filter by Task", task_list, default=task_list)
sel_stores = st.sidebar.multiselect("Filter by Store", store_list, default=[])

filtered = week_df[week_df['Task name'].isin(sel_tasks)]
if sel_stores:
    filtered = filtered[filtered['Store'].isin(sel_stores)]

# --- Key Metrics ---
total   = len(filtered)
on_time = (filtered['Days Before Due'] >= 0).sum()
avg_spd = filtered['Days Before Due'].mean()
overdue  = int(filtered['Overdue'].sum())
adhoc    = (filtered['Store'].value_counts() == 1).sum()
avg_csat = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered.columns else None

st.title("Task Performance Dashboard")
st.markdown("### Key Metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Tasks", total)
c2.metric("% On Time", f"{on_time/total:.0%}" if total else "N/A")
c3.metric("Avg Early/Late (days)", f"{avg_spd:.1f}" if not np.isnan(avg_spd) else "N/A")
c4.metric("Overdue Tasks", overdue)
c5.metric("Ad Hoc Tasks", adhoc)
if avg_csat is not None:
    c6.metric("Avg CSAT", f"{avg_csat:.1f}")

# --- Store Status Summary ---
sb = filtered.groupby('Store').agg(
    Total_Tasks=('Store','size'),
    Overdue_Rate=('Overdue','mean'),
    Avg_Days     =('Days Before Due','mean')
)
if 'CSAT Score' in filtered.columns:
    sb['CSAT'] = filtered.groupby('Store')['CSAT Score'].mean()
if 'Cleanliness Score' in filtered.columns:
    sb['Cleanliness'] = filtered.groupby('Store')['Cleanliness Score'].mean()
if 'Sales vs Target (%)' in filtered.columns:
    sb['Sales_vs_Target'] = filtered.groupby('Store')['Sales vs Target (%)'].mean()

summary = sb.reset_index()
summary['Performance'] = summary['Avg_Days'].apply(lambda x: 'Early' if x>0 else ('On Time' if x==0 else 'Late'))
counts = summary['Performance'].value_counts().to_dict()

st.markdown("### Store Status Summary")
s1, s2, s3 = st.columns(3)
s1.metric("Stores Early", counts.get('Early',0))
s2.metric("Stores On Time", counts.get('On Time',0))
s3.metric("Stores Late", counts.get('Late',0))

# --- Smart, Non-Obvious Insights ---
st.markdown("### Smart, Non-Obvious Insights")
insights = []

# Effort vs Results
if 'Expected duration' in filtered.columns and 'CSAT Score' in filtered.columns:
    effort = filtered.groupby('Store')['Expected duration'].sum()
    csat   = filtered.groupby('Store')['CSAT Score'].mean()
    high_eff = effort[effort > effort.quantile(0.8)].index.tolist()
    insights.append(f"**Effort vs Results:** Stores {', '.join(high_eff)} in top 20% effort.")

# Task Saturation
sat = (
    filtered.set_index('End date')
            .groupby('Store')['Store']
            .rolling('3D')
            .count()
            .reset_index(name='cnt')
)
high_sat = sat[sat['cnt'] > 10]['Store'].unique().tolist()
insights.append(f"**Task Saturation:** {', '.join(high_sat)} had >10 tasks in any 3-day span.")

# Planned vs Reactive
tc      = filtered.groupby(['Store','Task name']).size().reset_index(name='Count')
react   = tc[tc['Count']==1].groupby('Store')['Count'].count()
plan    = tc[tc['Count']>1].groupby('Store')['Count'].count().add(0)
unbal   = react[react > plan].index.tolist()
insights.append(f"**Planned vs Reactive:** {', '.join(unbal)} more reactive than planned tasks.")

# Focus Drift
cats  = filtered.groupby('Store')['Task category'].nunique()
drift = cats[cats > 5].index.tolist()
insights.append(f"**Focus Drift:** {', '.join(drift)} handle >5 task categories.")

for item in insights:
    st.markdown(f"- {item}", unsafe_allow_html=True)

# --- Task Effort & Performance by Task ---
st.markdown("### Task Effort & Performance by Task")
task_summary = (
    filtered.groupby('Task name')
            .agg(
                Task_Count   = ('Task name','size'),
                Total_Effort = ('Expected duration','sum'),
                Overdue_Rate = ('Overdue','mean'),
                Avg_Days     = ('Days Before Due','mean')
            )
            .reset_index()
            .sort_values('Task_Count', ascending=False)
)
task_summary['Overdue_Rate'] = task_summary['Overdue_Rate'].apply(lambda x: f"{x:.0%}")
task_summary['Avg_Days']     = task_summary['Avg_Days'].round(1)
st.dataframe(task_summary, use_container_width=True)

st.markdown("")  # spacer

# --- Store Performance Snapshot Table ---
st.markdown("### Store Performance Snapshot")
cols = ['Store','Total_Tasks','Overdue_Rate','Avg_Days','Performance']
for k in ['CSAT','Cleanliness','Sales_vs_Target']:
    if k in summary.columns:
        cols.append(k)
df_display = summary[cols].rename(columns={'Avg_Days':'Avg Days Relative'})
st.dataframe(df_display, use_container_width=True)

# --- Store Detail Lookup ---
query = st.sidebar.text_input("🔍 Store Details")
if query:
    try:
        detail = filtered[filtered['Store'].str.contains(query, case=False)]
        if detail.empty:
            st.warning(f"No data found for '{query}'")
        else:
            avg  = detail['Days Before Due'].mean()
            perf = 'Early' if avg>0 else ('On Time' if avg==0 else 'Late')
            st.markdown(f"### Details for {query}")
            st.write(f"- Performance: **{perf}**")
            st.write(f"- Avg Days Relative: {avg:.1f}")
            st.write(f"- Total Tasks: {len(detail)}")
            st.write(f"- Overdue Tasks: {int(detail['Overdue'].sum())}")
            if 'CSAT Score' in detail.columns:
                st.write(f"- Avg CSAT: {detail['CSAT Score'].mean():.1f}")
    except Exception as e:
        st.error(f"Error loading store details: {e}")
