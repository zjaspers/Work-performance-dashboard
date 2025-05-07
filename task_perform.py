import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(page_title="Task Performance Dashboard", layout="wide")

# ─── Styling ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding: 2rem; font-family: 'Inter', sans-serif; }
  .metric-card {
    background: white; border-radius: 8px; padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
    margin-bottom: 1rem;
  }
  .metric-label { font-size: 0.9rem; color: #555; }
  .metric-value { font-size: 2rem; font-weight: bold; }
  .insight-item {
    padding: 0.5rem; background-color: #f7f7f7;
    border-radius: 4px; margin-bottom: 0.5rem;
  }
  .dataframe tbody tr:hover { background-color: #f1f1f1; }
</style>
""", unsafe_allow_html=True)

def metric_card(label, value):
    st.markdown(f"""
      <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
      </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_csv(fpath):
    return pd.read_csv(fpath)

def preprocess(df):
    df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
    df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')
    today = pd.Timestamp.now().normalize()
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    missing = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing, 'Days Before Due'] = -((df.loc[missing,'End date'] - today).dt.days)
    df['Overdue'] = (df['Days Before Due'] < 0)
    df['Region'] = df['Level 1'].fillna('Unknown')
    df['Store'] = df['Location name']
    df = df[~df['Store'].isin(['JameTrade','Midwest'])]
    df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)
    return df

# ─── Sidebar ─────────────────────────────────────────────────────────
st.sidebar.header("Upload & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file  = st.sidebar.file_uploader("📊 Store KPI CSV (optional)", type="csv")

if not task_file:
    st.info("Please upload the Task CSV to proceed.")
    st.stop()

# ─── Load & Prepare ──────────────────────────────────────────────────
df = load_csv(task_file)
df = preprocess(df)
if kpi_file:
    kpi = load_csv(kpi_file).rename(columns={'Location ID': 'Location external ID'})
    df = df.merge(kpi, on=['Location external ID','Store'], how='left')

# ─── Week Selector ───────────────────────────────────────────────────
weeks = sorted(df['Week Start'].dropna().unique(), reverse=True)
labels = [f"{w.date()}–{(w+timedelta(days=6)).date()}" for w in weeks]
choice = st.sidebar.selectbox("Select Week", labels)
start  = weeks[labels.index(choice)]
week_df = df[df['Week Start'] == start]

# ─── Filters ─────────────────────────────────────────────────────────
tasks  = sorted(week_df['Task name'].unique())
stores = sorted(week_df['Store'].unique())
sel_tasks  = st.sidebar.multiselect("Filter by Task", tasks, default=tasks)
sel_stores = st.sidebar.multiselect("Filter by Store", stores)

filtered = week_df[week_df['Task name'].isin(sel_tasks)]
if sel_stores:
    filtered = filtered[filtered['Store'].isin(sel_stores)]

# ─── Key Metrics ─────────────────────────────────────────────────────
st.title("Task Performance Dashboard")
st.markdown("### Key Metrics")

# Total unique tasks by Task ID
total_tasks     = filtered['Task ID'].nunique()
# On-time tasks: Task ID groups where max Days Before Due ≥ 0
on_time_tasks   = filtered.groupby('Task ID')['Days Before Due'].max().ge(0).sum()
avg_days_before = filtered.groupby('Task ID')['Days Before Due'].mean().mean().round(1)
overdue_tasks   = total_tasks - on_time_tasks
# Ad hoc tasks: Task IDs assigned to exactly one store
ad_hoc_tasks    = filtered.groupby('Task ID')['Store'].nunique().eq(1).sum()
avg_csat        = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered.columns else None

c1, c2, c3, c4, c5, c6 = st.columns(6)
metric_card("Total Tasks", total_tasks)
metric_card("% On Time", f"{on_time_tasks/total_tasks:.0%}")
metric_card("Avg Days Before Due", avg_days_before)
metric_card("Overdue Tasks", overdue_tasks)
metric_card("Ad Hoc Tasks", ad_hoc_tasks)
if avg_csat is not None:
    metric_card("Avg CSAT", f"{avg_csat:.1f}")

# ─── Store Status Summary ───────────────────────────────────────────
st.markdown("### Store Status Summary")
sb = filtered.groupby('Store').agg(
    Total_Tasks     = ('Task ID','nunique'),
    Overdue_Rate    = ('Overdue','mean'),
    Avg_Days_Rel    = ('Days Before Due','mean')
)
# merge KPIs if present
if 'CSAT Score' in filtered: sb['CSAT'] = filtered.groupby('Store')['CSAT Score'].mean()
if 'Sales vs Target (%)' in filtered: sb['Sales'] = filtered.groupby('Store')['Sales vs Target (%)'].mean()

sb = sb.reset_index()
sb['Performance'] = sb['Avg_Days_Rel'].apply(lambda x: 'Early' if x>0 else ('On Time' if x==0 else 'Late'))

# Barplot
plt.figure(figsize=(8,4))
sns.barplot(
    data=sb, x='Overdue_Rate', y='Store', hue='Performance', dodge=False,
    palette={'Early':'green','On Time':'gold','Late':'red'}
)
plt.xlabel("Overdue Rate")
st.pyplot(plt.gcf())

st.dataframe(
    sb.style.format({
        'Overdue_Rate':'{:.0%}',
        'Avg_Days_Rel':'{:.1f}',
        'CSAT':'{:.1f}',
        'Sales':'{:.1f}%'
    }), use_container_width=True
)
