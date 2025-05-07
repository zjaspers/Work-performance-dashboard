import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Task Performance Dashboard",
    layout="wide",
)

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

# ─── Utility Functions ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(f):
    return pd.read_csv(f)

def preprocess_tasks(df):
    df['End date']       = pd.to_datetime(df['End date'], errors='coerce')
    df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')
    today = pd.Timestamp.now().normalize()
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    missing = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing, 'Days Before Due'] = -((df.loc[missing,'End date'] - today).dt.days)
    df['Overdue'] = (df['End date'] < pd.Timestamp.now()) & (df['Task status'] != 'Completed')
    df['Region'] = df['Level 1'].fillna('Unknown')
    df['Store']  = df['Location name']
    df = df[~df['Store'].isin(['JameTrade','Midwest'])]
    df['Week Start'] = df['End date'].dt.to_period('W').apply(lambda r: r.start_time)
    return df

def metric_card(label, value):
    st.markdown(f"""
      <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
      </div>
    """, unsafe_allow_html=True)

# ─── Sidebar: Upload & Filters ────────────────────────────────────────
st.sidebar.header("Data & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file  = st.sidebar.file_uploader("📊 Store KPI CSV (optional)", type="csv")

if not task_file:
    st.sidebar.info("Please upload your Task CSV.")
    st.stop()

# ─── Load & Prepare Data ─────────────────────────────────────────────
df = load_csv(task_file)
df = preprocess_tasks(df)
if kpi_file:
    kpi = load_csv(kpi_file).rename(columns={'Location ID':'Location external ID'})
    df = df.merge(kpi, on=['Location external ID','Store'], how='left')

# Week selector
weeks  = sorted(df['Week Start'].dropna().unique(), reverse=True)
labels = [f"{w.date()}–{(w+timedelta(days=6)).date()}" for w in weeks]
sel    = st.sidebar.selectbox("Select Week", labels)
start  = weeks[labels.index(sel)]
week_df = df[df['Week Start'] == start]

# Task & Store filters
task_list  = sorted(week_df['Task name'].unique())
store_list = sorted(week_df['Store'].unique())
sel_tasks  = st.sidebar.multiselect("Filter by Task",  task_list,  default=task_list)
sel_stores = st.sidebar.multiselect("Filter by Store", store_list)

filtered = week_df[week_df['Task name'].isin(sel_tasks)]
if sel_stores:
    filtered = filtered[filtered['Store'].isin(sel_stores)]

# ─── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Key Metrics", 
    "🏬 Store Performance", 
    "🛠 Task Analysis", 
    "💡 Recommendations"
])

# ─── Tab 1: Key Metrics ──────────────────────────────────────────────
with tab1:
    st.header("Key Metrics")
    total    = len(filtered)
    on_time  = (filtered['Days Before Due'] >= 0).sum()
    avg_spd  = filtered['Days Before Due'].mean()
    overdue  = int(filtered['Overdue'].sum())
    adhoc    = (filtered['Store'].value_counts() == 1).sum()
    avg_csat = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered else None

    cols = st.columns(6)
    with cols[0]: metric_card("Total Tasks",           total)
    with cols[1]: metric_card("% On Time",            f"{on_time/total:.0%}" if total else "N/A")
    with cols[2]: metric_card("Avg Early/Late (days)", f"{avg_spd:.1f}" if not np.isnan(avg_spd) else "N/A")
    with cols[3]: metric_card("Overdue Tasks",        overdue)
    with cols[4]: metric_card("Ad Hoc Tasks",         adhoc)
    if avg_csat is not None:
        with cols[5]: metric_card("Avg CSAT",         f"{avg_csat:.1f}")

# ─── Tab 2: Store Performance ────────────────────────────────────────
with tab2:
    st.header("Store Performance")
    sb = filtered.groupby('Store').agg(
        Total_Tasks=('Store','size'),
        Overdue_Rate=('Overdue','mean'),
        Avg_Days=('Days Before Due','mean')
    )
    if 'CSAT Score' in filtered.columns:
        sb['CSAT'] = filtered.groupby('Store')['CSAT Score'].mean()
    if 'Sales vs Target (%)' in filtered.columns:
        sb['Sales'] = filtered.groupby('Store')['Sales vs Target (%)'].mean()

    summary = sb.reset_index()
    summary['Performance'] = summary['Avg_Days'].apply(
        lambda x: 'Early' if x>0 else ('On Time' if x==0 else 'Late')
    )

    # Bar chart of Overdue Rate
    plt.figure(figsize=(8,4))
    sns.barplot(
        data=summary, x='Overdue_Rate', y='Store',
        hue='Performance', dodge=False,
        palette={'Early':'green','On Time':'gold','Late':'red'}
    )
    plt.xlabel("Overdue Rate")
    st.pyplot(plt.gcf())

    st.dataframe(
        summary.style.format({
            'Overdue_Rate':'{:.0%}',
            'Avg_Days':'{:.1f}',
            'CSAT':'{:.1f}',
            'Sales':'{:.1f}%'
        }),
        use_container_width=True
    )

# ─── Tab 3: Task Analysis ────────────────────────────────────────────
with tab3:
    st.header("Task Effort & Performance by Task")
    ta = filtered.groupby('Task name').agg(
        Count=('Task name','size'),
        Effort=('Expected duration','sum'),
        Overdue=('Overdue','mean'),
        Speed=('Days Before Due','mean')
    ).reset_index()

    plt.figure(figsize=(6,6))
    sizes = (ta['Count'] / ta['Count'].max()) * 300
    plt.scatter(ta['Effort'], ta['Overdue'], s=sizes, alpha=0.6)
    for i,row in ta.iterrows():
        plt.text(row['Effort'], row['Overdue'], row['Task name'], fontsize=8)
    plt.xlabel("Total Effort (hrs)")
    plt.ylabel("Overdue Rate")
    st.pyplot(plt.gcf())

    ta['Overdue'] = ta['Overdue'].map("{:.0%}".format)
    ta['Speed']   = ta['Speed'].round(1)
    st.dataframe(ta, use_container_width=True)

# ─── Tab 4: Recommendations ─────────────────────────────────────────
with tab4:
    st.header("Recommendations & Insights")
    insights = []

    # Correlation
    if 'CSAT Score' in filtered.columns:
        corr = filtered['Days Before Due'].corr(filtered['CSAT Score'])
        insights.append(f"- **Completion vs CSAT correlation:** {corr:.2f}")

    # Alert late stores
    late = summary[summary['Performance']=='Late']['Store'].tolist()
    if late:
        insights.append(f"- **Alert:** Stores behind: {', '.join(late)}")

    for ins in insights:
        st.markdown(ins)
