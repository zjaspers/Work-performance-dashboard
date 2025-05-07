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

# ─── Helpers ─────────────────────────────────────────────────────────
@st.cache_data
def load_csv(f): return pd.read_csv(f)

def preprocess(df):
    df['End date']       = pd.to_datetime(df['End date'], errors='coerce')
    df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')
    today = pd.Timestamp.now().normalize()
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    missing = df['Date completed'].isna() & df['End date'].notna()
    df.loc[missing,'Days Before Due'] = -((df.loc[missing,'End date'] - today).dt.days)
    df['Overdue'] = df['Days Before Due'] < 0
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

# ─── Sidebar: Upload & Filters ───────────────────────────────────────
st.sidebar.header("Data & Filters")
task_file = st.sidebar.file_uploader("➕ Task CSV", type="csv")
kpi_file  = st.sidebar.file_uploader("📊 Store KPI CSV (optional)", type="csv")

if not task_file:
    st.sidebar.info("Please upload Task CSV.")
    st.stop()

# ─── Load & Prepare Data ─────────────────────────────────────────────
df = load_csv(task_file)
df = preprocess(df)
if kpi_file:
    kpi = load_csv(kpi_file).rename(columns={'Location ID':'Location external ID'})
    df = df.merge(kpi, on=['Location external ID','Store'], how='left')

# ─── Week Selector & Filters ─────────────────────────────────────────
weeks  = sorted(df['Week Start'].dropna().unique(), reverse=True)
labels = [f"{w.date()}–{(w+timedelta(days=6)).date()}" for w in weeks]
sel    = st.sidebar.selectbox("Select Week", labels)
start  = weeks[labels.index(sel)]
week_df = df[df['Week Start']==start]

tasks  = sorted(week_df['Task name'].unique())
stores = sorted(week_df['Store'].unique())
sel_tasks  = st.sidebar.multiselect("Filter by Task", tasks, default=tasks)
sel_stores = st.sidebar.multiselect("Filter by Store", stores)

filtered = week_df[week_df['Task name'].isin(sel_tasks)]
if sel_stores:
    filtered = filtered[filtered['Store'].isin(sel_stores)]

# ─── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Key Metrics & Recommendations",
    "🏬 Store Performance",
    "🛠 Task Analysis"
])

# ─── Tab 1: Key Metrics & Recommendations ───────────────────────────
with tab1:
    st.header("Key Metrics")
    total_tasks   = filtered['Task ID'].nunique()
    on_time_count = filtered.groupby('Task ID')['Days Before Due'].max().ge(0).sum()
    avg_days      = filtered.groupby('Task ID')['Days Before Due'].mean().mean().round(1)
    overdue_count = total_tasks - on_time_count
    adhoc        = filtered.groupby('Task ID')['Store'].nunique().eq(1).sum()
    avg_csat     = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered else None

    cols = st.columns(6)
    with cols[0]: metric_card("Total Tasks", total_tasks)
    with cols[1]: metric_card("% On Time", f"{on_time_count/total_tasks:.0%}")
    with cols[2]: metric_card("Avg Days Before Due", avg_days)
    with cols[3]: metric_card("Overdue Tasks", overdue_count)
    with cols[4]: metric_card("Ad Hoc Tasks", adhoc)
    if avg_csat is not None:
        with cols[5]: metric_card("Avg CSAT", f"{avg_csat:.1f}")

    st.markdown("### Recommendations")
    recos = []
    # Correlation
    if 'CSAT Score' in filtered:
        corr = filtered['Days Before Due'].corr(filtered['CSAT Score'])
        recos.append(f"- Completion speed vs CSAT correlation: **{corr:.2f}**")
    # Underperformers
    sb = filtered.groupby('Store').agg(
        Overdue_Rate=('Overdue','mean'),
        Avg_Days=('Days Before Due','mean')
    ).reset_index()
    late = sb[sb['Avg_Days']<0]['Store'].tolist()
    if late:
        recos.append(f"- Underperforming stores: **{', '.join(late)}**")
    # Effort vs CSAT
    if 'Expected duration' in filtered and 'CSAT Score' in filtered:
        effort = filtered.groupby('Store')['Expected duration'].sum()
        csat   = filtered.groupby('Store')['CSAT Score'].mean()
        high_eff = effort[effort>effort.quantile(0.8)].index.tolist()
        recos.append(f"- High effort but lower CSAT: **{', '.join(high_eff)}**")
    for r in recos:
        st.markdown(r)

# ─── Tab 2: Store Performance ────────────────────────────────────────
with tab2:
    st.header("Store Performance")
    sb = filtered.groupby('Store').agg(
        Total_Tasks=('Task ID','nunique'),
        Overdue_Rate=('Overdue','mean'),
        Avg_Days=('Days Before Due','mean')
    )
    if 'CSAT Score' in filtered:
        sb['CSAT'] = filtered.groupby('Store')['CSAT Score'].mean()
    sb = sb.reset_index()
    sb['Performance'] = sb['Avg_Days'].apply(
        lambda x: 'Early' if x>0 else ('On Time' if x==0 else 'Late')
    )

    # dynamic height
    n = len(sb)
    fig, ax = plt.subplots(figsize=(8, max(4, n*0.3)))
    sns.barplot(
        data=sb.sort_values('Overdue_Rate', ascending=False),
        x='Overdue_Rate', y='Store', hue='Performance', dodge=False,
        palette={'Early':'#2ca02c','On Time':'#ff7f0e','Late':'#d62728'},
        ax=ax
    )
    ax.set_xlabel("Overdue Rate")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(title="Performance", bbox_to_anchor=(1.02,1), loc='upper left')
    st.pyplot(fig)

    st.dataframe(
        sb.style.format({
            'Overdue_Rate':'{:.0%}','Avg_Days':'{:.1f}','CSAT':'{:.1f}'
        }),
        use_container_width=True
    )

# ─── Tab 3: Task Analysis ────────────────────────────────────────────
with tab3:
    st.header("Task Effort & Performance")
    ta = filtered.groupby('Task name').agg(
        Count=('Task ID','nunique'),
        Effort=('Expected duration','sum'),
        Overdue=('Overdue','mean'),
        Speed=('Days Before Due','mean')
    ).reset_index()
    plt.figure(figsize=(6,6))
    sizes = (ta['Count']/ta['Count'].max())*300
    plt.scatter(ta['Effort'], ta['Overdue'], s=sizes, alpha=0.6)
    for _,r in ta.iterrows():
        plt.text(r['Effort'], r['Overdue'], r['Task name'], fontsize=8)
    plt.xlabel("Total Effort (hrs)")
    plt.ylabel("Overdue Rate")
    st.pyplot(plt.gcf())

    ta['Overdue'] = ta['Overdue'].map("{:.0%}".format)
    ta['Speed']   = ta['Speed'].round(1)
    st.dataframe(ta, use_container_width=True)
