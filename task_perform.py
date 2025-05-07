import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta

# ─── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Task Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Styling ────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding: 2rem; font-family: 'Inter', sans-serif; }
  .metric-card {
    background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center; margin-bottom: 1rem;
  }
  .metric-label { font-size: 0.9rem; color: #555; }
  .metric-value { font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── Utility Functions ─────────────────────────────────────────────

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

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

def metric_card(label, value, delta=None):
    """Renders a single metric card."""
    cols = st.columns([1])
    with cols[0]:
        st.markdown(f"<div class='metric-card'>"
                    f"<div class='metric-value'>{value}</div>"
                    f"<div class='metric-label'>{label}</div>"
                    f"</div>", unsafe_allow_html=True)

# ─── Sidebar Uploads & Filters ─────────────────────────────────────
st.sidebar.header("Data Upload & Filters")
task_file = st.sidebar.file_uploader("➕ WorkJam Task CSV", type="csv")
kpi_file  = st.sidebar.file_uploader("📊 Store KPI CSV (optional)", type="csv")

if not task_file:
    st.sidebar.info("Please upload your Task CSV to proceed.")
    st.stop()

# Load & preprocess
df_tasks = load_csv(task_file)
df = preprocess_tasks(df_tasks)

if kpi_file:
    df_kpi = load_csv(kpi_file).rename(columns={'Location ID':'Location external ID'})
    df = df.merge(df_kpi, on=['Location external ID','Store'], how='left')

# Week selector
weeks  = sorted(df['Week Start'].dropna().unique(), reverse=True)
labels = [f"{w.date()}–{(w + timedelta(days=6)).date()}" for w in weeks]
sel    = st.sidebar.selectbox("Select Week", labels)
start  = weeks[labels.index(sel)]
week_df = df[df['Week Start']==start]

# Task & Store filters
task_opts  = sorted(week_df['Task name'].unique())
store_opts = sorted(week_df['Store'].unique())
sel_tasks  = st.sidebar.multiselect("Filter by Task", task_opts, default=task_opts)
sel_stores = st.sidebar.multiselect("Filter by Store", store_opts)

filtered = week_df[week_df['Task name'].isin(sel_tasks)]
if sel_stores:
    filtered = filtered[filtered['Store'].isin(sel_stores)]

# ─── Tabs: Organize the Dashboard ──────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Key Metrics", 
    "🏬 Store Performance", 
    "🛠 Task Analysis", 
    "💡 Recommendations"
])

# ─── Tab 1: Key Metrics ────────────────────────────────────────────
with tab1:
    st.header("Key Metrics")
    total    = len(filtered)
    on_time  = (filtered['Days Before Due']>=0).sum()
    avg_spd  = filtered['Days Before Due'].mean()
    overdue  = int(filtered['Overdue'].sum())
    adhoc    = (filtered['Store'].value_counts()==1).sum()
    avg_csat = filtered['CSAT Score'].mean() if 'CSAT Score' in filtered.columns else None

    cols = st.columns(6)
    metric_card("Total Tasks",           total)
    metric_card("% On Time",            f"{on_time/total:.0%}" if total else "N/A")
    metric_card("Avg Early/Late (days)", f"{avg_spd:.1f}")
    metric_card("Overdue Tasks",        overdue)
    metric_card("Ad Hoc Tasks",         adhoc)
    if avg_csat is not None:
        metric_card("Avg CSAT",         f"{avg_csat:.1f}")

# ─── Tab 2: Store Performance ─────────────────────────────────────
with tab2:
    st.header("Store Status Summary")
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

    # Interactive bar chart of Overdue Rate by store
    fig = px.bar(
        summary, x='Store', y='Overdue_Rate', color='Performance',
        color_discrete_map={'Early':'green','On Time':'gold','Late':'red'},
        labels={'Overdue_Rate':'Overdue Rate'}, title="Overdue Rate by Store"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        summary.style.format({
            'Overdue_Rate':'{:.0%}', 'Avg_Days':'{:.1f}', 'CSAT':'{:.1f}', 'Sales':'{:.1f}%'
        }),
        use_container_width=True
    )

# ─── Tab 3: Task Analysis ─────────────────────────────────────────
with tab3:
    st.header("Task Effort & Performance by Task")
    ta = filtered.groupby('Task name').agg(
        Count=('Task name','size'),
        Effort=('Expected duration','sum'),
        Overdue=('Overdue','mean'),
        Speed=('Days Before Due','mean')
    ).reset_index()
    ta['Overdue'] = ta['Overdue'].map("{:.0%}".format)
    ta['Speed']   = ta['Speed'].round(1)

    fig2 = px.scatter(
        ta, x='Effort', y='Overdue', size='Count', hover_name='Task name',
        labels={'Effort':'Total Effort (hrs)','Overdue':'Overdue Rate'},
        title="Effort vs. Overdue Rate by Task"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(ta, use_container_width=True)

# ─── Tab 4: Recommendations ────────────────────────────────────────
with tab4:
    st.header("Smart Insights & Recommendations")
    reco = []

    # Correlation analysis
    if 'CSAT Score' in filtered.columns:
        corr = filtered['Days Before Due'].corr(filtered['CSAT Score'])
        reco.append(f"- **Correlation:** Completion speed vs. CSAT: {corr:.2f}")

    # Quadrant analysis: Effort vs. Overdue
    ta['Overdue_pct'] = ta['Overdue'].str.rstrip('%').astype(float)
    q_fig = px.scatter(
        ta, x='Effort', y='Overdue_pct', 
        labels={'Effort':'Total Effort','Overdue_pct':'Overdue %'},
        title="Quadrant: Effort vs Overdue %"
    )
    st.plotly_chart(q_fig, use_container_width=True)

    # Alerts
    poor = summary[summary['Performance']=='Late']['Store'].tolist()
    if poor:
        reco.append(f"- **Alert:** Stores behind: {', '.join(poor)}")

    # Display recommendations
    for r in reco:
        st.markdown(r)
