import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Task Performance Dashboard", layout="wide")
st.markdown("""<style>.st-emotion-cache-1avcm0n {padding-top: 0rem;}</style>""", unsafe_allow_html=True)

# Upload CSV with a clean "plus" button style
st.title("Task Performance Dashboard")
uploaded_file = st.file_uploader("➕ Upload WorkJam CSV", type="csv", label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
    df['Date completed'] = pd.to_datetime(df['Date completed'], errors='coerce')
    df['Days Before Due'] = (df['End date'] - df['Date completed']).dt.days
    df['Overdue'] = (df['End date'] < pd.Timestamp.now()) & (df['Task status'] != 'Completed')

    # Clean location groupings
    df['Region'] = df['Level 1'].fillna('Unknown')
    df['Store'] = df['Location name']
    df['Company'] = df['Store'].apply(lambda x: 'JameTrade' if 'JameTrade' in x else 'Other')

    # Aggregate KPIs
    total_tasks = len(df)
    overdue_tasks = df['Overdue'].sum()
    completed_on_time = df[df['Days Before Due'] >= 0].shape[0]
    ad_hoc_tasks = df['Store'].value_counts()[df['Store'].value_counts() == 1].shape[0]
    avg_days_before_due = df['Days Before Due'].dropna().mean()
    common_category = df['Task category'].mode()[0] if not df['Task category'].mode().empty else 'N/A'

    st.markdown("### Company-Level KPIs (JameTrade)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tasks", total_tasks)
    col2.metric("Completed On Time", f"{completed_on_time / total_tasks:.0%}")
    col3.metric("Avg Days Before Due", f"{avg_days_before_due:.1f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Overdue Tasks", overdue_tasks)
    col5.metric("Ad Hoc Tasks", ad_hoc_tasks)
    col6.metric("Top Task Category", common_category)

    # Regional insights
    st.markdown("### Regional Performance Summary")
    region_summary = df.groupby('Region').agg({
        'Task name': 'count',
        'Overdue': 'sum',
        'Days Before Due': 'mean'
    }).rename(columns={
        'Task name': 'Total Tasks',
        'Overdue': 'Overdue Tasks',
        'Days Before Due': 'Avg Days Before Due'
    }).sort_values(by='Overdue Tasks', ascending=False)

    st.dataframe(region_summary.style.format({
        'Avg Days Before Due': "{:.1f}",
        'Total Tasks': "{:,.0f}",
        'Overdue Tasks': "{:,.0f}"
    }), use_container_width=True)

    # Store-level insights: show only notables
    st.markdown("### Most Proactive Stores (Completed Tasks Early)")
    store_speed = df.dropna(subset=['Days Before Due'])
    store_speed = store_speed.groupby('Store')['Days Before Due'].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=store_speed.values, y=store_speed.index, palette='Greens_d', ax=ax)
    ax.set_title("Top 10 Proactive Stores (Avg Days Before Due)")
    ax.set_xlabel("Avg Days Early")
    st.pyplot(fig)

    st.markdown("### Stores with Most Overdue Tasks")
    overdue_counts = df[df['Overdue']].groupby('Store').size().sort_values(ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=overdue_counts.values, y=overdue_counts.index, palette='Reds_d', ax=ax2)
    ax2.set_title("Top 10 Stores with Overdue Tasks")
    ax2.set_xlabel("Overdue Task Count")
    st.pyplot(fig2)

else:
    st.info("Use the ➕ button above to upload your WorkJam task CSV.")
