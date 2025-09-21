import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time

# ------------------- Streamlit Setup -------------------
st.set_page_config(page_title="Classroom Emotion Dashboard", layout="wide")
st.title("📊 Classroom Emotion Dashboard (Enhanced Live)")

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

# ------------------- Connect to DB -------------------
conn = sqlite3.connect("emotions.db")
df = pd.read_sql_query("SELECT * FROM emotions", conn)

if not df.empty:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].dt.floor('T')

    # ------------------- Live Faces Detected -------------------
    latest_time = df['timestamp'].max()
    live_faces_count = df[df['timestamp'] == latest_time].shape[0]

    # ------------------- Last 10 Minutes Data -------------------
    last_time = df['minute'].max()
    start_time = last_time - pd.Timedelta(minutes=9)
    last_10min_df = df[(df['minute'] >= start_time) & (df['minute'] <= last_time)]

    happy_count = last_10min_df[last_10min_df['emotion'] == 'Happy'].shape[0]
    stress_count = last_10min_df[last_10min_df['emotion'].isin(['Sad','Angry','Fear','Disgust'])].shape[0]
    most_frequent = last_10min_df['emotion'].mode()[0] if not last_10min_df.empty else "N/A"

    # ------------------- Summary Cards / KPIs -------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Faces Detected (Live)", live_faces_count)
    col2.metric("Happy Count (Last 10 min)", happy_count)
    col3.metric("Stress Count (Last 10 min)", stress_count)
    col4.metric("Most Frequent Emotion (Last 10 min)", most_frequent)

    # ------------------- Emotion Distribution Bar Chart -------------------
    st.subheader("Emotion Distribution (Last 10 Minutes)")
    if not last_10min_df.empty:
        emotion_counts = last_10min_df['emotion'].value_counts().reset_index()
        emotion_counts.columns = ['emotion', 'count']
        fig_bar = px.bar(
            emotion_counts,
            x='emotion',
            y='count',
            color='emotion',
            text='count',
            color_discrete_map={
                'Happy':'#2ca02c',
                'Neutral':'#1f77b4',
                'Sad':'#ff7f0e',
                'Angry':'#d62728',
                'Fear':'#9467bd',
                'Disgust':'#8c564b',
                'Surprise':'#e377c2'
            },
        )
        fig_bar.update_layout(yaxis_title="Number of Detections", xaxis_title="Emotion", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No emotion data in the last 10 minutes.")

    # ------------------- Dominant Emotion Curve Graph -------------------
    st.subheader("Dominant Emotion Trend (Last 10 Minutes)")
    if not last_10min_df.empty:
        dominant_df = last_10min_df.groupby('minute')['emotion'].agg(lambda x: x.value_counts().idxmax()).reset_index()
        all_emotions = df['emotion'].unique().tolist()
        emotion_order = {e:i for i, e in enumerate(all_emotions)}
        dominant_df['emotion_code'] = dominant_df['emotion'].map(emotion_order)

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=dominant_df['minute'],
            y=dominant_df['emotion_code'],
            mode='lines+markers',
            line=dict(shape='spline', width=3, color='blue'),
            marker=dict(size=8),
            name='Dominant Emotion'
        ))
        fig_curve.update_yaxes(
            tickvals=list(emotion_order.values()),
            ticktext=list(emotion_order.keys()),
            title_text="Emotion"
        )
        fig_curve.update_xaxes(title_text="Time (per minute)")
        fig_curve.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig_curve, use_container_width=True)

    # ------------------- Overall Emotion Percentage (9 AM - 5 PM) -------------------
    st.subheader("Overall Emotion Percentage (9 AM - 5 PM)")
    df['time_only'] = df['timestamp'].dt.time
    start_day = time(9,0,0)
    end_day = time(17,0,0)
    day_df = df[(df['time_only'] >= start_day) & (df['time_only'] <= end_day)]

    if not day_df.empty:
        day_counts = day_df['emotion'].value_counts().reset_index()
        day_counts.columns = ['emotion', 'count']
        fig_pie = px.pie(
            day_counts,
            names='emotion',
            values='count',
            color='emotion',
            color_discrete_map={
                'Happy':'#2ca02c',
                'Neutral':'#1f77b4',
                'Sad':'#ff7f0e',
                'Angry':'#d62728',
                'Fear':'#9467bd',
                'Disgust':'#8c564b',
                'Surprise':'#e377c2'
            },
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No emotion data recorded between 9 AM - 5 PM.")

else:
    st.info("No emotion data recorded yet.")

conn.close()
