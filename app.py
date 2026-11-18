import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Page Configuration
st.set_page_config(page_title="SupeRR Selector Dashboard", layout="wide")
st.title("🏏 Cricket Pressure Performance Dashboard")
st.markdown("Interactive analysis of Top 10 Batters and Bowlers under high-pressure match situations (2022 onwards).")

# 2. Load & Process Data (Cached so the app runs fast)
@st.cache_data
def load_and_process_data():
    bb_file = "6af855c0e8386_ball_by_ball.csv" if os.path.exists("6af855c0e8386_ball_by_ball.csv") else "ball_by_ball.csv"
    ball_by_ball = pd.read_csv(bb_file)
    matches = pd.read_csv("matches.csv")

    df = ball_by_ball.merge(matches[['match_id', 'match_date']], on='match_id', how='inner')
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df[df['match_date'].dt.year >= 2022].copy()

    # Innings totals & match state reconstruction
    inn_totals = df.groupby(['match_id', 'innings'])['runs'].sum().reset_index().rename(columns={'runs': 'total_innings_runs'})
    df = df.merge(inn_totals, on=['match_id', 'innings'], how='left')

    df = df.sort_values(['match_id', 'innings', 'overid', 'over_ball_num'])
    df['cum_runs_prior'] = df.groupby(['match_id', 'innings'])['runs'].cumsum() - df['runs']
    df['wicket_int'] = df['is_wicket'].astype(int)
    df['cum_wickets_prior'] = df.groupby(['match_id', 'innings'])['wicket_int'].cumsum() - df['wicket_int']

    balls_bowled = (df['overid'] - 1) * 6 + df['over_ball_num'] - 1
    df['balls_remaining'] = (120 - balls_bowled).clip(lower=1)

    first_inn = inn_totals[inn_totals['innings'] == 1][['match_id', 'total_innings_runs']].rename(columns={'total_innings_runs': 'target_1st_inn'})
    df = df.merge(first_inn, on='match_id', how='left')
    df['target'] = df['target_1st_inn'] + 1

    df['runs_required'] = np.where(df['innings'] == 2, df['target'] - df['cum_runs_prior'], np.nan)
    df['rrr'] = np.where((df['innings'] == 2) & (df['balls_remaining'] > 0), (df['runs_required'] / df['balls_remaining']) * 6, np.nan)

    # Pressure Definition
    p1 = (df['innings'] == 1) & ((df['overid'] >= 16) | ((df['cum_wickets_prior'] >= 3) & (df['overid'] <= 12)))
    p2 = (df['innings'] == 2) & ((df['rrr'] >= 9.0) | (df['cum_wickets_prior'] >= 4) | ((df['balls_remaining'] <= 36) & (df['rrr'] >= 8.0)))
    df['is_pressure'] = p1 | p2

    # Aggregating Batter Stats
    bat_df = df.groupby(['striker_id', 'striker_name']).apply(lambda g: pd.Series({
        'pressure_balls': g['is_pressure'].sum(),
        'normal_balls': (~g['is_pressure']).sum(),
        'pressure_runs': g.loc[g['is_pressure'], 'runs'].sum(),
        'normal_runs': g.loc[~g['is_pressure'], 'runs'].sum(),
        'pressure_boundaries': (g.loc[g['is_pressure'], 'is_four'] | g.loc[g['is_pressure'], 'is_six']).sum(),
        'pressure_dismissals': (g['is_pressure'] & g['is_wicket']).sum()
    })).reset_index()

    # Aggregating Bowler Stats
    bowl_df = df.groupby(['bowler_id', 'bowler_name']).apply(lambda g: pd.Series({
        'pressure_balls': g['is_pressure'].sum(),
        'normal_balls': (~g['is_pressure']).sum(),
        'pressure_runs': g.loc[g['is_pressure'], 'runs'].sum(),
        'normal_runs': g.loc[~g['is_pressure'], 'runs'].sum(),
        'pressure_wickets': (g['is_pressure'] & g['is_wicket']).sum(),
        'pressure_dots': (g['is_pressure'] & (g['runs'] == 0)).sum()
    })).reset_index()

    return df, bat_df, bowl_df

with st.spinner("Processing ball-by-ball data..."):
    df_merged, bat_raw, bowl_raw = load_and_process_data()

# 3. Sidebar Interactive Controls
st.sidebar.header("⚙️ Filter Thresholds")
min_bat_balls = st.sidebar.slider("Min Pressure Balls (Batters)", min_value=50, max_value=400, value=180, step=10)
min_bowl_balls = st.sidebar.slider("Min Pressure Balls (Bowlers)", min_value=50, max_value=400, value=140, step=10)

# 4. Calculate Dynamic Rankings
# Batters
bat_filtered = bat_raw[bat_raw['pressure_balls'] >= min_bat_balls].copy()
bat_filtered['Normal SR'] = (bat_filtered['normal_runs'] / bat_filtered['normal_balls']) * 100
bat_filtered['Pressure SR'] = (bat_filtered['pressure_runs'] / bat_filtered['pressure_balls']) * 100
bat_filtered['SR Delta'] = bat_filtered['Pressure SR'] - bat_filtered['Normal SR']
bat_filtered['Boundary %'] = (bat_filtered['pressure_boundaries'] / bat_filtered['pressure_balls']) * 100
bat_filtered['Dismissal Rate'] = bat_filtered['pressure_dismissals'] / bat_filtered['pressure_balls']

bat_filtered['Clutch Score'] = (
    0.35 * bat_filtered['Pressure SR'].rank(pct=True) +
    0.25 * bat_filtered['SR Delta'].rank(pct=True) +
    0.25 * bat_filtered['Boundary %'].rank(pct=True) +
    0.15 * (1 - bat_filtered['Dismissal Rate'].rank(pct=True))
)
top_10_bat = bat_filtered.sort_values(by='Clutch Score', ascending=False).head(10).reset_index(drop=True)

# Bowlers
bowl_filtered = bowl_raw[bowl_raw['pressure_balls'] >= min_bowl_balls].copy()
bowl_filtered['Pressure Economy'] = (bowl_filtered['pressure_runs'] / bowl_filtered['pressure_balls']) * 6
bowl_filtered['Normal Economy'] = (bowl_filtered['normal_runs'] / bowl_filtered['normal_balls']) * 6
bowl_filtered['Economy Delta'] = bowl_filtered['Normal Economy'] - bowl_filtered['Pressure Economy']
bowl_filtered['Wicket Rate'] = bowl_filtered['pressure_wickets'] / bowl_filtered['pressure_balls']
bowl_filtered['Dot Ball %'] = (bowl_filtered['pressure_dots'] / bowl_filtered['pressure_balls']) * 100

bowl_filtered['Containment Score'] = (
    0.35 * (1 - bowl_filtered['Pressure Economy'].rank(pct=True)) +
    0.25 * bowl_filtered['Economy Delta'].rank(pct=True) +
    0.25 * bowl_filtered['Wicket Rate'].rank(pct=True) +
    0.15 * bowl_filtered['Dot Ball %'].rank(pct=True)
)
top_10_bowl = bowl_filtered.sort_values(by='Containment Score', ascending=False).head(10).reset_index(drop=True)

# 5. Top KPI Summary Cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Deliveries Analyzed", f"{len(df_merged):,}")
col2.metric("High-Pressure Deliveries", f"{df_merged['is_pressure'].sum():,}")
col3.metric("Top Clutch Batter", top_10_bat.loc[0, 'striker_name'] if not top_10_bat.empty else "N/A")

st.divider()

# 6. Dashboard Layout: Two Columns for Batters and Bowlers
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🔥 Top 10 Batters Under Pressure")
    if not top_10_bat.empty:
        # Minimalist Pink Bar Chart
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        bp = top_10_bat.iloc[::-1]
        bars1 = ax1.barh(bp['striker_name'], bp['Pressure SR'], color='#D95F87', height=0.6)
        ax1.set_xlabel('Pressure Strike Rate (Runs per 100 Balls)', fontweight='bold')
        ax1.grid(axis='x', linestyle='--', alpha=0.4)
        ax1.set_xlim(0, bp['Pressure SR'].max() * 1.2)
        for bar in bars1:
            w = bar.get_width()
            ax1.text(w + 2, bar.get_y() + bar.get_height()/2, f'{w:.1f}', va='center', fontsize=9, color='#8C2749', fontweight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)

        # Clean Display Table
        display_bat = top_10_bat[['striker_name', 'pressure_balls', 'Pressure SR', 'SR Delta', 'Clutch Score']].copy()
        display_bat.columns = ['Batter', 'Balls', 'Pressure SR', 'SR Delta', 'Score']
        st.dataframe(display_bat.style.format({'Pressure SR': '{:.1f}', 'SR Delta': '{:+.1f}', 'Score': '{:.3f}'}), use_container_width=True)

with right_col:
    st.subheader("🎯 Top 10 Bowlers Under Pressure")
    if not top_10_bowl.empty:
        # Minimalist Slate Blue Bar Chart
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        b_p = top_10_bowl.iloc[::-1]
        bars2 = ax2.barh(b_p['bowler_name'], b_p['Pressure Economy'], color='#2B5C8F', height=0.6)
        ax2.set_xlabel('Pressure Economy Rate (Lower is Better)', fontweight='bold')
        ax2.grid(axis='x', linestyle='--', alpha=0.4)
        ax2.set_xlim(0, b_p['Pressure Economy'].max() * 1.22)
        for bar in bars2:
            w = bar.get_width()
            ax2.text(w + 0.1, bar.get_y() + bar.get_height()/2, f'{w:.2f}', va='center', fontsize=9, color='#1A3A5C', fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)

        # Clean Display Table
        display_bowl = top_10_bowl[['bowler_name', 'pressure_balls', 'Pressure Economy', 'Dot Ball %', 'Containment Score']].copy()
        display_bowl.columns = ['Bowler', 'Balls', 'Economy', 'Dot %', 'Score']
        st.dataframe(display_bowl.style.format({'Economy': '{:.2f}', 'Dot %': '{:.1f}%', 'Score': '{:.3f}'}), use_container_width=True)