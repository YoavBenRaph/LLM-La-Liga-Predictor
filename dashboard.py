import streamlit as st
import pandas as pd
import joblib
import numpy as np
import unicodedata
from datetime import datetime, timedelta

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="⚽ La Liga Predictor",
    page_icon="🏆",
    layout="wide"
)

# ===== LA LIGA COLORS =====
LIGA_RED = "#FF4B44"
LIGA_DARK_RED = "#CC3A34"
LIGA_WHITE = "#FFFFFF"
LIGA_GRAY = "#F5F5F5"


# ===== HELPER FUNCTIONS =====

def clean_team_name(name):
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    name = ' '.join(name.split())
    return name


def get_team_stats(team, df, n_matches=5):
    team_matches = df[
        (df['home_team'] == team) | (df['away_team'] == team)
        ].tail(n_matches)

    if len(team_matches) == 0:
        return {'form': 0.0, 'goals_avg': 1.0, 'xg_avg': 1.0}

    points = 0
    goals = 0
    xg = 0

    for _, match in team_matches.iterrows():
        if match['home_team'] == team:
            if match['result'] == 'Home Win':
                points += 3
            elif match['result'] == 'Draw':
                points += 1
            goals += match['home_goals']
            xg += match['home_xg']
        else:
            if match['result'] == 'Away Win':
                points += 3
            elif match['result'] == 'Draw':
                points += 1
            goals += match['away_goals']
            xg += match['away_xg']

    return {
        'form': points / n_matches,
        'goals_avg': goals / n_matches,
        'xg_avg': xg / n_matches
    }


def get_head_to_head(home_team, away_team, df):
    h2h_matches = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ]

    if len(h2h_matches) == 0:
        return 0.0, 0.0, 0.0, []

    home_wins = 0
    draws = 0
    away_wins = 0
    recent_matches = []

    for _, match in h2h_matches.iterrows():
        if match['home_team'] == home_team:
            if match['result'] == 'Home Win':
                home_wins += 1
            elif match['result'] == 'Draw':
                draws += 1
            else:
                away_wins += 1
            recent_matches.append({
                'date': match['date'],
                'home': match['home_team'],
                'away': match['away_team'],
                'home_goals': int(match['home_goals']),
                'away_goals': int(match['away_goals']),
                'result': match['result']
            })
        else:
            if match['result'] == 'Away Win':
                home_wins += 1
            elif match['result'] == 'Draw':
                draws += 1
            else:
                away_wins += 1
            recent_matches.append({
                'date': match['date'],
                'home': match['home_team'],
                'away': match['away_team'],
                'home_goals': int(match['home_goals']),
                'away_goals': int(match['away_goals']),
                'result': match['result']
            })

    total = home_wins + draws + away_wins
    if total == 0:
        return 0.0, 0.0, 0.0, []

    recent_matches = sorted(recent_matches, key=lambda x: x['date'], reverse=True)[:5]
    return home_wins / total, draws / total, away_wins / total, recent_matches


# ===== LOAD MODEL AND DATA =====

@st.cache_resource
def load_models():
    try:
        model = joblib.load('laliga_model.pkl')
        le = joblib.load('label_encoder.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        return model, le, feature_cols
    except:
        return None, None, None


model, le, feature_cols = load_models()


@st.cache_data
def load_team_data():
    try:
        df = pd.read_csv('matches_prepared.csv')
        df['home_team'] = df['home_team'].apply(clean_team_name)
        df['away_team'] = df['away_team'].apply(clean_team_name)
        teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        return df, teams
    except:
        return None, []


df, teams = load_team_data()


def get_upcoming_matches():
    today = datetime.now()
    return [
        {"home": "Barcelona", "away": "Real Madrid", "date": (today + timedelta(days=3)).strftime("%b %d, %Y")},
        {"home": "Atletico Madrid", "away": "Sevilla", "date": (today + timedelta(days=4)).strftime("%b %d, %Y")},
        {"home": "Villarreal", "away": "Athletic Club", "date": (today + timedelta(days=5)).strftime("%b %d, %Y")},
        {"home": "Real Sociedad", "away": "Betis", "date": (today + timedelta(days=6)).strftime("%b %d, %Y")},
        {"home": "Valencia", "away": "Celta Vigo", "date": (today + timedelta(days=7)).strftime("%b %d, %Y")},
    ]


# ================================================
# ===== PAGE LAYOUT =====
# ================================================

st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0 1.5rem 0; background: linear-gradient(180deg, {LIGA_RED}, {LIGA_DARK_RED}); border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="font-size: 4rem; margin-bottom: 0; color: {LIGA_WHITE};">⚽ La Liga Predictor</h1>
        <p style="font-size: 1.3rem; color: {LIGA_WHITE}; opacity: 0.9;">AI-Powered Match Outcome Predictor</p>
    </div>
""", unsafe_allow_html=True)

if df is None or len(teams) == 0:
    st.warning("⚠️ No team data found. Run prepare_data.py first!")
    st.stop()

if model is None:
    st.warning("⚠️ Model not found! Run train_model.py first!")
    st.stop()

# ================================================
# ===== TWO COLUMN LAYOUT =====
# ================================================

col_left, col_right = st.columns([2, 1], gap="large")

# ================================================
# ===== LEFT COLUMN: CUSTOM MATCH SETUP =====
# ================================================

with col_left:
    st.markdown(f"""
        <div style="background: {LIGA_GRAY}; padding: 1.5rem; border-radius: 15px; border-left: 5px solid {LIGA_RED}; margin-bottom: 1rem;">
            <h2 style="text-align: center; color: {LIGA_DARK_RED};">🎯 Custom Match Setup</h2>
        </div>
    """, unsafe_allow_html=True)

    col_home, col_away = st.columns(2)

    with col_home:
        home_team = st.selectbox(
            "🏠 Home Team",
            teams,
            key="home_team_select"
        )

    with col_away:
        away_team = st.selectbox(
            "✈️ Away Team",
            teams,
            key="away_team_select"
        )

    if home_team == away_team:
        st.warning("⚠️ Home and Away teams must be different!")
        st.stop()

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button(
            "🚀 Predict Match",
            type="primary",
            use_container_width=True,
            key="predict_btn"
        )

# ================================================
# ===== RIGHT COLUMN: UPCOMING MATCHES =====
# ================================================

with col_right:
    st.markdown(f"""
        <div style="background: {LIGA_GRAY}; padding: 1.5rem; border-radius: 15px; border-left: 5px solid {LIGA_RED}; margin-bottom: 1rem;">
            <h2 style="text-align: center; color: {LIGA_DARK_RED};">📅 Upcoming Matches</h2>
            <p style="text-align: center; color: #666; font-size: 0.9rem;">Click a match to predict</p>
        </div>
    """, unsafe_allow_html=True)

    upcoming_matches = get_upcoming_matches()

    for match in upcoming_matches:
        col_match, col_btn = st.columns([4, 1])

        with col_match:
            st.markdown(f"""
                <div style="
                    background: {LIGA_WHITE};
                    padding: 0.5rem 0.75rem;
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                ">
                    <strong style="color: {LIGA_RED};">{match['home']}</strong>
                    <span style="color: #666;"> vs </span>
                    <strong style="color: {LIGA_RED};">{match['away']}</strong>
                    <br>
                    <span style="color: #666; font-size: 0.7rem;">📅 {match['date']}</span>
                </div>
            """, unsafe_allow_html=True)

        with col_btn:
            clean_home = clean_team_name(match['home'])
            clean_away = clean_team_name(match['away'])

            if st.button("🔮", key=f"quick_{clean_home}_{clean_away}",
                         help=f"Predict {match['home']} vs {match['away']}"):
                # Store the match in session state
                st.session_state.quick_home = match['home']
                st.session_state.quick_away = match['away']
                st.rerun()

# ================================================
# ===== PREDICTION RESULTS (Shared between both methods) =====
# ================================================

# Check if we need to auto-predict from quick match
auto_predict = False
if 'quick_home' in st.session_state and 'quick_away' in st.session_state:
    # Clear the quick match after using it
    auto_predict = True
    home_team = st.session_state.quick_home
    away_team = st.session_state.quick_away
    # Clear the session state to prevent auto-predict on every refresh
    del st.session_state.quick_home
    del st.session_state.quick_away
    # Set the dropdowns via a different method
    # We'll use the auto_predict flag instead of modifying session state directly

# If predict button was clicked OR quick match was selected
if predict_clicked or auto_predict:
    # Use the current home_team and away_team from the dropdowns
    # If auto_predict is True, we need to set them differently
    if auto_predict:
        # Find the matching team names in the dropdown
        home_match = next((t for t in teams if clean_team_name(t) == clean_team_name(home_team)), home_team)
        away_match = next((t for t in teams if clean_team_name(t) == clean_team_name(away_team)), away_team)
        # We can't modify the dropdowns, so we'll use the values directly
        current_home = home_match
        current_away = away_match
    else:
        current_home = home_team
        current_away = away_team

    # Calculate stats
    home_stats = get_team_stats(current_home, df)
    away_stats = get_team_stats(current_away, df)
    h2h_home, h2h_draw, h2h_away, recent_matches = get_head_to_head(current_home, current_away, df)

    with st.spinner("Analyzing..."):
        features = np.array([[
            home_stats['form'],
            away_stats['form'],
            home_stats['goals_avg'],
            away_stats['goals_avg'],
            home_stats['xg_avg'],
            away_stats['xg_avg'],
            h2h_home,
            h2h_draw,
            h2h_away
        ]])

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        outcome = le.inverse_transform([pred])[0]

        st.markdown("---")

        st.markdown(f"""
            <div style="display: flex; justify-content: space-around; text-align: center; padding: 0.5rem 0;">
                <div style="background: {LIGA_RED}; padding: 1rem 2rem; border-radius: 12px; flex: 1; margin: 0 0.25rem;">
                    <h3 style="color: {LIGA_WHITE}; margin: 0;">🏠 Home Win</h3>
                    <p style="color: {LIGA_WHITE}; font-size: 1.5rem; font-weight: bold; margin: 0;">{proba[2] * 100:.1f}%</p>
                </div>
                <div style="background: {LIGA_DARK_RED}; padding: 1rem 2rem; border-radius: 12px; flex: 1; margin: 0 0.25rem;">
                    <h3 style="color: {LIGA_WHITE}; margin: 0;">🤝 Draw</h3>
                    <p style="color: {LIGA_WHITE}; font-size: 1.5rem; font-weight: bold; margin: 0;">{proba[1] * 100:.1f}%</p>
                </div>
                <div style="background: {LIGA_RED}; padding: 1rem 2rem; border-radius: 12px; flex: 1; margin: 0 0.25rem;">
                    <h3 style="color: {LIGA_WHITE}; margin: 0;">✈️ Away Win</h3>
                    <p style="color: {LIGA_WHITE}; font-size: 1.5rem; font-weight: bold; margin: 0;">{proba[0] * 100:.1f}%</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if outcome == 'Home Win':
            st.success(f"🏠 **Prediction: {current_home} wins!**")
        elif outcome == 'Draw':
            st.warning(f"🤝 **Prediction: Draw!**")
        else:
            st.info(f"✈️ **Prediction: {current_away} wins!**")

        with st.expander("📊 Team Stats", expanded=False):
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.write(f"**🏠 {current_home}**")
                st.write(f"Recent form: {home_stats['form']:.1f} pts/match")
                st.write(f"Avg goals: {home_stats['goals_avg']:.2f}")
                st.write(f"Avg xG: {home_stats['xg_avg']:.2f}")
            with col_stat2:
                st.write(f"**✈️ {current_away}**")
                st.write(f"Recent form: {away_stats['form']:.1f} pts/match")
                st.write(f"Avg goals: {away_stats['goals_avg']:.2f}")
                st.write(f"Avg xG: {away_stats['xg_avg']:.2f}")

        with st.expander("📈 Head-to-Head", expanded=False):
            st.write(f"**{current_home} wins:** {h2h_home * 100:.1f}%")
            st.write(f"**Draws:** {h2h_draw * 100:.1f}%")
            st.write(f"**{current_away} wins:** {h2h_away * 100:.1f}%")

        with st.expander("📋 Last 5 Meetings", expanded=False):
            if recent_matches:
                for match in recent_matches:
                    if match['result'] == 'Home Win':
                        emoji = "🏠"
                    elif match['result'] == 'Away Win':
                        emoji = "✈️"
                    else:
                        emoji = "🤝"
                    st.write(
                        f"{emoji} **{match['date']}**: {match['home']} {match['home_goals']} - {match['away_goals']} {match['away']}")
            else:
                st.write("No previous meetings found.")

else:
    # Show placeholder when no prediction
    st.markdown(f"""
        <div style="background: {LIGA_GRAY}; padding: 2rem; border-radius: 15px; text-align: center; border: 2px dashed {LIGA_RED}; margin-top: 1rem;">
            <p style="color: {LIGA_DARK_RED}; font-size: 1.1rem;">⚡ Select two teams and click</p>
            <p style="color: {LIGA_RED}; font-size: 1.3rem; font-weight: bold;">"🚀 Predict Match"</p>
            <p style="color: {LIGA_DARK_RED}; font-size: 0.9rem;">or click 🔮 on an upcoming match</p>
        </div>
    """, unsafe_allow_html=True)

st.caption("⚽ Click 'Predict' on an upcoming match to quickly set up the prediction")