# Create advanced features

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("Loading prepared data...")
df = pd.read_csv('matches_prepared.csv')
print(f"Loaded {len(df)} matches")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# Get all unique teams
teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
print(f"Teams found: {len(teams)}")


def get_team_form(team, date, n_matches=5):
    """Calculate team's form over last N matches"""
    # Get all matches for this team before this date
    past_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
        ].tail(n_matches)

    if len(past_matches) == 0:
        return 0.0

    # Calculate points per match (3 for win, 1 for draw, 0 for loss)
    points = 0
    for _, match in past_matches.iterrows():
        if match['home_team'] == team:
            if match['result'] == 'Home Win':
                points += 3
            elif match['result'] == 'Draw':
                points += 1
        else:  # away team
            if match['result'] == 'Away Win':
                points += 3
            elif match['result'] == 'Draw':
                points += 1

    return points / n_matches


def get_head_to_head(home_team, away_team, date):
    """Get head-to-head history between two teams"""
    # Get all previous matches between these two teams
    h2h_matches = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ]
    h2h_matches = h2h_matches[h2h_matches['date'] < date]

    if len(h2h_matches) == 0:
        return 0.0, 0.0, 0.0  # No history

    home_wins = 0
    draws = 0
    away_wins = 0

    for _, match in h2h_matches.iterrows():
        if match['home_team'] == home_team:
            if match['result'] == 'Home Win':
                home_wins += 1
            elif match['result'] == 'Draw':
                draws += 1
            else:
                away_wins += 1
        else:  # home_team is away in this match
            if match['result'] == 'Away Win':
                home_wins += 1
            elif match['result'] == 'Draw':
                draws += 1
            else:
                away_wins += 1

    total = home_wins + draws + away_wins
    if total == 0:
        return 0.0, 0.0, 0.0

    return home_wins / total, draws / total, away_wins / total


def get_team_strength(team, date, n_matches=10):
    """Calculate team's average goals and xG over last N matches"""
    past_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
        ].tail(n_matches)

    if len(past_matches) == 0:
        return 1.0, 1.0

    total_goals = 0
    total_xg = 0

    for _, match in past_matches.iterrows():
        if match['home_team'] == team:
            total_goals += match['home_goals']
            total_xg += match['home_xg']
        else:
            total_goals += match['away_goals']
            total_xg += match['away_xg']

    return total_goals / len(past_matches), total_xg / len(past_matches)


print("\nEngineering features...")

# Apply features to each match
features = []

for idx, match in df.iterrows():
    if idx % 500 == 0:
        print(f"Processing match {idx}/{len(df)}")

    home_team = match['home_team']
    away_team = match['away_team']
    date = match['date']

    # Home team features
    home_form = get_team_form(home_team, date)
    away_form = get_team_form(away_team, date)

    home_goals_avg, home_xg_avg = get_team_strength(home_team, date)
    away_goals_avg, away_xg_avg = get_team_strength(away_team, date)

    home_win_pct, draw_pct, away_win_pct = get_head_to_head(home_team, away_team, date)

    features.append({
        'date': date,
        'home_team': home_team,
        'away_team': away_team,
        'home_form': home_form,
        'away_form': away_form,
        'home_goals_avg': home_goals_avg,
        'away_goals_avg': away_goals_avg,
        'home_xg_avg': home_xg_avg,
        'away_xg_avg': away_xg_avg,
        'h2h_home_win_pct': home_win_pct,
        'h2h_draw_pct': draw_pct,
        'h2h_away_win_pct': away_win_pct,
        'actual_result': match['result']
    })

# Create feature DataFrame
feature_df = pd.DataFrame(features)
print(f"\nCreated {len(feature_df)} feature rows")

# Clean up - remove rows with missing values
feature_df = feature_df.dropna()
print(f"After removing missing values: {len(feature_df)} rows")

# Save features
feature_df.to_csv('matches_features.csv', index=False)
print("Saved to 'matches_features.csv'")

# Show sample
print("\nSample features:")
print(feature_df.head())

print("\nFeature statistics:")
print(feature_df.describe())