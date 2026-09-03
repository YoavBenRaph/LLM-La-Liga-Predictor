#Convert dataset to match-level format

import pandas as pd
import numpy as np
import unicodedata


def clean_team_name(name):
    """Remove accents and standardize team names"""
    if isinstance(name, str):
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
        name = ' '.join(name.split())
    return name


print("Loading data...")
df = pd.read_csv('matches_laliga.csv')

print(f"Raw data: {len(df)} rows")

# Filter only La Liga
df = df[df['comp'] == 'La Liga'].copy()

# DROP 2025 (incomplete season)
df = df[df['season'] != 2025].copy()
print(f"Seasons kept: {sorted(df['season'].unique())}")
print(f"Rows: {len(df)}")
print(f"Matches: {len(df) // 2}")

# Clean team names in the original data
df['team'] = df['team'].apply(clean_team_name)
df['opponent'] = df['opponent'].apply(clean_team_name)

# Create match-level data
matches = {}

for idx, row in df.iterrows():
    date = row['date']
    team = row['team']
    opponent = row['opponent']

    # Ensure consistent ordering
    team1 = min(team, opponent)
    team2 = max(team, opponent)
    match_key = f"{date}_{team1}_{team2}"

    if match_key not in matches:
        matches[match_key] = {
            'date': date,
            'season': row['season'],
            'team1': team1,
            'team2': team2,
            'team1_home': None,
            'team2_home': None,
            'team1_goals': 0,
            'team2_goals': 0,
            'team1_xg': 0,
            'team2_xg': 0,
            'team1_poss': 0,
            'team2_poss': 0,
            'team1_shots': 0,
            'team2_shots': 0,
            'team1_sot': 0,
            'team2_sot': 0,
            'result': None
        }

    # Store the data for this team's perspective
    if row['venue'] == 'Home':
        matches[match_key][f'{team}_home'] = True
        matches[match_key][f'{team}_goals'] = int(row['gf']) if pd.notna(row['gf']) else 0
        matches[match_key][f'{team}_xg'] = float(row['xg']) if pd.notna(row['xg']) else 0
        matches[match_key][f'{team}_poss'] = float(row['poss']) if pd.notna(row['poss']) else 0
        matches[match_key][f'{team}_shots'] = int(row['sh']) if pd.notna(row['sh']) else 0
        matches[match_key][f'{team}_sot'] = int(row['sot']) if pd.notna(row['sot']) else 0
    else:
        matches[match_key][f'{team}_home'] = False
        matches[match_key][f'{team}_goals'] = int(row['gf']) if pd.notna(row['gf']) else 0
        matches[match_key][f'{team}_xg'] = float(row['xg']) if pd.notna(row['xg']) else 0
        matches[match_key][f'{team}_poss'] = float(row['poss']) if pd.notna(row['poss']) else 0
        matches[match_key][f'{team}_shots'] = int(row['sh']) if pd.notna(row['sh']) else 0
        matches[match_key][f'{team}_sot'] = int(row['sot']) if pd.notna(row['sot']) else 0

# Convert to match-level DataFrame
match_list = []
for match_key, data in matches.items():
    # Determine which team is home
    if data.get(f"{data['team1']}_home") == True:
        home_team = data['team1']
        away_team = data['team2']
        home_goals = data.get(f"{data['team1']}_goals", 0)
        away_goals = data.get(f"{data['team2']}_goals", 0)
        home_xg = data.get(f"{data['team1']}_xg", 0)
        away_xg = data.get(f"{data['team2']}_xg", 0)
        home_poss = data.get(f"{data['team1']}_poss", 0)
        away_poss = data.get(f"{data['team2']}_poss", 0)
        home_shots = data.get(f"{data['team1']}_shots", 0)
        away_shots = data.get(f"{data['team2']}_shots", 0)
        home_sot = data.get(f"{data['team1']}_sot", 0)
        away_sot = data.get(f"{data['team2']}_sot", 0)
    elif data.get(f"{data['team2']}_home") == True:
        home_team = data['team2']
        away_team = data['team1']
        home_goals = data.get(f"{data['team2']}_goals", 0)
        away_goals = data.get(f"{data['team1']}_goals", 0)
        home_xg = data.get(f"{data['team2']}_xg", 0)
        away_xg = data.get(f"{data['team1']}_xg", 0)
        home_poss = data.get(f"{data['team2']}_poss", 0)
        away_poss = data.get(f"{data['team1']}_poss", 0)
        home_shots = data.get(f"{data['team2']}_shots", 0)
        away_shots = data.get(f"{data['team1']}_shots", 0)
        home_sot = data.get(f"{data['team2']}_sot", 0)
        away_sot = data.get(f"{data['team1']}_sot", 0)
    else:
        continue  # Skip if no home team info

    # Determine result
    if home_goals > away_goals:
        result = 'Home Win'
    elif home_goals < away_goals:
        result = 'Away Win'
    else:
        result = 'Draw'

    match_list.append({
        'date': data['date'],
        'season': data['season'],
        'home_team': home_team,
        'away_team': away_team,
        'home_goals': int(home_goals),
        'away_goals': int(away_goals),
        'home_xg': float(home_xg),
        'away_xg': float(away_xg),
        'home_poss': float(home_poss),
        'away_poss': float(away_poss),
        'home_shots': int(home_shots),
        'away_shots': int(away_shots),
        'home_sot': int(home_sot),
        'away_sot': int(away_sot),
        'result': result
    })

matches_df = pd.DataFrame(match_list)

print(f"\nCreated {len(matches_df)} matches")

# Verify Alaves vs Atletico Madrid
print("\nVerifying Alaves vs Atletico Madrid:")
alaves_atleti = matches_df[
    ((matches_df['home_team'] == 'Alaves') & (matches_df['away_team'] == 'Atletico Madrid')) |
    ((matches_df['home_team'] == 'Atletico Madrid') & (matches_df['away_team'] == 'Alaves'))
    ]
for _, match in alaves_atleti.iterrows():
    print(f"{match['date']}: {match['home_team']} {match['home_goals']} - {match['away_goals']} {match['away_team']}")

# Save
matches_df.to_csv('matches_prepared.csv', index=False)
print("\n Saved to 'matches_prepared.csv'")