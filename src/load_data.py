import os
import glob
import re
import pandas as pd

KEEP_COLS = {
    "Team": "team",
    "Calls": "calls",
    "New BB": "broadband",
    "New Mobile": "mobile",
    "New TV": "tv",
    "Other Transactions": "regrades",
    "HH Orders": "hh_orders",
    "Total HH Value": "hh_value"    
}

# FUNCTION - Extracting day number from file name (Day 2 = 2)
def parse_day_number(filename):
    match = re.search(r'[Dd]ay[\s_]?(\d+)', filename)
    return int(match.group(1)) if match else None

# FUNCTION - Removing £ and commas from currency strings
def clean_currency(value):
    if isinstance(value, str):
        return float(re.sub(r'[£,]', '', value.strip()))
    return value

# FUNCTION - Loading single day's CSV and return clean dataframe
def load_day_file(filepath):

    df = pd.read_csv(filepath, skiprows=1, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    # Keeping only necessary columns
    df = df[[col for col in KEEP_COLS.keys() if col in df.columns]]
    df = df.rename(columns=KEEP_COLS)

    # Dropping rows with no team name
    df = df.dropna(subset=['team'])
    df = df[df['team'].str.strip() != '']

    # Cleaning currency column
    df['hh_value'] = df['hh_value'].apply(clean_currency)

    # Converting numeric columns
    numeric_cols = ['calls', 'broadband', 'mobile', 'tv', 'regrades', 'hh_orders']
    existing_numeric = [col for col in numeric_cols if col in df.columns]
    df[existing_numeric] = df[existing_numeric].apply(pd.to_numeric, errors='coerce')

    # Adding day number
    day = parse_day_number(os.path.basename(filepath))
    df['day'] = day

    return df

# FUNCTION - Loading all days but skipping MTD and inactive teams 
def load_all_days(data_dir):
    # Only loading files with day number in file name
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    day_files = [f for f in all_files if parse_day_number(os.path.basename(f)) is not None]
    day_files = sorted(day_files, key=lambda f: parse_day_number(os.path.basename(f)))

    print(f"Found {len(day_files)} daily CSV files\n")

    dfs = []
    for filepath in day_files:
        try:
            day_df = load_day_file(filepath)
            dfs.append(day_df)
            print(f"Day {day_df['day'].iloc[0]:>2} - {len(day_df)} teams loaded")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(['day', 'team']).reset_index(drop=True)

    # Removing inactive teams from the dataset by checking the totals across the entire month
    # This is to ignore days a team doesn't work 
    team_totals = df_all.groupby('team')[['broadband', 'mobile', 'tv', 'hh_orders']].sum().sum(axis=1)
    inactive_teams = team_totals[team_totals == 0].index.tolist()

    if inactive_teams:
        print(f"\nRemoving inactive teams (zero new sales across full month): {inactive_teams}")
        df_all = df_all[~df_all['team'].isin(inactive_teams)]
    else:
        print("\nNo inactive teams found")

    print(f"\nFinal shape: {df_all.shape}")
    print(f"Teams: {sorted(df_all['team'].unique())}")
    print(f"Days: {sorted(df_all['day'].unique())}")
    
    return df_all