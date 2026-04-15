import pandas as pd
import numpy as np

np.random.seed(42)

SYNTHETIC_TEAMS = [
    'Alex Team',
    'Blair Team', 
    'Casey Team',
    'Dana Team',
    'Ellis Team',
    'Finlay Team',
    'Glen Team',
    'Harper Team'
]

TEAM_PROFILES = {
    'Alex Team':    {'calls': (180, 40), 'bb_rate': 0.170, 'mob_rate': 0.145, 'tv_rate': 0.062, 'hh_rate': 0.26, 'avg_hh_val': 32.0},
    'Blair Team':   {'calls': (180, 40), 'bb_rate': 0.170, 'mob_rate': 0.145, 'tv_rate': 0.062, 'hh_rate': 0.26, 'avg_hh_val': 32.0},
    'Casey Team':   {'calls': (180, 40), 'bb_rate': 0.170, 'mob_rate': 0.145, 'tv_rate': 0.062, 'hh_rate': 0.26, 'avg_hh_val': 32.0},
    'Dana Team':    {'calls': (180, 40), 'bb_rate': 0.170, 'mob_rate': 0.145, 'tv_rate': 0.062, 'hh_rate': 0.26, 'avg_hh_val': 32.0},
    'Ellis Team':   {'calls': (180, 40), 'bb_rate': 0.170, 'mob_rate': 0.145, 'tv_rate': 0.062, 'hh_rate': 0.26, 'avg_hh_val': 32.0},
    'Finlay Team':  {'calls': (180, 40), 'bb_rate': 0.170, 'mob_rate': 0.145, 'tv_rate': 0.062, 'hh_rate': 0.26, 'avg_hh_val': 32.0},
    'Glen Team':    {'calls': (180, 40), 'bb_rate': 0.170, 'mob_rate': 0.145, 'tv_rate': 0.062, 'hh_rate': 0.26, 'avg_hh_val': 32.0},
    'Harper Team':  {'calls': (180, 40), 'bb_rate': 0.170, 'mob_rate': 0.145, 'tv_rate': 0.062, 'hh_rate': 0.26, 'avg_hh_val': 32.0},
}

# Probability a team has a reduced staffing day (one or few advisors on shift)
# On these days calls and sales are much lower but never truly zero
LOW_STAFFING_PROB = 0.12  # roughly 1 in 8 days per team

rows = []
for day in range(1, 29):
    for team, profile in TEAM_PROFILES.items():

        # Determine if this is a low-staffing day for this team
        low_staffing = np.random.random() < LOW_STAFFING_PROB

        if low_staffing:
            # One or two advisors on shift - much lower volume
            calls = max(1, int(np.random.normal(profile['calls'][0] * 0.15, 10)))
        else:
            calls = max(0, int(np.random.normal(profile['calls'][0], profile['calls'][1])))

        # Generate sales from calls using conversion rates
        broadband = max(0, int(np.random.binomial(calls, profile['bb_rate'])))
        mobile    = max(0, int(np.random.binomial(calls, profile['mob_rate'])))
        tv        = max(0, int(np.random.binomial(calls, profile['tv_rate'])))
        regrades  = max(0, int(np.random.binomial(calls, 0.03)))

        # HH orders
        total_sales = broadband + mobile + tv
        hh_orders = max(0, int(total_sales * profile['hh_rate'] * np.random.uniform(0.8, 1.2)))
        hh_orders = min(hh_orders, total_sales)

        # HH value
        if hh_orders > 0:
            hh_value = round(hh_orders * profile['avg_hh_val'] * np.random.uniform(0.85, 1.15), 2)
        else:
            hh_value = 0.0

        rows.append({
            'team':      team,
            'day':       day,
            'calls':     calls,
            'broadband': broadband,
            'mobile':    mobile,
            'tv':        tv,
            'regrades':  regrades,
            'hh_orders': hh_orders,
            'hh_value':  hh_value
        })

df_synthetic = pd.DataFrame(rows)

df_synthetic.to_csv('data/clean/sales_data_synthetic.csv', index=False)
print(f"Synthetic dataset saved: {df_synthetic.shape}")
print(f"\nTeams: {sorted(df_synthetic['team'].unique())}")
print(f"\nSample:")
print(df_synthetic.head(16))