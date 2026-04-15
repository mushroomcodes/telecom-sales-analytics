import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# PAGE CONFIG
st.set_page_config(
    page_title="Sales Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DESIGN SYSTEM
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* BASE */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0a0e1a;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #0f1420;
    border-right: 1px solid #1e2535;
}

[data-testid="stSidebar"] .stRadio label {
    color: #8892a4;
    font-size: 0.85rem;
    padding: 6px 0;
    transition: color 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #00d4ff;
}

/* METRIC CARDS */
[data-testid="metric-container"] {
    background: #0f1420;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 20px 24px;
    transition: border-color 0.2s, transform 0.2s;
}

[data-testid="metric-container"]:hover {
    border-color: #00d4ff40;
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: #8892a4 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* HEADINGS */
h1 {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.02em !important;
}

h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2535;
    border-radius: 12px;
    overflow: hidden;
}

/* SELECTBOX */
[data-testid="stSelectbox"] > div > div {
    background: #0f1420;
    border: 1px solid #1e2535;
    border-radius: 8px;
    color: #e2e8f0;
}

/* DIVIDER */
hr {
    border-color: #1e2535 !important;
}

/* PLOTLY CHART CONTAINERS */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* SIDEBAR NAV PILLS */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 4px;
}

/* PAGE HEADER ACCENT */
.page-header {
    border-left: 3px solid #00d4ff;
    padding-left: 14px;
    margin-bottom: 24px;
}

.page-header h1 {
    margin: 0 !important;
}

.page-header p {
    color: #8892a4;
    font-size: 0.85rem;
    margin: 4px 0 0 0;
}

/* STAT BADGE */
.stat-badge {
    display: inline-block;
    background: #00d4ff15;
    color: #00d4ff;
    border: 1px solid #00d4ff30;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}

.stat-badge.green {
    background: #00ff8815;
    color: #00ff88;
    border-color: #00ff8830;
}

.stat-badge.red {
    background: #ff475715;
    color: #ff4757;
    border-color: #ff475730;
}
</style>
""", unsafe_allow_html=True)

# PLOTLY DARK TEMPLATE 
PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='#0f1420',
        plot_bgcolor='#0f1420',
        font=dict(family='DM Sans', color='#8892a4'),
        title=dict(font=dict(color='#e2e8f0', size=14)),
        xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535', tickcolor='#8892a4'),
        yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535', tickcolor='#8892a4'),
        legend=dict(bgcolor='#0f1420', bordercolor='#1e2535', borderwidth=1),
        coloraxis_colorbar=dict(tickcolor='#8892a4', outlinecolor='#1e2535')
    )
)

ACCENT = '#00d4ff'
GREEN = '#00ff88'
ORANGE = '#ff9f43'
RED = '#ff4757'
PURPLE = '#a29bfe'
COLOURS = [ACCENT, GREEN, ORANGE, RED, PURPLE, '#fd79a8']

# TARGETS 
TARGETS = {
    'hh_value': 28.50,
    'bb_conv': 0.1800,
    'mob_conv': 0.1420,
    'tv_conv': 0.0580,
    'combined_conv': 0.3220
}

# DATA 
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['total_new_sales'] = df['broadband'] + df['mobile'] + df['tv']
    df['total_transactions'] = df['broadband'] + df['mobile'] + df['tv'] + df['regrades']
    df['bb_conv'] = df['broadband'] / df['calls'].replace(0, np.nan)
    df['mob_conv'] = df['mobile'] / df['calls'].replace(0, np.nan)
    df['tv_conv'] = df['tv'] / df['calls'].replace(0, np.nan)
    df['combined_conv'] = (df['broadband'] + df['mobile']) / df['calls'].replace(0, np.nan)
    df['avg_hh_value'] = df['hh_value'] / df['hh_orders'].replace(0, np.nan)
    df['products_per_hh'] = df['total_transactions'] / df['hh_orders'].replace(0, np.nan)
    df['value_per_call'] = df['hh_value'] / df['calls'].replace(0, np.nan)
    return df

@st.cache_data
def build_monthly(df):
    monthly = df.groupby('team').agg(
        total_calls=('calls', 'sum'),
        total_broadband=('broadband', 'sum'),
        total_mobile=('mobile', 'sum'),
        total_tv=('tv', 'sum'),
        total_regrades=('regrades', 'sum'),
        total_hh_orders=('hh_orders', 'sum'),
        total_hh_value=('hh_value', 'sum'),
        total_new_sales=('total_new_sales', 'sum'),
        total_transactions=('total_transactions', 'sum'),
    ).reset_index()
    monthly['bb_conv'] = monthly['total_broadband'] / monthly['total_calls']
    monthly['mob_conv'] = monthly['total_mobile'] / monthly['total_calls']
    monthly['tv_conv'] = monthly['total_tv'] / monthly['total_calls']
    monthly['combined_conv'] = (monthly['total_broadband'] + monthly['total_mobile']) / monthly['total_calls']
    monthly['avg_hh_value'] = monthly['total_hh_value'] / monthly['total_hh_orders']
    monthly['products_per_hh'] = monthly['total_transactions'] / monthly['total_hh_orders']
    monthly['value_per_call'] = monthly['total_hh_value']  / monthly['total_calls']
    monthly['bb_vs_target'] = monthly['bb_conv'] - TARGETS['bb_conv']
    monthly['mob_vs_target'] = monthly['mob_conv'] - TARGETS['mob_conv']
    monthly['tv_vs_target'] = monthly['tv_conv'] - TARGETS['tv_conv']
    monthly['combined_vs_target'] = monthly['combined_conv'] - TARGETS['combined_conv']
    monthly['hh_value_vs_target'] = monthly['avg_hh_value'] - TARGETS['hh_value']
    return monthly

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = load_data(os.path.join(BASE_DIR, 'data', 'clean', 'sales_data_synthetic.csv'))
monthly = build_monthly(df)

# SIDEBAR 
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0'>
        <div style='font-size:1.2rem; font-weight:700; color:#ffffff; letter-spacing:-0.02em'>
            Sales Teams Analytics
        </div>
        <div style='font-size:0.75rem; color:#8892a4; margin-top:4px'>
            Sales Centre Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Leaderboard", "Daily Trends",
         "Target Performance", "Team Deep Dive", "ML Insights"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='margin:20px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.72rem; color:#8892a4; line-height:1.8'>
        <div><span style='color:#ffffff; font-weight:600'>Period</span> &nbsp; Days 1–{df['day'].max()}</div>
        <div><span style='color:#ffffff; font-weight:600'>Teams</span> &nbsp; {df['team'].nunique()} active</div>
        <div><span style='color:#ffffff; font-weight:600'>Records</span> &nbsp; {len(df):,} daily rows</div>
    </div>
    """, unsafe_allow_html=True)

# HELPERS
def page_header(title, subtitle=""):
    st.markdown(f"""
    <div class='page-header'>
        <h1>{title}</h1>
        {'<p>' + subtitle + '</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def styled_chart(fig, height=420):
    fig.update_layout(**PLOT_TEMPLATE['layout'], height=height, margin=dict(l=16, r=16, t=40, b=16))
    st.plotly_chart(fig, use_container_width=True)

# PAGE 1: OVERVIEW 
if page == "Overview":
    page_header("Centre Overview", "Monthly performance summary across all teams - All numbers and names in this dashboard are fake to protect real data")

    total_value = monthly['total_hh_value'].sum()
    total_bb = monthly['total_broadband'].sum()
    total_mob = monthly['total_mobile'].sum()
    total_tv = monthly['total_tv'].sum()
    total_hh = monthly['total_hh_orders'].sum()
    centre_comb_conv = (monthly['total_broadband'].sum() + monthly['total_mobile'].sum()) / monthly['total_calls'].sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total HH Value", f"£{total_value:,.0f}")
    c2.metric("Combined Conv Rate", f"{centre_comb_conv:.1%}",
              delta=f"{centre_comb_conv - TARGETS['combined_conv']:+.1%} vs target")
    c3.metric("Broadband Sold", f"{total_bb:,.0f}")
    c4.metric("Mobile Sold", f"{total_mob:,.0f}")
    c5.metric("TV Sold", f"{total_tv:,.0f}")
    c6.metric("HH Orders", f"{total_hh:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Product Mix by Team")
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Broadband', x=monthly['team'],
                             y=monthly['total_broadband'], marker_color=ACCENT))
        fig.add_trace(go.Bar(name='Mobile', x=monthly['team'],
                             y=monthly['total_mobile'],    marker_color=GREEN))
        fig.add_trace(go.Bar(name='TV', x=monthly['team'],
                             y=monthly['total_tv'], marker_color=ORANGE))
        fig.update_layout(barmode='group', xaxis_tickangle=-30,
                          legend=dict(orientation='h', y=1.1))
        styled_chart(fig, height=380)

    with col_right:
        st.markdown("#### Revenue Share by Team")
        fig2 = px.pie(
            monthly, values='total_hh_value', names='team',
            color_discrete_sequence=COLOURS,
            hole=0.6
        )
        fig2.update_traces(textposition='outside', textinfo='label+percent',
                           textfont_color='#8892a4')
        fig2.update_layout(showlegend=False)
        styled_chart(fig2, height=380)

    st.markdown("#### Daily HH Value Heatmap")
    heatmap_data = df.pivot_table(index='team', columns='day', values='hh_value', aggfunc='sum')
    fig3 = px.imshow(
        heatmap_data,
        color_continuous_scale='Blues',
        labels=dict(x="Day", y="Team", color="HH Value (£)"),
        aspect='auto', text_auto='.0f'
    )
    fig3.update_traces(textfont_size=8)
    styled_chart(fig3, height=340)

# PAGE 2: LEADERBOARD 
elif page == "Leaderboard":
    page_header("Team Leaderboard", "Rank teams across any performance metric")

    METRIC_OPTIONS = {
        "Total HH Value (£)": "total_hh_value",
        "BB Conversion Rate": "bb_conv",
        "Mobile Conversion Rate": "mob_conv",
        "TV Conversion Rate": "tv_conv",
        "Combined BB+Mob Conv": "combined_conv",
        "Avg HH Value (£)": "avg_hh_value",
        "Products per Household": "products_per_hh",
        "Value per Call (£)": "value_per_call"
    }

    selected_label = st.selectbox("Rank by", list(METRIC_OPTIONS.keys()))
    selected_metric = METRIC_OPTIONS[selected_label]
    ranked = monthly.sort_values(selected_metric, ascending=True)

    fig = px.bar(
        ranked, x=selected_metric, y='team',
        orientation='h',
        color=selected_metric,
        color_continuous_scale=[[0, '#1e2535'], [1, ACCENT]],
        labels={selected_metric: selected_label, 'team': ''},
        text=selected_metric
    )
    fig.update_traces(
        texttemplate='%{text:.2f}' if 'conv' in selected_metric or 'per' in selected_metric.lower() else '%{text:,.0f}',
        textposition='outside',
        textfont_color='#e2e8f0'
    )
    fig.update_layout(coloraxis_showscale=False)
    styled_chart(fig, height=420)

    st.markdown("#### Full Monthly Summary")
    display = monthly[[
        'team', 'total_hh_value', 'bb_conv', 'mob_conv',
        'tv_conv', 'combined_conv', 'avg_hh_value', 'products_per_hh', 'value_per_call'
    ]].sort_values('total_hh_value', ascending=False).reset_index(drop=True)
    display.index += 1
    display.columns = ['Team', 'HH Value', 'BB Conv', 'Mob Conv',
                       'TV Conv', 'Combined', 'Avg HH £', 'Prod/HH', 'Val/Call']
    st.dataframe(display.style.format({
        'HH Value': '£{:,.2f}',
        'BB Conv': '{:.1%}', 'Mob Conv': '{:.1%}',
        'TV Conv': '{:.1%}', 'Combined': '{:.1%}',
        'Avg HH £': '£{:.2f}',
        'Prod/HH': '{:.2f}', 'Val/Call': '£{:.2f}'
    }), use_container_width=True)   

# PAGE 3: DAILY TRENDS 
elif page == "Daily Trends":
    page_header("Daily Trends", "Sales output over the reporting period by team or centre-wide")

    col1, col2 = st.columns([2, 1])
    with col1:
        teams       = ['All Teams'] + sorted(df['team'].unique().tolist())
        sel_team    = st.selectbox("Team", teams)
    with col2:
        metric_opts = {'HH Value (£)': 'hh_value', 'Broadband': 'broadband',
                       'Mobile': 'mobile', 'TV': 'tv'}
        sel_metric  = st.selectbox("Highlight metric", list(metric_opts.keys()))

    if sel_team == 'All Teams':
        plot_df = df.groupby('day')[['broadband', 'mobile', 'tv', 'hh_value']].sum().reset_index()
    else:
        plot_df = df[df['team'] == sel_team].sort_values('day')

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Product Sales by Day', f'{sel_metric} by Day'),
        row_heights=[0.55, 0.45], vertical_spacing=0.14
    )
    fig.add_trace(go.Scatter(x=plot_df['day'], y=plot_df['broadband'],
                             name='Broadband', line=dict(color=ACCENT, width=2),
                             mode='lines+markers', marker=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['day'], y=plot_df['mobile'],
                             name='Mobile', line=dict(color=GREEN, width=2),
                             mode='lines+markers', marker=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['day'], y=plot_df['tv'],
                             name='TV', line=dict(color=ORANGE, width=2),
                             mode='lines+markers', marker=dict(size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['day'], y=plot_df[metric_opts[sel_metric]],
                             name=sel_metric, line=dict(color=PURPLE, width=2),
                             fill='tozeroy', fillcolor='rgba(162, 155, 254, 0.12)',
                             mode='lines+markers', marker=dict(size=5)), row=2, col=1)
    fig.update_xaxes(gridcolor='#1e2535', linecolor='#1e2535')
    fig.update_yaxes(gridcolor='#1e2535', linecolor='#1e2535')
    styled_chart(fig, height=560)

# PAGE 4: TARGET PERFORMANCE
elif page == "Target Performance":
    page_header("Target Performance", "How each team tracks against monthly targets")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BB Target", f"{TARGETS['bb_conv']:.1%}")
    c2.metric("Mobile Target", f"{TARGETS['mob_conv']:.1%}")
    c3.metric("TV Target", f"{TARGETS['tv_conv']:.1%}")
    c4.metric("Combined Target", f"{TARGETS['combined_conv']:.1%}")
    c5.metric("HH Value Target", f"£{TARGETS['hh_value']:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Variance vs Target (% above or below)")
    st.caption("Each cell shows how far above or below target as a % of that target. Green = above, Red = below.")

    variance_data = monthly.set_index('team').copy()
    normalised = pd.DataFrame({
        'BB': (variance_data['bb_conv'] - TARGETS['bb_conv']) / TARGETS['bb_conv'] * 100,
        'Mobile': (variance_data['mob_conv'] - TARGETS['mob_conv']) / TARGETS['mob_conv'] * 100,
        'TV': (variance_data['tv_conv'] - TARGETS['tv_conv']) / TARGETS['tv_conv'] * 100,
        'Combined': (variance_data['combined_conv'] - TARGETS['combined_conv'])/ TARGETS['combined_conv'] * 100,
        'HH Value': (variance_data['avg_hh_value'] - TARGETS['hh_value']) / TARGETS['hh_value'] * 100,
    })

    fig = px.imshow(
        normalised,
        color_continuous_scale=[[0, RED], [0.5, '#1e2535'], [1, GREEN]],
        color_continuous_midpoint=0,
        aspect='auto', text_auto='.1f',
        labels=dict(color="% vs Target")
    )
    fig.update_traces(textfont_size=11)
    styled_chart(fig, height=400)

# PAGE 5: TEAM DEEP DIVE 
elif page == "Team Deep Dive":
    page_header("Team Deep Dive", "Detailed performance breakdown for a single team")

    selected = st.selectbox("Select team", sorted(df['team'].unique()))
    team_df = df[df['team'] == selected].sort_values('day').copy()
    team_monthly = monthly[monthly['team'] == selected].iloc[0]

    # KPI CARDS
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total HH Value", f"£{team_monthly['total_hh_value']:,.2f}")
    c2.metric("BB Conversion", f"{team_monthly['bb_conv']:.1%}",
              delta=f"{team_monthly['bb_vs_target']:+.1%} vs target")
    c3.metric("Mobile Conv", f"{team_monthly['mob_conv']:.1%}",
              delta=f"{team_monthly['mob_vs_target']:+.1%} vs target")
    c4.metric("Products / HH", f"{team_monthly['products_per_hh']:.2f}")
    c5.metric("Value / Call", f"£{team_monthly['value_per_call']:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    # DAILY PRODUCT TREND
    with col_left:
        st.markdown("#### Daily Product Sales")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=team_df['day'], y=team_df['broadband'],
            name='Broadband', mode='lines+markers',
            line=dict(color=ACCENT, width=2), marker=dict(size=5),
            fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.06)'
        ))
        fig.add_trace(go.Scatter(
            x=team_df['day'], y=team_df['mobile'],
            name='Mobile', mode='lines+markers',
            line=dict(color=GREEN, width=2), marker=dict(size=5),
            fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.06)'
        ))
        fig.add_trace(go.Scatter(
            x=team_df['day'], y=team_df['tv'],
            name='TV', mode='lines+markers',
            line=dict(color=ORANGE, width=2), marker=dict(size=5),
            fill='tozeroy', fillcolor='rgba(255, 159, 67, 0.06)'
        ))
        fig.update_layout(legend=dict(orientation='h', y=1.1))
        styled_chart(fig, height=360)

    # PRODUCT MIX DONUT
    with col_right:
        st.markdown("#### Product Mix")
        mix_labels = ['Broadband', 'Mobile', 'TV']
        mix_values = [
            team_monthly['total_broadband'],
            team_monthly['total_mobile'],
            team_monthly['total_tv']
        ]
        fig2 = go.Figure(go.Pie(
            labels=mix_labels,
            values=mix_values,
            hole=0.62,
            marker=dict(colors=[ACCENT, GREEN, ORANGE]),
            textinfo='label+percent',
            textfont=dict(color='#e2e8f0', size=12),
            hovertemplate='%{label}: %{value:,.0f} units<extra></extra>'
        ))
        fig2.add_annotation(
            text=f"<b>{int(sum(mix_values)):,}</b><br><span style='font-size:10px'>total</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color='#ffffff', size=18)
        )
        fig2.update_layout(showlegend=False)
        styled_chart(fig2, height=360)

    # DAILY HH VALUE BAR
    st.markdown("#### Daily HH Value")
    fig3 = go.Figure(go.Bar(
        x=team_df['day'],
        y=team_df['hh_value'],
        marker=dict(
            color=team_df['hh_value'],
            colorscale=[[0, '#1e2535'], [1, ACCENT]],
            showscale=False
        ),
        hovertemplate='Day %{x}: £%{y:,.2f}<extra></extra>'
    ))
    fig3.add_hline(
        y=team_monthly['total_hh_value'] / df['day'].max(),
        line_dash='dash', line_color='#8892a4',
        annotation_text='Daily average',
        annotation_font_color='#8892a4'
    )
    styled_chart(fig3, height=280)

# PAGE 6: ML INSIGHTS 
elif page == "ML Insights":
    page_header("ML Insights", "K-means clustering (k=3) — teams grouped by performance profile")

    features = ['bb_conv', 'mob_conv', 'tv_conv', 'avg_hh_value', 'products_per_hh', 'value_per_call']
    X = monthly[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    monthly['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_names = {0: 'Efficient Converters', 1: 'Core Performers', 2: 'High Bundlers'}
    cluster_cols  = {
        'Efficient Converters': ACCENT,
        'Core Performers': GREEN,
        'High Bundlers': ORANGE
    }
    monthly['cluster_label'] = monthly['cluster'].map(cluster_names)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Team Clusters — Value Efficiency vs Bundling Rate")
        fig = px.scatter(
            monthly, x='value_per_call', y='products_per_hh',
            color='cluster_label', text='team',
            color_discrete_map=cluster_cols,
            labels={'value_per_call': 'Value per Call (£)', 'products_per_hh': 'Products per Household',
                    'cluster_label': 'Cluster'},
        )
        fig.update_traces(textposition='top center', marker=dict(size=16),
                          textfont=dict(color='#e2e8f0', size=11))
        fig.update_layout(legend=dict(orientation='h', y=-0.15))
        styled_chart(fig, height=440)

    with col_right:
        st.markdown("#### Cluster Descriptions")
        descriptions = {
            'Efficient Converters': ('Highest value per call. Convert calls into revenue most efficiently.', ACCENT),
            'Core Performers': ('Solid across all metrics. Largest cluster and the centre baseline.', GREEN),
            'High Bundlers': ('Exceptional products per HH. Maximise value per household interaction.', ORANGE)
        }
        for label, (desc, colour) in descriptions.items():
            teams_in_cluster = monthly[monthly['cluster_label'] == label]['team'].tolist()
            st.markdown(f"""
            <div style='background:#0a0e1a; border:1px solid #1e2535; border-left:3px solid {colour};
                        border-radius:8px; padding:16px; margin-bottom:12px'>
                <div style='color:{colour}; font-size:0.75rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em'>{label}</div>
                <div style='color:#e2e8f0; font-size:0.85rem; margin:6px 0'>{desc}</div>
                <div style='color:#8892a4; font-size:0.75rem'>{' · '.join(teams_in_cluster)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Cluster Profiles")
        profiles = monthly.groupby('cluster_label')[['bb_conv', 'mob_conv', 'tv_conv',
                                                      'avg_hh_value', 'products_per_hh',
                                                      'value_per_call']].mean().round(3)
        st.dataframe(profiles, use_container_width=True)