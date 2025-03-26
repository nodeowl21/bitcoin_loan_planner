import streamlit as st
import datetime
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("🔁 BTC Loan Planner")

# ---------- State Helper ----------

def get_state_value(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# ---------- Fetch Live Price ----------
def get_live_btc_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        data = response.json()
        return data["bitcoin"]["usd"]
    except:
        return None

# ---------- Simulated Price Path ----------
def generate_random_walk(years=5, annual_return=0.5, daily_volatility=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
    days = years * 365
    dt = 1 / 365
    mu = annual_return
    sigma = daily_volatility
    prices = [btc_price]
    for _ in range(days):
        shock = np.random.normal(loc=(mu * dt), scale=sigma * np.sqrt(dt))
        prices.append(prices[-1] * np.exp(shock))
    start = datetime.date.today()
    dates = pd.date_range(start=start, periods=days + 1)
    return pd.DataFrame({'price': prices}, index=dates)

# ---------- 📋 Loan Setup ----------
st.header("📋 Setup Loan Plan")

live_price = get_live_btc_price()
default_price = live_price if live_price else 50000

btc_owned = st.number_input("BTC Holdings", value=get_state_value("btc_owned", 1.0), key="btc_owned")
btc_price = st.number_input("BTC Price (USD)", value=get_state_value("btc_price", default_price), key="btc_price")
ltv = st.slider("Target LTV (%)", 10, 90, get_state_value("ltv", 20), key="ltv") / 100

max_rebalance_threshold_sell = round(max(0.001, 1.0 - ltv - 0.01), 3)
rebalance_threshold_sell = st.slider("Rebalancing Threshold for Sell (%)", 1, int(max_rebalance_threshold_sell * 100),
                                     get_state_value("rebalance_sell", 20), key="rebalance_sell") / 100
rebalance_threshold_buy = st.slider("Rebalancing Threshold for Buy (%)", 1, 50,
                                    get_state_value("rebalance_buy", 10), key="rebalance_buy") / 100
interest_rate = st.slider("Loan Interest Rate (% p.a.)", 0, 20, get_state_value("interest", 10), key="interest") / 100

safe_loan = (ltv * btc_owned * btc_price) / (1 - ltv)
btc_bought = safe_loan / btc_price
total_btc = btc_owned + btc_bought
yearly_interest = safe_loan * interest_rate

st.subheader("💰 Initial Loan Summary")
st.markdown(f"- **Loan Amount (USD):** `${safe_loan:,.2f}`")
st.markdown(f"- **BTC Purchased with Loan:** `{btc_bought:.6f}` BTC")
st.markdown(f"- **Total BTC after Loan:** `{total_btc:.6f}` BTC")
st.markdown(f"- **Annual Interest:** `${yearly_interest:,.2f}`")

# ---------- 📈 Simulation ----------
st.header("📈 Price Simulation & Rebalancing")

loan = {
    'btc_owned': btc_owned,
    'btc_price': btc_price,
    'rebalance_threshold_sell': rebalance_threshold_sell,
    'rebalance_threshold_buy': rebalance_threshold_buy,
    'interest_rate': interest_rate,
    'safe_loan': safe_loan,
    'btc_bought': btc_bought,
    'total_btc': total_btc,
    'zinsen_fix': 0.0
}

sim_mode = st.radio("Choose Price Source", ["Historical", "Generated"], key="sim_mode")

if sim_mode == "Generated":
    years = st.slider("Number of Simulation Years", 1, 20, get_state_value("sim_years", 5), key="sim_years")
    expected_return = st.slider("Expected Annual Return (%)", -100, 200,
                                get_state_value("exp_return", 50), key="exp_return") / 100
    volatility = st.slider("Daily Volatility (%)", 1, 30,
                           get_state_value("volatility", 5), key="volatility") / 100
    df = generate_random_walk(years=years, annual_return=expected_return, daily_volatility=volatility)
else:
    num_years = st.slider("Historical Timeframe (years)", 1, 10, get_state_value("sim_years", 5), key="sim_years")
    df_raw = pd.read_csv("btc-usd-max.csv")
    df_raw["snapped_at"] = pd.to_datetime(df_raw["snapped_at"])
    df = df_raw.set_index("snapped_at")["price"].sort_index()
    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(years=num_years)
    price_series = df.loc[start_date:end_date]
    price_rel = price_series / price_series.iloc[0]
    simulated_prices = price_rel * loan['btc_price']
    future_dates = pd.date_range(start=datetime.date.today(), periods=len(simulated_prices), freq='D')
    df = pd.DataFrame({'price': simulated_prices.values}, index=future_dates)

interval = st.selectbox("Rebalancing Interval", ["Daily", "Weekly", "Monthly", "Yearly"],
                        index=["Daily", "Weekly", "Monthly", "Yearly"].index(
                            get_state_value("interval", "Weekly")),
                        key="interval")

rebalance_days = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Yearly": 365}[interval]

# ---------- 🔄 Simulation Engine ----------
current_btc = loan['total_btc']
current_loan = loan['safe_loan']
zinsen_fix = 0.0
start_day = df.index[0].date()
data = []
rebalancing_log = []
liquidated = False

for i, date in enumerate(df.index):
    days_passed = (date.date() - start_day).days
    price = df.loc[date, 'price']

    zinsteil = current_loan * loan['interest_rate'] * days_passed / 365
    total_debt = current_loan + zinsen_fix + zinsteil
    collateral = current_btc * price
    current_ltv = total_debt / collateral if collateral > 0 else float('inf')
    rebalanced = False
    action = ""
    delta_btc = 0.0

    if not liquidated and i % rebalance_days == 0:
        abw = current_ltv - ltv
        if abw > loan['rebalance_threshold_sell']:
            D, P, B = total_debt, price, current_btc
            btc_to_sell = (D - ltv * B * P) / (P * (1 - ltv))
            btc_to_sell = max(0, btc_to_sell)

            if btc_to_sell > current_btc:
                rebalancing_log.append({
                    "Date": date.date(),
                    "Action": "Liquidation",
                    "BTC Δ": f'{-btc_to_sell:.6f} BTC',
                    "BTC after": f'{current_btc - btc_to_sell:.6f} BTC',
                    "Loan": f'{current_loan:.2f} $',
                    "Net BTC": f'{((current_btc - btc_to_sell) - (current_loan / price)):.6f} BTC',
                    "Price": f'{price:.2f} $',
                    "LTV": current_ltv
                })
                liquidated = True
                st.error(f"❌ Liquidation on {date.date()} – BTC collateral insufficient")
            else:
                current_btc -= btc_to_sell
                current_loan -= btc_to_sell * P
                zinsen_fix += zinsteil
                start_day = date.date()
                rebalanced = True
                action = "Sell"
                delta_btc = -btc_to_sell

        elif abw < -loan['rebalance_threshold_buy']:
            new_credit = (ltv * collateral - total_debt) / (1 - ltv)
            new_credit = max(0, new_credit)
            btc_to_buy = new_credit / price
            current_btc += btc_to_buy
            current_loan += new_credit
            zinsen_fix += zinsteil
            start_day = date.date()
            rebalanced = True
            action = "Buy"
            delta_btc = btc_to_buy

    if rebalanced:
        rebalancing_log.append({
            "Date": date.date(),
            "Action": action,
            "BTC Δ": f'{delta_btc:.6f} BTC',
            "BTC after": f'{current_btc:.6f} BTC',
            "Loan": f'{current_loan:.2f} $',
            "Net BTC": f'{(current_btc - (current_loan / price)):.6f} BTC',
            "Price": f'{price:.2f} $',
            "LTV": current_ltv
        })

    data.append({
        'Date': date,
        'Price': price,
        'BTC': current_btc,
        'Loan': total_debt,
        'Fixed Interest': zinsen_fix,
        'LTV': current_ltv
    })

results = pd.DataFrame(data).set_index('Date')
results["Net Worth"] = results["BTC"] * results["Price"] - results["Loan"]

# ---------- 📉 LTV Chart ----------
st.subheader("📉 LTV Development (Interactive)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=results.index, y=results['LTV'], mode='lines', name='LTV', hovertemplate=
"Date: %{x|%Y-%m-%d}<br>LTV: %{y:,.2f}"))
fig.add_trace(go.Scatter(
    x=results.index,
    y=[ltv] * len(results),
    mode='lines',
    name='Target LTV',
    line=dict(dash='dash'),
    hoverinfo='skip'  # No hover on this line
))
fig.add_trace(go.Scatter(
    x=results.index,
    y=results['Price'],
    mode='lines',
    name='BTC Price',
    yaxis='y2',
    line=dict(color='orange'),
    customdata=np.stack((
        results["BTC"],
        results["Loan"],
        results["Net Worth"]
    ), axis=-1),
    hovertemplate=
    "Date: %{x|%Y-%m-%d}<br>" +
    "BTC Price: $%{y:,.2f}<br>" +
    "BTC Holdings: %{customdata[0]:.6f} BTC<br>" +
    "Loan incl. interest: $%{customdata[1]:,.2f}<br>" +
    "Net Worth: $%{customdata[2]:,.2f}"
))

for _, row in pd.DataFrame(rebalancing_log).iterrows():
    fig.add_trace(go.Scatter(
        x=[row['Date']],
        y=[row['LTV']],
        mode='markers',
        name=row['Action'],
        marker=dict(
            size=12,
            symbol='x' if row['Action'] == 'Liquidation' else 'circle',
            color='red' if row['Action'] == 'Liquidation' else None
        ),
        hovertemplate=f"{row['Action']} on {row['Date']}<br>BTC Δ: {row['BTC Δ']}<br>Loan: ${row['Loan']}<br>Price: ${row['Price']}<br>LTV: {row['LTV']}"
    ))

fig.update_layout(
    yaxis=dict(title='LTV'),
    yaxis2=dict(title='BTC Price (USD)', overlaying='y', side='right'),
    title='LTV & BTC Price with Rebalancing Events',
    legend=dict(orientation="h", y=-0.2)
)

st.plotly_chart(fig, use_container_width=True)

# ---------- 📘 Rebalancing Log ----------
st.subheader("📘 Rebalancing Log")
st.dataframe(pd.DataFrame(rebalancing_log))

# ---------- 📊 Leverage Chart ----------
#st.subheader("📊 Leverage Development (Loan per BTC)")

#results["Loan per BTC"] = results["Loan"] / results["BTC"]

#fig_leverage = go.Figure()
#fig_leverage.add_trace(go.Scatter(
#    x=results.index,
#    y=results["Loan per BTC"],
#    mode='lines',
#    name='Loan per BTC',
#    line=dict(color='red')
#))

#fig_leverage.update_layout(
#    yaxis=dict(title='Loan per BTC (USD)'),
#    title='Leverage Over Time',
#    showlegend=True
#)

#st.plotly_chart(fig_leverage, use_container_width=True)