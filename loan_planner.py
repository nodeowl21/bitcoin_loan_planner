import datetime
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.title("🟠 Bitcoin Loan Planner")


# ---------- State Helper ----------

def get_state_value(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---------- Fetch Live Price ----------
def get_live_btc_price():
    try:
        currency = st.session_state.get("currency", "EUR").lower()
        url = f"https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies={currency}"
        response = requests.get(url, timeout=5)
        data = response.json()
        return data["bitcoin"][currency]
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


def get_strategy_config() -> dict:
    return {
        "ltv": st.session_state.get("ltv", 0.20),
        "rebalance_buy": st.session_state.get("rebalance_buy", 0.10),
        "rebalance_sell": st.session_state.get("rebalance_sell", 0.10),
        "rebalance_buy_factor": st.session_state.get("rebalance_buy_factor", 1.0),
        "rebalance_sell_factor": st.session_state.get("rebalance_sell_factor", 1.0),
        "enable_buy": st.session_state.get("enable_buy", True),
        "enable_sell": st.session_state.get("enable_sell", True),
        "ltv_relative_to_ath": st.session_state.get("ltv_relative_to_ath", False),
    }


st.markdown("""
This is a **Bitcoin Loan Planner** for simulating credit strategies aimed at accumulating more Bitcoin over time.
The core idea: BTC is purchased using borrowed capital, and added to the collateral securing the loan.
As part of the strategy, rebalancing actions can be simulated – selling BTC to reduce liquidation risk or using rising collateral value to accumulate more.
""")

# ---------- 📋 Loan Setup ----------
st.header("📋 Loan Plan Settings")

# ---------- 🎯 Strategy Presets ----------
preset_descriptions = {
    "Defensive HODL": "Minimal risk, no rebalancing. Loan is taken once and held. Ideal for conservative holders.",
    "Balanced Rebalancer": "Moderate LTV, active buy & sell rebalancing. Grows BTC stack with balanced risk.",
    "Aggressive Stacker": "High LTV with aggressive buy-ins and active rebalancing. Maximum exposure to upside.",
    "Crash Resilient": "Start with low leverage. Sell if LTV drifts too high. Designed to survive downturns by staying conservative and reducing risk."
}

if "strategy_presets" not in st.session_state:
    st.session_state["strategy_presets"] = {
        "Custom": {
            "ltv": 20,
            "enable_buy": True,
            "enable_sell": True,
            "rebalance_buy": 10,
            "rebalance_sell": 10
        },
        "Defensive HODL": {
            "ltv": 10,
            "enable_buy": False,
            "enable_sell": False,
        },
        "Balanced Rebalancer": {
            "ltv": 20,
            "rebalance_buy": 10,
            "rebalance_sell": 10,
            "enable_buy": True,
            "enable_sell": True,
        },
        "Aggressive Stacker": {
            "ltv": 35,
            "rebalance_buy": 5,
            "enable_buy": True,
            "enable_sell": False,
        },
        "Crash Resilient": {
            "ltv": 15,
            "rebalance_sell": 10,
            "enable_buy": False,
            "enable_sell": True,
        }
    }

live_price = get_live_btc_price()
default_price = live_price if live_price else 50000

st.sidebar.markdown("## 📂 Portfolio Summary")

st.subheader("Portfolio")

btc_owned_input = st.number_input("BTC Holdings", value=get_state_value("btc_owned", 1.0), step=0.1)
currency_input = st.selectbox("Currency", options=["EUR", "USD"],
                              index=["EUR", "USD"].index(get_state_value("currency", "EUR")))
currency_symbol = "$" if currency_input == "USD" else "€"

btc_price_input = st.number_input(
    f"BTC Price ({currency_symbol})",
    value=get_state_value("btc_price", default_price),
    step=1000
)

if st.button("📝 Save Portfolio"):
    st.session_state["btc_owned"] = btc_owned_input
    st.session_state["currency"] = currency_input
    st.session_state["btc_price"] = btc_price_input
    st.session_state["portfolio_saved"] = True

st.subheader("Existing Loans")

if "portfolio_loans" not in st.session_state:
    st.session_state["portfolio_loans"] = [{
        "platform": "Firefish",
        "amount": 10000.0,
        "interest": 5.0,
        "start_date": datetime.date(2025, 4, 25),
        "term_months": 12,
        "liquidation_ltv": 100}
    ]

new_platform = st.text_input("Platform / Lender", key="new_platform")
new_amount = st.number_input(f"Loan Amount ({currency_symbol})", key="new_amount", step=1000)
new_interest = st.number_input("Interest Rate (% p.a.)", min_value=0.0, max_value=50.0, value=5.0, key="new_interest",
                               step=0.1)
new_start = st.date_input("Start Date", value=datetime.date.today(), key="new_start")
term_mode = st.selectbox("Loan Term", ["Unlimited", "Set duration"])
if term_mode == "Set duration":
    new_term = st.number_input("Duration (months)", min_value=1, max_value=360, value=12, key="new_term")
else:
    new_term = None
new_liquidation_ltv = st.slider(
    "Liquidation LTV (%)", 50, 100,
    100,
    key="new_liquidation_ltv",
) / 100
if st.button("➕ Add Loan"):
    new_loan = {
        "platform": new_platform,
        "amount": new_amount,
        "interest": new_interest,
        "start_date": new_start,
        "term_months": new_term,
        "liquidation_ltv": new_liquidation_ltv}
    st.session_state["portfolio_loans"].append(new_loan)

if st.session_state["portfolio_loans"]:
    st.markdown("Current Loans")
    for i, loan in enumerate(st.session_state["portfolio_loans"]):
        cols = st.columns([1, 0.1])
        with cols[0]:
            if loan.get("term_months"):
                end_date = loan["start_date"] + pd.DateOffset(months=loan["term_months"])
                duration_str = f"{loan['term_months']} months (ends {end_date.date()})"
            else:
                duration_str = "Unlimited"

            st.markdown(
                f"**{loan['platform']}** – {currency_symbol}{loan['amount']:,.0f} at {loan['interest']}% p.a. — {duration_str}"
            )
        with cols[1]:
            if st.button("🗑️", key=f"delete_loan_{i}"):
                st.session_state["portfolio_loans"].pop(i)
                st.rerun()
btc_owned = st.session_state["btc_owned"]
total_btc = st.session_state["btc_owned"]
btc_price = st.session_state["btc_price"]
total_loan = 0
total_debt = 0

for loan in st.session_state.get("portfolio_loans", []):
    total_loan += loan["amount"]
    today = datetime.date.today()
    if loan.get("term_months"):
        end_date = loan["start_date"] + pd.DateOffset(months=loan["term_months"])
        effective_end = min(end_date.date(), today)
    else:
        effective_end = today
    days_passed = (effective_end - loan["start_date"]).days
    interest = loan["amount"] * loan["interest"] * days_passed / 365 / 100
    total_debt += loan["amount"] + interest

portfolio_value = total_btc * st.session_state["btc_price"]
ltv = total_debt / portfolio_value if portfolio_value > 0 else float("inf")

st.sidebar.metric("📈 Total BTC", f"{total_btc:.6f}")
st.sidebar.metric("💵 Total Debt", f"{currency_symbol}{total_debt:,.2f}")
st.sidebar.metric("📊 LTV", f"{ltv:.2%}")

st.subheader("Strategies")

strategy_presets = st.session_state["strategy_presets"]

selected_preset = st.selectbox(
    "Choose Preset",
    list(strategy_presets.keys()),
    key="preset_name"
)

preset_changed = st.session_state.get("preset_name") != st.session_state.get("last_preset")
preset_config = strategy_presets[st.session_state["preset_name"]]

if preset_changed:
    st.session_state["last_preset"] = st.session_state["preset_name"]
    st.session_state["ltv_input"] = int(preset_config.get("ltv", 20))
    st.session_state["ltv_relative_to_ath_input"] = preset_config.get("ltv_relative_to_ath", False)
    st.session_state["enable_sell_input"] = preset_config.get("enable_sell", True)
    st.session_state["rebalance_sell_input"] = int(preset_config.get("rebalance_sell", 20))
    st.session_state["rebalance_sell_factor_input"] = int(preset_config.get("rebalance_sell_factor", 100))
    st.session_state["enable_buy_input"] = preset_config.get("enable_buy", True)
    st.session_state["rebalance_buy_input"] = int(preset_config.get("rebalance_buy", 10))
    st.session_state["rebalance_buy_factor_input"] = int(preset_config.get("rebalance_buy_factor", 100))
    st.session_state["strategy_name_input"] = st.session_state["preset_name"]

ltv_input = st.slider(
    "Target LTV (%)", 1, 100,
    int(preset_config.get("ltv", 20)),
    key="ltv_input"
)

ltv_relative_to_ath_input = st.checkbox(
    "Rebalance LTV relative to BTC All-Time-High",
    value=preset_config.get("ltv_relative_to_ath", False),
    key="ltv_relative_to_ath_input"
)

enable_sell_input = st.checkbox("Enable Sell-Rebalancing", value=preset_config.get("enable_sell", True),
                                key="enable_sell_input")
if enable_sell_input:
    max_rebalance_threshold_sell = round(max(0.001, 1.0 - ltv_input / 100 - 0.01), 3)
    rebalance_sell_input = st.slider(
        "Sell Threshold (%)",
        1, int(max_rebalance_threshold_sell * 100),
        int(preset_config.get("rebalance_sell", 20)),
        key="rebalance_sell_input"
    )
    rebalance_sell_factor_input = st.slider(
        "Sell Rebalancing Intensity (%)",
        1, 100,
        int(preset_config.get("rebalance_sell_factor", 100)),
        key="rebalance_sell_factor_input"
    )
else:
    rebalance_sell_input = 0
    rebalance_sell_factor_input = 1.0

enable_buy_input = st.checkbox("Enable Buy-Rebalancing", value=preset_config.get("enable_buy", True),
                               key="enable_buy_input")
if enable_buy_input:
    max_buy_threshold = ltv_input - 1
    rebalance_buy_input = st.slider(
        "Buy Threshold (%)",
        1, max_buy_threshold,
        int(preset_config.get("rebalance_buy", min(10, max_buy_threshold))),
        key="rebalance_buy_input"
    )
    rebalance_buy_factor_input = st.slider(
        "Buy Rebalancing Intensity (%)",
        1, 100,
        int(preset_config.get("rebalance_buy_factor", 100)),
        key="rebalance_buy_factor_input"
    )
else:
    rebalance_buy_input = 0
    rebalance_buy_factor_input = 1.0

strategy_name_input = st.text_input(
    label="Name",
    value=st.session_state["preset_name"],
    key="strategy_name_input",
    placeholder="Preset name"
)

left_col, right_col = st.columns([1, 1])

with left_col:
    col_save_delete = st.columns(2)
    with col_save_delete[0]:
        if st.button("💾 Save Preset"):
            name = strategy_name_input.strip()
            if name:
                st.session_state["strategy_presets"][name] = {
                    "ltv": ltv_input,
                    "ltv_relative_to_ath": ltv_relative_to_ath_input,
                    "enable_sell": enable_sell_input,
                    "rebalance_sell": rebalance_sell_input,
                    "rebalance_sell_factor": rebalance_sell_factor_input,
                    "enable_buy": enable_buy_input,
                    "rebalance_buy": rebalance_buy_input,
                    "rebalance_buy_factor": rebalance_buy_factor_input
                }
                st.session_state["preset_to_select"] = name
                st.session_state["last_preset"] = None
                st.rerun()
    with col_save_delete[1]:
        if strategy_name_input not in ["Custom"]:
            if st.button("🗑️ Delete Preset"):
                del st.session_state["strategy_presets"][strategy_name_input]
                st.session_state["preset_to_select"] = "Custom"
                st.session_state["last_preset"] = None
                st.rerun()
with right_col:
    with st.container():
        col_imp_exp = st.columns(2)
        with col_imp_exp[0]:
            export_data = json.dumps(st.session_state["strategy_presets"], indent=2)
            st.download_button(
                label="⬇️ Export Presets",
                data=export_data,
                file_name="all_presets.json",
                mime="application/json",
            )
        with col_imp_exp[1]:
            if st.toggle("⬆️ Import Presets", value=False):
                import_file = st.file_uploader("Upload JSON file", type="json", key="file_uploader")
                if import_file is not None:
                    try:
                        imported_presets = json.load(import_file)
                        st.session_state["strategy_presets"].update(imported_presets)
                        del st.session_state["file_uploader"]
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to import presets: {e}")

df_raw = pd.read_csv("btc-usd-max.csv")
btc_ath = df_raw["price"].max()

# ---------- 📈 Simulation ----------
st.header("📈 Simulation & Rebalancing")

sim_mode = st.radio(
    "Choose Price Source",
    ["Historical", "Generated"],
    key="sim_mode",
    help="Choose between using historical BTC prices or a simulated random walk (based on expected return and volatility)."
)

if sim_mode == "Generated":
    years = st.slider("Number of Simulation Years", 1, 20, get_state_value("sim_years", 5), key="sim_years")
    expected_return = st.slider("Expected Annual Return (%)", -100, 200,
                                get_state_value("exp_return", 50), key="exp_return") / 100
    volatility = st.slider("Daily Volatility (%)", 1, 30,
                           get_state_value("volatility", 5), key="volatility") / 100
    df = generate_random_walk(years=years, annual_return=expected_return, daily_volatility=volatility)
else:
    num_years = st.slider("Historical Timeframe (years)", 1, 10, get_state_value("sim_years", 5), key="sim_years")
    df_raw["snapped_at"] = pd.to_datetime(df_raw["snapped_at"])
    df = df_raw.set_index("snapped_at")["price"].sort_index()
    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(years=num_years)
    price_series = df.loc[start_date:end_date]
    price_rel = price_series / price_series.iloc[0]
    simulated_prices = price_rel * btc_price
    future_dates = pd.date_range(start=datetime.date.today(), periods=len(simulated_prices), freq='D')
    df = pd.DataFrame({'price': simulated_prices.values}, index=future_dates)

interval = st.selectbox(
    "Rebalancing Interval",
    ["Daily", "Weekly", "Monthly", "Yearly"],
    index=["Daily", "Weekly", "Monthly", "Yearly"].index(get_state_value("interval", "Weekly")),
    key="interval",
    help="How often rebalancing is evaluated and potentially executed."
)

rebalance_days = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Yearly": 365}[interval]

interest_rate = st.number_input(
    "Loan Interest Rate (% p.a.)",
    min_value=0.0,
    max_value=20.0,
    value=get_state_value("interest", 12.5),
    step=0.1,
    key="interest"
) / 100

liquidation_ltv = st.slider(
    "Liquidation LTV (%)", 50, 100,
    get_state_value("liquidation_ltv", 100),
    key="liquidation_ltv",
    help="If the actual LTV exceeds this value, forced liquidation is triggered."
) / 100

selected_sim_strategy = st.selectbox(
    "Choose strategy for simulation:",
    list(strategy_presets.keys()),
    key="selected_sim_strategy"
)


# ---------- 🔄 Simulation Engine ----------
def run_simulation(config: dict, current_btc, price_df: pd.DataFrame, reference_value: float, loans: list):
    ltv = config.get("ltv", 0.2) / 100
    enable_buy = config.get("enable_buy", True)
    rebalance_buy = config.get("rebalance_buy", 100) / 100
    rebalance_buy_factor = config.get("rebalance_buy_factor", 100) / 100
    enable_sell = config.get("enable_sell", True)
    rebalance_sell = config.get("rebalance_sell", 100) / 100
    rebalance_sell_factor = config.get("rebalance_sell_factor", 100) / 100
    fixed_interest = 0.0
    data = []
    rebalancing_log = []
    liquidated = False
    active_loans = []
    for loan in loans:
        if loan.get("term_months"):
            end_date = loan["start_date"] + pd.DateOffset(months=loan["term_months"])
        else:
            end_date = None
        active_loans.append({
            **loan,
            "end_date": end_date,
            "paid": False,
            "accrued_interest": 0.0
        })

    total_debt = 0.0
    sim_start_date = price_df.index[0].date()

    for loan in active_loans:
        if loan.get("end_date") and loan["end_date"].date() < sim_start_date:
            continue

        if loan["start_date"] < sim_start_date:
            days_running = (sim_start_date - loan["start_date"]).days
            accrued = loan["amount"] * loan["interest"] * days_running / 365 / 100
            loan["accrued_interest"] += accrued
            total_debt += loan["amount"] + accrued
        else:
            total_debt += loan["amount"]

    st.markdown(total_debt)
    for i, date in enumerate(price_df.index):
        price = df.loc[date, 'price']

        for loan in active_loans:
            if loan["paid"] or not loan["end_date"]:
                continue
            if pd.Timestamp(date.date()) >= loan["end_date"]:
                repayment_amount = loan["amount"] + loan["accrued_interest"]
                btc_to_sell = repayment_amount / price
                current_btc -= btc_to_sell
                loan["paid"] = True
                action = f"Repay: {loan['platform']}"
                rebalancing_log.append({
                    "Date": date.date(),
                    "Action": action,
                    "BTC Δ": f"-{btc_to_sell:.6f} BTC",
                    "Price": f"{price:.2f} {currency_symbol}",
                    f"{currency_symbol} Spent": f'{- repayment_amount:.2f} {currency_symbol}',
                    "New Total BTC": f"{current_btc:.6f} BTC",
                    "New Total Debt": f"{(total_debt - repayment_amount):.2f} `{currency_symbol}`",
                    "LTV before": real_ltv,
                    "LTV after": real_ltv
                })
                total_debt -= repayment_amount

        accrued_interest = 0.0

        for loan in active_loans:
            if loan["paid"]:
                continue

            daily_interest = loan["amount"] * loan["interest"] / 365 / 100
            loan["accrued_interest"] += daily_interest
            accrued_interest += daily_interest
            total_debt += daily_interest

        fixed_interest += accrued_interest
        real_collateral = current_btc * price
        real_ltv = total_debt / real_collateral if real_collateral > 0 else float('inf')

        if config.get("ltv_relative_to_ath", False):
            reference_value = max(btc_ath, max(reference_value, price))
            rebalance_collateral = current_btc * reference_value
            rebalance_ltv = total_debt / rebalance_collateral if rebalance_collateral > 0 else float('inf')
        else:
            reference_value = price
            rebalance_collateral = real_collateral
            rebalance_ltv = real_ltv

        rebalanced = False
        action = ""
        delta_btc = 0.0

        if not liquidated:
            if real_ltv > liquidation_ltv:
                delta_btc = -current_btc
                current_btc = 0.0
                action = "Liquidation"
                liquidated = True
                rebalanced = True

            elif i % rebalance_days == 0:
                abw = rebalance_ltv - ltv

                if enable_sell and abw > rebalance_sell:
                    D, P, B = total_debt, reference_value, current_btc
                    adjusted_ltv = ltv + rebalance_sell * (1 - rebalance_sell_factor)
                    btc_to_sell = (D - adjusted_ltv * B * P) / (P * (1 - adjusted_ltv))
                    btc_to_sell = max(0, btc_to_sell)

                    if btc_to_sell > 0:
                        usd_available = btc_to_sell * reference_value

                        sorted_loans = sorted(
                            [loan for loan in active_loans if not loan["paid"]],
                            key=lambda l: (
                                -l["interest"], l["term_months"] if l.get("term_months") is not None else float('inf'))
                        )

                        for loan in sorted_loans:
                            if usd_available <= 0:
                                break
                            outstanding = loan["amount"] + loan["accrued_interest"]
                            repay_amount = min(usd_available, outstanding)
                            usd_available -= repay_amount
                            loan["amount"] -= repay_amount
                            if loan["amount"] <= 0.01:
                                loan["paid"] = True

                        current_btc -= btc_to_sell
                        total_debt -= (btc_to_sell * reference_value - usd_available)
                        rebalanced = True
                        action = "Sell"
                        delta_btc = -btc_to_sell

                elif enable_buy and abw < -rebalance_buy:
                    adjusted_ltv = ltv - rebalance_buy * (1 - rebalance_buy_factor)
                    new_credit = (adjusted_ltv * rebalance_collateral - total_debt) / (1 - adjusted_ltv)
                    new_credit = max(0, new_credit)
                    btc_to_buy = new_credit / reference_value
                    if btc_to_buy > 0:
                        new_loan = {
                            "platform": "Simulated",
                            "amount": new_credit,
                            "interest": interest_rate,
                            "start_date": date.date(),
                            "term_months": None,
                            "liquidation_ltv": liquidation_ltv,
                            "paid": False,
                            "accrued_interest": 0.0,
                            "end_date": None
                        }
                        active_loans.append(new_loan)
                        current_btc += btc_to_buy
                        total_debt += new_credit
                        rebalanced = True
                        action = "Buy"
                        delta_btc = btc_to_buy

        if rebalanced:
            ltv_after = total_debt / (current_btc * reference_value) if current_btc > 0 else float('inf')

            rebalancing_log.append({
                "Date": date.date(),
                "Action": action,
                "LTV before": rebalance_ltv,
                "LTV after": ltv_after,
                "BTC Δ": f'{delta_btc:+.6f} BTC',
                "Price": f'{price:.2f} {currency_symbol}',
                f"{currency_symbol} Spent": f'{delta_btc * price:.2f} {currency_symbol}',
                "New Total BTC": f'{current_btc:.6f} BTC',
                "New Total Debt": f'{total_debt:.2f} `{currency_symbol}`'
            })

        data.append({
            'Date': date,
            'Price': price,
            'BTC': current_btc,
            'Total Debt': total_debt,
            'Total Interest': fixed_interest,
            'LTV': rebalance_ltv,
            'Real LTV': real_ltv,
            'Fixed Interest': fixed_interest,
            "Accrued Interest": accrued_interest,
        })

    results = pd.DataFrame(data).set_index('Date')
    results["Net Worth"] = results["BTC"] * results["Price"] - results["Total Debt"]
    results["Net BTC"] = results["BTC"] - (results["Total Debt"] / results["Price"])
    return results, rebalancing_log


selected_strategy_cfg = strategy_presets[selected_sim_strategy]

results, rebalancing_log = run_simulation(selected_strategy_cfg, total_btc, df, btc_ath,
                                          st.session_state["portfolio_loans"])

target_ltv = selected_strategy_cfg.get("ltv", 20) / 100
# ---------- 📉 LTV Chart ----------
st.subheader("📉 LTV Development")

fig = go.Figure()
ltv_line_name = "LTV relative to ATH" if selected_strategy_cfg.get("ltv_relative_to_ath", False) else "LTV"
fig.add_trace(go.Scatter(
    x=results.index,
    y=results['LTV'],
    mode='lines',
    name=ltv_line_name,
    hovertemplate="Date: %{x|%Y-%m-%d}<br>LTV: %{y:,.2f}"
))

if selected_strategy_cfg.get("ltv_relative_to_ath", False):
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['Real LTV'],
        mode='lines',
        name='Real LTV',
        line=dict(dash='dash'),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Real LTV: %{y:.2%}"
    ))
fig.add_trace(go.Scatter(
    x=results.index,
    y=[target_ltv] * len(results),
    mode='lines',
    name='Target LTV',
    line=dict(dash='dash'),
    hoverinfo='skip'
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
        results["Total Debt"],
        results["Net Worth"],
        results["Net BTC"]
    ), axis=-1),
    hovertemplate=
    "Date: %{x|%Y-%m-%d}<br>" +
    "BTC Price: {currency_symbol}%{y:,.2f}<br>" +
    "BTC Holdings: %{customdata[0]:.6f} BTC<br>" +
    "Total Debt: {currency_symbol}%{customdata[1]:,.2f}<br>" +
    "Net Worth: {currency_symbol}%{customdata[2]:,.2f}<br>" +
    "Net BTC: %{customdata[3]:.6f} BTC"
))

rebal_df = pd.DataFrame(rebalancing_log)

if "Action" in rebal_df.columns:
    buy_mask = rebal_df["Action"] == "Buy"
    sell_mask = rebal_df["Action"] == "Sell"
    liq_mask = rebal_df["Action"] == "Liquidation"
    repay_mask = rebal_df["Action"].str.startswith("Repay")

    fig.add_trace(go.Scatter(
        x=rebal_df[buy_mask]["Date"],
        y=rebal_df[buy_mask]["LTV before"],
        mode='markers',
        name="Buy",
        marker=dict(size=12, symbol='circle', color='green'),
        hovertext=[
            f"Buy on {row['Date']}<br>BTC Δ: {row['BTC Δ']}<br>Total Debt: {row['New Total Debt']}<br>Price: {row['Price']}<br>LTV: {row['LTV before']}"
            for _, row in rebal_df[buy_mask].iterrows()
        ],
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=rebal_df[sell_mask]["Date"],
        y=rebal_df[sell_mask]["LTV before"],
        mode='markers',
        name="Sell",
        marker=dict(size=12, symbol='circle', color='gold'),
        hovertext=[
            f"Sell on {row['Date']}<br>BTC Δ: {row['BTC Δ']}<br>Total Debt: {row['New Total Debt']}<br>Price: {row['Price']}<br>LTV: {row['LTV before']}"
            for _, row in rebal_df[sell_mask].iterrows()
        ],
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=rebal_df[liq_mask]["Date"],
        y=rebal_df[liq_mask]["LTV before"],
        mode='markers',
        name="Liquidation",
        marker=dict(size=12, symbol='x', color='red'),
        hovertext=[
            f"Liquidation on {row['Date']}<br>BTC Δ: {row['BTC Δ']}<br>Total Debt: {row['New Total Debt']}<br>Price: {row['Price']}<br>LTV: {row['LTV before']}"
            for _, row in rebal_df[liq_mask].iterrows()
        ],
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter(
        x=rebal_df[repay_mask]["Date"],
        y=rebal_df[repay_mask]["LTV before"],
        mode='markers',
        name="Repay",
        marker=dict(size=12, symbol='diamond', color='blue'),
        hovertext=[...]  # auch hier passende Texte
    ))

fig.update_layout(
    yaxis=dict(title='LTV'),
    yaxis2=dict(title=f'BTC Price ({currency_symbol})', overlaying='y', side='right'),
    title='LTV & BTC Price with Rebalancing Events',
    legend=dict(orientation="h", y=-0.2)
)

st.plotly_chart(fig, use_container_width=True)

# ---------- 📘 Rebalancing Log ----------
if not rebal_df.empty:
    st.subheader("📘 Rebalancing Log")
    st.dataframe(pd.DataFrame(rebalancing_log))

st.markdown("## ✅ Loan Plan Summary")

liquidated = False
if not rebal_df.empty and "Liquidation" in rebal_df["Action"].values:
    last_liq = rebal_df[rebal_df["Action"] == "Liquidation"].iloc[-1]
    end_price = float(str(last_liq["Price"]).replace("`{currency_symbol}`", "").replace(",", ""))
    end_total_debt = float(str(last_liq["New Total Debt"]).replace("`{currency_symbol}`", "").replace(",", ""))
    original_btc = float(str(last_liq["BTC Δ"]).replace(" BTC", "").replace("+", "").replace(",", "").lstrip("-"))
    liquidation_value = original_btc * end_price
    remaining_value = max(liquidation_value - end_total_debt, 0)
    net_btc = remaining_value / end_price
    end_btc = net_btc
    liquidated = True
else:
    end_price = df.iloc[-1]["price"]
    end_total_debt = results["Total Debt"].iloc[-1]
    end_btc = results["BTC"].iloc[-1]
    net_btc = results["Net BTC"].iloc[-1]

total_interest = results["Total Interest"].iloc[-1]
start_price = df.iloc[0]["price"]
start_btc = btc_owned
start_value = start_btc * start_price
end_value = end_btc * end_price
value_diff = end_value - start_value
btc_diff = end_btc - start_btc
net_value = net_btc * end_price
net_value_diff = net_value - start_value
net_btc_diff = net_btc - start_btc

col1, col2 = st.columns(2)
with col1:
    st.metric("Total BTC", f"{end_btc:.6f} BTC", f"{btc_diff:+.6f} BTC")
    st.metric("Net BTC", f"{net_btc:.6f} BTC", f"{net_btc_diff:+.6f} BTC")
    st.metric("Total Debt (incl. interest)", f"{currency_symbol}{end_total_debt:,.2f}")
with col2:
    st.metric("Total Value", f"{currency_symbol}{end_value:,.2f}", f"{value_diff:+,.2f} {currency_symbol}")
    st.metric("Net Value", f"{currency_symbol}{net_value:,.2f}", f"{net_value_diff:+,.2f} {currency_symbol}")
    st.metric("Total Interest Paid", f"{currency_symbol}{total_interest:,.2f}")

max_ltv = results['Real LTV'].max()
ltv_buffer = (liquidation_ltv - max_ltv) / liquidation_ltv

if liquidated:
    liquidation_risk = "❌ High"
elif ltv_buffer < 0.20:
    liquidation_risk = "⚠️ Medium"
else:
    liquidation_risk = "🟢 Low"

st.metric("Liquidation Risk", liquidation_risk)

st.header("📊 Strategy Comparison")

selected_strategies = st.multiselect(
    "Select strategies to compare:",
    options=list(strategy_presets.keys()),
    default=["Custom"]
)

comparison_data = []

for strat_name in selected_strategies:
    strat_cfg = strategy_presets[strat_name]
    strat_label = strat_name

    btc_owned = strat_cfg.get("btc_owned", st.session_state.get("btc_owned", 1.0))
    btc_price = strat_cfg.get("btc_price", st.session_state.get("btc_price", 50000))
    ltv = strat_cfg.get("ltv", st.session_state.get("ltv", 0.20)) / 100
    ltv_relative_to_ath = strat_cfg.get("ltv_relative_to_ath", st.session_state.get("ltv_relative_to_ath", False))

    initial_ltv = ltv * (btc_ath / btc_price) if ltv_relative_to_ath else ltv
    safe_loan = (initial_ltv * btc_owned * btc_price) / (1 - initial_ltv)
    btc_bought = safe_loan / btc_price
    total_btc = btc_owned + btc_bought

    results, _ = run_simulation(strat_cfg, total_btc, df, btc_ath, st.session_state["portfolio_loans"])

    net_btc = results["Net BTC"].copy()
    net_worth = results["Net Worth"].copy()
    net_worth[net_worth < 0] = 0

    comparison_data.append({
        "name": strat_name,
        "dates": results.index,
        "net_worth": net_worth,
        "net_btc": net_btc,
        "ltv_series": results["LTV"].copy()
    })

comparison_view = st.radio(
    "View Mode",
    options=["Net Worth", "LTV"],
    horizontal=True
)
fig_compare = go.Figure()
for strat in comparison_data:
    y_data = strat["net_worth"] if comparison_view == "Net Worth" else strat["ltv_series"]

    fig_compare.add_trace(go.Scatter(
        x=strat["dates"],
        y=y_data,
        mode='lines',
        name=strat["name"],
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Net Value: {currency_symbol}%{customdata[0]:,.2f}<br>"
            "Net BTC: %{customdata[1]:.6f}<br>"
            "LTV: %{customdata[2]:.2%}"
        ),
        customdata=np.stack((
            strat["net_worth"],
            strat["net_btc"],
            strat["ltv_series"]
        ), axis=-1)
    ))

fig_compare.update_layout(
    title=f"📊 Strategy Comparison – {comparison_view}",
    xaxis_title="Date",
    yaxis_title=comparison_view,
    legend=dict(orientation="h", y=-0.2)
)

st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("## ⚠️ Disclaimers & Assumptions")
st.markdown("""
- **Historical data is no guarantee for future performance.**  
  Do not rely on the simulation results or optimize your strategy based on past data alone.

- **Not your keys, not your coins.**  
  By taking out loans, you introduce third-party risk. Your broker or custodian could become insolvent in a crisis.

- **The simulation excludes taxes, fees, spreads and edge conditions.**  
  Penalties for early repayment or forced liquidation are not modeled.

- **This tool does not constitute financial advice.**  
  Use it for educational purposes and make your own informed decisions.
""")

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 1.1em">
  🧡 If you like this tool, you can support it with a ⚡ Lightning tip:<br>
  <a href="https://strike.me/nodeowl21" target="_blank"><strong>strike.me/nodeowl21</strong></a>
  <br><br>
  <small>
    © 2025 <a href="mailto:nodeowl21@proton.me">nodeowl21</a> — Open Source • No Data Collected • Bitcoin only
  </small>
</div>
""", unsafe_allow_html=True)
