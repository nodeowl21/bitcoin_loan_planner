import datetime
import json
import numpy as np
from scipy.optimize import curve_fit

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import uuid

from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args

st.title("🟠 Bitcoin Loan Planner")


# ---------- State Helper ----------

def get_state_value(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---------- Fetch Live Price ----------
def get_live_btc_price(currency):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies={currency}"
        response = requests.get(url, timeout=5)
        data = response.json()
        return data["bitcoin"][currency.lower()]
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


# ---------- Power Law BTC Price Model ----------

def btc_price_model_power_law(start_year=2025, years=50):
    # Daten laden
    df = pd.read_csv("bitcoin_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Value"] > 0].copy()  # Nur valide Preise
    df["Days"] = (df["Date"] - pd.Timestamp("2009-01-09")).dt.days

    # Regressionsfunktion: log(price) = a + b * log(days)
    def log_func(x, a, b):
        return a + b * np.log(x)

    # Fit berechnen
    x_data = df["Days"].values
    y_data = np.log(df["Value"].values)
    popt, _ = curve_fit(log_func, x_data, y_data)
    a_fit, b_fit = popt

    # Prognose erstellen
    start_date = pd.Timestamp(f"{start_year}-01-01")
    future_dates = pd.date_range(start=start_date, periods=years * 365 + 1, freq="D")
    x_future = (future_dates - pd.Timestamp("2009-01-09")).days
    x_future = np.where(x_future > 0, x_future, 1)
    prices = np.exp(log_func(x_future, a_fit, b_fit))

    return pd.Series(prices, index=future_dates)


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


def export_user_data():
    # Loans vorbereiten: Datum zu String konvertieren
    loans_serialized = []
    for loan in st.session_state.get("portfolio_loans", []):
        serialized_loan = loan.copy()
        if isinstance(serialized_loan.get("start_date"), datetime.date):
            serialized_loan["start_date"] = serialized_loan["start_date"].isoformat()
        if isinstance(serialized_loan.get("end_date"), datetime.date):
            serialized_loan["end_date"] = serialized_loan["end_date"].isoformat()
        loans_serialized.append(serialized_loan)

    data = {
        "portfolio": {
            "btc_owned": st.session_state.get("btc_owned", 1.0),
            "currency": st.session_state.get("currency", "USD"),
            "income_per_year": st.session_state.get("income_per_year", 0.0),
            "btc_saving_rate_percent": st.session_state.get("btc_saving_rate_percent", 0.0),
            "other_assets": st.session_state.get("other_assets", 0.0),
        },
        "loans": loans_serialized,
        "strategies": {
            "presets": st.session_state.get("strategy_presets", {}),
            "default": st.session_state.get("default_strategy", "Custom"),
        },
        "simulation": {
            "sim_mode": st.session_state.get("sim_mode", "Historical"),
            "sim_years": st.session_state.get("sim_years", 5),
            "exp_return": st.session_state.get("exp_return", 0.0),
            "volatility": st.session_state.get("volatility", 0.00),
            "interval": st.session_state.get("interval", "Weekly"),
            "interest": st.session_state.get("interest", 12.5),
            "liquidation_ltv": st.session_state.get("liquidation_ltv", 100),
            "selected_sim_strategy": st.session_state.get("selected_sim_strategy", "Custom"),
            "enable_btc_saving": st.session_state.get("enable_btc_saving", True),
        }
    }
    return data


def import_user_data(uploaded_file):
    try:
        imported_data = json.load(uploaded_file)

        for k, v in imported_data.get("portfolio", {}).items():
            st.session_state[k] = v

        loans = imported_data.get("loans", [])
        deserialized_loans = []
        for loan in loans:
            deserialized_loan = loan.copy()
            if "start_date" in loan:
                deserialized_loan["start_date"] = datetime.date.fromisoformat(loan["start_date"])
            if "end_date" in loan and loan["end_date"] is not None:
                deserialized_loan["end_date"] = datetime.date.fromisoformat(loan["end_date"])
            deserialized_loans.append(deserialized_loan)
        st.session_state["portfolio_loans"] = deserialized_loans

        strategies = imported_data.get("strategies", {})
        if "presets" in strategies:
            st.session_state["strategy_presets"] = strategies["presets"]
        if "default" in strategies:
            st.session_state["default_strategy"] = strategies["default"]

        simulation = imported_data.get("simulation", {})
        for k, v in simulation.items():
            st.session_state[k] = v

    except Exception as e:
        st.error(f"❌ Failed to import data: {e}")


if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = str(uuid.uuid4())

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
        }}
    st.session_state["default_strategy"] = "Custom"

st.sidebar.markdown("## 📂 Portfolio Summary")

live_price = get_live_btc_price(st.session_state.get("currency", "USD"))

if live_price:
    st.session_state["btc_price"] = live_price
prev_currency = st.session_state.get("prev_currency", get_state_value("currency", "USD"))

st.subheader("General")

currency_input = st.selectbox(
    "Currency",
    options=["EUR", "USD"],
    index=["EUR", "USD"].index(get_state_value("currency", "USD"))
)

currency_changed = currency_input != prev_currency

if currency_changed:
    live_price = get_live_btc_price(currency_input)
    if live_price:
        st.session_state["temp_btc_price"] = live_price
    st.session_state["prev_currency"] = currency_input

temp_currency_symbol = "$" if currency_input == "USD" else "€"

temp_price = st.session_state.get("temp_btc_price")
btc_price_input = st.number_input(
    f"BTC Price ({temp_currency_symbol})",
    value=temp_price if temp_price else get_state_value("btc_price", 100000),
    step=1000
)

st.session_state["currency"] = currency_input
st.session_state["btc_price"] = btc_price_input

st.subheader("Portfolio")

btc_owned_input = st.number_input(
    "BTC Holdings",
    value=get_state_value("btc_owned", 1.0),
    step=0.1,
    format="%.6f"
)

income_per_year_input = st.number_input(
    f"Annual Income ({temp_currency_symbol})",
    value=get_state_value("income_per_year", 0.0),
    step=1000.0,
    format="%.2f"
)

btc_saving_rate_input = st.slider(
    "BTC Saving Rate (% of Income)",
    min_value=0.0,
    max_value=100.0,
    value=get_state_value("btc_saving_rate_percent", 0.0),
    step=0.5,
    format="%.1f"
)

other_assets_input = st.number_input(
    f"Other Assets ({temp_currency_symbol})",
    value=get_state_value("other_assets", 0.0),
    step=1000.0,
    format="%.2f"
)

if st.button("Save Portfolio"):
    st.session_state["btc_owned"] = btc_owned_input
    st.session_state["income_per_year"] = income_per_year_input
    st.session_state["btc_saving_rate_percent"] = btc_saving_rate_input
    st.session_state["other_assets"] = other_assets_input
    st.session_state["portfolio_saved"] = True

currency_symbol = "$" if get_state_value("currency", "USD") == "USD" else "€"

st.subheader("Existing Loans")

if "portfolio_loans" not in st.session_state:
    st.session_state["portfolio_loans"] = []

editing_loan = next(
    (loan for loan in st.session_state["portfolio_loans"] if loan.get("id") == st.session_state.get("edit_loan_id")),
    None)
if editing_loan:
    new_platform_default = editing_loan.get("platform", "")
    new_amount_default = editing_loan.get("amount", 0.0)
    new_interest_default = editing_loan.get("interest", 5.0)
    new_btc_bought_default = editing_loan.get("btc_bought", 0.0)
    new_start_default = editing_loan.get("start_date", datetime.date.today())
    new_liquidation_ltv_default = editing_loan.get("liquidation_ltv", 100)
    new_term_mode_default = "Set duration" if editing_loan.get("term_months") else "Unlimited"
    new_term_default = editing_loan.get("term_months", 12)
else:
    new_platform_default = st.session_state.get("new_platform", "")
    new_amount_default = st.session_state.get("new_amount", 0.0)
    new_interest_default = st.session_state.get("new_interest", 5.0)
    new_btc_bought_default = st.session_state.get("new_btc_bought", 0.0)
    new_start_default = st.session_state.get("new_start", datetime.date.today())
    new_liquidation_ltv_default = st.session_state.get("new_liquidation_ltv", 100)
    new_term_mode_default = st.session_state.get("new_term_mode", "Unlimited")
    new_term_default = st.session_state.get("new_term", 12)

new_platform = st.text_input("Platform / Lender", key="new_platform", value=new_platform_default)
new_amount = st.number_input(f"Loan Amount ({currency_symbol})", key="new_amount", step=1000.0,
                             value=new_amount_default)
new_interest = st.number_input(
    "Interest Rate (% p.a.)", min_value=0.0, max_value=50.0, value=new_interest_default, key="new_interest", step=0.1
)
new_btc_bought = st.number_input(
    "BTC Bought",
    min_value=0.0,
    step=0.0001,
    format="%.6f",
    key="new_btc_bought",
    value=new_btc_bought_default
)
new_start = st.date_input("Start Date", value=new_start_default, key="new_start")
term_mode = st.selectbox("Loan Term", ["Unlimited", "Set duration"],
                         index=0 if new_term_mode_default == "Unlimited" else 1)
if term_mode == "Set duration":
    new_term = st.number_input("Duration (months)", min_value=1, max_value=360, value=new_term_default, key="new_term")
else:
    new_term = None
new_liquidation_ltv = st.slider(
    "Liquidation LTV (%)", 50, 100,
    int(new_liquidation_ltv_default),
    key="new_liquidation_ltv",
)
import uuid

if st.button("Save Loan"):
    loan_id = st.session_state.get("edit_loan_id") or str(uuid.uuid4())
    new_loan = {
        "id": loan_id,
        "platform": new_platform,
        "amount": new_amount,
        "interest": new_interest,
        "start_date": new_start,
        "term_months": new_term,
        "liquidation_ltv": new_liquidation_ltv,
        "btc_bought": new_btc_bought
    }
    existing_ids = [loan.get("id") for loan in st.session_state["portfolio_loans"] if "id" in loan]
    if new_loan["id"] in existing_ids:
        st.session_state["portfolio_loans"] = [
            (new_loan if loan.get("id") == new_loan["id"] else loan)
            for loan in st.session_state["portfolio_loans"]
        ]
    else:
        st.session_state["portfolio_loans"].append(new_loan)
    st.session_state.pop("edit_loan_id", None)
    st.rerun()

if st.session_state["portfolio_loans"]:
    st.markdown("Current Loans")
    for i, loan in enumerate(st.session_state["portfolio_loans"]):
        cols = st.columns([1, 0.1, 0.1])
        with cols[0]:
            start_date = loan["start_date"]
            if loan.get("term_months"):
                end_date = start_date + pd.DateOffset(months=loan["term_months"])
                duration_str = f"{loan['term_months']} months (from {start_date} to {end_date.date()})"
            else:
                duration_str = f"Unlimited (since {start_date})"

            st.markdown(
                f"**{loan['platform']}** – {currency_symbol}{loan['amount']:,.0f} at {loan['interest']:.2f}% p.a. — {duration_str}<br>"
                f"BTC Bought: {loan.get('btc_bought', 0.0):.6f} BTC — Liquidation LTV: {loan.get('liquidation_ltv', 100):.0f}%",
                unsafe_allow_html=True
            )
        with cols[1]:
            if st.button("✏️", key=f"edit_loan_{i}"):
                st.session_state["edit_loan_id"] = loan.get("id")
                st.rerun()
        with cols[2]:
            if st.button("🗑️", key=f"delete_loan_{i}"):
                st.session_state["portfolio_loans"].pop(i)
                st.rerun()

btc_owned = st.session_state["btc_owned"]

btc_from_loans = sum(loan.get("btc_bought", 0.0) for loan in st.session_state.get("portfolio_loans", []))

total_btc = btc_owned + btc_from_loans
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
    interest = loan["amount"] * loan["interest"] / 100 * days_passed / 365
    total_debt += loan["amount"] + interest

portfolio_value = total_btc * st.session_state["btc_price"]
ltv = total_debt / portfolio_value if portfolio_value > 0 else float("inf")

income_per_year = st.session_state.get("income_per_year", 0.0)
btc_saving_rate_percent = st.session_state.get("btc_saving_rate_percent", 0.0)
other_assets = st.session_state.get("other_assets", 0.0)
total_assets = portfolio_value + other_assets
net_assets = portfolio_value + other_assets - total_debt
btc_exposure = portfolio_value / total_assets if total_assets > 0 else 0

st.sidebar.metric("Total BTC", f"{total_btc:.6f}")
st.sidebar.metric("Total BTC Value", f"{currency_symbol}{portfolio_value:.2f}")
st.sidebar.metric("Total Debt", f"{currency_symbol}{total_debt:,.2f}")
st.sidebar.metric("LTV", f"{ltv:.2%}")
st.sidebar.metric("Annual Income", f"{currency_symbol}{income_per_year:,.2f}")
st.sidebar.metric("Other Assets", f"{currency_symbol}{other_assets:,.2f}")
st.sidebar.metric("BTC Saving Rate", f"{btc_saving_rate_percent:.1f}%")

st.sidebar.markdown("---")
st.sidebar.metric("Total Value", f"{currency_symbol}{total_assets:,.2f}")
st.sidebar.metric("Net Value", f"{currency_symbol}{net_assets:,.2f}")
st.sidebar.metric("BTC Exposure", f"{btc_exposure:.2%}")

st.subheader("Strategies")

strategy_presets = st.session_state["strategy_presets"]

preset_list = list(strategy_presets.keys())

selected_preset = st.selectbox(
    "Choose Preset",
    options=preset_list,
    index=preset_list.index(st.session_state.get("default_strategy", "Custom")),
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
    rebalance_sell_factor_input = 100

enable_buy_input = st.checkbox("Enable Buy-Rebalancing", value=preset_config.get("enable_buy", True),
                               key="enable_buy_input")
if enable_buy_input:
    max_buy_threshold = ltv_input - 1
    rebalance_buy_input = st.slider(
        "Buy Threshold (%)",
        0, max_buy_threshold,
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
    rebalance_buy_factor_input = 100

strategy_name_input = st.text_input(
    label="Name",
    value=st.session_state["preset_name"],
    key="strategy_name_input",
    placeholder="Preset name"
)

set_default_checkbox = st.checkbox(
    "Default Strategy",
    value=(st.session_state.get("default_strategy") == st.session_state.get("preset_name")),
    help="Mark this strategy as the default for future selections."
)

if set_default_checkbox:
    st.session_state["default_strategy"] = st.session_state.get("preset_name")
left_col, right_col = st.columns([1, 1])

with left_col:
    col_save_delete = st.columns(2)
    with col_save_delete[0]:
        if st.button("Save Strategy"):
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
                if set_default_checkbox:
                    st.session_state["default_strategy"] = name
                st.session_state["preset_to_select"] = name
                st.session_state["last_preset"] = None
                st.rerun()

with right_col:
    with col_save_delete[1]:
        if strategy_name_input not in ["Custom"]:
            if st.button("🗑️ Delete Strategy"):
                del st.session_state["strategy_presets"][strategy_name_input]
                st.session_state["preset_to_select"] = "Custom"
                st.session_state["last_preset"] = None
                st.rerun()

df_raw = pd.read_csv("btc-usd-max.csv")
btc_ath = df_raw["price"].max()

# ---------- 📈 Simulation ----------
st.header("📈 Simulation & Rebalancing")

sim_mode_input = st.radio(
    "Choose Price Source",
    ["Historical", "Generated", "Power-Law"],
    index=["Historical", "Generated", "Power-Law"].index(st.session_state.get("sim_mode", "Historical")),
    help="Choose between historical, generated, or power-law based prices."
)

if sim_mode_input == "Generated":
    sim_years_input = st.slider(
        "Number of Simulation Years", 1, 20,
        value=st.session_state.get("sim_years", 5),
    )

    expected_return_input = st.slider(
        "Expected Annual Return (%)", -100, 200,
        value=int(st.session_state.get("exp_return", 50)),
    )

    volatility_input = st.slider(
        "Daily Volatility (%)", 1, 100,
        value=int(st.session_state.get("volatility", 5)),
    )

    df = generate_random_walk(
        years=sim_years_input,
        annual_return=expected_return_input / 100,
        daily_volatility=volatility_input / 100,
        seed=42
    )

elif sim_mode_input == "Power-Law":
    sim_years_input = st.slider(
        "Number of Simulation Years", 1, 20,
        value=st.session_state.get("sim_years", 5),
    )
    expected_return_input = 0
    volatility_input = 0
    df = pd.DataFrame({
        "price": btc_price_model_power_law(
            start_year=datetime.date.today().year,
            years=sim_years_input
        )
    })

else:
    sim_years_input = st.slider(
        "Historical Timeframe (years)", 1, 10,
        value=st.session_state.get("sim_years", 5),
    )

    expected_return_input = 0
    volatility_input = 0

    df_raw["snapped_at"] = pd.to_datetime(df_raw["snapped_at"])
    df = df_raw.set_index("snapped_at")["price"].sort_index()
    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(years=sim_years_input)
    price_series = df.loc[start_date:end_date]
    price_rel = price_series / price_series.iloc[0]
    simulated_prices = price_rel * btc_price
    future_dates = pd.date_range(start=datetime.date.today(), periods=len(simulated_prices), freq='D')
    df = pd.DataFrame({'price': simulated_prices.values}, index=future_dates)

interval_input = st.selectbox(
    "Rebalancing Interval",
    ["Daily", "Weekly", "Monthly", "Yearly"],
    index=["Daily", "Weekly", "Monthly", "Yearly"].index(st.session_state.get("interval", "Weekly")),
)

interest_input = st.number_input(
    "Loan Interest Rate (% p.a.)",
    min_value=0.0,
    max_value=20.0,
    value=st.session_state.get("interest", 12.5),
)

liquidation_ltv_input = st.slider(
    "Liquidation LTV (%)", 50, 100,
    value=int(st.session_state.get("liquidation_ltv", 100)),
)

enable_btc_saving_input = st.checkbox(
    "Enable BTC Saving (daily)",
    value=st.session_state.get("enable_btc_saving", True),
)

selected_sim_strategy_input = st.selectbox(
    "Choose strategy for simulation:",
    options=list(st.session_state["strategy_presets"].keys()),
    index=list(st.session_state["strategy_presets"].keys()).index(
        st.session_state.get("selected_sim_strategy", "Custom")),
    key="selected_sim_strategy_input"
)

def saveSimulationParametersToState():
    st.session_state["sim_mode"] = sim_mode_input
    st.session_state["sim_years"] = sim_years_input
    st.session_state["exp_return"] = expected_return_input
    st.session_state["volatility"] = volatility_input
    st.session_state["interval"] = interval_input
    st.session_state["interest"] = interest_input
    st.session_state["liquidation_ltv"] = liquidation_ltv_input
    st.session_state["enable_btc_saving"] = enable_btc_saving_input
    st.session_state["selected_sim_strategy"] = selected_sim_strategy_input


if st.button("Run Optimization"):
    saveSimulationParametersToState()

    st.session_state["simulation_ready"] = False
    st.session_state["optimization_triggered"] = True

    st.rerun()

if st.button("Run Simulation"):
    saveSimulationParametersToState()

    st.session_state["simulation_ready"] = True
    st.session_state["optimization_results"] = False

    st.rerun()


# ---------- 🔄 Simulation Engine ----------
def run_simulation(config: dict, current_btc, price_df: pd.DataFrame, reference_value: float, loans: list):
    ltv = config.get("ltv", 20) / 100
    enable_buy = config.get("enable_buy", True)
    rebalance_buy = config.get("rebalance_buy", 10) / 100
    rebalance_buy_factor = config.get("rebalance_buy_factor", 100) / 100
    enable_sell = config.get("enable_sell", True)
    rebalance_sell = config.get("rebalance_sell", 10) / 100
    rebalance_sell_factor = config.get("rebalance_sell_factor", 100) / 100
    rebalance_days = {"Daily": 1, "Weekly": 7, "Monthly": 30, "Yearly": 365}[
        st.session_state.get("interval", "Weekly")]

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

        this_liquidation_ltv = loan.get("liquidation_ltv", 100) / 100
        this_interest = loan.get("interest", 5.0) / 100
        active_loans.append({
            **loan,
            "liquidation_ltv": this_liquidation_ltv,
            "interest": this_interest,
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
            accrued = loan["amount"] * loan["interest"] * days_running / 365
            loan["accrued_interest"] += accrued
            total_debt += loan["amount"] + accrued
        else:
            total_debt += loan["amount"]

    for i, date in enumerate(price_df.index):
        price = df.loc[date, 'price']

        currency_symbol = "$" if st.session_state.get("currency", "USD") == "USD" else "€"

        if st.session_state["enable_btc_saving"]:
            btc_saving_rate_percent = st.session_state.get("btc_saving_rate_percent", 0.0)
            income_per_year = st.session_state.get("income_per_year", 0.0)

            daily_income = income_per_year / 365
            daily_saving_fiat = (btc_saving_rate_percent / 100) * daily_income

            if daily_saving_fiat > 0 and price > 0:
                daily_btc_bought = daily_saving_fiat / price
                current_btc += daily_btc_bought

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
                    "New Total Debt": f"{(total_debt - repayment_amount):.2f} {currency_symbol}",
                    "LTV before": real_ltv,
                    "LTV after": real_ltv
                })
                total_debt -= repayment_amount

        accrued_interest = 0.0

        for loan in active_loans:
            if loan["paid"]:
                continue

            daily_interest = loan["amount"] * loan["interest"] / 365
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
            unpaid_loans = [loan for loan in active_loans if not loan.get("paid", False)]
            if unpaid_loans:
                weighted_liq_ltv = sum(
                    loan["liquidation_ltv"] * loan["amount"]
                    for loan in unpaid_loans
                ) / sum(loan["amount"] for loan in unpaid_loans)
            else:
                weighted_liq_ltv = st.session_state["liquidation_ltv"] / 100

            if real_ltv > weighted_liq_ltv:
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
                            "interest": st.session_state["interest"] / 100,
                            "start_date": date.date(),
                            "term_months": None,
                            "liquidation_ltv": st.session_state["liquidation_ltv"] / 100,
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
                "New Total Debt": f'{total_debt:.2f} {currency_symbol}'
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


param_space = [
    Integer(1, 45, name="ltv"),
    Integer(1, 10, name="rebalance_buy"),
    Integer(50, 100, name="rebalance_buy_factor"),
    Integer(1, 10, name="rebalance_sell"),
    Integer(50, 100, name="rebalance_sell_factor"),
    Categorical([True, False], name="enable_sell"),
    Categorical([True, False], name="ltv_relative_to_ath")
]


@use_named_args(param_space)
def simulate_strategy(ltv, rebalance_buy, rebalance_buy_factor, rebalance_sell, rebalance_sell_factor, enable_sell,
                      ltv_relative_to_ath):
    strategy_config = {
        "ltv": ltv,
        "rebalance_buy": rebalance_buy,
        "rebalance_buy_factor": rebalance_buy_factor,
        "rebalance_sell": rebalance_sell,
        "rebalance_sell_factor": rebalance_sell_factor,
        "enable_sell": enable_sell,
        "ltv_relative_to_ath": ltv_relative_to_ath
    }

    df.index = pd.date_range(
        start=datetime.date.today(),
        periods=len(df),
        freq="D"
    )

    results, rebalancing_log = run_simulation(
        strategy_config,
        btc_owned,
        df,
        btc_ath,
        []
    )

    rebal_df = pd.DataFrame(rebalancing_log)

    if not rebal_df.empty and "Liquidation" in rebal_df["Action"].values:
        net_btc = 0
    else:
        net_btc = results["Net BTC"].iloc[-1]

    print(
        f"Simulated strategy: LTV={ltv}, Buy Threshold={rebalance_buy}, Buy Intensity={rebalance_buy_factor}, Sell Rebalancing={enable_sell}, Sell Threshold={rebalance_sell}, Sell Intensity={rebalance_sell_factor}")
    print(f"Net BTC: {net_btc:.6f}")
    return -(net_btc - btc_owned)

if st.session_state.get("optimization_triggered"):
    with st.spinner("Optimizing..."):
        result = gp_minimize(
            simulate_strategy,
            dimensions=param_space,
            n_calls=50,
            random_state=42
        )
        best_strategies = sorted(zip(result.func_vals, result.x_iters), key=lambda x: x[0])[:5]
        keys = [dim.name for dim in param_space]
        optimization_results = []
        for i, (score, params) in enumerate(best_strategies, 1):
            strategy = dict(zip(keys, params))
            optimization_results.append({"params": strategy, "net_btc_delta": -score})
        st.session_state["optimization_results"] = optimization_results
    # Only run once per trigger
    st.session_state["optimization_triggered"] = False


if st.session_state.get("optimization_results"):
    st.subheader("📈 Optimized Strategies")
    # Convert optimization results to a list of strategies
    optimized_strategies = []
    for res in st.session_state["optimization_results"]:
        strat = res["params"].copy()
        # Ensure all relevant fields exist, with sensible defaults
        strat.setdefault("ltv", 20)
        strat.setdefault("ltv_relative_to_ath", False)
        strat.setdefault("enable_buy", True)
        strat.setdefault("rebalance_buy", 10)
        strat.setdefault("rebalance_buy_factor", 100)
        strat.setdefault("enable_sell", True)
        strat.setdefault("rebalance_sell", 10)
        strat.setdefault("rebalance_sell_factor", 100)
        optimized_strategies.append(strat)

    for i, strat in enumerate(optimized_strategies):
        # Prepare user-friendly mapping
        mapping = {
            "Target LTV (%)": strat["ltv"],
            "Rebalance LTV relative to BTC All-Time-High": strat.get("ltv_relative_to_ath", False),
            "Enable Buy-Rebalancing": strat.get("enable_buy", False),
            "Buy Threshold (%)": strat.get("rebalance_buy", 10),
            "Buy Rebalancing Intensity (%)": strat.get("rebalance_buy_factor", 100),
            "Enable Sell-Rebalancing": strat.get("enable_sell", False),
            "Sell Threshold (%)": strat.get("rebalance_sell", 10),
            "Sell Rebalancing Intensity (%)": strat.get("rebalance_sell_factor", 100),
        }
        net_btc_delta = st.session_state["optimization_results"][i].get("net_btc_delta", 0)
        st.markdown(f"**Optimized #{i + 1} (Net BTC Δ: {net_btc_delta * 100:.2f}%)**")
        st.json(mapping)
        if st.button(f"➕ Add 'Optimized #{i + 1}' to Presets", key=f"add_opt_{i}"):
            if "strategy_presets" not in st.session_state:
                st.session_state["strategy_presets"] = {}
            st.session_state["strategy_presets"][f"Optimized #{i + 1}"] = {
                "ltv": strat["ltv"],
                "ltv_relative_to_ath": strat.get("ltv_relative_to_ath", False),
                "enable_buy": strat.get("enable_buy", False),
                "rebalance_buy": strat.get("rebalance_buy", 10),
                "rebalance_buy_factor": strat.get("rebalance_buy_factor", 100),
                "enable_sell": strat.get("enable_sell", False),
                "rebalance_sell": strat.get("rebalance_sell", 10),
                "rebalance_sell_factor": strat.get("rebalance_sell_factor", 100),
            }
            st.success(f"'Optimized #{i + 1}' added to strategy presets.")
            st.rerun()

if st.session_state.get("simulation_ready", False):

    selected_strategy_cfg = strategy_presets[st.session_state["selected_sim_strategy"]]

    results, rebalancing_log = run_simulation(
        selected_strategy_cfg,
        total_btc,
        df,
        btc_ath,
        st.session_state["portfolio_loans"]
    )

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
        f"BTC Price: {currency_symbol}%{{y:,.2f}}<br>" +
        "BTC Holdings: %{customdata[0]:.6f} BTC<br>" +
        f"Total Debt: {currency_symbol}%{{customdata[1]:,.2f}}<br>" +
        f"Net Worth: {currency_symbol}%{{customdata[2]:,.2f}}<br>" +
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
        end_price = float(str(last_liq["Price"]).replace(f"{currency_symbol}", "").replace(",", ""))
        end_total_debt = float(str(last_liq["New Total Debt"]).replace(f"{currency_symbol}", "").replace(",", ""))
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

    max_ltv = results['Real LTV'].max()
    ltv_buffer = (st.session_state["liquidation_ltv"] - max_ltv) / st.session_state["liquidation_ltv"]

    if liquidated:
        liquidation_risk = "❌ High"
    elif ltv_buffer < 0.20:
        liquidation_risk = "⚠️ Medium"
    else:
        liquidation_risk = "🟢 Low"

    income_per_year = st.session_state.get("income_per_year", 0.0)
    other_assets = st.session_state.get("other_assets", 0.0)

    epsilon = 1e-6

    if abs(end_total_debt) < epsilon:
        if income_per_year + other_assets > 0:
            debt_coverage_ratio = float('inf')
        else:
            debt_coverage_ratio = 1
    else:
        debt_coverage_ratio = (income_per_year + other_assets) / end_total_debt

    if debt_coverage_ratio == float('inf'):
        dcr_value_display = "∞"
    else:
        dcr_value_display = f"{debt_coverage_ratio:.2f}"

    if debt_coverage_ratio >= 1.5:
        dcr_status = "🟢 Low Risk"
    elif 1.0 <= debt_coverage_ratio < 1.5:
        dcr_status = "🟡 Medium Risk"
    else:
        dcr_status = "🔴 High Risk"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total BTC", f"{end_btc:.6f} BTC", f"{btc_diff:+.6f} BTC")
        st.metric("Net BTC", f"{net_btc:.6f} BTC", f"{net_btc_diff:+.6f} BTC")
        st.metric("Total Debt (incl. interest)", f"{currency_symbol}{end_total_debt:,.2f}")
        st.metric("Liquidation Risk", liquidation_risk)
    with col2:
        st.metric("Total Value", f"{currency_symbol}{end_value:,.2f}", f"{value_diff:+,.2f} {currency_symbol}")
        st.metric("Net Value", f"{currency_symbol}{net_value:,.2f}", f"{net_value_diff:+,.2f} {currency_symbol}")
        st.metric("Total Interest Paid", f"{currency_symbol}{total_interest:,.2f}")
        st.metric(
            label="Debt Coverage Ratio (DCR)",
            value=dcr_value_display,
            delta=dcr_status,
            help="Debt Coverage Ratio (DCR) = (Annual Income + Other Assets) / Total Debt.\n\n"
                 "Indicates how easily you could cover your outstanding debt with your non-BTC assets and income.\n\n"
                 "Higher values mean lower risk. A DCR above 1.5 is considered safe, between 1.0 and 1.5 moderate, and below 1.0 risky."
        )
    st.header("📊 Strategy Comparison")

    selected_strategies = st.multiselect(
        "Select strategies to compare:",
        options=list(strategy_presets.keys()),
        default=[st.session_state.get("default_strategy", "Custom")]
    )

    comparison_data = []

    for strat_name in selected_strategies:
        strat_cfg = strategy_presets[strat_name]
        strat_label = strat_name

        btc_owned = strat_cfg.get("btc_owned", st.session_state.get("btc_owned", 1.0))
        btc_price = strat_cfg.get("btc_price", st.session_state.get("btc_price", 50000))
        ltv = strat_cfg.get("ltv", st.session_state.get("ltv", 20)) / 100
        ltv_relative_to_ath = strat_cfg.get("ltv_relative_to_ath", st.session_state.get("ltv_relative_to_ath", False))

        initial_ltv = ltv * (btc_ath / btc_price) if ltv_relative_to_ath else ltv
        safe_loan = (initial_ltv * btc_owned * btc_price) / (1 - initial_ltv)
        btc_bought = safe_loan / btc_price
        total_btc = btc_owned + btc_bought

        results, _ = run_simulation(strat_cfg, total_btc, df, btc_ath, st.session_state["portfolio_loans"])
        sim_df = results
        if "LTV" in sim_df.columns and (sim_df["LTV"] > 1).any():
            sim_df = sim_df.loc[:sim_df["LTV"].gt(1).idxmax()]
        # Nur sim_df verwenden, keine anderen DataFrames (wie results oder df) für Visualisierungsdaten
        # Sicherheitsprüfung nach dem Abschneiden
        if sim_df.empty or "Net BTC" not in sim_df.columns or "Net Worth" not in sim_df.columns:
            continue

        net_btc = sim_df["Net BTC"]

        net_btc_delta_pct = pd.Series(index=sim_df.index, dtype="float64")

        for idx in sim_df.index:
            if sim_df.loc[idx, "BTC"] > 0:
                net_btc_delta_pct.loc[idx] = ((net_btc.loc[idx] - btc_owned) / btc_owned)
            else:
                net_btc_delta_pct.loc[idx] = -1.0

        net_worth = sim_df["Net Worth"]

        net_worth = net_worth.copy()
        net_worth[net_worth < 0] = 0

        comparison_data.append({
            "name": strat_name,
            "dates": sim_df.index,
            "net_worth": net_worth,
            "net_btc": net_btc,
            "ltv_series": sim_df["LTV"].copy(),
            "net_btc_delta_pct": net_btc_delta_pct
        })

    comparison_view = st.radio(
        "View Mode",
        options=["Net Worth", "LTV", "Net BTC Δ (%)"],
        horizontal=True
    )

    comparison_hovertemplate = {
        "Net Worth": "Date: %{x|%Y-%m-%d}<br>Net Value: " + currency_symbol + "%{customdata[0]:,.2f}<br>Net BTC: %{customdata[1]:.6f}<br>LTV: %{customdata[2]:.2%}",
        "LTV": "Date: %{x|%Y-%m-%d}<br>LTV: %{customdata[2]:.2%}<br>Net BTC: %{customdata[1]:.6f}<br>Net Value: " + currency_symbol + "%{customdata[0]:,.2f}",
        "Net BTC Δ (%)": "Date: %{x|%Y-%m-%d}<br>Net BTC Δ: %{y:.2%}"
    }

    fig_compare = go.Figure()
    for strat in comparison_data:
        # Skip if no data (should not occur, but safety)
        if len(strat["dates"]) == 0:
            continue
        if comparison_view == "Net Worth":
            y_data = strat["net_worth"]
        elif comparison_view == "LTV":
            y_data = strat["ltv_series"]
        else:
            y_data = strat["net_btc_delta_pct"]

        if comparison_view in ["Net Worth", "LTV"]:
            customdata = np.stack((
                strat["net_worth"],
                strat["net_btc"],
                strat["ltv_series"]
            ), axis=-1)
        else:
            customdata = None

        fig_compare.add_trace(go.Scatter(
            x=strat["dates"],
            y=y_data,
            mode='lines',
            name=strat["name"],
            hovertemplate=comparison_hovertemplate[comparison_view],
            customdata=customdata
        ))

    fig_compare.update_layout(
        title=f"📊 Strategy Comparison – {comparison_view}",
        xaxis_title="Date",
        yaxis_title=comparison_view,
        legend=dict(orientation="h", y=-0.2)
    )
    # Format Y-axis as percent for Net BTC Δ (%)
    if comparison_view == "Net BTC Δ (%)":
        fig_compare.update_yaxes(tickformat=".0%", title="Net BTC Δ")

    st.plotly_chart(fig_compare, use_container_width=True)

    # ---------- 📋 Strategy Comparison Table (Yearly) ----------
    st.markdown("### 💹 Total Delta at Year")

    table_data = []
    for strat in comparison_data:
        # Skip if no data
        if len(strat["dates"]) == 0:
            continue
        name = strat["name"]
        sim_df = pd.DataFrame({
            "Date": strat["dates"],
            "Net Worth": strat["net_worth"],
            "Net BTC Δ (%)": strat["net_btc_delta_pct"],
            "LTV": strat["ltv_series"]
        })
        sim_df["Year"] = sim_df["Date"].dt.year
        df_yearly = sim_df.groupby("Year").last().reset_index()
        df_yearly["Strategy"] = name
        table_data.append(df_yearly)

    if table_data:
        table_df = pd.concat(table_data)
        table_df = table_df[["Strategy", "Year", "Net Worth", "Net BTC Δ (%)", "LTV"]]

        # Pivot for comparison
        pivot_df = table_df.pivot(index="Strategy", columns="Year", values="Net BTC Δ (%)")
        # Fill NAs with 'liq' to indicate liquidation
        summary_table = pivot_df.fillna("liquidated")
        # Format as percent with 2 decimals where possible, otherwise keep as 'liq'
        summary_table = summary_table.applymap(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
        # Sort by order in selected_strategies
        summary_table = summary_table.reindex(selected_strategies)

        st.dataframe(summary_table, use_container_width=True)

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

with st.sidebar.expander("Import/Export"):
    import_file = st.file_uploader(
        "Import Settings",
        type="json",
        key=st.session_state["upload_key"]
    )

    if import_file:
        try:
            import_user_data(import_file)

            st.session_state["upload_key"] = str(uuid.uuid4())  # generate a new key!
            st.rerun()
        except Exception as e:
            st.error(f"❌ Import failed: {e}")

    st.download_button(
        label="⬇️ Export Settings",
        data=json.dumps(export_user_data(), indent=2),
        file_name="loan_planner_backup.json",
        mime="application/json"
    )
