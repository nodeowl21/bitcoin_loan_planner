# filename: btc_loan_sim.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("BTC Loan Strategietool")

# Eingabeparameter
initial_btc = st.number_input("Start BTC", value=1.0)
btc_price = st.number_input("Startpreis (USD)", value=50000)
days = st.slider("Simulationsdauer (Tage)", 30, 730, 365)
drawdown = st.slider("Crash-Sicherheit (% max. Kursverlust)", 10, 90, 80) / 100
liquidation_threshold = st.slider("Liquidationsgrenze (% vom Collateral)", 50, 95, 85) / 100
interest_rate = st.slider("Kreditzins p.a. (%)", 0, 20, 10) / 100
rebalance_threshold = st.slider("Rebalancing-Schwelle (% Kursänderung)", 1, 50, 10) / 100
daily_volatility = st.slider("Tägliche Kursvolatilität (%)", 0, 10, 2) / 100

# Initialwerte
btc_owned = initial_btc
usd_debt = 0
price_history = [btc_price]
btc_history = [btc_owned]
debt_history = [usd_debt]
status_history = ["OK"]
actions = []
last_rebalance_price = btc_price

# Simulation
for day in range(1, days + 1):
    # Simulierter Preis (random walk)
    change = np.random.normal(0, daily_volatility)
    new_price = price_history[-1] * (1 + change)
    price_history.append(new_price)

    # Zinsen auf den Kredit
    usd_debt *= (1 + interest_rate / 365)

    # Sicherheitsprüfung nach Kursverfall
    crash_price = new_price * (1 - drawdown)
    collateral_value = btc_owned * crash_price
    required_collateral = usd_debt / liquidation_threshold
    liquidationsgefahr = collateral_value < required_collateral

    # Rebalancing Entscheidung
    price_diff = abs(new_price - last_rebalance_price) / last_rebalance_price

    if price_diff >= rebalance_threshold:
        if new_price > last_rebalance_price:
            # BTC kaufen per Kredit
            max_new_loan = (btc_owned * new_price - usd_debt)
            new_btc = max_new_loan / new_price
            btc_owned += new_btc
            usd_debt += max_new_loan
            actions.append((day, new_price, "Kauf", new_btc))
            last_rebalance_price = new_price
        else:
            # BTC verkaufen um Schulden zu reduzieren
            required_value = usd_debt / liquidation_threshold
            needed_collateral = required_value / new_price
            sell_btc = max(0, btc_owned - needed_collateral)
            btc_owned -= sell_btc
            usd_debt -= sell_btc * new_price
            actions.append((day, new_price, "Verkauf", sell_btc))
            last_rebalance_price = new_price

    btc_history.append(btc_owned)
    debt_history.append(usd_debt)
    status_history.append("Liquidationsgefahr" if liquidationsgefahr else "OK")

# Ergebnis-Tabelle
df = pd.DataFrame({
    "Tag": list(range(days + 1)),
    "BTC-Preis": price_history,
    "BTC-Bestand": btc_history,
    "USD-Schulden": debt_history,
    "Status": status_history
})

st.subheader("Simulationsergebnisse")

fig, ax = plt.subplots()
ax.plot(df["Tag"], df["BTC-Preis"], label="BTC-Preis")
ax.set_ylabel("Preis (USD)")
ax.set_xlabel("Tag")
ax.legend()
st.pyplot(fig)

fig2, ax2 = plt.subplots()
ax2.plot(df["Tag"], df["BTC-Bestand"], label="BTC-Bestand")
ax2.plot(df["Tag"], df["USD-Schulden"], label="USD-Schulden")
ax2.set_ylabel("Wert")
ax2.set_xlabel("Tag")
ax2.legend()
st.pyplot(fig2)

st.subheader("Rebalancing-Aktionen")
if actions:
    for tag, preis, art, menge in actions:
        st.write(f"Tag {tag}: {art} von {menge:.6f} BTC bei ${preis:.2f}")
else:
    st.write("Keine Aktionen notwendig.")

st.subheader("Letzter Status")
st.success("System stabil") if status_history[-1] == "OK" else st.error("Liquidationsgefahr!")