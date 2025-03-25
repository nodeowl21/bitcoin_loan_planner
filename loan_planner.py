import streamlit as st
import datetime

st.title("🔁 BTC Loan Planer")

# Session State initialisieren
if 'loan_data' not in st.session_state:
    st.session_state.loan_data = None

# Sidebar-Box für aktuellen Zustand
with st.sidebar:
    st.header("📦 Aktueller Zustand")
    if st.session_state.loan_data:
        loan = st.session_state.loan_data
        st.markdown(f"**BTC-Bestand:** `{loan['total_btc']:.6f}` BTC")
        st.markdown(f"**Kreditbetrag:** `${loan['safe_loan']:,.2f}` USD")
    else:
        st.info("Noch kein Plan gespeichert.")

st.header("1️⃣ Plan erstellen")

# Eingaben
btc_owned = st.number_input("BTC-Bestand", value=1.0)
btc_price = st.number_input("BTC-Preis (USD)", value=50000)
drawdown = st.slider("Absicherung gegen Kurssturz (%)", 10, 90, 80) / 100
rebalance_threshold = st.slider("Rebalancing-Schwelle (%)", 1, 50, 10) / 100
interest_rate = st.slider("Kreditzins p.a. (%)", 0, 20, 10) / 100
start_date = st.date_input("Startdatum", value=datetime.date.today())

if drawdown == 0:
    st.error("⚠️ Drawdown darf nicht 0 % sein.")
else:
    total_btc = btc_owned / drawdown
    btc_bought = total_btc - btc_owned
    safe_loan = btc_bought * btc_price
    yearly_interest = safe_loan * interest_rate
    price_up = btc_price * (1 + rebalance_threshold)
    price_down = btc_price * (1 - rebalance_threshold)

    st.subheader("📊 Ergebnis:")
    st.markdown(f"- **Maximaler Kreditbetrag (USD):** `${safe_loan:,.2f}`")
    st.markdown(f"- **Gekaufte BTC durch Kredit:** `{btc_bought:.6f}` BTC")
    st.markdown(f"- **Neuer BTC-Gesamtbestand:** `{total_btc:.6f}` BTC")
    st.markdown(f"- **Jährliche Zinskosten:** `${yearly_interest:,.2f}`")

    if st.button("✅ Plan speichern"):
        st.session_state.loan_data = {
            'btc_owned': btc_owned,
            'btc_price': btc_price,
            'drawdown': drawdown,
            'rebalance_threshold': rebalance_threshold,
            'interest_rate': interest_rate,
            'start_date': start_date,
            'safe_loan': safe_loan,
            'btc_bought': btc_bought,
            'total_btc': total_btc,
            'zinsen_fix': 0.0
        }
        st.success("Plan gespeichert!")
        st.rerun()

st.header("2️⃣ Loan Übersicht & Rebalancing")

if st.session_state.loan_data:
    loan = st.session_state.loan_data

    st.markdown(f"**Startdatum:** {loan['start_date']}")
    st.markdown(f"**BTC-Startbestand:** {loan['btc_owned']} BTC")
    st.markdown(f"**Kreditbetrag:** ${loan['safe_loan']:,.2f}")
    st.markdown(f"**Gekaufte BTC:** {loan['btc_bought']:.6f} BTC")
    st.markdown(f"**Gesamt-BTC:** {loan['total_btc']:.6f} BTC")
    st.markdown(f"**Zinssatz:** {loan['interest_rate']*100:.2f} % p.a.")

    current_price = st.number_input("Aktueller BTC-Preis (USD)", value=loan['btc_price'], key="current_price")
    today = datetime.date.today()
    days_passed = (today - loan['start_date']).days
    interest_due_new = loan['safe_loan'] * loan['interest_rate'] * (days_passed / 365)
    total_interest = loan.get('zinsen_fix', 0.0) + interest_due_new
    total_debt = loan['safe_loan'] + total_interest
    collateral_value = loan['total_btc'] * current_price
    ltv = total_debt / collateral_value if collateral_value > 0 else float('inf')

    st.markdown(f"**Vergangene Tage:** {days_passed} Tage")
    st.markdown(f"**Aufgelaufene Zinsen:** `${total_interest:,.2f}`")
    st.markdown(f"**Gesamtschulden:** `${total_debt:,.2f}`")
    st.markdown(f"**Aktueller Sicherheitswert:** `${collateral_value:,.2f}`")
    st.markdown(f"**Loan-to-Value (LTV):** {ltv*100:.2f} %")

    target_ltv = 0.20
    z = loan['interest_rate']
    t = days_passed

    if ltv > target_ltv:
        D = total_debt
        P = current_price
        B = loan['total_btc']
        btc_to_sell = (5 * D - B * P) / (4 * P)
        btc_to_sell = max(0, btc_to_sell)
        st.warning(f"📉 Preis gefallen → Verkaufe `{btc_to_sell:.6f}` BTC, um LTV wieder auf 20% zu senken.")

        if st.button("🔻 Rebalancing ausführen (Verkauf)"):
            loan['total_btc'] -= btc_to_sell
            loan['safe_loan'] -= btc_to_sell * P
            loan['btc_bought'] = loan['total_btc'] - loan['btc_owned']
            loan['zinsen_fix'] += interest_due_new
            loan['start_date'] = today
            st.session_state.loan_data = loan
            st.rerun()

    elif ltv < target_ltv:
        D = total_debt
        C = collateral_value
        new_credit = (0.2 * C - D) / 0.8
        new_credit = max(0, new_credit)
        btc_to_buy = new_credit / current_price
        st.success(f"📈 Preis gestiegen → Du kannst für etwa ${new_credit:,.2f} nachbeleihen und `{btc_to_buy:.6f}` BTC kaufen.")

        if st.button("🔺 Rebalancing ausführen (Kauf)"):
            loan['total_btc'] += btc_to_buy
            loan['safe_loan'] += new_credit
            loan['btc_bought'] = loan['total_btc'] - loan['btc_owned']
            loan['zinsen_fix'] += interest_due_new
            loan['start_date'] = today
            st.session_state.loan_data = loan
            st.rerun()

    else:
        st.info("✅ LTV ist genau bei 20%. Kein Rebalancing nötig.")
else:
    st.warning("⚠️ Bitte zuerst einen Plan erstellen.")