# 🟠 Bitcoin Loan Planner

**Simulate leveraged BTC strategies with dynamic rebalancing, risk control, and long-term planning.**  
This open-source tool helps you explore what happens when you use Bitcoin-backed loans to stack more sats.

---

🧠 Motivation

Bitcoin-backed loans are becoming increasingly attractive:
⚖️ lower rates, fairer terms.

As Bitcoin goes up forever, these loans allow you to cover everyday expenses — all while sticking to the principle of “never sell your Bitcoin.”

But you can also use them to leverage up and stack more sats — borrowing to buy more BTC and using that as additional collateral.
Naturally, that comes with risk.

I built this simulator to explore the dynamics of such strategies:
How far can you push it? When do you get liquidated? And how much BTC do you actually end up with?

---

## 🚀 Features

- 🧪 Simulate loan-based BTC accumulation over time
- 📈 Model BTC price paths (historical or random walk)
- ⚖️ Custom target LTV & dynamic rebalancing thresholds (buy/sell)
- 📉 Visualize loan performance, liquidation events & net BTC
- 🔁 Adjustable interest rate, volatility, expected return

---

## 🖥️ Try it out

> 👉 [Link to Live App (Streamlit Cloud or custom URL)](https://your-link-here.com)

---

## 🔧 Installation (Local)

```bash
git clone https://github.com/yourusername/bitcoin-loan-planner.git
cd bitcoin-loan-planner
pip install -r requirements.txt
streamlit run loan_planner.py