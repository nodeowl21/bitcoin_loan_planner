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
```

## FastAPI + React App

The Streamlit app is still available, but the migrated web stack lives in `backend/`
and `frontend/`.

Start the API:

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

Start the React frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173.

The dev server proxies `/health`, `/btc-price`, `/simulate`, and `/optimize` to
`http://127.0.0.1:8000`, so you do **not** need `VITE_API_BASE_URL` for local
development. If you prefer a direct URL instead, set e.g.
`VITE_API_BASE_URL=http://127.0.0.1:8000` in `frontend/.env.local`.

The API exposes:

- `GET /health`
- `GET /btc-price?currency=USD`
- `POST /simulate`
- `POST /optimize`

When `frontend/dist` exists (e.g. after `npm run build`), the same FastAPI app
also serves the built SPA from `/` so a single process can host UI + API (used
for Docker / Render).

## Deploy on Render (free tier)

You need:

1. A **GitHub** (or GitLab) account and this repository pushed there.
2. A **Render** account ([render.com](https://render.com)) — sign in with GitHub.
3. The repo must contain **`btc-usd-max.csv`** and **`bitcoin_data.csv`** at
   the root (the API reads them for historical / power-law modes).

**Option A — Blueprint (recommended)**

1. In Render: **New** → **Blueprint**.
2. Connect the repository and select the branch.
3. Render reads [`render.yaml`](render.yaml) and creates a **Web Service**
   (Docker, free plan).
4. After the first deploy, open the service URL (e.g. `https://loan-planner.onrender.com`).

**Option B — Manual Web Service**

1. **New** → **Web Service** → connect the repo.
2. **Runtime:** Docker (or “Dockerfile”).
3. **Instance type:** Free.
4. **Dockerfile path:** `Dockerfile` (root).
5. Deploy. Health check can use path `/health`.

The **Dockerfile** builds the frontend and copies `frontend/dist` into the
image; **uvicorn** serves the API and the static UI on the same host, so no CORS
configuration is required for production.

**Caveats (free tier):** The service **spins down** after idle time; the first
request after that can take **~30–60 seconds** (cold start). Heavy `/optimize`
runs may occasionally hit a **request timeout** on very small instance types.

Optional: set **`CORS_ORIGINS`** in Render’s environment if you ever split
frontend and API onto different origins (comma-separated URLs).

## Tests

### Backend

The backend ships with a `pytest` suite that pins down the simulation
mathematics (liquidation handling, rebalancing, ATH-mode, loan lifecycle,
savings, multi-loan repayment priorities, summary aggregation), the price
modelling (`Generated`, `Historical`, `Power-Law`), `optimize_strategy`, and the
FastAPI endpoints.

Install the development dependencies and run the suite from the project root:

```bash
pip install -r requirements-dev.txt
pytest
```

### Frontend

The frontend uses [Vitest](https://vitest.dev/) (`happy-dom` environment) to
test the pure utilities that were extracted out of `App.tsx`: formatters,
default settings/presets, portfolio totals and import/export round-trips.

```bash
cd frontend
npm install
npm test          # one-shot run
npm run test:watch
```