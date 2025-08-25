# Agentic-News-Trader
### News Trader — Agentic AI Demo

A tiny, end-to-end demo of an **agentic LLM** that:

1. fetches recent news,
2. runs a simple sentiment analysis,
3. decides `BUY/SELL/HOLD` for a few tickers, and
4. lets you place paper trades via **Interactive Brokers**.

> **Disclaimer**
> There are many factors that drive a trading decision. For the demo, only **news sentiment** is used. This is for learning cloud-native agentic patterns — **not financial advice**.

---

## Quickstart

### 1) Clone

```bash
git clone https://github.com/KayalvizhiT513/Agentic-News-Trader.git
cd Agentic-News-Trader
```

### 2) Configure API keys

Create a `.env` file in the repo root:

```ini
# .env
OPENAI_API_KEY=sk-...
NEWS_API_KEY=your_newsapi_key
# Optional
TICKERS=AAPL,TSLA,BK
```

### 3) Prerequisites

* **Python 3.10+** (Windows uses `py`, macOS/Linux uses `python`)
* **Docker Desktop** (for `docker compose` services used by the supervisor)
* **Interactive Brokers** TWS or IB Gateway **paper** environment running locally (default host/port `127.0.0.1:7497`)
* `pip install -r requirements.txt` (includes `ib_insync`, etc.)

### 4) Run the supervisor (builds observations → writes actions)

```bash
# Windows PowerShell / CMD
py supervisor_agent.py

# macOS / Linux
python supervisor_agent.py
```

The supervisor:

* runs `docker compose` services:

  * `news_fetcher` → writes `data/news.json`
  * `sentiment` → writes `data/sentiment.json`
  * `trader` (internal tool) → produces the decisions JSON
* writes final decisions to **`data/actions.json`** (atomically)

### 5) Inspect the JSONs (optional but recommended)

**Windows (PowerShell/CMD):**

```powershell
type data\news.json
type data\sentiment.json
type data\actions.json
```

**macOS/Linux:**

```bash
cat data/news.json
cat data/sentiment.json
cat data/actions.json
```

### 6) Place paper trades (confirm quantity)

**Windows:**

```bat
type data\actions.json | py broker_demo.py -q 5
```

**macOS/Linux:**

```bash
cat data/actions.json | python broker_demo.py -q 5
```

If you omit `-q`, the script will prompt you interactively (TTY only).
You can also set `ORDER_QTY=5` as an environment variable.

---

## How it works

```
+----------------+      +-------------------+      +------------------+
|  news_fetcher  | ---> |   news.json       |      |                  |
+----------------+      +-------------------+      |                  |
                                                    v
+----------------+      +-------------------+   +-------+    final decisions
|   sentiment    | ---> | sentiment.json    |-->| LLM   |--> {"AAPL":"BUY",...}
+----------------+      +-------------------+   +-------+          |
                                                                  writes
                                                                  v
                                                           data/actions.json
```

* **`supervisor_agent.py`**: an agentic planner using OpenAI to decide when to fetch news, run sentiment, and finalize trade actions.
* **`broker_demo.py`**: reads actions from **stdin** or `data/actions.json`, shows decided actions, asks/accepts quantity, and places **market** orders via `ib_insync` to Paper TWS.

---

## Configuration

* **Tickers**: set a comma-separated list in `.env` as `TICKERS=AAPL,TSLA,BK`.
* **IB connection** (defaults suitable for Paper TWS):

  * Host: `127.0.0.1` (`--host` or `IB_HOST`)
  * Port: `7497` (`--port` or `IB_PORT`)
  * Client ID: `42` (`--client-id` or `IB_CLIENT_ID`)

---

## Project structure

```
.
├─ data/
│  ├─ news.json
│  ├─ sentiment.json
│  └─ actions.json
├─ supervisor_agent.py
├─ broker_demo.py
├─ docker-compose.yml
├─ requirements.txt
└─ .env
```

---

## Troubleshooting

* **`JSONDecodeError` in broker**
  Ensure `data/actions.json` exists and contains valid JSON. Re-run the supervisor.

* **Empty JSON files**
  Check Docker is running and `docker compose` is available. Verify your `NEWS_API_KEY`.

* **IB connection errors**
  Make sure TWS/IB Gateway **paper** is running locally, API enabled, and the port matches (`7497`).

* **No trades placed**
  Only `BUY`/`SELL` trigger orders. `HOLD` is skipped by design.

---

## Safety & Scope

This repo is educational. It focuses on the **agentic orchestration** pattern and **basic sentiment** from news. Use at your own risk and **do not** trade real money based only on this demo.

---
