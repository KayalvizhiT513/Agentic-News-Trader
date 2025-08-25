import os, json, requests, time

API_KEY = os.getenv("NEWS_API_KEY")
TICKERS = [t.strip() for t in os.getenv("TICKERS","AAPL,TSLA,BK").split(",")]
EXCLUDE_SOURCES = {"pr newswire","prnewswire","business wire","businesswire","globenewswire","accesswire"}
OUT = "/data/news.json"

def get_news_for(ticker):
    # Keep it simple: titles+descriptions are enough for the demo
    url = ("https://newsapi.org/v2/everything"
           f"?q={ticker}&language=en&sortBy=publishedAt&pageSize=25&apiKey={API_KEY}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    keep = []
    for a in r.json().get("articles", []):
        src = (a.get("source",{}).get("name") or "").lower()
        if any(s in src for s in EXCLUDE_SOURCES):  # drop press-release wires
            continue
        keep.append({
            "ticker": ticker,
            "source": a["source"]["name"],
            "author": a.get("author"),
            "title": a.get("title"),
            "desc": a.get("description"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt")
        })
    return keep

if __name__ == "__main__":
    all_news = []
    for t in TICKERS:
        try:
            all_news += get_news_for(t); time.sleep(0.8)  # be gentle
        except Exception as e:
            print(f"[WARN] {t}: {e}")
    os.makedirs("/data", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(all_news, f)
    print(f"[news_fetcher] saved {len(all_news)} articles to {OUT}")
