import os, json, sys
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
INP, OUT = "/data/news.json", "/data/sentiment.json"

SENT_SYS = (
    "You label finance news snippets as Positive, Negative, or Neutral for short-term stock moves. "
    "Return ONLY one of: positive|negative|neutral."
)

def classify(txt):
    if not txt or not txt.strip():
        return "neutral"
    msg = [{"role":"system","content":SENT_SYS},
           {"role":"user","content":txt[:800]}]
    out = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0)
    lab = out.choices[0].message.content.strip().lower()
    if "positive" in lab: return "positive"
    if "negative" in lab: return "negative"
    return "neutral"

if __name__ == "__main__":
    try:
        news = json.load(open(INP))
    except FileNotFoundError:
        print("[sentiment] no news.json found"); sys.exit(0)

    for a in news:
        text = (a.get("title") or "") + " " + (a.get("desc") or "")
        a["sentiment"] = classify(text)

    with open(OUT, "w") as f:
        json.dump(news, f, indent=2)
    print(f"[sentiment] labeled {len(news)} items -> {OUT}")
