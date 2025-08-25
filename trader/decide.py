import json, collections, os
INP = "/data/sentiment.json"

def action_from(sentiments):
    score = sentiments.count("positive") - sentiments.count("negative")
    if score > 0: return "BUY"
    if score < 0: return "SELL"
    return "HOLD"

if __name__ == "__main__":
    data = json.load(open(INP))
    grouped = collections.defaultdict(list)
    for a in data:
        grouped[a["ticker"]].append(a["sentiment"])

    decisions = {}
    for t, sents in grouped.items():
        decisions[t] = action_from(sents)

    print(json.dumps(decisions, indent=2))
