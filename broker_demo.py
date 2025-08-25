# broker_demo.py  (interactive-friendly)
import os, sys, json, pathlib, argparse
from ib_insync import IB, Stock, MarketOrder

ACTIONS_PATH = pathlib.Path(os.getenv("ACTIONS_PATH", "data/actions.json"))


def load_actions():
    """
    Load actions from stdin if present; otherwise from data/actions.json.
    """
    raw = sys.stdin.read()
    if raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"STDIN JSON error: {e}. Falling back to {ACTIONS_PATH}", file=sys.stderr)

    with open(ACTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def decide_qty(args):
    """
    Determine order quantity:
    - CLI: --qty / -q
    - ENV: ORDER_QTY
    - Interactive prompt if TTY
    - Default: 1
    """
    # 1) CLI flag
    if args.qty is not None:
        return max(1, int(args.qty))

    # 2) ENV var
    env_qty = os.getenv("ORDER_QTY")
    if env_qty:
        try:
            return max(1, int(env_qty))
        except ValueError:
            print(f"Ignoring invalid ORDER_QTY={env_qty!r}", file=sys.stderr)

    # 3) Interactive prompt only if stdin & stdout are TTYs
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            s = input("Enter quantity PER order for BUY/SELL (default 1): ").strip()
            if not s:
                return 1
            return max(1, int(s))
        except Exception:
            return 1

    # 4) Fallback default
    return 1


def main():
    parser = argparse.ArgumentParser(description="Place IB paper orders from actions JSON.")
    parser.add_argument("-q", "--qty", type=int, help="Quantity per order (overrides prompt/env).")
    parser.add_argument("--client-id", type=int, default=int(os.getenv("IB_CLIENT_ID", "42")),
                        help="IB clientId (default 42 or IB_CLIENT_ID env).")
    parser.add_argument("--host", default=os.getenv("IB_HOST", "127.0.0.1"),
                        help="IB host (default 127.0.0.1 or IB_HOST env).")
    parser.add_argument("--port", type=int, default=int(os.getenv("IB_PORT", "7497")),
                        help="IB port (default 7497 or IB_PORT env).")
    args = parser.parse_args()

    actions = load_actions()

    # Show decided actions
    print("Decided actions:")
    for t, a in actions.items():
        print(f"  - {t}: {a}")
    print()

    qty = decide_qty(args)
    print(f"Using quantity per order: {qty}\n")

    # Connect to IB (TWS Paper by default)
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)

    # Place orders
    for ticker, act in actions.items():
        if act not in {"BUY", "SELL"}:
            print(f"{ticker}: HOLD -> skip")
            continue

        contract = Stock(ticker, 'SMART', 'USD')
        ib.qualifyContracts(contract)  # good practice

        order = MarketOrder(act, qty)
        trade = ib.placeOrder(contract, order)
        print(f"{ticker}: {act} {qty} @ market submitted (id={trade.order.orderId})")


if __name__ == "__main__":
    main()
