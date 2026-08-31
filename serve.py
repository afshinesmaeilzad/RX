#!/usr/bin/env python3
"""Start the RX API.

    RX_AUTH_TOKEN=secret python3 serve.py --host 0.0.0.0 --port 8077

The model is not loaded at startup — the first /detect pulls it, or POST
/model/load pulls it on demand. That keeps the box answering /health while ~9 GB
of weights are still coming down.
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import uvicorn
    from rxapi import config

    config.ensure_dirs()
    print(f"RX API on {args.host}:{args.port}  ({config.summary()})")
    uvicorn.run("rxapi.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
