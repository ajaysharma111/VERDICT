# VERDICT market tick WebSocket

## What is available

- Dashboard client connects to `ws://127.0.0.1:8000/ws/market` (or `wss` when served over HTTPS).
- `GET /api/market/stream/status` reports the stream provider, connected dashboard clients, and last tick.
- In `MARKET_DATA_MODE=demo`, the backend publishes synthetic, explicitly labelled `DEMO` ticks for the watchlist.
- In `MARKET_DATA_MODE=live`, the backend uses the Angel One SmartAPI market-data WebSocket after the user explicitly connects. The adapter is data-only; no order-placement API is implemented.

## Angel One configuration

Install dependencies with `pip install -r requirements.txt`, then configure these values in the backend `.env` file:

```dotenv
MARKET_DATA_MODE=live
ANGELONE_API_KEY=your_api_key
ANGELONE_CLIENT_CODE=your_client_code
ANGELONE_PIN=your_pin
ANGELONE_TOTP_SECRET=your_totp_secret
ANGELONE_INSTRUMENT_TOKENS={"RELIANCE":"verified_token","TCS":"verified_token"}
```

`ANGELONE_INSTRUMENT_TOKENS` must contain symbol-to-token values copied from the current official Angel One instrument master. Do not guess or reuse stale tokens. The current adapter subscribes to NSE cash-market tokens in LTP mode. Start the app, then call `POST /api/auth/angelone/connect` from the local dashboard flow to authenticate and start the feed.

Never commit `.env`, expose credentials in the browser, or share API keys, PINs, or TOTP secrets. If credentials are missing, invalid, or instrument tokens are absent, live connection reports an error; it does not fall back to synthetic data.

## Verification and limitations

Run `python -m pytest -q` from `backend/`. The suite includes a DEMO WebSocket integration test that checks a real tick payload over the FastAPI test WebSocket client.

The Angel One WebSocket path is implemented but has not been authenticated or validated against a live brokerage account in this environment. Live verification requires the account owner's credentials, current instrument tokens, and any Angel One network/static-IP requirements. The project remains paper-trading-only: market data never places a real order.
