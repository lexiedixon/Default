import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime

STOCK_UNIVERSE = {
    "ETFs": ["SPY", "QQQ", "DIA", "IWM", "VTI", "ARKK"],
    "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN", "AMD"],
    "Finance": ["JPM", "BAC", "GS", "V", "MA", "BRK-B"],
    "Energy": ["XOM", "CVX", "SLB", "OXY"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "LLY"],
    "Sector ETFs": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLY", "XLB"],
}

COMMODITY_UNIVERSE = {
    "Gold": "GLD",
    "Silver": "SLV",
    "Crude Oil": "USO",
    "Natural Gas": "UNG",
    "Copper": "CPER",
    "Wheat": "WEAT",
    "Corn": "CORN",
}

COINGECKO_URL = "https://api.coingecko.com/api/v3"

POSITIVE_WORDS = {
    "surge", "jump", "gain", "rise", "bull", "growth", "record", "beat",
    "profit", "revenue", "strong", "upgrade", "rally", "boom", "outperform",
    "exceed", "milestone", "expansion", "innovation", "launch", "breakout",
    "high", "positive", "opportunity", "momentum", "breakthrough",
}
NEGATIVE_WORDS = {
    "drop", "fall", "crash", "bear", "loss", "miss", "decline", "cut",
    "sell", "slump", "plunge", "weak", "downgrade", "recession", "inflation",
    "risk", "concern", "fear", "warning", "crisis", "layoff", "fraud",
    "lawsuit", "penalty", "investigate", "default", "bankruptcy", "low",
}


def score_news_sentiment(news_items: list) -> float:
    """Return sentiment score in [-1, 1] from news title keywords."""
    if not news_items:
        return 0.0
    total = 0
    for item in news_items:
        title = item.get("title", "").lower()
        words = set(title.split())
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        total += pos - neg
    max_possible = len(news_items) * 3
    return max(-1.0, min(1.0, total / max_possible)) if max_possible else 0.0


def fetch_ticker_data(ticker: str) -> dict | None:
    """Fetch price history, info snapshot, and news for one ticker."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="3mo")
        if hist.empty or len(hist) < 10:
            return None
        info = t.fast_info
        news = getattr(t, "news", [])[:8]
        return {
            "ticker": ticker,
            "history": hist,
            "fast_info": info,
            "news": news,
        }
    except Exception:
        return None


def fetch_all_stocks(selected_categories: list[str]) -> tuple[list[dict], bool]:
    """Fetch data for all tickers in selected categories. Returns (data, is_live)."""
    tickers = []
    for cat in selected_categories:
        tickers.extend(STOCK_UNIVERSE.get(cat, []))
    tickers = list(dict.fromkeys(tickers))

    results = []
    for ticker in tickers:
        data = fetch_ticker_data(ticker)
        if data:
            results.append(data)

    if results:
        return results, True

    # Fall back to demo data if live fetch fails
    from demo_data import get_mock_stocks
    return get_mock_stocks(selected_categories), False


def fetch_all_commodities() -> tuple[list[dict], bool]:
    """Fetch data for all commodity proxy ETFs. Returns (data, is_live)."""
    results = []
    for name, ticker in COMMODITY_UNIVERSE.items():
        data = fetch_ticker_data(ticker)
        if data:
            data["display_name"] = name
            results.append(data)

    if results:
        return results, True

    from demo_data import get_mock_commodities
    return get_mock_commodities(), False


def fetch_crypto_data() -> tuple[list[dict], bool]:
    """Fetch top cryptocurrencies from CoinGecko. Returns (data, is_live)."""
    try:
        resp = requests.get(
            f"{COINGECKO_URL}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 50,
                "page": 1,
                "price_change_percentage": "7d,30d",
                "sparkline": False,
            },
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return data, True
    except Exception:
        pass

    from demo_data import get_mock_crypto
    return get_mock_crypto(), False


def fetch_crypto_news() -> list[dict]:
    """Fetch general crypto news from CoinGecko."""
    try:
        resp = requests.get(f"{COINGECKO_URL}/news", timeout=10)
        resp.raise_for_status()
        return resp.json().get("data", [])[:20]
    except Exception:
        return []
