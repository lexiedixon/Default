"""Realistic synthetic market data for demo / offline mode."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)


def _make_price_history(start_price: float, days: int = 90, drift: float = 0.0002, vol: float = 0.015) -> pd.DataFrame:
    """Simulate a price series with GBM."""
    returns = RNG.normal(drift, vol, days)
    prices = start_price * np.exp(np.cumsum(returns))
    dates = [datetime.today() - timedelta(days=days - i) for i in range(days)]
    df = pd.DataFrame(index=pd.DatetimeIndex(dates))
    df["Close"] = prices
    df["Open"] = prices * (1 + RNG.normal(0, 0.003, days))
    df["High"] = np.maximum(df["Close"], df["Open"]) * (1 + RNG.uniform(0, 0.008, days))
    df["Low"] = np.minimum(df["Close"], df["Open"]) * (1 - RNG.uniform(0, 0.008, days))
    df["Volume"] = RNG.integers(1_000_000, 50_000_000, days).astype(float)
    return df


MOCK_STOCKS = [
    {"ticker": "NVDA", "display_name": "NVIDIA Corp", "start": 880, "drift": 0.0008, "vol": 0.025},
    {"ticker": "AAPL", "display_name": "Apple Inc", "start": 185, "drift": 0.0003, "vol": 0.012},
    {"ticker": "MSFT", "display_name": "Microsoft Corp", "start": 415, "drift": 0.0004, "vol": 0.011},
    {"ticker": "AMZN", "display_name": "Amazon.com Inc", "start": 192, "drift": 0.0005, "vol": 0.018},
    {"ticker": "GOOGL", "display_name": "Alphabet Inc", "start": 175, "drift": 0.0003, "vol": 0.014},
    {"ticker": "META", "display_name": "Meta Platforms", "start": 510, "drift": 0.0006, "vol": 0.022},
    {"ticker": "TSLA", "display_name": "Tesla Inc", "start": 180, "drift": 0.0001, "vol": 0.035},
    {"ticker": "AMD", "display_name": "Advanced Micro Devices", "start": 155, "drift": 0.0007, "vol": 0.030},
    {"ticker": "SPY", "display_name": "SPDR S&P 500 ETF", "start": 510, "drift": 0.0002, "vol": 0.008},
    {"ticker": "QQQ", "display_name": "Invesco QQQ Trust", "start": 440, "drift": 0.0003, "vol": 0.010},
    {"ticker": "IWM", "display_name": "iShares Russell 2000 ETF", "start": 210, "drift": 0.0001, "vol": 0.012},
    {"ticker": "JPM", "display_name": "JPMorgan Chase", "start": 205, "drift": 0.0002, "vol": 0.013},
    {"ticker": "GS", "display_name": "Goldman Sachs", "start": 490, "drift": 0.0003, "vol": 0.016},
    {"ticker": "XOM", "display_name": "Exxon Mobil", "start": 115, "drift": 0.0001, "vol": 0.014},
    {"ticker": "LLY", "display_name": "Eli Lilly", "start": 780, "drift": 0.0005, "vol": 0.020},
    {"ticker": "UNH", "display_name": "UnitedHealth Group", "start": 525, "drift": 0.0002, "vol": 0.012},
]

MOCK_COMMODITIES = [
    {"ticker": "GLD", "display_name": "Gold", "start": 225, "drift": 0.0002, "vol": 0.008},
    {"ticker": "SLV", "display_name": "Silver", "start": 27, "drift": 0.0001, "vol": 0.014},
    {"ticker": "USO", "display_name": "Crude Oil", "start": 74, "drift": -0.0001, "vol": 0.018},
    {"ticker": "UNG", "display_name": "Natural Gas", "start": 13, "drift": -0.0002, "vol": 0.025},
    {"ticker": "CPER", "display_name": "Copper", "start": 24, "drift": 0.0003, "vol": 0.015},
    {"ticker": "WEAT", "display_name": "Wheat", "start": 5.8, "drift": -0.0001, "vol": 0.016},
    {"ticker": "CORN", "display_name": "Corn", "start": 5.2, "drift": 0.0000, "vol": 0.014},
]

MOCK_CRYPTO = [
    {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "current_price": 67400, "market_cap": 1_320_000_000_000, "total_volume": 28_000_000_000, "price_change_percentage_24h": 2.1, "price_change_percentage_7d_in_currency": 8.4, "price_change_percentage_30d_in_currency": 22.5, "image": ""},
    {"id": "ethereum", "symbol": "eth", "name": "Ethereum", "current_price": 3520, "market_cap": 422_000_000_000, "total_volume": 15_000_000_000, "price_change_percentage_24h": 3.2, "price_change_percentage_7d_in_currency": 12.1, "price_change_percentage_30d_in_currency": 18.0, "image": ""},
    {"id": "solana", "symbol": "sol", "name": "Solana", "current_price": 168, "market_cap": 75_000_000_000, "total_volume": 3_200_000_000, "price_change_percentage_24h": 5.8, "price_change_percentage_7d_in_currency": 18.3, "price_change_percentage_30d_in_currency": 35.2, "image": ""},
    {"id": "binancecoin", "symbol": "bnb", "name": "BNB", "current_price": 598, "market_cap": 88_000_000_000, "total_volume": 1_800_000_000, "price_change_percentage_24h": 1.4, "price_change_percentage_7d_in_currency": 5.2, "price_change_percentage_30d_in_currency": 10.8, "image": ""},
    {"id": "ripple", "symbol": "xrp", "name": "XRP", "current_price": 0.58, "market_cap": 32_000_000_000, "total_volume": 1_200_000_000, "price_change_percentage_24h": -1.2, "price_change_percentage_7d_in_currency": -4.5, "price_change_percentage_30d_in_currency": 8.3, "image": ""},
    {"id": "cardano", "symbol": "ada", "name": "Cardano", "current_price": 0.44, "market_cap": 15_600_000_000, "total_volume": 380_000_000, "price_change_percentage_24h": 2.3, "price_change_percentage_7d_in_currency": 7.1, "price_change_percentage_30d_in_currency": -5.2, "image": ""},
    {"id": "avalanche", "symbol": "avax", "name": "Avalanche", "current_price": 36, "market_cap": 14_800_000_000, "total_volume": 520_000_000, "price_change_percentage_24h": 4.1, "price_change_percentage_7d_in_currency": 14.2, "price_change_percentage_30d_in_currency": 28.7, "image": ""},
    {"id": "chainlink", "symbol": "link", "name": "Chainlink", "current_price": 14.8, "market_cap": 9_100_000_000, "total_volume": 480_000_000, "price_change_percentage_24h": -0.8, "price_change_percentage_7d_in_currency": 3.4, "price_change_percentage_30d_in_currency": 12.4, "image": ""},
    {"id": "polkadot", "symbol": "dot", "name": "Polkadot", "current_price": 7.2, "market_cap": 9_900_000_000, "total_volume": 320_000_000, "price_change_percentage_24h": 1.6, "price_change_percentage_7d_in_currency": 6.3, "price_change_percentage_30d_in_currency": -8.1, "image": ""},
    {"id": "dogecoin", "symbol": "doge", "name": "Dogecoin", "current_price": 0.162, "market_cap": 23_400_000_000, "total_volume": 1_500_000_000, "price_change_percentage_24h": 6.4, "price_change_percentage_7d_in_currency": 22.3, "price_change_percentage_30d_in_currency": 45.1, "image": ""},
]


def get_mock_stocks(categories: list[str]) -> list[dict]:
    return [
        {
            "ticker": s["ticker"],
            "display_name": s["display_name"],
            "asset_class": "Stock/ETF",
            "history": _make_price_history(s["start"], drift=s["drift"], vol=s["vol"]),
            "fast_info": {},
            "news": [
                {"title": f"{s['display_name']} reports strong quarterly earnings beat", "link": "#", "publisher": "Reuters", "providerPublishTime": int(datetime.now().timestamp()) - 3600},
                {"title": f"Analysts upgrade {s['ticker']} on growth momentum", "link": "#", "publisher": "Bloomberg", "providerPublishTime": int(datetime.now().timestamp()) - 86400},
            ],
        }
        for s in MOCK_STOCKS
    ]


def get_mock_commodities() -> list[dict]:
    return [
        {
            "ticker": c["ticker"],
            "display_name": c["display_name"],
            "asset_class": "Commodity",
            "history": _make_price_history(c["start"], drift=c["drift"], vol=c["vol"]),
            "fast_info": {},
            "news": [],
        }
        for c in MOCK_COMMODITIES
    ]


def get_mock_crypto() -> list[dict]:
    return MOCK_CRYPTO
