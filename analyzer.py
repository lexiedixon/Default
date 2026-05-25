import pandas as pd
import numpy as np
from data_fetcher import score_news_sentiment, COMMODITY_UNIVERSE

RISK_WEIGHTS = {
    "Conservative": {
        "return_7d": 0.15,
        "return_30d": 0.30,
        "rsi": 0.25,
        "volume": 0.10,
        "sentiment": 0.10,
        "volatility_penalty": 0.10,
    },
    "Moderate": {
        "return_7d": 0.25,
        "return_30d": 0.25,
        "rsi": 0.20,
        "volume": 0.15,
        "sentiment": 0.15,
        "volatility_penalty": 0.00,
    },
    "Aggressive": {
        "return_7d": 0.40,
        "return_30d": 0.15,
        "rsi": 0.15,
        "volume": 0.20,
        "sentiment": 0.10,
        "volatility_penalty": 0.00,
    },
}


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0


def score_rsi(rsi: float) -> float:
    """Map RSI to a 0-1 score. Peak at 50, zero at extremes (<20 or >80)."""
    if rsi <= 20 or rsi >= 80:
        return 0.0
    if rsi <= 50:
        return (rsi - 20) / 30
    return (80 - rsi) / 30


def score_return(pct: float, clip_low: float = -0.5, clip_high: float = 0.5) -> float:
    """Normalize a return percentage to [0, 1]."""
    clipped = max(clip_low, min(clip_high, pct))
    return (clipped - clip_low) / (clip_high - clip_low)


def score_volume(hist: pd.DataFrame) -> float:
    """Score based on recent volume vs 30-day average. Capped at 1.0."""
    if "Volume" not in hist.columns or len(hist) < 5:
        return 0.5
    avg_vol = hist["Volume"].mean()
    recent_vol = hist["Volume"].iloc[-5:].mean()
    if avg_vol == 0:
        return 0.5
    ratio = recent_vol / avg_vol
    return min(1.0, ratio / 2.0)  # ratio of 2x maps to score 1.0


def analyze_yfinance_asset(data: dict, risk_level: str) -> dict:
    """Score a yfinance-fetched asset and return enriched record."""
    hist = data["history"]
    closes = hist["Close"]
    ticker = data["ticker"]
    weights = RISK_WEIGHTS[risk_level]

    ret_7d = (closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) >= 6 else 0.0
    ret_30d = (closes.iloc[-1] / closes.iloc[-22] - 1) if len(closes) >= 22 else 0.0
    volatility = closes.pct_change().std() * np.sqrt(252)
    rsi = calculate_rsi(closes)

    s_ret7 = score_return(ret_7d)
    s_ret30 = score_return(ret_30d)
    s_rsi = score_rsi(rsi)
    s_vol = score_volume(hist)
    s_news = (score_news_sentiment(data.get("news", [])) + 1) / 2  # [-1,1] -> [0,1]

    vol_penalty = min(1.0, volatility / 0.8) if weights["volatility_penalty"] > 0 else 0.0

    raw_score = (
        weights["return_7d"] * s_ret7
        + weights["return_30d"] * s_ret30
        + weights["rsi"] * s_rsi
        + weights["volume"] * s_vol
        + weights["sentiment"] * s_news
        - weights["volatility_penalty"] * vol_penalty
    )
    score = round(max(0.0, min(100.0, raw_score * 100)), 1)

    signals = []
    if ret_7d > 0.05:
        signals.append(f"+{ret_7d:.1%} this week")
    elif ret_7d < -0.05:
        signals.append(f"{ret_7d:.1%} this week")
    if ret_30d > 0.10:
        signals.append(f"+{ret_30d:.1%} this month")
    elif ret_30d < -0.10:
        signals.append(f"{ret_30d:.1%} this month")
    if rsi > 65:
        signals.append(f"RSI {rsi:.0f} — nearing overbought")
    elif rsi < 35:
        signals.append(f"RSI {rsi:.0f} — nearing oversold")
    if s_vol > 0.7:
        signals.append("High volume surge")
    if s_news > 0.6:
        signals.append("Positive news sentiment")
    elif s_news < 0.35:
        signals.append("Negative news sentiment")

    commodity_names = {v: k for k, v in COMMODITY_UNIVERSE.items()}
    display = data.get("display_name") or commodity_names.get(ticker) or ticker

    return {
        "ticker": ticker,
        "name": display,
        "asset_class": data.get("asset_class", "Stock/ETF"),
        "price": round(float(closes.iloc[-1]), 2),
        "return_7d": ret_7d,
        "return_30d": ret_30d,
        "rsi": round(rsi, 1),
        "volatility": round(volatility, 3),
        "score": score,
        "signals": signals,
        "history": hist,
        "news": data.get("news", []),
    }


def analyze_crypto_asset(coin: dict, risk_level: str) -> dict:
    """Score a CoinGecko coin record."""
    weights = RISK_WEIGHTS[risk_level]

    ret_7d = (coin.get("price_change_percentage_7d_in_currency") or 0) / 100
    ret_30d = (coin.get("price_change_percentage_30d_in_currency") or 0) / 100
    ret_24h = (coin.get("price_change_percentage_24h") or 0) / 100

    vol_24h = coin.get("total_volume") or 0
    mcap = coin.get("market_cap") or 1
    vol_mcap_ratio = vol_24h / mcap  # proxy for volume surge

    s_ret7 = score_return(ret_7d, -0.6, 0.6)
    s_ret30 = score_return(ret_30d, -0.7, 0.7)
    s_vol = min(1.0, vol_mcap_ratio / 0.5)
    s_news = 0.5  # no per-coin news without API key

    raw_score = (
        weights["return_7d"] * s_ret7
        + weights["return_30d"] * s_ret30
        + weights["rsi"] * 0.5  # neutral RSI for crypto
        + weights["volume"] * s_vol
        + weights["sentiment"] * s_news
    )
    score = round(max(0.0, min(100.0, raw_score * 100)), 1)

    signals = []
    if ret_24h > 0.05:
        signals.append(f"+{ret_24h:.1%} today")
    elif ret_24h < -0.05:
        signals.append(f"{ret_24h:.1%} today")
    if ret_7d > 0.10:
        signals.append(f"+{ret_7d:.1%} this week")
    elif ret_7d < -0.10:
        signals.append(f"{ret_7d:.1%} this week")
    if vol_mcap_ratio > 0.2:
        signals.append("High trading volume vs market cap")

    return {
        "ticker": coin["symbol"].upper(),
        "name": coin["name"],
        "asset_class": "Crypto",
        "price": coin.get("current_price", 0),
        "return_7d": ret_7d,
        "return_30d": ret_30d,
        "rsi": 50.0,
        "volatility": abs(ret_7d) * 2,
        "score": score,
        "signals": signals,
        "history": None,
        "news": [],
        "image": coin.get("image", ""),
        "market_cap": coin.get("market_cap", 0),
    }


def get_top_picks(
    stock_data: list[dict],
    commodity_data: list[dict],
    crypto_raw: list[dict],
    risk_level: str,
    include_crypto: bool,
    include_commodities: bool,
    top_n: int,
) -> list[dict]:
    """Analyze all assets and return the top N by score."""
    scored = []

    for d in stock_data:
        d["asset_class"] = "Stock/ETF"
        try:
            scored.append(analyze_yfinance_asset(d, risk_level))
        except Exception:
            continue

    if include_commodities:
        for d in commodity_data:
            d["asset_class"] = "Commodity"
            try:
                scored.append(analyze_yfinance_asset(d, risk_level))
            except Exception:
                continue

    if include_crypto:
        for coin in crypto_raw:
            try:
                scored.append(analyze_crypto_asset(coin, risk_level))
            except Exception:
                continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
