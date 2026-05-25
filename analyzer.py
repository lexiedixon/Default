import pandas as pd
import numpy as np
from data_fetcher import score_news_sentiment, COMMODITY_UNIVERSE

# Short-term: momentum, volume, and news drive the score.
# Long-term: sustained trend and low volatility matter most.
HORIZON_WEIGHTS = {
    "short": {
        "Conservative": {"return_7d": 0.25, "return_30d": 0.10, "return_90d": 0.00, "rsi": 0.30, "volume": 0.20, "sentiment": 0.15, "vol_penalty": 0.00},
        "Moderate":     {"return_7d": 0.35, "return_30d": 0.05, "return_90d": 0.00, "rsi": 0.20, "volume": 0.25, "sentiment": 0.15, "vol_penalty": 0.00},
        "Aggressive":   {"return_7d": 0.45, "return_30d": 0.00, "return_90d": 0.00, "rsi": 0.10, "volume": 0.30, "sentiment": 0.15, "vol_penalty": 0.00},
    },
    "long": {
        "Conservative": {"return_7d": 0.00, "return_30d": 0.20, "return_90d": 0.30, "rsi": 0.15, "volume": 0.05, "sentiment": 0.05, "vol_penalty": 0.25},
        "Moderate":     {"return_7d": 0.05, "return_30d": 0.25, "return_90d": 0.35, "rsi": 0.10, "volume": 0.10, "sentiment": 0.05, "vol_penalty": 0.10},
        "Aggressive":   {"return_7d": 0.10, "return_30d": 0.25, "return_90d": 0.40, "rsi": 0.10, "volume": 0.10, "sentiment": 0.05, "vol_penalty": 0.00},
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
    """Map RSI to 0-1. Peak at 50, zero at extremes (<20 or >80)."""
    if rsi <= 20 or rsi >= 80:
        return 0.0
    if rsi <= 50:
        return (rsi - 20) / 30
    return (80 - rsi) / 30


def score_return(pct: float, clip_low: float = -0.5, clip_high: float = 0.5) -> float:
    clipped = max(clip_low, min(clip_high, pct))
    return (clipped - clip_low) / (clip_high - clip_low)


def score_volume(hist: pd.DataFrame) -> float:
    if "Volume" not in hist.columns or len(hist) < 5:
        return 0.5
    avg_vol = hist["Volume"].mean()
    recent_vol = hist["Volume"].iloc[-5:].mean()
    if avg_vol == 0:
        return 0.5
    return min(1.0, (recent_vol / avg_vol) / 2.0)


def _build_signals(
    ret_7d: float,
    ret_30d: float,
    ret_90d: float,
    rsi: float,
    vol_score: float,
    sentiment_score: float,
    volatility: float,
    time_horizon: str,
) -> list[str]:
    signals = []
    if time_horizon == "short":
        if ret_7d > 0.08:
            signals.append(f"+{ret_7d:.1%} this week — strong momentum")
        elif ret_7d > 0.03:
            signals.append(f"+{ret_7d:.1%} this week")
        elif ret_7d < -0.05:
            signals.append(f"{ret_7d:.1%} this week")
        if vol_score > 0.7:
            signals.append("Volume surge — confirming move")
        if rsi > 70:
            signals.append(f"RSI {rsi:.0f} — overbought, caution")
        elif 45 <= rsi <= 62:
            signals.append(f"RSI {rsi:.0f} — building momentum")
        elif rsi < 35:
            signals.append(f"RSI {rsi:.0f} — potential bounce")
        if sentiment_score > 0.6:
            signals.append("Positive news flow")
        elif sentiment_score < 0.35:
            signals.append("Negative news sentiment")
    else:
        if ret_90d > 0.20:
            signals.append(f"+{ret_90d:.1%} over 3 months — strong trend")
        elif ret_90d > 0.08:
            signals.append(f"+{ret_90d:.1%} quarterly gain")
        elif ret_90d < -0.10:
            signals.append(f"{ret_90d:.1%} over 3 months — weak trend")
        if ret_90d > 0 and ret_30d > 0:
            signals.append("Consistent uptrend (30d & 90d positive)")
        if volatility < 0.15 and ret_30d > 0:
            signals.append("Low volatility growth — stable hold")
        elif volatility > 0.50:
            signals.append("High volatility — risky long-term hold")
        if 40 <= rsi <= 60:
            signals.append(f"RSI {rsi:.0f} — healthy neutral zone")
        elif rsi > 70:
            signals.append(f"RSI {rsi:.0f} — may be extended")
    return signals


def analyze_yfinance_asset(data: dict, risk_level: str, time_horizon: str) -> dict:
    hist = data["history"]
    closes = hist["Close"]
    ticker = data["ticker"]
    weights = HORIZON_WEIGHTS[time_horizon][risk_level]

    ret_7d  = (closes.iloc[-1] / closes.iloc[-6]  - 1) if len(closes) >= 6  else 0.0
    ret_30d = (closes.iloc[-1] / closes.iloc[-22] - 1) if len(closes) >= 22 else 0.0
    ret_90d = (closes.iloc[-1] / closes.iloc[0]   - 1) if len(closes) >= 60 else ret_30d
    volatility = closes.pct_change().std() * np.sqrt(252)
    rsi = calculate_rsi(closes)

    s_ret7  = score_return(ret_7d,  -0.5,  0.5)
    s_ret30 = score_return(ret_30d, -0.5,  0.5)
    s_ret90 = score_return(ret_90d, -0.6,  0.6)
    s_rsi   = score_rsi(rsi)
    s_vol   = score_volume(hist)
    s_news  = (score_news_sentiment(data.get("news", [])) + 1) / 2

    vol_penalty = min(1.0, volatility / 0.8) if weights["vol_penalty"] > 0 else 0.0

    raw_score = (
        weights["return_7d"]  * s_ret7
        + weights["return_30d"]  * s_ret30
        + weights["return_90d"]  * s_ret90
        + weights["rsi"]         * s_rsi
        + weights["volume"]      * s_vol
        + weights["sentiment"]   * s_news
        - weights["vol_penalty"] * vol_penalty
    )
    score = round(max(0.0, min(100.0, raw_score * 100)), 1)

    commodity_names = {v: k for k, v in COMMODITY_UNIVERSE.items()}
    display = data.get("display_name") or commodity_names.get(ticker) or ticker

    return {
        "ticker": ticker,
        "name": display,
        "asset_class": data.get("asset_class", "Stock/ETF"),
        "price": round(float(closes.iloc[-1]), 2),
        "return_7d": ret_7d,
        "return_30d": ret_30d,
        "return_90d": ret_90d,
        "rsi": round(rsi, 1),
        "volatility": round(volatility, 3),
        "score": score,
        "signals": _build_signals(ret_7d, ret_30d, ret_90d, rsi, s_vol, s_news, volatility, time_horizon),
        "history": hist,
        "news": data.get("news", []),
    }


def analyze_crypto_asset(coin: dict, risk_level: str, time_horizon: str) -> dict:
    weights = HORIZON_WEIGHTS[time_horizon][risk_level]

    ret_7d  = (coin.get("price_change_percentage_7d_in_currency")  or 0) / 100
    ret_30d = (coin.get("price_change_percentage_30d_in_currency") or 0) / 100
    ret_24h = (coin.get("price_change_percentage_24h") or 0) / 100
    ret_90d = ret_30d  # CoinGecko free tier doesn't provide 90d; use 30d as proxy

    vol_24h = coin.get("total_volume") or 0
    mcap    = coin.get("market_cap") or 1
    s_vol   = min(1.0, (vol_24h / mcap) / 0.5)

    s_ret7  = score_return(ret_7d,  -0.6, 0.6)
    s_ret30 = score_return(ret_30d, -0.7, 0.7)
    s_ret90 = score_return(ret_90d, -0.7, 0.7)
    s_news  = 0.5

    raw_score = (
        weights["return_7d"]  * s_ret7
        + weights["return_30d"]  * s_ret30
        + weights["return_90d"]  * s_ret90
        + weights["rsi"]         * 0.5
        + weights["volume"]      * s_vol
        + weights["sentiment"]   * s_news
    )
    score = round(max(0.0, min(100.0, raw_score * 100)), 1)

    signals = []
    if time_horizon == "short":
        if ret_24h > 0.05:
            signals.append(f"+{ret_24h:.1%} today — momentum")
        elif ret_24h < -0.05:
            signals.append(f"{ret_24h:.1%} today")
        if ret_7d > 0.12:
            signals.append(f"+{ret_7d:.1%} this week — strong move")
        if s_vol > 0.7:
            signals.append("High volume vs market cap")
    else:
        if ret_30d > 0.15:
            signals.append(f"+{ret_30d:.1%} this month — sustained growth")
        elif ret_30d < -0.10:
            signals.append(f"{ret_30d:.1%} this month — weak trend")
        if ret_7d > 0 and ret_30d > 0:
            signals.append("Positive across 7d & 30d")
        mcap_b = (coin.get("market_cap") or 0) / 1e9
        if mcap_b > 50:
            signals.append(f"Large cap (${mcap_b:.0f}B) — lower risk")

    return {
        "ticker": coin["symbol"].upper(),
        "name": coin["name"],
        "asset_class": "Crypto",
        "price": coin.get("current_price", 0),
        "return_7d": ret_7d,
        "return_30d": ret_30d,
        "return_90d": ret_90d,
        "rsi": 50.0,
        "volatility": abs(ret_30d) * 2,
        "score": score,
        "signals": signals,
        "history": None,
        "news": [],
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
    time_horizon: str = "short",
) -> list[dict]:
    scored = []

    for d in stock_data:
        d["asset_class"] = "Stock/ETF"
        try:
            scored.append(analyze_yfinance_asset(d, risk_level, time_horizon))
        except Exception:
            continue

    if include_commodities:
        for d in commodity_data:
            d["asset_class"] = "Commodity"
            try:
                scored.append(analyze_yfinance_asset(d, risk_level, time_horizon))
            except Exception:
                continue

    if include_crypto:
        for coin in crypto_raw:
            try:
                scored.append(analyze_crypto_asset(coin, risk_level, time_horizon))
            except Exception:
                continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
