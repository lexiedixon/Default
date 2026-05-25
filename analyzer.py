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

# Sell scoring weights: higher score = stronger exit/avoid signal.
SELL_WEIGHTS = {
    "short": {
        "Conservative": {"rsi_ext": 0.35, "neg_7d": 0.25, "neg_30d": 0.15, "neg_90d": 0.00, "neg_sent": 0.15, "vol_risk": 0.10},
        "Moderate":     {"rsi_ext": 0.25, "neg_7d": 0.35, "neg_30d": 0.15, "neg_90d": 0.00, "neg_sent": 0.15, "vol_risk": 0.10},
        "Aggressive":   {"rsi_ext": 0.15, "neg_7d": 0.45, "neg_30d": 0.15, "neg_90d": 0.00, "neg_sent": 0.15, "vol_risk": 0.10},
    },
    "long": {
        "Conservative": {"rsi_ext": 0.15, "neg_7d": 0.00, "neg_30d": 0.30, "neg_90d": 0.40, "neg_sent": 0.05, "vol_risk": 0.10},
        "Moderate":     {"rsi_ext": 0.10, "neg_7d": 0.05, "neg_30d": 0.25, "neg_90d": 0.40, "neg_sent": 0.05, "vol_risk": 0.15},
        "Aggressive":   {"rsi_ext": 0.05, "neg_7d": 0.10, "neg_30d": 0.20, "neg_90d": 0.40, "neg_sent": 0.05, "vol_risk": 0.20},
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


def generate_explanation(asset: dict, time_horizon: str) -> str:
    """Build a plain-English explanation of why this asset scored as it did."""
    name     = asset["name"]
    score    = asset["score"]
    ret_7d   = asset["return_7d"]
    ret_30d  = asset["return_30d"]
    ret_90d  = asset["return_90d"]
    rsi      = asset["rsi"]
    vol      = asset["volatility"]
    is_crypto = asset["asset_class"] == "Crypto"

    # Opening sentence based on score band
    if score >= 70:
        opener = f"**{name}** ranks as a strong opportunity this cycle"
    elif score >= 55:
        opener = f"**{name}** presents a solid opportunity"
    elif score >= 40:
        opener = f"**{name}** shows a moderate opportunity"
    else:
        opener = f"**{name}** scores below average — worth monitoring but not a clear buy signal"

    reasons, caveats = [], []

    if time_horizon == "short":
        if ret_7d > 0.08:
            reasons.append(f"an exceptional **+{ret_7d:.1%} gain this week** signals strong near-term buying pressure")
        elif ret_7d > 0.03:
            reasons.append(f"a **+{ret_7d:.1%} weekly gain** shows emerging upward momentum")
        elif ret_7d < -0.08:
            reasons.append(f"a **{ret_7d:.1%} weekly dip** has pushed it into a potential bounce zone")
        elif ret_7d < -0.03:
            reasons.append(f"recent weakness (**{ret_7d:.1%} this week**) may be creating a short-term entry point")

        if not is_crypto:
            if 45 <= rsi <= 62:
                reasons.append(f"an RSI of **{rsi:.0f}** sits in a healthy momentum zone with room to run before becoming overbought")
            elif rsi < 35:
                reasons.append(f"an RSI of **{rsi:.0f}** signals oversold conditions — a near-term reversal is statistically likely")
            elif rsi > 70:
                caveats.append(f"the RSI of **{rsi:.0f}** is in overbought territory, meaning a pullback is possible before the next leg up")

        if vol > 0.45:
            caveats.append(f"annualized volatility of **{vol:.0%}** makes this a higher-risk trade — size positions accordingly")
        if ret_30d < -0.10:
            caveats.append(f"the 30-day trend is still negative (**{ret_30d:.1%}**), so confirm momentum before entering")

    else:  # long
        if ret_90d > 0.25:
            reasons.append(f"a **+{ret_90d:.1%} return over 3 months** demonstrates a strong, sustained uptrend")
        elif ret_90d > 0.08:
            reasons.append(f"a steady **+{ret_90d:.1%} quarterly gain** shows consistent upward momentum")
        elif ret_90d < -0.10:
            caveats.append(f"a **{ret_90d:.1%} decline over 3 months** is a headwind — the long-term thesis needs a catalyst to reverse")

        if ret_30d > 0 and ret_90d > 0:
            reasons.append("both the **30-day and 90-day returns are positive**, confirming the uptrend is holding across timeframes")
        elif ret_30d < 0 < ret_90d:
            caveats.append("the **30-day return has turned negative** even as the 90-day remains positive — watch for trend continuation")

        if not is_crypto and vol < 0.15:
            reasons.append(f"low annualized volatility of **{vol:.0%}** makes this a stable, lower-risk long-term hold")
        elif vol > 0.50:
            caveats.append(f"annualized volatility of **{vol:.0%}** is high — appropriate for an aggressive allocation only")

        if not is_crypto and rsi > 70:
            caveats.append(f"an RSI of **{rsi:.0f}** suggests the asset is extended — consider scaling in gradually rather than buying all at once")
        elif not is_crypto and 40 <= rsi <= 60:
            reasons.append(f"an RSI of **{rsi:.0f}** is in a neutral, healthy zone — not overextended")

    # Score sentence
    if score >= 70:
        score_line = f"These signals align strongly, producing an opportunity score of **{score}/100**."
    elif score >= 55:
        score_line = f"On balance the positives outweigh the risks, resulting in a score of **{score}/100**."
    elif score >= 40:
        score_line = f"Mixed signals keep the score at **{score}/100** — this is a watchlist candidate rather than a high-conviction pick."
    else:
        score_line = f"Weak signals across the board produce a low score of **{score}/100** — consider waiting for a clearer setup."

    # Assemble
    parts = [opener + "."]
    if reasons:
        parts.append("It was selected because " + (", and ".join(reasons)) + ".")
    if caveats:
        parts.append("Note: " + " Additionally, ".join(caveats) + ".")
    parts.append(score_line)
    return " ".join(parts)


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

    asset = {
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
    asset["explanation"] = generate_explanation(asset, time_horizon)
    return asset


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

    asset = {
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
    asset["explanation"] = generate_explanation(asset, time_horizon)
    return asset


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


# ── Sell / Exit Signal Scoring ────────────────────────────────────────────────

def _score_rsi_extended(rsi: float) -> float:
    """0 = neutral/oversold, 1 = heavily overbought (RSI ≥ 82)."""
    if rsi <= 60:
        return 0.0
    if rsi >= 82:
        return 1.0
    return (rsi - 60) / 22


def _score_neg_return(pct: float, worst: float = -0.40) -> float:
    """0 = zero or positive, 1 = at-or-below worst."""
    if pct >= 0:
        return 0.0
    return min(1.0, abs(pct) / abs(worst))


def _score_vol_risk(volatility: float) -> float:
    return min(1.0, volatility / 0.8)


def _build_sell_signals(
    ret_7d: float,
    ret_30d: float,
    ret_90d: float,
    rsi: float,
    sentiment_norm: float,
    volatility: float,
    time_horizon: str,
) -> list[str]:
    signals = []
    if time_horizon == "short":
        if rsi > 75:
            signals.append(f"RSI {rsi:.0f} — heavily overbought")
        elif rsi > 68:
            signals.append(f"RSI {rsi:.0f} — entering overbought zone")
        if ret_7d < -0.08:
            signals.append(f"{ret_7d:.1%} this week — sharp decline")
        elif ret_7d < -0.03:
            signals.append(f"{ret_7d:.1%} this week — weakening")
        if ret_7d > 0.12 and rsi > 68:
            signals.append(f"Up +{ret_7d:.1%} but RSI {rsi:.0f} — overextended")
        if sentiment_norm < 0.35:
            signals.append("Negative news flow")
        if volatility > 0.45:
            signals.append(f"High volatility ({volatility:.0%}) — elevated risk")
    else:
        if ret_90d < -0.15:
            signals.append(f"{ret_90d:.1%} over 3 months — sustained downtrend")
        elif ret_90d < -0.05:
            signals.append(f"{ret_90d:.1%} quarterly — trend deteriorating")
        if ret_30d < 0 and ret_90d < 0:
            signals.append("Both 30d & 90d negative — downtrend confirmed")
        if rsi > 72:
            signals.append(f"RSI {rsi:.0f} — may be topping out")
        if volatility > 0.50:
            signals.append(f"Very high volatility ({volatility:.0%})")
    return signals


def generate_sell_explanation(asset: dict, time_horizon: str) -> str:
    name    = asset["name"]
    score   = asset["score"]
    ret_7d  = asset["return_7d"]
    ret_30d = asset["return_30d"]
    ret_90d = asset["return_90d"]
    rsi     = asset["rsi"]
    vol     = asset["volatility"]
    is_crypto = asset["asset_class"] == "Crypto"

    if score >= 70:
        opener = f"**{name}** shows strong exit signals"
    elif score >= 50:
        opener = f"**{name}** shows moderate exit signals worth considering"
    else:
        opener = f"**{name}** has some cautionary flags, though conviction is low"

    reasons, caveats = [], []

    if time_horizon == "short":
        if not is_crypto:
            if rsi > 75:
                reasons.append(f"an RSI of **{rsi:.0f}** is deeply overbought — historically a precursor to a pullback")
            elif rsi > 68:
                reasons.append(f"an RSI of **{rsi:.0f}** is entering overbought territory, limiting near-term upside")
        if ret_7d < -0.08:
            reasons.append(f"a sharp **{ret_7d:.1%} decline this week** signals accelerating selling pressure")
        elif ret_7d < -0.03:
            reasons.append(f"a **{ret_7d:.1%} weekly decline** shows near-term momentum has turned negative")
        if ret_7d > 0.12 and rsi > 68:
            reasons.append(f"after a **+{ret_7d:.1%} surge this week** against an overbought RSI, profit-taking risk is elevated")
        if vol > 0.45:
            caveats.append(f"high volatility (**{vol:.0%}** annualized) means a snap-back rally is also possible")
    else:
        if ret_90d < -0.15:
            reasons.append(f"a **{ret_90d:.1%} decline over 3 months** represents a sustained and confirmed downtrend")
        elif ret_90d < -0.05:
            reasons.append(f"a **{ret_90d:.1%} quarterly return** shows the long-term uptrend has stalled")
        if ret_30d < 0 and ret_90d < 0:
            reasons.append("both the **30-day and 90-day returns are negative**, confirming this is not a temporary dip")
        if not is_crypto and rsi > 72:
            reasons.append(f"an RSI of **{rsi:.0f}** suggests the asset is overextended relative to its weakening trend")
        if vol > 0.50:
            reasons.append(f"annualized volatility of **{vol:.0%}** makes large drawdowns likely for long-term holders")

    if score >= 70:
        score_line = f"These factors combine for a sell signal score of **{score}/100** — a high-conviction exit."
    elif score >= 50:
        score_line = f"The exit case is moderately strong at **{score}/100** — consider reducing or exiting the position."
    else:
        score_line = f"With a score of **{score}/100**, treat this as a watchlist flag rather than an urgent exit."

    parts = [opener + "."]
    if reasons:
        parts.append("Key reasons: " + (", and ".join(reasons)) + ".")
    if caveats:
        parts.append("Caveat: " + " ".join(caveats) + ".")
    parts.append(score_line)
    return " ".join(parts)


def _sell_yfinance(data: dict, risk_level: str, time_horizon: str) -> dict:
    hist   = data["history"]
    closes = hist["Close"]
    ticker = data["ticker"]
    w      = SELL_WEIGHTS[time_horizon][risk_level]

    ret_7d  = (closes.iloc[-1] / closes.iloc[-6]  - 1) if len(closes) >= 6  else 0.0
    ret_30d = (closes.iloc[-1] / closes.iloc[-22] - 1) if len(closes) >= 22 else 0.0
    ret_90d = (closes.iloc[-1] / closes.iloc[0]   - 1) if len(closes) >= 60 else ret_30d
    volatility = closes.pct_change().std() * np.sqrt(252)
    rsi = calculate_rsi(closes)
    sentiment_norm = (score_news_sentiment(data.get("news", [])) + 1) / 2

    raw = (
        w["rsi_ext"]  * _score_rsi_extended(rsi)
        + w["neg_7d"]   * _score_neg_return(ret_7d,  -0.30)
        + w["neg_30d"]  * _score_neg_return(ret_30d, -0.40)
        + w["neg_90d"]  * _score_neg_return(ret_90d, -0.50)
        + w["neg_sent"] * (1.0 - sentiment_norm)
        + w["vol_risk"] * _score_vol_risk(volatility)
    )
    score = round(max(0.0, min(100.0, raw * 100)), 1)

    commodity_names = {v: k for k, v in COMMODITY_UNIVERSE.items()}
    display = data.get("display_name") or commodity_names.get(ticker) or ticker

    asset = {
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
        "signals": _build_sell_signals(ret_7d, ret_30d, ret_90d, rsi, sentiment_norm, volatility, time_horizon),
        "history": hist,
        "news": data.get("news", []),
    }
    asset["explanation"] = generate_sell_explanation(asset, time_horizon)
    return asset


def _sell_crypto(coin: dict, risk_level: str, time_horizon: str) -> dict:
    w = SELL_WEIGHTS[time_horizon][risk_level]

    ret_7d  = (coin.get("price_change_percentage_7d_in_currency")  or 0) / 100
    ret_30d = (coin.get("price_change_percentage_30d_in_currency") or 0) / 100
    ret_24h = (coin.get("price_change_percentage_24h") or 0) / 100
    ret_90d = ret_30d

    raw = (
        w["rsi_ext"]  * 0.5
        + w["neg_7d"]   * _score_neg_return(ret_7d,  -0.40)
        + w["neg_30d"]  * _score_neg_return(ret_30d, -0.50)
        + w["neg_90d"]  * _score_neg_return(ret_90d, -0.50)
        + w["neg_sent"] * 0.5
        + w["vol_risk"] * min(1.0, abs(ret_30d) * 2)
    )
    score = round(max(0.0, min(100.0, raw * 100)), 1)

    signals = []
    if time_horizon == "short":
        if ret_24h < -0.05:
            signals.append(f"{ret_24h:.1%} today — sharp drop")
        if ret_7d < -0.12:
            signals.append(f"{ret_7d:.1%} this week — momentum negative")
        if ret_7d > 0.15:
            signals.append(f"Up +{ret_7d:.1%} this week — overextended?")
    else:
        if ret_30d < -0.15:
            signals.append(f"{ret_30d:.1%} this month — sustained weakness")
        if ret_7d < 0 and ret_30d < 0:
            signals.append("Both 7d & 30d negative — confirmed downtrend")

    asset = {
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
    asset["explanation"] = generate_sell_explanation(asset, time_horizon)
    return asset


def get_sell_picks(
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
            scored.append(_sell_yfinance(d, risk_level, time_horizon))
        except Exception:
            continue
    if include_commodities:
        for d in commodity_data:
            d["asset_class"] = "Commodity"
            try:
                scored.append(_sell_yfinance(d, risk_level, time_horizon))
            except Exception:
                continue
    if include_crypto:
        for coin in crypto_raw:
            try:
                scored.append(_sell_crypto(coin, risk_level, time_horizon))
            except Exception:
                continue
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
