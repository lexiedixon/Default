import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from data_fetcher import (
    fetch_all_stocks,
    fetch_all_commodities,
    fetch_crypto_data,
    STOCK_UNIVERSE,
)
from analyzer import get_top_picks

st.set_page_config(
    page_title="Investment Finder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
    }
    .score-high { background: #d4edda; color: #155724; }
    .score-mid  { background: #fff3cd; color: #856404; }
    .score-low  { background: #f8d7da; color: #721c24; }
    .signal-chip {
        display: inline-block;
        background: #e9ecef;
        color: #495057;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.8em;
    }
    .metric-up { color: #28a745; font-weight: 600; }
    .metric-down { color: #dc3545; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.1em; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Asset Classes")
    include_stocks = st.checkbox("Stocks & ETFs", value=True)
    include_crypto = st.checkbox("Cryptocurrencies", value=True)
    include_commodities = st.checkbox("Commodities", value=True)

    if include_stocks:
        st.subheader("Stock Categories")
        selected_categories = st.multiselect(
            "Include",
            list(STOCK_UNIVERSE.keys()),
            default=["ETFs", "Tech", "Finance", "Energy", "Healthcare"],
        )
    else:
        selected_categories = []

    st.subheader("Strategy")
    risk_level = st.select_slider(
        "Risk Tolerance",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate",
    )

    top_n = st.slider("Top recommendations to show", 5, 20, 10)

    st.divider()
    refresh_clicked = st.button("🔄 Refresh Data", type="primary", use_container_width=True)
    st.caption("Data auto-caches for 5 minutes.")
    st.caption("Stocks & commodities: Yahoo Finance\nCrypto: CoinGecko")
    st.markdown("---")
    st.markdown(
        "⚠️ **Disclaimer:** This app is for informational purposes only. "
        "Not financial advice. Always do your own research."
    )


# ── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_all_data(categories: tuple, _include_crypto: bool, _include_commodities: bool):
    live_flags = []

    if categories:
        stock_data, s_live = fetch_all_stocks(list(categories))
        live_flags.append(s_live)
    else:
        stock_data, s_live = [], False

    if _include_commodities:
        commodity_data, c_live = fetch_all_commodities()
        live_flags.append(c_live)
    else:
        commodity_data = []

    if _include_crypto:
        crypto_raw, cr_live = fetch_crypto_data()
        live_flags.append(cr_live)
    else:
        crypto_raw = []

    any_live = any(live_flags)
    return stock_data, commodity_data, crypto_raw, any_live


if refresh_clicked:
    st.cache_data.clear()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📈 Investment Opportunity Finder")
st.markdown(
    f"Real-time analysis across stocks, crypto, and commodities — "
    f"updated {datetime.now().strftime('%b %d, %Y %H:%M')}"
)

if not (include_stocks or include_crypto or include_commodities):
    st.warning("Select at least one asset class in the sidebar.")
    st.stop()

with st.spinner("Fetching real-time market data…"):
    stock_data, commodity_data, crypto_raw, any_live = load_all_data(
        tuple(selected_categories),
        include_crypto,
        include_commodities,
    )

if not any_live:
    st.info(
        "⚠️ **Demo Mode** — Could not reach live market APIs from this environment. "
        "Showing realistic synthetic data. Run the app locally for real-time prices.",
        icon="📊",
    )

if not stock_data and not commodity_data and not crypto_raw:
    st.error("Could not fetch any market data. Check your internet connection and try refreshing.")
    st.stop()

short_picks = get_top_picks(
    stock_data, commodity_data, crypto_raw,
    risk_level=risk_level, include_crypto=include_crypto,
    include_commodities=include_commodities, top_n=top_n, time_horizon="short",
)
long_picks = get_top_picks(
    stock_data, commodity_data, crypto_raw,
    risk_level=risk_level, include_crypto=include_crypto,
    include_commodities=include_commodities, top_n=top_n, time_horizon="long",
)


# ── Market Pulse ─────────────────────────────────────────────────────────────
st.subheader("📊 Market Pulse")

pulse_tickers = {"S&P 500": "SPY", "Nasdaq": "QQQ", "Gold": "GLD"}
all_data_map = {d["ticker"]: d for d in stock_data + commodity_data}
crypto_map = {c["symbol"].upper(): c for c in crypto_raw}

pulse_cols = st.columns(5)
pulse_items = list(pulse_tickers.items())

for i, (label, ticker) in enumerate(pulse_items):
    d = all_data_map.get(ticker)
    if d:
        closes = d["history"]["Close"]
        price = closes.iloc[-1]
        chg = (closes.iloc[-1] / closes.iloc[-2] - 1) if len(closes) >= 2 else 0
        color = "normal" if chg >= 0 else "inverse"
        pulse_cols[i].metric(label, f"${price:,.2f}", f"{chg:+.2%}", delta_color=color)

# Add BTC and ETH to pulse
for i, sym in enumerate(["BTC", "ETH"]):
    coin = crypto_map.get(sym)
    if coin and include_crypto:
        price = coin.get("current_price", 0)
        chg = (coin.get("price_change_percentage_24h") or 0) / 100
        col_idx = len(pulse_items) + i
        if col_idx < 5:
            color = "normal" if chg >= 0 else "inverse"
            pulse_cols[col_idx].metric(sym, f"${price:,.0f}", f"{chg:+.2%}", delta_color=color)


# ── Recommendations Tabs ─────────────────────────────────────────────────────
st.subheader(f"🏆 Top Recommendations — {risk_level} Strategy")

tab_short, tab_long = st.tabs(["⚡ Short-Term  (Days – Weeks)", "📈 Long-Term  (Months – Years)"])


def render_picks(picks: list[dict], time_horizon: str):
    if not picks:
        st.info("No recommendations available with current filters.")
        return

    if time_horizon == "short":
        st.caption(
            "Ranked by recent momentum, volume surge, and news sentiment. "
            "Best suited for active traders looking for near-term price moves."
        )
    else:
        st.caption(
            "Ranked by sustained 90-day trend, consistency, and low volatility. "
            "Best suited for investors building positions to hold over months."
        )

    for rank, asset in enumerate(picks, 1):
        score = asset["score"]
        badge_class = "score-high" if score >= 65 else ("score-mid" if score >= 40 else "score-low")

        with st.expander(
            f"#{rank}  {asset['name']} ({asset['ticker']})  —  "
            f"{asset['asset_class']}  |  Score: {score}/100",
            expanded=(rank <= 3),
        ):
            col1, col2, col3, col4 = st.columns(4)
            price = asset["price"]

            col1.metric("Price", f"${price:,.4f}" if price < 1 else f"${price:,.2f}")
            if time_horizon == "short":
                ret7 = asset["return_7d"]
                ret30 = asset["return_30d"]
                col2.metric("7-Day Return", f"{ret7:+.2%}", delta_color="normal" if ret7 >= 0 else "inverse")
                col3.metric("30-Day Return", f"{ret30:+.2%}", delta_color="normal" if ret30 >= 0 else "inverse")
            else:
                ret30 = asset["return_30d"]
                ret90 = asset["return_90d"]
                col2.metric("30-Day Return", f"{ret30:+.2%}", delta_color="normal" if ret30 >= 0 else "inverse")
                col3.metric("90-Day Return", f"{ret90:+.2%}", delta_color="normal" if ret90 >= 0 else "inverse")
            col4.metric("RSI", f"{asset['rsi']:.1f}")

            if asset["signals"]:
                st.markdown(
                    " ".join(f'<span class="signal-chip">{s}</span>' for s in asset["signals"]),
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<span class="score-badge {badge_class}">Opportunity Score: {score}/100</span>',
                unsafe_allow_html=True,
            )

            if asset["history"] is not None:
                hist = asset["history"].tail(60)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name=asset["ticker"],
                    increasing_line_color="#28a745",
                    decreasing_line_color="#dc3545",
                ))
                fig.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_rangeslider_visible=False,
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            news = asset.get("news", [])
            if news:
                st.markdown("**Recent News**")
                for item in news[:4]:
                    title = item.get("title", "")
                    link = item.get("link", "#")
                    publisher = item.get("publisher", "")
                    ts = item.get("providerPublishTime", 0)
                    date_str = datetime.fromtimestamp(ts).strftime("%b %d") if ts else ""
                    st.markdown(f"- [{title}]({link})  <small style='color:gray'>{publisher} · {date_str}</small>", unsafe_allow_html=True)


with tab_short:
    render_picks(short_picks, "short")

with tab_long:
    render_picks(long_picks, "long")


# ── All Assets Table ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Full Asset Scorecard")

all_short = get_top_picks(
    stock_data, commodity_data, crypto_raw,
    risk_level=risk_level, include_crypto=include_crypto,
    include_commodities=include_commodities, top_n=500, time_horizon="short",
)
all_long = get_top_picks(
    stock_data, commodity_data, crypto_raw,
    risk_level=risk_level, include_crypto=include_crypto,
    include_commodities=include_commodities, top_n=500, time_horizon="long",
)
all_scored = all_short  # use short for the scatter plot reference

long_score_map = {a["ticker"]: a["score"] for a in all_long}

if all_short:
    df = pd.DataFrame([
        {
            "Ticker": a["ticker"],
            "Name": a["name"],
            "Class": a["asset_class"],
            "Price": a["price"],
            "7D Return": a["return_7d"],
            "30D Return": a["return_30d"],
            "90D Return": a["return_90d"],
            "RSI": a["rsi"],
            "Short-Term Score": a["score"],
            "Long-Term Score": long_score_map.get(a["ticker"], 0.0),
        }
        for a in all_short
    ])

    def color_return(val):
        color = "#28a745" if val > 0 else "#dc3545"
        return f"color: {color}; font-weight: 600"

    def color_score(val):
        if val >= 65:
            return "background-color: #d4edda; color: #155724"
        if val >= 40:
            return "background-color: #fff3cd; color: #856404"
        return "background-color: #f8d7da; color: #721c24"

    styled = (
        df.style
        .format({
            "Price": lambda v: f"${v:,.4f}" if v < 1 else f"${v:,.2f}",
            "7D Return": "{:+.2%}",
            "30D Return": "{:+.2%}",
            "90D Return": "{:+.2%}",
            "RSI": "{:.1f}",
            "Short-Term Score": "{:.1f}",
            "Long-Term Score": "{:.1f}",
        })
        .map(color_return, subset=["7D Return", "30D Return", "90D Return"])
        .map(color_score, subset=["Short-Term Score", "Long-Term Score"])
    )
    st.dataframe(styled, use_container_width=True, height=450)


# ── Scatter Plot ──────────────────────────────────────────────────────────────
st.subheader("📉 Risk vs. Return (30-Day)")

if all_scored:
    scatter_df = pd.DataFrame([
        {
            "Name": a["name"],
            "Ticker": a["ticker"],
            "30D Return": a["return_30d"] * 100,
            "Volatility": a["volatility"] * 100,
            "Score": a["score"],
            "Class": a["asset_class"],
        }
        for a in all_scored
    ])

    fig2 = px.scatter(
        scatter_df,
        x="Volatility",
        y="30D Return",
        color="Class",
        size="Score",
        hover_name="Name",
        hover_data={"Ticker": True, "Score": True, "Volatility": ":.1f", "30D Return": ":.2f"},
        labels={"Volatility": "Annualized Volatility (%)", "30D Return": "30-Day Return (%)"},
        title="",
        height=400,
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Data sources: Yahoo Finance (stocks, ETFs, commodities), CoinGecko (crypto). "
    "Scores are algorithmic and not financial advice."
)
