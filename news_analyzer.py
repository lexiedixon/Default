"""Claude API-powered news analysis for investment recommendations."""

import json
import anthropic
import streamlit as st


def _build_prompt(ticker: str, asset_name: str, news_items: list) -> str:
    headlines = []
    for item in news_items[:6]:
        title = item.get("title", "")
        publisher = item.get("publisher", "")
        if title:
            headlines.append(f"- {title}" + (f" ({publisher})" if publisher else ""))
    if not headlines:
        return ""
    return f"""Analyze these recent news headlines for {asset_name} ({ticker}) and provide a brief investment-relevant summary.

Headlines:
{chr(10).join(headlines)}

Respond with ONLY a JSON object — no markdown fences, no extra text:
{{
  "sentiment": <float from -1.0 to 1.0>,
  "summary": "<2-3 sentence summary of what's happening and its investment relevance>",
  "key_events": ["<concise event 1>", "<concise event 2>"],
  "outlook": "<one of: Bullish, Mildly Bullish, Neutral, Mildly Bearish, Bearish>"
}}"""


def _call_claude(prompt: str, api_key: str) -> dict | None:
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Strip accidental markdown code fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _analyze_cached(ticker: str, asset_name: str, headlines_json: str, api_key: str) -> dict | None:
    """30-minute cached Claude call. headlines_json is a stable JSON string of headlines."""
    try:
        news_items = json.loads(headlines_json)
    except Exception:
        return None
    prompt = _build_prompt(ticker, asset_name, news_items)
    if not prompt:
        return None
    return _call_claude(prompt, api_key)


def enrich_picks_with_news_analysis(picks: list[dict], api_key: str) -> list[dict]:
    """Add Claude news analysis to each pick that has news. Mutates picks in place."""
    if not api_key:
        return picks
    for pick in picks:
        news = pick.get("news", [])
        if not news:
            continue
        headlines_json = json.dumps([
            {"title": n.get("title", ""), "publisher": n.get("publisher", "")}
            for n in news[:6]
        ])
        analysis = _analyze_cached(pick["ticker"], pick["name"], headlines_json, api_key)
        if analysis:
            pick["news_analysis"] = analysis
    return picks
