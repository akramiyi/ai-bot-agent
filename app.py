import os
import re
import json
import requests
import urllib.parse
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
from openai import OpenAI
import yt_dlp
from dotenv import load_dotenv
from functools import wraps

import sys
sys.modules['google.generativeai'] = None
sys.modules['anthropic'] = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'astra-level-9-super-secret')

PROFILES_DIR = 'profiles'
os.makedirs(PROFILES_DIR, exist_ok=True)

# ---------- Simple TTL Cache ----------
def ttl_cache(seconds):
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            if key in cache and cache[key]['expires'] > now:
                return cache[key]['value']
            result = func(*args, **kwargs)
            cache[key] = {'value': result, 'expires': now + seconds}
            return result
        return wrapper
    return decorator

# ---------- NVIDIA AI Client ----------
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

SYSTEM_PROMPT = """You are Astra, a Level 9 AI assistant created for Akram Ansari.
You are smart, helpful, and speak in Hinglish (Hindi + English mix).
You have deep knowledge of stocks, crypto, coding, study topics, and general life advice.
Keep replies concise, useful, and conversational. Use emojis naturally.
User Profile: Akram Ansari, B.Tech CS student at Brainware University (2024-2028), from Chhapra Bihar.
Friends: Rosidul Islam (Best Friend), Aryan Raj (Editor), Kaif Ali, Munshi Insiyat."""

# ---------- Conversation Memory ----------
conversation_history = []
MAX_HISTORY = 10

def ask_nvidia_stream(prompt, system_message=None):
    global conversation_history
    if not system_message:
        system_message = SYSTEM_PROMPT
    
    conversation_history.append({"role": "user", "content": prompt})
    if len(conversation_history) > MAX_HISTORY * 2:
        conversation_history = conversation_history[-MAX_HISTORY * 2:]

    try:
        messages = [{"role": "system", "content": system_message}] + conversation_history
        response = client.chat.completions.create(
            model="meta/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            stream=True
        )
        full_reply = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_reply += content
                yield content
        conversation_history.append({"role": "assistant", "content": full_reply})
    except Exception as e:
        yield f"AI Error: {str(e)}"

# ---------- Financial Module ----------
@ttl_cache(300)
def get_stock_price(symbol):
    try:
        indian = ['RELIANCE', 'TCS', 'INFY', 'WIPRO', 'HDFCBANK', 'TATAMOTORS', 'ADANI', 'BAJAJ']
        if symbol.upper() in indian:
            symbol = f"{symbol.upper()}.NS"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        res = data['chart']['result'][0]['meta']
        price = res['regularMarketPrice']
        currency = res['currency']
        change = res.get('regularMarketChange', 0)
        pct = res.get('regularMarketChangePercent', 0)
        arrow = "📈" if change >= 0 else "📉"
        return f"{arrow} **{symbol.replace('.NS','')}**\nPrice: {currency} {price:,.2f}\nChange: {change:+.2f} ({pct:+.2f}%)"
    except:
        return f"Could not fetch stock {symbol}. Check symbol name."

@ttl_cache(300)
def get_crypto_price(coin):
    try:
        coin_id = coin.lower().strip()
        mapping = {
            'btc': 'bitcoin', 'eth': 'ethereum', 'doge': 'dogecoin',
            'sol': 'solana', 'bnb': 'binancecoin', 'xrp': 'ripple',
            'ada': 'cardano', 'avax': 'avalanche-2', 'matic': 'matic-network'
        }
        coin_id = mapping.get(coin_id, coin_id)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd,inr&include_24hr_change=true"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if coin_id not in data:
            return f"Crypto '{coin}' not found."
        prices = data[coin_id]
        change = prices.get('usd_24h_change', 0)
        arrow = "📈" if change >= 0 else "📉"
        return f"{arrow} **{coin_id.upper()}**\n💰 ${prices['usd']:,.2f} USD ({change:+.2f}%)\n🇮🇳 ₹{prices['inr']:,.2f} INR"
    except Exception as e:
        return f"Crypto price check failed: {str(e)}"

@ttl_cache(300)
def get_portfolio_summary():
    """Get a quick market overview"""
    coins = ['bitcoin', 'ethereum', 'solana']
    results = []
    try:
        ids = ','.join(coins)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true"
        data = requests.get(url, timeout=10).json()
        for c in coins:
            if c in data:
                p = data[c]
                ch = p.get('usd_24h_change', 0)
                arrow = "▲" if ch >= 0 else "▼"
                results.append(f"{c.upper()[:3]}: ${p['usd']:,.0f} {arrow}{abs(ch):.1f}%")
        return " | ".join(results)
    except:
        return "Market data unavailable"

# ---------- News Module ----------
@ttl_cache(1800)
def get_news(query=None, country="in"):
    api_key = os.getenv('GNEWS_API_KEY') or os.getenv('NEWS_API_KEY')
    if not api_key:
        return "⚠️ News API key missing. Add GNEWS_API_KEY to .env"
    if query:
        url = f"https://gnews.io/api/v4/search?q={urllib.parse.quote(query)}&token={api_key}&lang=en&max=4"
    else:
        url = f"https://gnews.io/api/v4/top-headlines?country={country}&token={api_key}&max=4"
    try:
        data = requests.get(url, timeout=10).json()
        articles = data.get('articles', [])
        if not articles:
            return "No news found."
        lines = []
        for a in articles:
            lines.append(f"📰 **{a['title']}**\n🔗 [Read More]({a['url']})\n")
        return "\n".join(lines)
    except:
        return "News error. Try again."

# ---------- Weather Module ----------
@ttl_cache(600)
def get_weather(city):
    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key:
        return "⚠️ Weather API key missing. Add WEATHER_API_KEY to .env"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        data = requests.get(url, timeout=10).json()
        if data.get('cod') != 200:
            return f"City '{city}' not found."
        temp = data['main']['temp']
        feels = data['main']['feels_like']
        humidity = data['main']['humidity']
        desc = data['weather'][0]['description'].title()
        icon_map = {
            'clear': '☀️', 'cloud': '☁️', 'rain': '🌧️',
            'thunder': '⛈️', 'snow': '❄️', 'mist': '🌫️', 'haze': '🌫️'
        }
        icon = '🌡️'
        for k, v in icon_map.items():
            if k in desc.lower():
                icon = v
                break
        return (f"{icon} **{city.title()} Weather**\n"
                f"🌡️ Temp: {temp}°C (feels {feels}°C)\n"
                f"💧 Humidity: {humidity}%\n"
                f"🌤️ {desc}")
    except:
        return "Weather check failed."

# ---------- Study Mode ----------
study_state = {"active": False, "remaining": 0, "total": 0, "subject": ""}
tasks = []  # Task list

def study_timer_thread(minutes, subject=""):
    global study_state
    study_state["active"] = True
    study_state["total"] = minutes * 60
    study_state["remaining"] = study_state["total"]
    study_state["subject"] = subject
    while study_state["remaining"] > 0 and study_state["active"]:
        time.sleep(1)
        study_state["remaining"] -= 1
    study_state["active"] = False

# ---------- Quiz Generator ----------
def generate_quiz(topic):
    prompt = f"""Generate 3 multiple choice quiz questions about: {topic}
Format EXACTLY like this (no extra text):
Q1: [question]
A) [option]
B) [option]  
C) [option]
D) [option]
Answer: [letter]

Q2: ...
Q3: ..."""
    try:
        response = client.chat.completions.create(
            model="meta/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": "You are a quiz generator. Output only the quiz, no extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=600,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Quiz generation failed: {str(e)}"

# ---------- YouTube Helper ----------
def get_youtube_embed_url(query):
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if 'entries' in info and info['entries']:
                video_id = info['entries'][0]['id']
                title = info['entries'][0].get('title', query)
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&controls=1&rel=0", title
    except:
        pass
    return None, None

# ---------- THE HTML ----------
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>▲ ASTRA LEVEL 9</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #00f0ff;
            --secondary: #ff00aa;
            --accent: #aaff00;
            --bg: #020408;
            --panel: rgba(0,240,255,0.04);
            --border: rgba(0,240,255,0.15);
            --text: #cce8f0;
            --text-dim: #3a6070;
            --glow: 0 0 20px rgba(0,240,255,0.5);
            --glow2: 0 0 20px rgba(255,0,170,0.5);
        }
        * { margin:0; padding:0; box-sizing:border-box; }
        html, body { height:100%; overflow:hidden; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Rajdhani', sans-serif;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }

        /* GRID BG */
        body::before {
            content:'';
            position:fixed; inset:0;
            background-image:
                linear-gradient(rgba(0,240,255,0.025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,240,255,0.025) 1px, transparent 1px);
            background-size: 44px 44px;
            pointer-events:none; z-index:0;
        }
        /* SCANLINES */
        body::after {
            content:'';
            position:fixed; inset:0;
            background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px);
            pointer-events:none; z-index:1;
        }

        .app {
            position: relative; z-index: 2;
            display: flex; flex-direction: column;
            height: 100vh; max-width: 960px;
            margin: 0 auto; width: 100%;
            padding: 10px 12px;
            gap: 8px;
        }

        /* HEADER */
        header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 10px 20px;
            border: 1px solid var(--border);
            background: var(--panel);
            backdrop-filter: blur(10px);
            clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 16px 100%, 0 calc(100% - 16px));
            flex-shrink: 0;
        }
        .logo {
            font-family: 'Orbitron', monospace;
            font-size: 1.3rem; font-weight: 900;
            color: var(--primary);
            text-shadow: var(--glow);
            letter-spacing: 3px;
        }
        .logo span { color: var(--secondary); }
        .header-right {
            display: flex; gap: 10px; align-items: center;
        }
        .status-pill {
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.65rem; color: var(--accent);
            border: 1px solid rgba(170,255,0,0.3);
            padding: 3px 10px; border-radius: 20px;
            background: rgba(170,255,0,0.06);
            animation: pulse 2s ease-in-out infinite;
        }
        @keyframes pulse { 0%,100%{opacity:0.6;} 50%{opacity:1;} }

        /* MARKET TICKER */
        .ticker-bar {
            background: rgba(0,240,255,0.04);
            border: 1px solid var(--border);
            padding: 6px 16px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.68rem;
            color: var(--text-dim);
            overflow: hidden;
            white-space: nowrap;
            flex-shrink: 0;
        }
        .ticker-inner { display: inline-block; animation: ticker 30s linear infinite; }
        @keyframes ticker { 0%{transform:translateX(100vw);} 100%{transform:translateX(-100%);} }
        .tick-up { color: var(--accent); }
        .tick-dn { color: var(--secondary); }

        /* NAV TABS */
        .nav-tabs {
            display: flex; gap: 6px;
            flex-shrink: 0;
        }
        .tab-btn {
            flex: 1;
            background: var(--panel);
            border: 1px solid var(--border);
            color: var(--text-dim);
            font-family: 'Orbitron', monospace;
            font-size: 0.6rem;
            padding: 7px 4px;
            cursor: pointer;
            letter-spacing: 1px;
            transition: all 0.2s;
            clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
        }
        .tab-btn.active, .tab-btn:hover {
            border-color: var(--primary);
            color: var(--primary);
            background: rgba(0,240,255,0.08);
            text-shadow: var(--glow);
        }

        /* PANELS */
        .panel {
            display: none; flex: 1;
            flex-direction: column;
            background: var(--panel);
            border: 1px solid var(--border);
            overflow: hidden;
            min-height: 0;
        }
        .panel.active { display: flex; }

        /* CHAT */
        #chatPanel { display: flex; flex-direction: column; }
        .chat-msgs {
            flex: 1; overflow-y: auto;
            padding: 14px;
            display: flex; flex-direction: column;
            gap: 10px;
            scroll-behavior: smooth;
            min-height: 0;
        }
        .chat-msgs::-webkit-scrollbar { width: 4px; }
        .chat-msgs::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 4px; }

        .msg {
            max-width: 82%; padding: 10px 16px;
            border-radius: 16px; font-size: 0.9rem;
            line-height: 1.5; word-wrap: break-word;
            animation: fadeUp 0.25s ease-out;
        }
        @keyframes fadeUp { from{opacity:0;transform:translateY(8px);} to{opacity:1;transform:translateY(0);} }
        .msg.user {
            align-self: flex-end;
            background: linear-gradient(135deg, var(--primary), #0088aa);
            color: #020408;
            font-weight: 600;
            border-bottom-right-radius: 4px;
        }
        .msg.bot {
            align-self: flex-start;
            background: rgba(0,240,255,0.05);
            border: 1px solid rgba(0,240,255,0.15);
            color: var(--text);
            border-bottom-left-radius: 4px;
        }
        .typing-dots {
            display: flex; gap: 5px; align-items: center;
            padding: 10px 16px;
            background: rgba(0,240,255,0.04);
            border: 1px solid rgba(0,240,255,0.12);
            border-radius: 16px; width: fit-content;
            border-bottom-left-radius: 4px;
        }
        .typing-dots span {
            width: 7px; height: 7px;
            background: var(--primary);
            border-radius: 50%;
            animation: bounce 1.2s infinite;
        }
        .typing-dots span:nth-child(2){animation-delay:0.2s;}
        .typing-dots span:nth-child(3){animation-delay:0.4s;}
        @keyframes bounce {
            0%,60%,100%{transform:translateY(0);opacity:0.4;}
            30%{transform:translateY(-7px);opacity:1;}
        }

        .chat-input-row {
            display: flex; gap: 8px;
            padding: 10px 12px;
            border-top: 1px solid var(--border);
            flex-shrink: 0;
        }
        .chat-input-row input {
            flex: 1;
            background: rgba(0,0,0,0.4);
            border: 1px solid rgba(0,240,255,0.25);
            border-radius: 30px;
            padding: 11px 18px;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.95rem;
            color: #fff;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .chat-input-row input:focus {
            border-color: var(--primary);
            box-shadow: var(--glow);
        }
        .icon-btn {
            background: rgba(0,240,255,0.08);
            border: 1px solid rgba(0,240,255,0.25);
            border-radius: 50%;
            width: 42px; height: 42px;
            color: var(--primary);
            font-size: 1rem; cursor: pointer;
            transition: all 0.2s; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
        }
        .icon-btn:hover { background: rgba(0,240,255,0.18); box-shadow: var(--glow); }
        .send-btn {
            background: linear-gradient(135deg, var(--primary), #0088aa);
            border: none;
            border-radius: 30px;
            padding: 0 20px;
            font-family: 'Orbitron', monospace;
            font-size: 0.7rem; font-weight: 700;
            color: #020408; cursor: pointer;
            transition: all 0.2s; flex-shrink: 0;
            letter-spacing: 1px;
        }
        .send-btn:hover { transform: scale(1.04); box-shadow: var(--glow); }

        /* STUDY PANEL */
        #studyPanel { padding: 16px; gap: 12px; overflow-y: auto; }
        .study-grid {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .study-card {
            background: rgba(0,240,255,0.04);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            display: flex; flex-direction: column; gap: 8px;
        }
        .study-card h3 {
            font-family: 'Orbitron', monospace;
            font-size: 0.7rem; color: var(--primary);
            letter-spacing: 2px;
        }
        .timer-display {
            font-family: 'Orbitron', monospace;
            font-size: 2.8rem; font-weight: 900;
            color: var(--primary);
            text-shadow: var(--glow);
            text-align: center;
        }
        .progress-bar-wrap {
            background: rgba(0,240,255,0.08);
            border-radius: 20px; height: 6px;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            border-radius: 20px;
            transition: width 1s linear;
        }
        .study-actions { display: flex; gap: 8px; flex-wrap: wrap; }
        .s-btn {
            flex: 1;
            background: rgba(0,240,255,0.08);
            border: 1px solid var(--border);
            color: var(--primary);
            font-family: 'Orbitron', monospace;
            font-size: 0.6rem;
            padding: 8px 6px; cursor: pointer;
            border-radius: 6px; letter-spacing: 1px;
            transition: all 0.2s;
        }
        .s-btn:hover { background: rgba(0,240,255,0.18); }
        .s-btn.danger { color: var(--secondary); border-color: rgba(255,0,170,0.3); }
        .s-btn.danger:hover { background: rgba(255,0,170,0.12); }
        .min-input {
            background: rgba(0,0,0,0.4);
            border: 1px solid var(--border);
            color: var(--primary);
            font-family: 'Orbitron', monospace;
            font-size: 0.85rem;
            padding: 8px 12px; border-radius: 6px;
            width: 100%; outline: none;
            text-align: center;
        }
        .task-list { display: flex; flex-direction: column; gap: 6px; }
        .task-item {
            display: flex; align-items: center; gap: 8px;
            padding: 8px 10px;
            background: rgba(0,240,255,0.03);
            border: 1px solid rgba(0,240,255,0.1);
            border-radius: 8px;
            font-size: 0.85rem;
        }
        .task-item.done { opacity: 0.4; text-decoration: line-through; }
        .task-check {
            width: 16px; height: 16px;
            border: 1px solid var(--primary);
            border-radius: 4px; cursor: pointer;
            flex-shrink: 0; display: flex; align-items: center; justify-content: center;
            font-size: 0.7rem; color: var(--primary);
        }
        .quiz-area {
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.78rem;
            color: var(--text);
            line-height: 1.8;
            white-space: pre-wrap;
            min-height: 120px;
        }
        .quiz-input-row { display: flex; gap: 8px; }
        .quiz-input {
            flex: 1;
            background: rgba(0,0,0,0.4);
            border: 1px solid var(--border);
            color: var(--primary);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.8rem;
            padding: 8px 12px; border-radius: 6px; outline: none;
        }

        /* STOCKS PANEL */
        #stocksPanel { padding: 16px; gap: 12px; overflow-y: auto; }
        .stocks-search-row { display: flex; gap: 8px; }
        .stock-input {
            flex: 1;
            background: rgba(0,0,0,0.4);
            border: 1px solid var(--border);
            color: var(--primary);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            padding: 10px 16px; border-radius: 8px; outline: none;
        }
        .stock-input:focus { border-color: var(--primary); box-shadow: var(--glow); }
        .stock-go-btn {
            background: linear-gradient(135deg, var(--primary), #0088aa);
            border: none; border-radius: 8px;
            padding: 0 18px;
            font-family: 'Orbitron', monospace;
            font-size: 0.65rem; font-weight: 700;
            color: #020408; cursor: pointer;
        }
        .quick-chips { display: flex; gap: 6px; flex-wrap: wrap; }
        .chip {
            background: rgba(0,240,255,0.06);
            border: 1px solid rgba(0,240,255,0.2);
            color: var(--primary);
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.7rem;
            padding: 5px 12px; border-radius: 20px;
            cursor: pointer; transition: all 0.2s;
        }
        .chip:hover { background: rgba(0,240,255,0.15); }
        .chip.crypto { border-color: rgba(255,0,170,0.3); color: var(--secondary); }
        .chip.crypto:hover { background: rgba(255,0,170,0.12); }
        .stock-result-box {
            background: rgba(0,240,255,0.03);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            line-height: 2;
            min-height: 80px;
        }

        /* MUSIC PANEL */
        #musicPanel { padding: 16px; gap: 12px; overflow-y: auto; }
        .music-search-row { display: flex; gap: 8px; }
        .music-input {
            flex: 1;
            background: rgba(0,0,0,0.4);
            border: 1px solid rgba(255,0,170,0.25);
            color: #fff;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.95rem;
            padding: 10px 16px; border-radius: 8px; outline: none;
        }
        .music-input:focus { border-color: var(--secondary); box-shadow: var(--glow2); }
        .music-go-btn {
            background: linear-gradient(135deg, var(--secondary), #880055);
            border: none; border-radius: 8px;
            padding: 0 18px;
            font-family: 'Orbitron', monospace;
            font-size: 0.65rem; font-weight: 700;
            color: #fff; cursor: pointer;
        }
        .music-moods { display: flex; gap: 6px; flex-wrap: wrap; }
        .mood-chip {
            background: rgba(255,0,170,0.06);
            border: 1px solid rgba(255,0,170,0.2);
            color: var(--secondary);
            font-size: 0.75rem;
            padding: 5px 14px; border-radius: 20px;
            cursor: pointer; transition: all 0.2s;
        }
        .mood-chip:hover { background: rgba(255,0,170,0.14); }
        .music-player-frame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,0,170,0.3);
            background: #000;
            aspect-ratio: 16/9;
        }
        .music-player-frame iframe {
            width: 100%; height: 100%; border: none;
        }
        .music-placeholder {
            width: 100%; aspect-ratio: 16/9;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            background: rgba(255,0,170,0.04);
            border: 1px dashed rgba(255,0,170,0.2);
            border-radius: 12px;
            color: rgba(255,0,170,0.4);
            font-family: 'Orbitron', monospace;
            font-size: 0.8rem;
            gap: 8px;
        }
        .music-placeholder .big-icon { font-size: 2.5rem; }

        /* STUDY TIMER FLOATING */
        #floatTimer {
            position: fixed;
            bottom: 80px; right: 16px;
            background: rgba(2,4,8,0.92);
            border: 1px solid var(--primary);
            border-radius: 14px;
            padding: 10px 16px;
            font-family: 'Orbitron', monospace;
            text-align: center;
            z-index: 999;
            display: none;
            box-shadow: var(--glow);
            min-width: 120px;
        }
        #floatTimer .f-label { font-size: 0.55rem; color: var(--primary); letter-spacing: 2px; }
        #floatTimer .f-time { font-size: 1.6rem; font-weight: 900; color: #fff; }
        #floatTimer .f-stop {
            background: var(--secondary);
            border: none; border-radius: 6px;
            padding: 4px 10px;
            color: #fff; font-size: 0.6rem;
            cursor: pointer; width: 100%; margin-top: 6px;
            font-family: 'Orbitron', monospace;
        }

        /* RESPONSIVE */
        @media (max-width: 600px) {
            .logo { font-size: 1rem; letter-spacing: 2px; }
            .study-grid { grid-template-columns: 1fr; }
            .msg { max-width: 92%; font-size: 0.83rem; }
            .tab-btn { font-size: 0.55rem; padding: 6px 2px; }
            .timer-display { font-size: 2.2rem; }
        }
    </style>
</head>
<body>
<div class="app">

    <!-- HEADER -->
    <header>
        <div class="logo">▲ ASTRA<span> L9</span></div>
        <div class="header-right">
            <div class="status-pill" id="statusPill">● ONLINE</div>
        </div>
    </header>

    <!-- TICKER -->
    <div class="ticker-bar">
        <div class="ticker-inner" id="tickerInner">Loading market data...</div>
    </div>

    <!-- NAV -->
    <div class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('chat', this)">💬 CHAT</button>
        <button class="tab-btn" onclick="switchTab('study', this)">📚 STUDY</button>
        <button class="tab-btn" onclick="switchTab('stocks', this)">📈 STOCKS</button>
        <button class="tab-btn" onclick="switchTab('music', this)">🎵 MUSIC</button>
    </div>

    <!-- CHAT PANEL -->
    <div class="panel active" id="chatPanel">
        <div class="chat-msgs" id="chatMsgs"></div>
        <div class="chat-input-row">
            <button class="icon-btn" onclick="startVoice()" title="Voice Input">🎤</button>
            <input type="text" id="chatInput" placeholder="Ask Astra anything..." autocomplete="off" enterkeyhint="send">
            <button class="send-btn" onclick="sendChat()">SEND</button>
        </div>
    </div>

    <!-- STUDY PANEL -->
    <div class="panel" id="studyPanel">
        <div class="study-grid">
            <!-- POMODORO TIMER -->
            <div class="study-card" style="grid-column: 1 / -1;">
                <h3>⏱ POMODORO TIMER</h3>
                <div class="timer-display" id="timerDisplay">25:00</div>
                <div class="progress-bar-wrap">
                    <div class="progress-bar-fill" id="progressBar" style="width:100%;"></div>
                </div>
                <input class="min-input" type="number" id="studyMins" value="25" min="1" max="120" placeholder="Minutes">
                <div class="study-actions">
                    <button class="s-btn" onclick="startStudyTimer()">▶ START</button>
                    <button class="s-btn danger" onclick="stopStudyTimer()">■ STOP</button>
                </div>
            </div>

            <!-- TASK LIST -->
            <div class="study-card">
                <h3>✅ TASK LIST</h3>
                <div class="task-list" id="taskList"></div>
                <div style="display:flex;gap:6px;margin-top:4px;">
                    <input class="min-input" type="text" id="taskInput" placeholder="Add task..." style="text-align:left;font-family:'Rajdhani',sans-serif;font-size:0.85rem;">
                    <button class="s-btn" style="flex:0;padding:8px 12px;" onclick="addTask()">+</button>
                </div>
            </div>

            <!-- QUIZ GENERATOR -->
            <div class="study-card">
                <h3>🧠 QUIZ GENERATOR</h3>
                <div class="quiz-input-row">
                    <input class="quiz-input" type="text" id="quizTopic" placeholder="Topic (e.g. Arrays, Python)">
                    <button class="s-btn" style="flex:0;padding:8px 14px;" onclick="generateQuiz()">GO</button>
                </div>
                <div class="quiz-area" id="quizArea">Enter a topic and click GO to generate quiz questions...</div>
            </div>
        </div>
    </div>

    <!-- STOCKS PANEL -->
    <div class="panel" id="stocksPanel">
        <div class="stocks-search-row">
            <input class="stock-input" type="text" id="stockInput" placeholder="Stock symbol (RELIANCE, NVDA, TSLA...)">
            <button class="stock-go-btn" onclick="fetchStock()">FETCH</button>
        </div>
        <div class="quick-chips">
            <span class="chip" onclick="quickStock('RELIANCE')">RELIANCE</span>
            <span class="chip" onclick="quickStock('TCS')">TCS</span>
            <span class="chip" onclick="quickStock('INFY')">INFY</span>
            <span class="chip" onclick="quickStock('NVIDIA')">NVIDIA</span>
            <span class="chip" onclick="quickStock('AAPL')">AAPL</span>
            <span class="chip" onclick="quickStock('TSLA')">TSLA</span>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <span class="chip crypto" onclick="quickCrypto('bitcoin')">₿ BTC</span>
            <span class="chip crypto" onclick="quickCrypto('ethereum')">Ξ ETH</span>
            <span class="chip crypto" onclick="quickCrypto('solana')">◎ SOL</span>
            <span class="chip crypto" onclick="quickCrypto('doge')">Ð DOGE</span>
        </div>
        <div class="stock-result-box" id="stockResult">
            <span style="color:var(--text-dim);">Select a stock or crypto above, or type a symbol...</span>
        </div>
    </div>

    <!-- MUSIC PANEL -->
    <div class="panel" id="musicPanel">
        <div class="music-search-row">
            <input class="music-input" type="text" id="musicInput" placeholder="Search song, artist, album...">
            <button class="music-go-btn" onclick="playMusic()">▶ PLAY</button>
        </div>
        <div class="music-moods">
            <span class="mood-chip" onclick="playMood('lofi hip hop study beats')">🎧 Lofi</span>
            <span class="mood-chip" onclick="playMood('arijit singh sad songs')">💔 Sad</span>
            <span class="mood-chip" onclick="playMood('bollywood workout motivation')">💪 Workout</span>
            <span class="mood-chip" onclick="playMood('classical music focus')">🎻 Focus</span>
            <span class="mood-chip" onclick="playMood('ap dhillon new songs 2025')">🌙 AP Dhillon</span>
            <span class="mood-chip" onclick="playMood('bollywood party hits')">🎉 Party</span>
        </div>
        <div id="musicPlayerWrap">
            <div class="music-placeholder" id="musicPlaceholder">
                <div class="big-icon">🎵</div>
                <div>Search a song to play</div>
            </div>
            <div class="music-player-frame" id="musicPlayerFrame" style="display:none;">
                <iframe id="musicIframe" src="" allow="autoplay; encrypted-media" allowfullscreen></iframe>
            </div>
        </div>
        <div id="nowPlayingLabel" style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:var(--secondary);display:none;"></div>
    </div>

</div>

<!-- FLOATING TIMER -->
<div id="floatTimer">
    <div class="f-label">📚 STUDY</div>
    <div class="f-time" id="floatTime">25:00</div>
    <button class="f-stop" onclick="stopStudyTimer()">■ STOP</button>
</div>

<script>
// ===================== TAB SYSTEM =====================
function switchTab(name, btn) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(name + 'Panel').classList.add('active');
    btn.classList.add('active');
}

// ===================== CHAT =====================
const chatMsgs = document.getElementById('chatMsgs');
const chatInput = document.getElementById('chatInput');

function addMsg(role, html, isTyping=false) {
    const d = document.createElement('div');
    d.className = 'msg ' + role;
    if (isTyping) {
        d.className = 'typing-dots';
        d.innerHTML = '<span></span><span></span><span></span>';
    } else {
        d.innerHTML = html;
    }
    chatMsgs.appendChild(d);
    chatMsgs.scrollTop = chatMsgs.scrollHeight;
    return d;
}

async function sendChat() {
    const text = chatInput.value.trim();
    if (!text) return;
    addMsg('user', escHtml(text));
    chatInput.value = '';
    const typingEl = addMsg('bot', '', true);

    try {
        const resp = await fetch('/ask-stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text})
        });

        typingEl.remove();
        const botEl = addMsg('bot', '');
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let full = '';
        let musicUrl = null;

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            const chunk = dec.decode(value);
            if (chunk.startsWith('{') && chunk.includes('music_url')) {
                try {
                    const data = JSON.parse(chunk);
                    if (data.music_url) musicUrl = data.music_url;
                    if (data.message) full += data.message;
                    if (data.title) full += `\n🎵 **${data.title}**`;
                } catch { full += chunk; }
            } else {
                full += chunk;
            }
            botEl.innerHTML = formatMsg(full);
            chatMsgs.scrollTop = chatMsgs.scrollHeight;
        }
        if (musicUrl) {
            switchToMusicPlayer(musicUrl);
        }
        if (!full) botEl.innerHTML = 'No response received.';
    } catch(e) {
        typingEl.remove();
        addMsg('bot', '❌ Network error. Please try again.');
    }
}

function escHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatMsg(t) {
    return t
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
        .replace(/\\[(.+?)\\]\\((.+?)\\)/g, '<a href="$2" target="_blank" style="color:var(--primary)">$1</a>')
        .replace(/\\n/g, '<br>');
}

chatInput.addEventListener('keypress', e => { if (e.key === 'Enter') sendChat(); });

// VOICE INPUT
let recognition;
function startVoice() {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        addMsg('bot', '❌ Your browser does not support voice input.');
        return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.lang = 'hi-IN';
    recognition.interimResults = false;
    recognition.onresult = e => {
        chatInput.value = e.results[0][0].transcript;
        sendChat();
    };
    recognition.onerror = () => addMsg('bot', '❌ Voice error. Try again.');
    recognition.start();
    document.getElementById('statusPill').textContent = '🎤 LISTENING';
    recognition.onend = () => {
        document.getElementById('statusPill').textContent = '● ONLINE';
    };
}

// ===================== STUDY TIMER =====================
let studyInterval = null;
let studyTotal = 0;
let studyRemaining = 0;
const tasks = [];

function startStudyTimer() {
    const mins = parseInt(document.getElementById('studyMins').value) || 25;
    studyTotal = mins * 60;
    studyRemaining = studyTotal;
    clearInterval(studyInterval);

    fetch('/start-study', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({minutes: mins})
    });

    document.getElementById('floatTimer').style.display = 'block';
    updateTimerDisplay();

    studyInterval = setInterval(() => {
        if (studyRemaining > 0) {
            studyRemaining--;
            updateTimerDisplay();
        } else {
            clearInterval(studyInterval);
            document.getElementById('floatTimer').style.display = 'none';
            document.getElementById('timerDisplay').textContent = '00:00';
            alert('🎉 Study session complete! Time for a break!');
        }
    }, 1000);
}

function stopStudyTimer() {
    clearInterval(studyInterval);
    studyRemaining = 0;
    document.getElementById('floatTimer').style.display = 'none';
    document.getElementById('timerDisplay').textContent = '00:00';
    document.getElementById('progressBar').style.width = '0%';
    fetch('/stop-study', {method: 'POST'});
}

function updateTimerDisplay() {
    const m = Math.floor(studyRemaining / 60);
    const s = studyRemaining % 60;
    const fmt = `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
    document.getElementById('timerDisplay').textContent = fmt;
    document.getElementById('floatTime').textContent = fmt;
    const pct = studyTotal > 0 ? (studyRemaining / studyTotal) * 100 : 100;
    document.getElementById('progressBar').style.width = pct + '%';
}

// TASKS
function addTask() {
    const inp = document.getElementById('taskInput');
    const text = inp.value.trim();
    if (!text) return;
    tasks.push({text, done: false});
    inp.value = '';
    renderTasks();
}

function renderTasks() {
    const list = document.getElementById('taskList');
    list.innerHTML = '';
    tasks.forEach((t, i) => {
        const d = document.createElement('div');
        d.className = 'task-item' + (t.done ? ' done' : '');
        d.innerHTML = `
            <div class="task-check" onclick="toggleTask(${i})">${t.done ? '✓' : ''}</div>
            <span>${escHtml(t.text)}</span>
            <span style="margin-left:auto;cursor:pointer;color:var(--secondary);font-size:0.8rem;" onclick="removeTask(${i})">✕</span>
        `;
        list.appendChild(d);
    });
}

function toggleTask(i) { tasks[i].done = !tasks[i].done; renderTasks(); }
function removeTask(i) { tasks.splice(i, 1); renderTasks(); }

document.getElementById('taskInput').addEventListener('keypress', e => {
    if (e.key === 'Enter') addTask();
});

// QUIZ
async function generateQuiz() {
    const topic = document.getElementById('quizTopic').value.trim();
    if (!topic) return;
    const area = document.getElementById('quizArea');
    area.textContent = '⏳ Generating quiz...';
    try {
        const resp = await fetch('/quiz', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({topic})
        });
        const data = await resp.json();
        area.textContent = data.quiz || 'Could not generate quiz.';
    } catch {
        area.textContent = 'Error generating quiz.';
    }
}

// ===================== STOCKS =====================
async function fetchStock() {
    const sym = document.getElementById('stockInput').value.trim().toUpperCase();
    if (!sym) return;
    const box = document.getElementById('stockResult');
    box.innerHTML = '<span style="color:var(--text-dim);">⏳ Fetching...</span>';
    try {
        const resp = await fetch('/stock', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol: sym})
        });
        const data = await resp.json();
        box.innerHTML = formatMsg(data.result || 'No data.');
    } catch {
        box.innerHTML = '❌ Error fetching stock data.';
    }
}

function quickStock(sym) {
    document.getElementById('stockInput').value = sym;
    fetchStock();
}

async function quickCrypto(coin) {
    const box = document.getElementById('stockResult');
    box.innerHTML = '<span style="color:var(--text-dim);">⏳ Fetching...</span>';
    try {
        const resp = await fetch('/crypto', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({coin})
        });
        const data = await resp.json();
        box.innerHTML = formatMsg(data.result || 'No data.');
    } catch {
        box.innerHTML = '❌ Error fetching crypto data.';
    }
}

// ===================== MUSIC =====================
async function playMusic() {
    const query = document.getElementById('musicInput').value.trim();
    if (!query) return;
    loadMusicFromQuery(query);
}

function playMood(mood) {
    document.getElementById('musicInput').value = mood;
    loadMusicFromQuery(mood);
}

async function loadMusicFromQuery(query) {
    document.getElementById('musicPlaceholder').style.display = 'flex';
    document.getElementById('musicPlaceholder').querySelector('div:last-child').textContent = '⏳ Loading...';
    document.getElementById('musicPlayerFrame').style.display = 'none';
    document.getElementById('nowPlayingLabel').style.display = 'none';

    try {
        const resp = await fetch('/play-music', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query})
        });
        const data = await resp.json();
        if (data.url) {
            switchToMusicPlayer(data.url, data.title);
        } else {
            document.getElementById('musicPlaceholder').querySelector('div:last-child').textContent = '❌ Could not load music.';
        }
    } catch {
        document.getElementById('musicPlaceholder').querySelector('div:last-child').textContent = '❌ Error loading music.';
    }
}

function switchToMusicPlayer(url, title) {
    document.getElementById('musicPlaceholder').style.display = 'none';
    document.getElementById('musicIframe').src = url;
    document.getElementById('musicPlayerFrame').style.display = 'block';
    if (title) {
        const lbl = document.getElementById('nowPlayingLabel');
        lbl.textContent = '🎵 Now Playing: ' + title;
        lbl.style.display = 'block';
    }
    // Switch to music tab
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById('musicPanel').classList.add('active');
    document.querySelectorAll('.tab-btn')[3].classList.add('active');
}

// ===================== TICKER =====================
async function updateTicker() {
    try {
        const resp = await fetch('/market-ticker');
        const data = await resp.json();
        document.getElementById('tickerInner').innerHTML = data.ticker || 'Market data loading...';
    } catch {}
}
updateTicker();
setInterval(updateTicker, 60000);

// ===================== BOOT MESSAGE =====================
window.addEventListener('load', () => {
    addMsg('bot', formatMsg(`🖖 **Assalamalekum Akram Bhai!**

✨ **Astra Level 9 - Upgraded** ✨

**New Features:**
• 💬 **Chat** — Multi-turn memory, voice input
• 📚 **Study** — Pomodoro timer + Task list + Quiz generator
• 📈 **Stocks** — Live stocks & crypto prices
• 🎵 **Music** — YouTube player with mood presets

*Type anything or use the tabs above!* 🚀`));
});
</script>
</body>
</html>
"""

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/ask-stream', methods=['POST'])
def ask_stream():
    data = request.get_json()
    input_text = data.get('message', '').strip()
    if not input_text:
        return Response("Boliye...", mimetype='text/plain')

    def generate():
        low = input_text.lower()

        # Stock commands in chat
        if 'stock' in low or 'share price' in low:
            for sym in ['RELIANCE','TCS','INFY','WIPRO','HDFCBANK','TATAMOTORS','NVIDIA','AAPL','TSLA','GOOGL','MSFT']:
                if sym.lower() in low:
                    yield get_stock_price(sym)
                    return
            yield get_stock_price('RELIANCE')
            return

        # Crypto in chat
        if any(x in low for x in ['bitcoin','btc','ethereum','eth','solana','sol','crypto']):
            coin = 'bitcoin'
            if 'ethereum' in low or ' eth' in low: coin = 'ethereum'
            elif 'solana' in low or ' sol' in low: coin = 'solana'
            elif 'dogecoin' in low or 'doge' in low: coin = 'dogecoin'
            yield get_crypto_price(coin)
            return

        # Music play command
        if low.startswith('play ') or 'gaana' in low or 'song play' in low:
            query = low.replace('play ', '').replace('gaana bajao', '').replace('song play', '').strip()
            if query:
                embed_url, title = get_youtube_embed_url(query + " song")
                if embed_url:
                    yield json.dumps({"message": f"🎵 Playing music for you!", "music_url": embed_url, "title": title or query})
                else:
                    yield f"🎵 Could not load player. Search on [YouTube](https://youtube.com/results?search_query={urllib.parse.quote(query)})"
            else:
                yield "Kaunsa gaana chahiye? Song name batao 🎵"
            return

        # Weather
        if 'weather' in low or 'mausam' in low:
            city = re.sub(r'weather|mausam|in|ka|of', '', low).strip() or 'Delhi'
            yield get_weather(city.strip())
            return

        # News
        if 'news' in low or 'khabar' in low:
            query = re.sub(r'news|khabar|aaj ki|latest|top', '', low).strip() or None
            yield get_news(query=query if query else None)
            return

        # Study timer (chat shortcut)
        if 'start study' in low or 'study start' in low or 'pomodoro' in low:
            match = re.search(r'(\d+)\s*min', low)
            mins = int(match.group(1)) if match else 25
            threading.Thread(target=study_timer_thread, args=(mins,), daemon=True).start()
            yield f"🎓 Study timer started for **{mins} minutes**! Switch to Study tab to see it. Focus karo Akram! 💪"
            return

        # General AI with memory
        for chunk in ask_nvidia_stream(input_text):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/start-study', methods=['POST'])
def start_study():
    data = request.get_json()
    mins = data.get('minutes', 25)
    threading.Thread(target=study_timer_thread, args=(mins,), daemon=True).start()
    return jsonify({"status": "started", "minutes": mins})

@app.route('/stop-study', methods=['POST'])
def stop_study():
    study_state["active"] = False
    study_state["remaining"] = 0
    return jsonify({"status": "stopped"})

@app.route('/health')
def health_check():
    return "Astra Level 9 is alive!", 200

@app.route('/stock', methods=['POST'])
def stock_route():
    data = request.get_json()
    symbol = data.get('symbol', 'RELIANCE').strip().upper()
    return jsonify({"result": get_stock_price(symbol)})

@app.route('/crypto', methods=['POST'])
def crypto_route():
    data = request.get_json()
    coin = data.get('coin', 'bitcoin').strip()
    return jsonify({"result": get_crypto_price(coin)})

@app.route('/play-music', methods=['POST'])
def play_music_route():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "No query"})
    embed_url, title = get_youtube_embed_url(query)
    if embed_url:
        return jsonify({"url": embed_url, "title": title or query})
    return jsonify({"error": "Could not find video"})

@app.route('/quiz', methods=['POST'])
def quiz_route():
    data = request.get_json()
    topic = data.get('topic', '').strip()
    if not topic:
        return jsonify({"quiz": "Please enter a topic."})
    quiz = generate_quiz(topic)
    return jsonify({"quiz": quiz})

@app.route('/market-ticker')
def market_ticker():
    try:
        summary = get_portfolio_summary()
        # Format as colored HTML
        parts = summary.split(' | ')
        html_parts = []
        for p in parts:
            if '▲' in p:
                html_parts.append(f'<span class="tick-up">{p}</span>')
            elif '▼' in p:
                html_parts.append(f'<span class="tick-dn">{p}</span>')
            else:
                html_parts.append(p)
        ticker = ' &nbsp;·&nbsp; '.join(html_parts)
        ticker += ' &nbsp;·&nbsp; <span style="color:var(--text-dim)">ASTRA LEVEL 9 · MARKET LIVE</span>'
        return jsonify({"ticker": ticker})
    except:
        return jsonify({"ticker": "Market data loading..."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
