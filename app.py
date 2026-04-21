import os
import re
import json
import requests
import urllib.parse
import threading
import time
import jwt
import hashlib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
from openai import OpenAI
import yt_dlp
from functools import wraps
from dotenv import load_dotenv
import asyncio
import edge_tts
import tempfile
import uuid

import sys
sys.modules['google.generativeai'] = None
sys.modules['anthropic'] = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'astra-level-9-super-secret-2025')

# ─── JWT AUTH ───
JWT_SECRET = os.getenv('JWT_SECRET', 'astra-jwt-secret-key-2025')
JWT_EXPIRY_HOURS = 24
USERS = {"akram": hashlib.sha256("1619".encode()).hexdigest()}

def generate_token(username):
    return jwt.encode({'username': username, 'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS), 'iat': datetime.utcnow()}, JWT_SECRET, algorithm='HS256')

def verify_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=['HS256'])['username']
    except:
        return None

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '') or request.cookies.get('astra_token', '')
        if not token or not verify_token(token):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

def get_username_from_request():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    return verify_token(token) or 'akram'

# ─── TTL CACHE ───
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

# ─── NVIDIA AI ───
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY) if NVIDIA_API_KEY else None
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")

# ─── FIREBASE ───
firebase_db = None
def init_firebase():
    global firebase_db
    try:
        import firebase_admin
        from firebase_admin import credentials, db
        cred_data = os.getenv("FIREBASE_CREDENTIALS")
        fb_url = os.getenv("FIREBASE_DB_URL", "https://astra-ai-2cc5a-default-rtdb.asia-southeast1.firebasedatabase.app")
        cred = None
        if cred_data:
            try: cred = credentials.Certificate(json.loads(cred_data))
            except: print("Error parsing FIREBASE_CREDENTIALS env var.")
        
        if not cred:
            paths = ["/etc/secrets/FIREBASE_CREDENTIALS", "firebase.json"]
            for p in paths:
                if os.path.exists(p):
                    try:
                        cred = credentials.Certificate(p)
                        break
                    except Exception as e:
                        print(f"Skipping {p}: Invalid certificate format.")
        
        if not cred:
            print("Firebase: No valid credentials found. Local fallback enabled.")
            return

        try: firebase_admin.get_app()
        except ValueError: firebase_admin.initialize_app(cred, {'databaseURL': fb_url})
        firebase_db = db
        print("Firebase Connected!")
    except Exception as e:
        print(f"Firebase Setup Error: {str(e)}")

init_firebase()
_local_memory = {}

def save_memory_cloud(username, key, value):
    _local_memory[f"{username}:{key}"] = value
    if firebase_db:
        def _s():
            try: firebase_db.reference(f"users/{username}/memory").update({key: value})
            except: pass
        threading.Thread(target=_s, daemon=True).start()

def get_memory_cloud(username, key=None):
    if firebase_db:
        try:
            data = firebase_db.reference(f"users/{username}/memory").get() or {}
            return data.get(key) if key else data
        except: pass
    if key: return _local_memory.get(f"{username}:{key}")
    prefix = f"{username}:"
    return {k.replace(prefix, ''): v for k, v in _local_memory.items() if k.startswith(prefix)}

def save_conv(username, history):
    if firebase_db:
        def _s():
            try: firebase_db.reference(f"users/{username}/conversation").set(history[-20:])
            except: pass
        threading.Thread(target=_s, daemon=True).start()

def load_conv(username):
    if firebase_db:
        try:
            data = firebase_db.reference(f"users/{username}/conversation").get()
            return data if isinstance(data, list) else []
        except: pass
    return []

_conv_histories = {}
MAX_HISTORY = 20

def get_history(username):
    if username not in _conv_histories:
        _conv_histories[username] = load_conv(username)
    return _conv_histories[username]

def build_system_prompt(username="akram"):
    memory = get_memory_cloud(username) or {}
    extras = []
    if memory.get("notes"): extras.append(f"Notes: {', '.join(str(n) for n in memory['notes'][-5:])}")
    if memory.get("preference"): extras.append(f"Likes: {memory['preference']}")
    if memory.get("friends"): extras.append(f"Friends: {', '.join(memory['friends'][-10:])}")
    return f"""You are Astra, a Level 9 AI assistant for Akram Ansari.
Speak Hinglish (Hindi+English mix). Be smart, warm, concise (2-4 lines). Use emojis.
User: Akram Ansari, B.Tech CS, Brainware University 2024-2028, Chhapra Bihar.
Friends: Rosidul Islam (Best Friend), Aryan Raj (Editor), Kaif Ali, Munshi Insiyat.
{chr(10).join(extras)}
Be like a smart best friend. Auto-learn from conversation."""

def ask_nvidia_stream(prompt, username="akram"):
    if not client:
        yield "⚠️ NVIDIA_API_KEY missing. Please configure environment variables."
        return
    history = get_history(username)
    history.append({"role": "user", "content": prompt})
    if len(history) > MAX_HISTORY * 2: history = history[-(MAX_HISTORY * 2):]
    _conv_histories[username] = history
    try:
        messages = [{"role": "system", "content": build_system_prompt(username)}] + history
        response = client.chat.completions.create(model=NVIDIA_MODEL, messages=messages, temperature=0.72, max_tokens=900, stream=True)
        full_reply = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_reply += content
                yield content
        history.append({"role": "assistant", "content": full_reply})
        _conv_histories[username] = history
        save_conv(username, history)
        low = prompt.lower()
        if "i like" in low or "mujhe pasand" in low:
            pref = re.sub(r'i like|mujhe pasand hai', '', low, flags=re.I).strip()
            if pref: save_memory_cloud(username, "preference", pref)
        if "my name is" in low or "mera naam" in low:
            name = re.sub(r'my name is|mera naam', '', low, flags=re.I).strip()
            if name: save_memory_cloud(username, "user_name", name)
    except Exception as e:
        yield f"⚠️ AI Error: {str(e)}"

# ─── FINANCIAL ───
@ttl_cache(300)
def get_stock_price(symbol):
    try:
        indian = ['RELIANCE','TCS','INFY','WIPRO','HDFCBANK','TATAMOTORS','ADANIPORTS','BAJAJFINSV','ICICIBANK','SBIN']
        sym_up = symbol.upper()
        if sym_up in indian: sym_up = f"{sym_up}.NS"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym_up}"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        res = resp.json()['chart']['result'][0]['meta']
        price = res['regularMarketPrice']
        currency = res['currency']
        change = res.get('regularMarketChange', 0)
        pct = res.get('regularMarketChangePercent', 0)
        high = res.get('regularMarketDayHigh', price)
        low_p = res.get('regularMarketDayLow', price)
        arrow = "📈" if change >= 0 else "📉"
        return f"{arrow} **{sym_up.replace('.NS','')}**\nPrice: {currency} {price:,.2f}\nChange: {change:+.2f} ({pct:+.2f}%)\nH: {high:,.2f}  L: {low_p:,.2f}"
    except:
        return f"❌ Could not fetch `{symbol}`. Check symbol."

@ttl_cache(300)
def get_crypto_price(coin):
    try:
        mapping = {'btc':'bitcoin','eth':'ethereum','doge':'dogecoin','sol':'solana','bnb':'binancecoin','xrp':'ripple','ada':'cardano','avax':'avalanche-2','matic':'matic-network','link':'chainlink','dot':'polkadot','shib':'shiba-inu'}
        coin_id = mapping.get(coin.lower(), coin.lower())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd,inr&include_24hr_change=true&include_market_cap=true"
        data = requests.get(url, timeout=10).json()
        if coin_id not in data: return f"❌ Crypto `{coin}` not found."
        p = data[coin_id]
        ch = p.get('usd_24h_change', 0)
        arrow = "📈" if ch >= 0 else "📉"
        mcap = p.get('usd_market_cap', 0)
        mcap_str = f"${mcap/1e9:.1f}B" if mcap > 1e9 else f"${mcap/1e6:.1f}M"
        return f"{arrow} **{coin_id.upper()}**\n💰 ${p['usd']:,.4f} ({ch:+.2f}%)\n🇮🇳 ₹{p['inr']:,.2f}\nMarket Cap: {mcap_str}"
    except Exception as e:
        return f"❌ Crypto error: {str(e)}"

@ttl_cache(300)
def get_portfolio_summary():
    try:
        ids = 'bitcoin,ethereum,solana,binancecoin'
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true"
        data = requests.get(url, timeout=8).json()
        names = {'bitcoin':'BTC','ethereum':'ETH','solana':'SOL','binancecoin':'BNB'}
        parts = []
        for cid, label in names.items():
            if cid in data:
                p = data[cid]; ch = p.get('usd_24h_change', 0)
                parts.append(f"{label}: ${p['usd']:,.0f} {'▲' if ch>=0 else '▼'}{abs(ch):.1f}%")
        return " | ".join(parts)
    except: return "Market data loading..."

@ttl_cache(1800)
def get_news(query=None, country="in"):
    api_key = os.getenv('GNEWS_API_KEY') or os.getenv('NEWS_API_KEY')
    if not api_key: return "⚠️ GNEWS_API_KEY missing in .env"
    url = (f"https://gnews.io/api/v4/search?q={urllib.parse.quote(query)}&token={api_key}&lang=en&max=4" if query else f"https://gnews.io/api/v4/top-headlines?country={country}&token={api_key}&max=4")
    try:
        articles = requests.get(url, timeout=10).json().get('articles', [])
        if not articles: return "No news found."
        return "\n".join([f"📰 **{a['title']}**\n🔗 [Read]({a['url']})\n" for a in articles])
    except: return "❌ News fetch failed."

@ttl_cache(600)
def get_weather(city):
    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key: return "⚠️ WEATHER_API_KEY missing"
    try:
        data = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric", timeout=10).json()
        if data.get('cod') != 200: return f"❌ '{city}' not found."
        temp = data['main']['temp']; feels = data['main']['feels_like']
        humidity = data['main']['humidity']; wind = data['wind']['speed']
        desc = data['weather'][0]['description'].title()
        icons = {'clear':'☀️','cloud':'☁️','rain':'🌧️','thunder':'⛈️','snow':'❄️','mist':'🌫️','haze':'🌫️','drizzle':'🌦️'}
        icon = next((v for k, v in icons.items() if k in desc.lower()), '🌡️')
        return f"{icon} **{city.title()}**\n🌡️ {temp}°C (feels {feels}°C)\n💧 Humidity: {humidity}% | 💨 Wind: {wind} m/s\n{desc}"
    except: return "❌ Weather fetch failed."

# ─── STUDY ───
study_state = {"active": False, "remaining": 0, "total": 0}

def study_timer_thread(minutes):
    study_state.update({"active": True, "total": minutes*60, "remaining": minutes*60})
    while study_state["remaining"] > 0 and study_state["active"]:
        time.sleep(1); study_state["remaining"] -= 1
    study_state["active"] = False

def run_ai(system_prompt, user_prompt, max_tokens=700, temp=0.4):
    if not client: return "❌ AI Error: NVIDIA_API_KEY missing."
    try:
        r = client.chat.completions.create(model=NVIDIA_MODEL, messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], temperature=temp, max_tokens=max_tokens, stream=False)
        return r.choices[0].message.content.strip()
    except Exception as e: return f"❌ AI Error: {str(e)}"

def generate_quiz(topic):
    return run_ai("Generate exactly 3 MCQ questions. Output ONLY the quiz, no extra text.", f"Topic: {topic}\n\nFormat:\nQ1: [question]\nA) [opt]\nB) [opt]\nC) [opt]\nD) [opt]\nAnswer: [letter]\n\nQ2:...\nQ3:...")

def generate_flashcards(topic):
    try:
        text = run_ai("Generate 5 flashcards. Output ONLY a JSON array, no markdown backticks.", f"Topic: {topic}\nFormat: [{{\"front\":\"question\",\"back\":\"answer\"}}, ...]", max_tokens=500, temp=0.3)
        text = re.sub(r'```json|```', '', text).strip()
        return json.loads(text)
    except Exception as e: return [{"front": "Error generating flashcards", "back": str(e)}]

def summarize_text(text):
    return run_ai("Summarize in clear bullet points. Be concise.", text[:3000], max_tokens=500)

def explain_code(code):
    return run_ai("Explain this code in Hinglish. State what it does, any bugs or improvements.", f"```\n{code[:2000]}\n```", max_tokens=600)

def generate_image_prompt(description):
    return run_ai("Generate a detailed AI image prompt for Midjourney/DALL-E. Be specific about style, lighting, composition.", f"Create an image prompt for: {description}", max_tokens=300, temp=0.8)

# ─── YOUTUBE ───
def get_youtube_embed_url(query):
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True, 'no_warnings': True}) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if 'entries' in info and info['entries']:
                vid = info['entries'][0]
                return f"https://www.youtube.com/embed/{vid['id']}?autoplay=1&controls=1&rel=0", vid.get('title', query), vid.get('duration', 0)
    except: pass
    return None, None, 0

# ─── HTML ───
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>▲ ASTRA LEVEL 9</title>
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#020408">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root{--p:#00f0ff;--s:#ff00aa;--a:#aaff00;--bg:#020408;--panel:rgba(0,240,255,0.04);--border:rgba(0,240,255,0.14);--text:#cce8f0;--dim:#3a6070;--glow:0 0 18px rgba(0,240,255,0.45);--glow2:0 0 18px rgba(255,0,170,0.45);--glow3:0 0 18px rgba(170,255,0,0.4);}
        *{margin:0;padding:0;box-sizing:border-box;-webkit-text-size-adjust:100%;}html,body{height:100%;overflow:hidden;}
        body{background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;display:flex;flex-direction:column;height:100vh;overflow-x:hidden;width:100vw;}
        body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,240,255,0.022) 1px,transparent 1px),linear-gradient(90deg,rgba(0,240,255,0.022) 1px,transparent 1px);background-size:44px 44px;pointer-events:none;z-index:0;}
        body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.07) 2px,rgba(0,0,0,0.07) 4px);pointer-events:none;z-index:1;}
        /* LOGIN */
        #loginScreen{position:fixed;inset:0;z-index:1000;background:var(--bg);display:flex;align-items:center;justify-content:center;flex-direction:column;gap:20px;}
        #loginScreen.hidden{display:none;}
        .login-box{background:rgba(0,240,255,0.04);border:1px solid var(--border);padding:44px 52px;border-radius:4px;text-align:center;min-width:340px;max-width:90vw;clip-path:polygon(0 0,calc(100% - 22px) 0,100% 22px,100% 100%,22px 100%,0 calc(100% - 22px));animation:loginIn .5s ease-out;}
        @keyframes loginIn{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
        .login-logo{font-family:'Orbitron',monospace;font-size:2.2rem;font-weight:900;color:var(--p);text-shadow:var(--glow);letter-spacing:5px;}
        .login-sub{font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:var(--dim);margin:6px 0 28px;letter-spacing:3px;}
        .login-field{width:100%;background:rgba(0,0,0,0.5);border:1px solid rgba(0,240,255,0.25);border-radius:4px;padding:13px 18px;color:#fff;font-family:'Share Tech Mono',monospace;font-size:1rem;outline:none;margin-bottom:10px;text-align:center;transition:all .2s;}
        .login-field:focus{border-color:var(--p);box-shadow:var(--glow);}
        .login-btn{width:100%;background:linear-gradient(135deg,var(--p),#007799);border:none;border-radius:4px;padding:14px;font-family:'Orbitron',monospace;font-size:0.78rem;font-weight:700;color:#020408;cursor:pointer;letter-spacing:2px;transition:all .2s;margin-top:6px;}
        .login-btn:hover{transform:scale(1.02);box-shadow:var(--glow);}
        .login-err{color:var(--s);font-size:0.78rem;font-family:'Share Tech Mono',monospace;min-height:18px;}
        /* APP */
        .app{position:relative;z-index:2;display:flex;flex-direction:column;height:100vh;max-width:980px;margin:0 auto;width:100%;padding:6px 8px;gap:6px;overflow-x:hidden;}
        header{display:flex;align-items:center;justify-content:space-between;padding:8px 18px;border:1px solid var(--border);background:var(--panel);backdrop-filter:blur(12px);clip-path:polygon(0 0,calc(100% - 14px) 0,100% 14px,100% 100%,14px 100%,0 calc(100% - 14px));flex-shrink:0;}
        .logo{font-family:'Orbitron',monospace;font-size:1.2rem;font-weight:900;color:var(--p);text-shadow:var(--glow);letter-spacing:3px;}.logo span{color:var(--s);}
        .header-right{display:flex;gap:8px;align-items:center;}
        .status-pill{font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:var(--a);border:1px solid rgba(170,255,0,0.28);padding:3px 9px;border-radius:20px;background:rgba(170,255,0,0.06);animation:blink 2s ease-in-out infinite;}
        @keyframes blink{0%,100%{opacity:.55;}50%{opacity:1;}}
        .logout-btn{font-family:'Orbitron',monospace;font-size:0.5rem;background:rgba(255,0,170,0.08);border:1px solid rgba(255,0,170,0.25);color:var(--s);padding:4px 10px;border-radius:4px;cursor:pointer;transition:all .2s;}
        .logout-btn:hover{background:rgba(255,0,170,0.18);}
        .ticker-bar{background:rgba(0,240,255,0.03);border:1px solid var(--border);padding:5px 14px;font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:var(--dim);overflow:hidden;white-space:nowrap;flex-shrink:0;}
        .ticker-inner{display:inline-block;animation:tick 35s linear infinite;}
        @keyframes tick{0%{transform:translateX(100vw);}100%{transform:translateX(-100%);}}
        .t-up{color:var(--a);}.t-dn{color:var(--s);}
        .nav-tabs{display:flex;gap:4px;flex-shrink:0;}
        .tab-btn{flex:1;background:var(--panel);border:1px solid var(--border);color:var(--dim);font-family:'Orbitron',monospace;font-size:0.55rem;padding:7px 3px;cursor:pointer;letter-spacing:.8px;transition:all .2s;clip-path:polygon(0 0,calc(100% - 7px) 0,100% 7px,100% 100%,7px 100%,0 calc(100% - 7px));}
        .tab-btn.active,.tab-btn:hover{border-color:var(--p);color:var(--p);background:rgba(0,240,255,0.07);text-shadow:var(--glow);}
        .panel{display:none;flex:1;flex-direction:column;background:var(--panel);border:1px solid var(--border);overflow:hidden;min-height:0;}
        .panel.active{display:flex;}
        /* CHAT */
        .chat-msgs{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;scroll-behavior:smooth;min-height:0;}
        .chat-msgs::-webkit-scrollbar{width:3px;}.chat-msgs::-webkit-scrollbar-thumb{background:var(--p);border-radius:3px;}
        .msg{max-width:82%;padding:9px 14px;border-radius:14px;font-size:0.88rem;line-height:1.5;word-wrap:break-word;animation:fadeUp .22s ease-out;}
        @keyframes fadeUp{from{opacity:0;transform:translateY(7px);}to{opacity:1;transform:translateY(0);}}
        .msg.user{align-self:flex-end;background:linear-gradient(135deg,var(--p),#007799);color:#020408;font-weight:600;border-bottom-right-radius:3px;}
        .msg.bot{align-self:flex-start;background:rgba(0,240,255,0.04);border:1px solid rgba(0,240,255,0.13);color:var(--text);border-bottom-left-radius:3px;}
        .msg.bot a{color:var(--p);}.msg.bot code{background:rgba(0,240,255,0.1);padding:1px 5px;border-radius:3px;font-family:monospace;font-size:.85em;}
        .typing-dots{display:flex;gap:5px;align-items:center;padding:9px 14px;background:rgba(0,240,255,0.03);border:1px solid rgba(0,240,255,0.1);border-radius:14px;width:fit-content;border-bottom-left-radius:3px;}
        .typing-dots span{width:6px;height:6px;background:var(--p);border-radius:50%;animation:bounce 1.2s infinite;}
        .typing-dots span:nth-child(2){animation-delay:.2s;}.typing-dots span:nth-child(3){animation-delay:.4s;}
        @keyframes bounce{0%,60%,100%{transform:translateY(0);opacity:.4;}30%{transform:translateY(-6px);opacity:1;}}
        .suggestions{display:flex;gap:5px;flex-wrap:wrap;padding:5px 10px;border-top:1px solid rgba(0,240,255,0.08);}
        .sug-chip{background:rgba(0,240,255,0.06);border:1px solid rgba(0,240,255,0.15);color:var(--p);font-size:0.68rem;padding:3px 9px;border-radius:14px;cursor:pointer;transition:all .2s;font-family:'Share Tech Mono',monospace;}
        .sug-chip:hover{background:rgba(0,240,255,0.15);}
        .chat-input-row{display:flex;gap:7px;padding:8px 10px;border-top:1px solid var(--border);flex-shrink:0;align-items:center;}
        .chat-input-row input{flex:1;background:rgba(0,0,0,0.4);border:1px solid rgba(0,240,255,0.22);border-radius:28px;padding:10px 16px;font-family:'Rajdhani',sans-serif;font-size:0.92rem;color:#fff;outline:none;transition:all .2s;}
        .chat-input-row input:focus{border-color:var(--p);box-shadow:var(--glow);}
        .icon-btn{background:rgba(0,240,255,0.07);border:1px solid rgba(0,240,255,0.22);border-radius:50%;width:38px;height:38px;color:var(--p);font-size:.9rem;cursor:pointer;transition:all .2s;flex-shrink:0;display:flex;align-items:center;justify-content:center;}
        .icon-btn:hover{background:rgba(0,240,255,0.16);box-shadow:var(--glow);}
        .icon-btn.tts-on{background:rgba(170,255,0,0.12);border-color:rgba(170,255,0,0.3);color:var(--a);box-shadow:var(--glow3);}
        .send-btn{background:linear-gradient(135deg,var(--p),#007799);border:none;border-radius:28px;padding:0 18px;font-family:'Orbitron',monospace;font-size:0.65rem;font-weight:700;color:#020408;cursor:pointer;transition:all .2s;flex-shrink:0;height:38px;letter-spacing:1px;}
        .send-btn:hover{transform:scale(1.04);box-shadow:var(--glow);}
        /* STUDY */
        #studyPanel{padding:10px;gap:8px;overflow-y:auto;}
        .sub-tabs{display:flex;gap:4px;flex-shrink:0;}
        .sub-tab{flex:1;background:rgba(0,240,255,0.04);border:1px solid var(--border);color:var(--dim);font-family:'Orbitron',monospace;font-size:0.5rem;padding:6px 3px;cursor:pointer;border-radius:4px;transition:all .2s;letter-spacing:.6px;}
        .sub-tab.active,.sub-tab:hover{border-color:var(--p);color:var(--p);}
        .sub-view{display:none;flex-direction:column;gap:8px;}.sub-view.active{display:flex;}
        .card{background:rgba(0,240,255,0.03);border:1px solid var(--border);border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:7px;}
        .card h3{font-family:'Orbitron',monospace;font-size:0.63rem;color:var(--p);letter-spacing:2px;}
        .timer-display{font-family:'Orbitron',monospace;font-size:2.5rem;font-weight:900;color:var(--p);text-shadow:var(--glow);text-align:center;}
        .prog-wrap{background:rgba(0,240,255,0.07);border-radius:20px;height:5px;overflow:hidden;}
        .prog-fill{height:100%;background:linear-gradient(90deg,var(--p),var(--a));border-radius:20px;transition:width 1s linear;}
        .btn-row{display:flex;gap:6px;}
        .sb{flex:1;background:rgba(0,240,255,0.07);border:1px solid var(--border);color:var(--p);font-family:'Orbitron',monospace;font-size:0.55rem;padding:7px 4px;cursor:pointer;border-radius:5px;letter-spacing:.7px;transition:all .2s;}
        .sb:hover{background:rgba(0,240,255,0.15);}
        .sb.red{color:var(--s);border-color:rgba(255,0,170,0.25);}.sb.red:hover{background:rgba(255,0,170,0.1);}
        .sb.green{color:var(--a);border-color:rgba(170,255,0,0.25);}.sb.green:hover{background:rgba(170,255,0,0.1);}
        .field{background:rgba(0,0,0,0.4);border:1px solid var(--border);color:var(--p);font-family:'Share Tech Mono',monospace;font-size:0.8rem;padding:8px 10px;border-radius:5px;outline:none;width:100%;}
        .field::placeholder{color:var(--dim);}
        .task-list{display:flex;flex-direction:column;gap:4px;max-height:150px;overflow-y:auto;}
        .task-item{display:flex;align-items:center;gap:6px;padding:6px 8px;background:rgba(0,240,255,0.02);border:1px solid rgba(0,240,255,0.08);border-radius:5px;font-size:0.82rem;}
        .task-item.done{opacity:.38;text-decoration:line-through;}
        .task-chk{width:14px;height:14px;border:1px solid var(--p);border-radius:3px;cursor:pointer;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.6rem;}
        .quiz-box{background:rgba(0,0,0,0.28);border:1px solid var(--border);border-radius:6px;padding:10px;font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:var(--text);line-height:1.9;white-space:pre-wrap;min-height:90px;max-height:210px;overflow-y:auto;}
        .flashcard-wrap{perspective:800px;}
        .flashcard{width:100%;min-height:100px;position:relative;transform-style:preserve-3d;transition:transform .5s;cursor:pointer;}
        .flashcard.flipped{transform:rotateY(180deg);}
        .card-front,.card-back{position:absolute;inset:0;background:rgba(0,240,255,0.05);border:1px solid var(--border);border-radius:8px;display:flex;align-items:center;justify-content:center;padding:14px;text-align:center;backface-visibility:hidden;font-size:.9rem;}
        .card-back{transform:rotateY(180deg);background:rgba(255,0,170,0.05);border-color:rgba(255,0,170,0.2);color:var(--s);}
        .fc-nav{display:flex;gap:8px;justify-content:center;margin-top:8px;}
        .sum-ta{background:rgba(0,0,0,0.4);border:1px solid var(--border);color:var(--text);font-family:'Rajdhani',sans-serif;font-size:0.85rem;padding:9px;border-radius:6px;width:100%;min-height:75px;outline:none;resize:vertical;}
        .result-box{background:rgba(0,0,0,0.28);border:1px solid var(--border);border-radius:6px;padding:10px;font-size:0.83rem;line-height:1.7;min-height:50px;max-height:180px;overflow-y:auto;}
        /* STOCKS */
        #stocksPanel{padding:10px;gap:8px;overflow-y:auto;}
        .search-row{display:flex;gap:6px;}
        .sf{flex:1;background:rgba(0,0,0,0.4);border:1px solid var(--border);color:var(--p);font-family:'Share Tech Mono',monospace;font-size:0.82rem;padding:9px 13px;border-radius:6px;outline:none;}
        .sf:focus{border-color:var(--p);box-shadow:var(--glow);}
        .go-btn{background:linear-gradient(135deg,var(--p),#007799);border:none;border-radius:6px;padding:0 15px;font-family:'Orbitron',monospace;font-size:0.58rem;font-weight:700;color:#020408;cursor:pointer;}
        .chips{display:flex;gap:4px;flex-wrap:wrap;}
        .chip{background:rgba(0,240,255,0.05);border:1px solid rgba(0,240,255,0.17);color:var(--p);font-family:'Share Tech Mono',monospace;font-size:0.65rem;padding:4px 10px;border-radius:15px;cursor:pointer;transition:all .2s;}
        .chip:hover{background:rgba(0,240,255,0.13);}
        .chip.c{border-color:rgba(255,0,170,0.25);color:var(--s);}.chip.c:hover{background:rgba(255,0,170,0.1);}
        .stock-result{background:rgba(0,240,255,0.02);border:1px solid var(--border);border-radius:8px;padding:13px;font-family:'Share Tech Mono',monospace;font-size:0.83rem;line-height:2;min-height:65px;}
        .watchlist{display:flex;flex-direction:column;gap:5px;}
        .wl-item{display:flex;align-items:center;justify-content:space-between;padding:7px 11px;background:rgba(0,240,255,0.03);border:1px solid rgba(0,240,255,0.09);border-radius:5px;font-family:'Share Tech Mono',monospace;font-size:0.77rem;}
        .wl-sym{color:var(--p);}.wl-del{color:var(--s);cursor:pointer;}
        /* MUSIC */
        #musicPanel{padding:10px;gap:8px;overflow-y:auto;}
        .mf{flex:1;background:rgba(0,0,0,0.4);border:1px solid rgba(255,0,170,0.22);color:#fff;font-family:'Rajdhani',sans-serif;font-size:.92rem;padding:9px 14px;border-radius:6px;outline:none;}
        .mf:focus{border-color:var(--s);box-shadow:var(--glow2);}
        .music-btn{background:linear-gradient(135deg,var(--s),#880055);border:none;border-radius:6px;padding:0 15px;font-family:'Orbitron',monospace;font-size:.58rem;font-weight:700;color:#fff;cursor:pointer;}
        .moods{display:flex;gap:4px;flex-wrap:wrap;}
        .mood{background:rgba(255,0,170,0.05);border:1px solid rgba(255,0,170,0.17);color:var(--s);font-size:0.7rem;padding:4px 11px;border-radius:15px;cursor:pointer;transition:all .2s;}
        .mood:hover{background:rgba(255,0,170,0.12);}
        .player-wrap{border-radius:10px;overflow:hidden;border:1px solid rgba(255,0,170,0.28);background:#000;aspect-ratio:16/9;}
        .player-wrap iframe{width:100%;height:100%;border:none;}
        .music-ph{width:100%;aspect-ratio:16/9;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(255,0,170,0.03);border:1px dashed rgba(255,0,170,0.17);border-radius:10px;color:rgba(255,0,170,0.33);font-family:'Orbitron',monospace;font-size:0.72rem;gap:7px;}
        .np-label{font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:var(--s);display:none;}
        /* TOOLS */
        #toolsPanel{padding:10px;gap:8px;overflow-y:auto;}
        .code-ta{background:rgba(0,0,0,0.5);border:1px solid var(--border);color:#aaff88;font-family:'Share Tech Mono',monospace;font-size:0.76rem;padding:9px;border-radius:6px;width:100%;min-height:90px;resize:vertical;outline:none;}
        /* MEMORY */
        #memoryPanel{padding:8px;gap:6px;overflow-y:auto;overflow-x:hidden;width:100%;}
        .mem-section{background:rgba(0,240,255,0.03);border:1px solid var(--border);border-radius:8px;padding:10px;width:100%;overflow:hidden;box-sizing:border-box;}
        .mem-section h3{font-family:'Orbitron',monospace;font-size:0.58rem;color:var(--p);letter-spacing:1px;margin-bottom:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
        .mem-item{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid rgba(0,240,255,0.05);font-size:0.8rem;}
        .mem-key{color:var(--dim);font-family:'Share Tech Mono',monospace;font-size:0.7rem;}.mem-val{color:var(--p);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
        .mem-save-row{display:flex;gap:4px;margin-top:7px;flex-wrap:wrap;}
        .mem-input{flex:1;min-width:80px;background:rgba(0,0,0,0.4);border:1px solid var(--border);color:#fff;font-family:'Rajdhani',sans-serif;font-size:0.82rem;padding:6px 9px;border-radius:4px;outline:none;box-sizing:border-box;}
        /* FLOAT */
        #floatTimer{position:fixed;bottom:70px;right:14px;background:rgba(2,4,8,0.93);border:1px solid var(--p);border-radius:12px;padding:9px 14px;font-family:'Orbitron',monospace;text-align:center;z-index:999;display:none;box-shadow:var(--glow);min-width:110px;}
        .f-lbl{font-size:0.5rem;color:var(--p);letter-spacing:2px;}.f-time{font-size:1.5rem;font-weight:900;color:#fff;}
        .f-stop{background:var(--s);border:none;border-radius:5px;padding:4px 10px;color:#fff;font-size:0.55rem;cursor:pointer;width:100%;margin-top:5px;font-family:'Orbitron',monospace;}
        #installBanner{display:none;position:fixed;bottom:14px;left:50%;transform:translateX(-50%);background:rgba(2,4,8,.96);border:1px solid var(--a);border-radius:10px;padding:10px 18px;z-index:998;align-items:center;gap:10px;font-family:'Orbitron',monospace;font-size:0.65rem;color:var(--a);white-space:nowrap;}
        #installBanner.show{display:flex;}
        #installBanner button{background:var(--a);border:none;color:#020408;font-family:'Orbitron',monospace;font-size:.6rem;font-weight:700;padding:5px 12px;border-radius:5px;cursor:pointer;}
        @media(max-width:600px){
            .logo{font-size:.9rem;}
            .msg{max-width:93%;font-size:.82rem;}
            .tab-btn{font-size:.42rem;padding:5px 1px;}
            .timer-display{font-size:2rem;}
            .mem-section h3{font-size:0.54rem;letter-spacing:0.5px;}
            .mem-item{flex-wrap:wrap;gap:4px;}
            .mem-val{max-width:150px;}
            header{padding:6px 10px;}
            .login-box{padding:30px 24px;min-width:unset;width:90vw;}
            .app{padding:4px 6px;}
        }
    </style>
</head>
<body>

<div id="loginScreen">
    <div class="login-box">
        <div class="login-logo">▲ ASTRA</div>
        <div class="login-sub">LEVEL 9 · SECURE ACCESS</div>
        <input type="text" id="lUser" class="login-field" placeholder="USERNAME" value="akram" autocomplete="username">
        <input type="password" id="lPass" class="login-field" placeholder="PASSWORD / PIN" autocomplete="current-password">
        <div class="login-err" id="lErr"></div>
        <button class="login-btn" onclick="doLogin()">▶ INITIALIZE SYSTEM</button>
    </div>
</div>

<div class="app">
    <header>
        <div class="logo">▲ ASTRA<span> L9</span></div>
        <div class="header-right">
            <div class="status-pill" id="statusPill">● ONLINE</div>
            <button class="logout-btn" onclick="doLogout()">LOGOUT</button>
        </div>
    </header>
    <div class="ticker-bar"><div class="ticker-inner" id="tickerInner">Loading market data...</div></div>
    <div class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('chat',this)">💬 CHAT</button>
        <button class="tab-btn" onclick="switchTab('study',this)">📚 STUDY</button>
        <button class="tab-btn" onclick="switchTab('stocks',this)">📈 STOCKS</button>
        <button class="tab-btn" onclick="switchTab('music',this)">🎵 MUSIC</button>
        <button class="tab-btn" onclick="switchTab('tools',this)">🛠 TOOLS</button>
        <button class="tab-btn" onclick="switchTab('memory',this)">🧠 MEM</button>
    </div>

    <!-- CHAT -->
    <div class="panel active" id="chatPanel">
        <div class="chat-msgs" id="chatMsgs"></div>
        <div class="suggestions">
            <span class="sug-chip" onclick="qa('Aaj kya padhna chahiye?')">📚 Study</span>
            <span class="sug-chip" onclick="qa('Bitcoin price?')">₿ BTC</span>
            <span class="sug-chip" onclick="qa('Motivate me bhai!')">🔥 Motivate</span>
            <span class="sug-chip" onclick="qa('Weather Delhi')">🌤️ Weather</span>
            <span class="sug-chip" onclick="qa('Top news today')">📰 News</span>
            <span class="sug-chip" onclick="qa('Code review tips for beginners')">💻 Coding</span>
        </div>
        <div class="chat-input-row">
            <button class="icon-btn" onclick="startVoice()" title="Voice input">🎤</button>
            <button class="icon-btn" id="ttsBtn" onclick="toggleTTS()" title="Voice reply">🔊</button>
            <input type="text" id="chatInput" placeholder="Ask Astra anything..." autocomplete="off" enterkeyhint="send">
            <button class="send-btn" onclick="sendChat()">SEND</button>
        </div>
    </div>

    <!-- STUDY -->
    <div class="panel" id="studyPanel">
        <div class="sub-tabs">
            <button class="sub-tab active" onclick="switchSub('timer',this)">⏱ TIMER</button>
            <button class="sub-tab" onclick="switchSub('tasks',this)">✅ TASKS</button>
            <button class="sub-tab" onclick="switchSub('quiz',this)">🧠 QUIZ</button>
            <button class="sub-tab" onclick="switchSub('flash',this)">🃏 FLASH</button>
            <button class="sub-tab" onclick="switchSub('summary',this)">📝 SUMM</button>
        </div>
        <div class="sub-view active" id="sv-timer">
            <div class="card"><h3>⏱ POMODORO</h3>
                <div class="timer-display" id="timerDisplay">25:00</div>
                <div class="prog-wrap"><div class="prog-fill" id="progressBar" style="width:100%"></div></div>
                <input class="field" type="number" id="studyMins" value="25" min="1" max="180" style="text-align:center">
                <div class="btn-row">
                    <button class="sb" onclick="startStudyTimer()">▶ START</button>
                    <button class="sb red" onclick="stopStudyTimer()">■ STOP</button>
                    <button class="sb green" onclick="resetTimer()">↺ RESET</button>
                </div>
            </div>
        </div>
        <div class="sub-view" id="sv-tasks">
            <div class="card"><h3>✅ TASK LIST</h3>
                <div class="task-list" id="taskList"></div>
                <div style="display:flex;gap:6px;margin-top:2px;">
                    <input class="field" type="text" id="taskInput" placeholder="Add task..." style="font-family:'Rajdhani',sans-serif">
                    <button class="sb" style="flex:0;padding:7px 11px;" onclick="addTask()">+</button>
                </div>
                <div class="btn-row">
                    <button class="sb red" onclick="clearDone()">Clear done</button>
                    <button class="sb" id="aiPriBtn" onclick="aiPrioritize()">🤖 AI Prioritize</button>
                </div>
            </div>
        </div>
        <div class="sub-view" id="sv-quiz">
            <div class="card"><h3>🧠 QUIZ GENERATOR</h3>
                <div style="display:flex;gap:6px;">
                    <input class="field" type="text" id="quizTopic" placeholder="Topic (Arrays, Chemistry, History...)">
                    <button class="sb" style="flex:0;padding:7px 13px;" onclick="doQuiz()">GO</button>
                </div>
                <div class="quiz-box" id="quizArea">Enter topic and click GO...</div>
            </div>
        </div>
        <div class="sub-view" id="sv-flash">
            <div class="card"><h3>🃏 FLASHCARDS</h3>
                <div style="display:flex;gap:6px;">
                    <input class="field" type="text" id="flashTopic" placeholder="Topic for flashcards...">
                    <button class="sb" style="flex:0;padding:7px 13px;" onclick="doFlash()">GO</button>
                </div>
                <div class="flashcard-wrap">
                    <div class="flashcard" id="flashcard" onclick="flipCard()">
                        <div class="card-front" id="cFront">Click GO to generate</div>
                        <div class="card-back" id="cBack"></div>
                    </div>
                </div>
                <div class="fc-nav">
                    <button class="sb" style="flex:0;padding:5px 14px;" onclick="prevCard()">◀</button>
                    <span id="cardCount" style="color:var(--dim);font-family:'Share Tech Mono',monospace;font-size:0.73rem;align-self:center;">0/0</span>
                    <button class="sb" style="flex:0;padding:5px 14px;" onclick="nextCard()">▶</button>
                </div>
            </div>
        </div>
        <div class="sub-view" id="sv-summary">
            <div class="card"><h3>📝 TEXT SUMMARIZER</h3>
                <textarea class="sum-ta" id="sumInput" placeholder="Paste text / notes / article here..."></textarea>
                <button class="sb green" onclick="doSummarize()">⚡ SUMMARIZE</button>
                <div class="result-box" id="sumResult">Summary will appear here...</div>
            </div>
        </div>
    </div>

    <!-- STOCKS -->
    <div class="panel" id="stocksPanel">
        <div class="sub-tabs">
            <button class="sub-tab active" onclick="switchSt('search',this)">🔍 SEARCH</button>
            <button class="sub-tab" onclick="switchSt('watchlist',this)">👁 WATCHLIST</button>
            <button class="sub-tab" onclick="switchSt('crypto',this)">🪙 CRYPTO</button>
        </div>
        <div class="sub-view active" id="stv-search">
            <div class="search-row">
                <input class="sf" type="text" id="stockInput" placeholder="RELIANCE, NVDA, TSLA...">
                <button class="go-btn" onclick="fetchStock()">FETCH</button>
                <button class="sb" style="flex:0;padding:0 9px;font-size:0.5rem;" onclick="addWL()">+WL</button>
            </div>
            <div class="chips">
                <span class="chip" onclick="qs('RELIANCE')">RELIANCE</span>
                <span class="chip" onclick="qs('TCS')">TCS</span>
                <span class="chip" onclick="qs('INFY')">INFY</span>
                <span class="chip" onclick="qs('NVIDIA')">NVIDIA</span>
                <span class="chip" onclick="qs('AAPL')">AAPL</span>
                <span class="chip" onclick="qs('TSLA')">TSLA</span>
                <span class="chip" onclick="qs('GOOGL')">GOOGL</span>
                <span class="chip" onclick="qs('ICICIBANK')">ICICI</span>
            </div>
            <div class="stock-result" id="stockResult"><span style="color:var(--dim)">Select or type stock symbol...</span></div>
        </div>
        <div class="sub-view" id="stv-watchlist">
            <div class="watchlist" id="wlEl"></div>
            <div id="wlHint" style="color:var(--dim);font-size:0.73rem;font-family:'Share Tech Mono',monospace;">Add from Search →</div>
        </div>
        <div class="sub-view" id="stv-crypto">
            <div class="search-row">
                <input class="sf" type="text" id="cryptoInput" placeholder="bitcoin, ethereum, solana...">
                <button class="go-btn" onclick="fetchCrypto()">FETCH</button>
            </div>
            <div class="chips">
                <span class="chip c" onclick="qc('bitcoin')">₿ BTC</span>
                <span class="chip c" onclick="qc('ethereum')">Ξ ETH</span>
                <span class="chip c" onclick="qc('solana')">◎ SOL</span>
                <span class="chip c" onclick="qc('doge')">Ð DOGE</span>
                <span class="chip c" onclick="qc('bnb')">⬡ BNB</span>
                <span class="chip c" onclick="qc('xrp')">✦ XRP</span>
            </div>
            <div class="stock-result" id="cryptoResult"><span style="color:var(--dim)">Select crypto...</span></div>
        </div>
    </div>

    <!-- MUSIC -->
    <div class="panel" id="musicPanel">
        <div style="display:flex;gap:7px;">
            <input class="mf" type="text" id="musicInput" placeholder="Song, artist, album...">
            <button class="music-btn" onclick="playMusic()">▶ PLAY</button>
        </div>
        <div class="moods">
            <span class="mood" onclick="playMood('lofi hip hop study beats')">🎧 Lofi</span>
            <span class="mood" onclick="playMood('arijit singh sad songs')">💔 Sad</span>
            <span class="mood" onclick="playMood('bollywood gym workout 2025')">💪 Gym</span>
            <span class="mood" onclick="playMood('classical piano focus music')">🎹 Focus</span>
            <span class="mood" onclick="playMood('ap dhillon latest 2025')">🌙 AP Dhillon</span>
            <span class="mood" onclick="playMood('bollywood party hits 2024 2025')">🎉 Party</span>
            <span class="mood" onclick="playMood('quran recitation soothing')">🕌 Quran</span>
            <span class="mood" onclick="playMood('coding synthwave background music')">💻 Code</span>
        </div>
        <div id="musicPH" class="music-ph"><div style="font-size:2rem">🎵</div><div>Search a song to play</div></div>
        <div class="player-wrap" id="playerWrap" style="display:none;"><iframe id="musicIframe" src="" allow="autoplay;encrypted-media" allowfullscreen></iframe></div>
        <div class="np-label" id="npLabel"></div>
    </div>

    <!-- TOOLS -->
    <div class="panel" id="toolsPanel">
        <div class="sub-tabs">
            <button class="sub-tab active" onclick="switchTl('code',this)">💻 CODE</button>
            <button class="sub-tab" onclick="switchTl('img',this)">🎨 IMG PROMPT</button>
            <button class="sub-tab" onclick="switchTl('trans',this)">🌐 TRANSLATE</button>
            <button class="sub-tab" onclick="switchTl('math',this)">🧮 MATH</button>
        </div>
        <div class="sub-view active" id="tlv-code">
            <div class="card"><h3>💻 CODE EXPLAINER</h3>
                <textarea class="code-ta" id="codeInput" placeholder="Paste your code here..."></textarea>
                <button class="sb green" onclick="doExplain()">⚡ EXPLAIN</button>
                <div class="result-box" id="codeResult">Paste code and click Explain...</div>
            </div>
        </div>
        <div class="sub-view" id="tlv-img">
            <div class="card"><h3>🎨 AI IMAGE PROMPT</h3>
                <textarea class="sum-ta" id="imgDesc" placeholder="Describe what you want to generate..."></textarea>
                <button class="sb green" onclick="doImgPrompt()">✨ GENERATE</button>
                <div class="result-box" id="imgResult">Prompt will appear here...</div>
                <button class="sb" id="copyBtn" onclick="copyPrompt()" style="display:none">📋 COPY</button>
            </div>
        </div>
        <div class="sub-view" id="tlv-trans">
            <div class="card"><h3>🌐 TRANSLATE</h3>
                <textarea class="sum-ta" id="transInput" placeholder="Text to translate..."></textarea>
                <div style="display:flex;gap:6px;">
                    <input class="field" type="text" id="transLang" placeholder="Target language (Hindi, French, Arabic...)">
                    <button class="sb" style="flex:0;padding:7px 12px;" onclick="doTranslate()">GO</button>
                </div>
                <div class="result-box" id="transResult">Translation here...</div>
            </div>
        </div>
        <div class="sub-view" id="tlv-math">
            <div class="card"><h3>🧮 MATH AI SOLVER</h3>
                <textarea class="sum-ta" id="mathInput" placeholder="Math problem... (e.g. solve x^2 + 5x + 6 = 0)"></textarea>
                <button class="sb green" onclick="doMath()">⚡ SOLVE</button>
                <div class="result-box" id="mathResult">Solution here...</div>
            </div>
        </div>
    </div>

    <!-- MEMORY -->
    <div class="panel" id="memoryPanel">
        <div style="padding:10px;display:flex;flex-direction:column;gap:10px;overflow-y:auto;flex:1;">
            <div class="mem-section"><h3>🧠 MEMORY</h3><div id="memList"><span style="color:var(--dim);font-size:0.76rem;">Loading...</span></div></div>
            <div class="mem-section"><h3>💾 SAVE</h3>
                <div class="mem-save-row">
                    <input class="mem-input" type="text" id="memKey" placeholder="Key">
                    <input class="mem-input" type="text" id="memVal" placeholder="Value">
                    <button class="sb" style="flex:0;padding:6px 11px;" onclick="saveMem()">SAVE</button>
                </div>
            </div>
            <div class="mem-section"><h3>📜 SESSION</h3>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:var(--dim);line-height:2;">
                    Messages: <span id="histCount" style="color:var(--p)">0</span><br>
                    Firebase: <span id="fbStatus" style="color:var(--a)">Checking...</span><br>
                    Model: <span style="color:var(--p)">llama-3.3-70b</span>
                </div>
            </div>
        </div>
    </div>
</div>

<div id="floatTimer">
    <div class="f-lbl">📚 STUDY</div>
    <div class="f-time" id="floatTime">25:00</div>
    <button class="f-stop" onclick="stopStudyTimer()">■ STOP</button>
</div>

<div id="installBanner">
    📱 Install Astra?
    <button onclick="installPWA()">INSTALL</button>
    <button style="background:transparent;color:var(--dim);border:none;cursor:pointer;" onclick="this.parentElement.classList.remove('show')">✕</button>
</div>

<script>
// ── AUTH ──
let token=localStorage.getItem('astra_token')||'';
async function doLogin(){
    const u=document.getElementById('lUser').value.trim(),p=document.getElementById('lPass').value.trim(),err=document.getElementById('lErr');
    if(!u||!p){err.textContent='Dono fields chahiye.';return;}
    try{
        const r=await fetch('/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})});
        const d=await r.json();
        if(d.token){token=d.token;localStorage.setItem('astra_token',token);document.getElementById('loginScreen').classList.add('hidden');onLogin(u);}
        else{err.textContent=d.error||'Login failed.';document.getElementById('lPass').value='';}
    }catch{err.textContent='Server error.';}
}
function doLogout(){token='';localStorage.removeItem('astra_token');location.reload();}
(async()=>{if(!token)return;try{const r=await fetch('/verify-token',{headers:{'Authorization':'Bearer '+token}});if(r.ok){document.getElementById('loginScreen').classList.add('hidden');onLogin('Akram');}else{localStorage.removeItem('astra_token');token='';}}catch{localStorage.removeItem('astra_token');token='';}})();
document.getElementById('lPass').addEventListener('keypress',e=>{if(e.key==='Enter')doLogin();});
function onLogin(u){loadMem();updateTicker();loadWL();setTimeout(()=>bootMsg(u),400);document.getElementById('fbStatus').textContent='Connected ✓';}

// ── FORMAT ──
function fmt(t){if(!t)return '';return t.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\*(.+?)\*/g,'<em>$1</em>').replace(/`(.+?)`/g,'<code>$1</code>').replace(/\[(.+?)\]\((.+?)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>').replace(/\n/g,'<br>');}
function esc(t){return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

// ── TTS ──
let ttsOn=localStorage.getItem('tts_on')==='true';
const ttsBtn=document.getElementById('ttsBtn');ttsBtn.classList.toggle('tts-on',ttsOn);
function toggleTTS(){ttsOn=!ttsOn;localStorage.setItem('tts_on',ttsOn);ttsBtn.classList.toggle('tts-on',ttsOn);if(ttsOn)speak('Voice reply enabled.');}
async function speak(text){if(!ttsOn)return;const clean=text.replace(/<[^>]*>/g,'').replace(/[*_`#]/g,'').slice(0,400);try{const r=await fetch(`/speak?text=${encodeURIComponent(clean)}`,{headers:{'Authorization':'Bearer '+token}});if(r.ok){const b=await r.blob();const u=URL.createObjectURL(b);const a=new Audio(u);a.play();}}catch(e){}}

// ── TABS ──
function switchTab(n,b){document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));document.getElementById(n+'Panel').classList.add('active');b.classList.add('active');if(n==='memory')loadMem();}
function switchSub(n,b){document.querySelectorAll('.sub-view').forEach(v=>v.classList.remove('active'));document.querySelectorAll('#studyPanel .sub-tab').forEach(b=>b.classList.remove('active'));document.getElementById('sv-'+n).classList.add('active');b.classList.add('active');}
function switchSt(n,b){document.querySelectorAll('#stocksPanel .sub-view').forEach(v=>v.classList.remove('active'));document.querySelectorAll('#stocksPanel .sub-tab').forEach(b=>b.classList.remove('active'));document.getElementById('stv-'+n).classList.add('active');b.classList.add('active');}
function switchTl(n,b){document.querySelectorAll('#toolsPanel .sub-view').forEach(v=>v.classList.remove('active'));document.querySelectorAll('#toolsPanel .sub-tab').forEach(b=>b.classList.remove('active'));document.getElementById('tlv-'+n).classList.add('active');b.classList.add('active');}

// ── CHAT ──
const chatMsgs=document.getElementById('chatMsgs'),chatInput=document.getElementById('chatInput');
let msgCount=0;
function addMsg(role,html,typing=false){const d=document.createElement('div');if(typing){d.className='typing-dots';d.innerHTML='<span></span><span></span><span></span>';}else{d.className='msg '+role;d.innerHTML=html;}chatMsgs.appendChild(d);chatMsgs.scrollTop=chatMsgs.scrollHeight;return d;}
function qa(q){chatInput.value=q;sendChat();}
function bootMsg(u){addMsg('bot',fmt(`🖖 **Assalamalekum ${u.charAt(0).toUpperCase()+u.slice(1)} Bhai!**\n\n✨ **Astra Level 9 — Fully Powered** ✨\n\n• 💬 Chat — Memory AI + Voice + TTS\n• 📚 Study — Timer + Tasks + Quiz + Flashcards + Summarizer\n• 📈 Stocks — Search + Watchlist + Crypto\n• 🎵 Music — YouTube + 8 Moods\n• 🛠 Tools — Code + Image Prompt + Translate + Math AI\n• 🧠 Memory — Firebase Cloud Sync\n\n*Kya karna hai aaj? Ready hoon!* 🚀`));}
async function sendChat(){
    const text=chatInput.value.trim();if(!text)return;
    addMsg('user',esc(text));chatInput.value='';msgCount+=2;document.getElementById('histCount').textContent=msgCount;
    const tp=addMsg('bot','',true);
    try{
        const resp=await fetch('/ask-stream',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({message:text})});
        if(resp.status===401){tp.remove();addMsg('bot','🔒 Session expired.');doLogout();return;}
        tp.remove();const botEl=addMsg('bot','');
        const reader=resp.body.getReader();const dec=new TextDecoder();let full='',musicUrl=null;
        while(true){const{done,value}=await reader.read();if(done)break;const chunk=dec.decode(value);
            if(chunk.startsWith('{')&&chunk.includes('music_url')){try{const d=JSON.parse(chunk);if(d.music_url)musicUrl=d.music_url;if(d.message)full+=d.message;if(d.title)full+=`\n🎵 **${d.title}**`;}catch{full+=chunk;}}else{full+=chunk;}
            botEl.innerHTML=fmt(full);chatMsgs.scrollTop=chatMsgs.scrollHeight;}
        if(musicUrl)toPlayer(musicUrl);if(!full)botEl.innerHTML='No response.';else speak(full);
    }catch{tp.remove();addMsg('bot','❌ Network error.');}
}
chatInput.addEventListener('keypress',e=>{if(e.key==='Enter')sendChat();});
function startVoice(){if(!('webkitSpeechRecognition'in window||'SpeechRecognition'in window)){addMsg('bot','❌ Browser voice not supported.');return;}const SR=window.SpeechRecognition||window.webkitSpeechRecognition;const r=new SR();r.lang='en-IN';r.interimResults=false;r.onresult=e=>{chatInput.value=e.results[0][0].transcript;sendChat();};r.onerror=()=>addMsg('bot','❌ Voice error.');r.start();document.getElementById('statusPill').textContent='🎤 LISTENING';r.onend=()=>document.getElementById('statusPill').textContent='● ONLINE';}

// ── STUDY ──
let studyInterval=null,studyTotal=0,studyRemaining=0;const tasks=[];
function startStudyTimer(){const mins=parseInt(document.getElementById('studyMins').value)||25;studyTotal=mins*60;studyRemaining=studyTotal;clearInterval(studyInterval);fetch('/start-study',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({minutes:mins})});document.getElementById('floatTimer').style.display='block';updTimer();studyInterval=setInterval(()=>{if(studyRemaining>0){studyRemaining--;updTimer();}else{clearInterval(studyInterval);document.getElementById('floatTimer').style.display='none';speak('Study complete! Take a break!');alert('🎉 Done! Take a break.');}},1000);}
function stopStudyTimer(){clearInterval(studyInterval);studyRemaining=0;document.getElementById('floatTimer').style.display='none';updTimer();fetch('/stop-study',{method:'POST',headers:{'Authorization':'Bearer '+token}});}
function resetTimer(){stopStudyTimer();const m=parseInt(document.getElementById('studyMins').value)||25;document.getElementById('timerDisplay').textContent=`${String(m).padStart(2,'0')}:00`;document.getElementById('progressBar').style.width='100%';}
function updTimer(){const m=Math.floor(studyRemaining/60),s=studyRemaining%60,f=`${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;document.getElementById('timerDisplay').textContent=f;document.getElementById('floatTime').textContent=f;document.getElementById('progressBar').style.width=(studyTotal>0?(studyRemaining/studyTotal)*100:100)+'%';}
function addTask(){const inp=document.getElementById('taskInput'),t=inp.value.trim();if(!t)return;tasks.push({text:t,done:false});inp.value='';renderTasks();}
function renderTasks(){const l=document.getElementById('taskList');l.innerHTML='';if(!tasks.length){l.innerHTML='<div style="color:var(--dim);font-size:0.76rem;padding:6px;">No tasks.</div>';return;}tasks.forEach((t,i)=>{const d=document.createElement('div');d.className='task-item'+(t.done?' done':'');d.innerHTML=`<div class="task-chk" onclick="toggleTask(${i})">${t.done?'✓':''}</div><span style="flex:1">${esc(t.text)}</span><span style="cursor:pointer;color:var(--s);font-size:0.73rem;" onclick="rmTask(${i})">✕</span>`;l.appendChild(d);});}
function toggleTask(i){tasks[i].done=!tasks[i].done;renderTasks();}
function rmTask(i){tasks.splice(i,1);renderTasks();}
function clearDone(){for(let i=tasks.length-1;i>=0;i--){if(tasks[i].done)tasks.splice(i,1);}renderTasks();}
document.getElementById('taskInput').addEventListener('keypress',e=>{if(e.key==='Enter')addTask();});
async function aiPrioritize(){if(!tasks.length){alert('Tasks add karo pehle!');return;}const btn=document.getElementById('aiPriBtn');btn.textContent='⏳...';btn.disabled=true;const list=tasks.map((t,i)=>`${i+1}. ${t.text}`).join('\n');try{const r=await fetch('/ask-stream',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({message:`Prioritize these tasks for me:\n${list}`})});const reader=r.body.getReader();const dec=new TextDecoder();let full='';while(true){const{done,value}=await reader.read();if(done)break;full+=dec.decode(value);}addMsg('bot',fmt(full));document.querySelectorAll('.tab-btn')[0].click();}catch{}btn.textContent='🤖 AI Prioritize';btn.disabled=false;}
async function doQuiz(){const t=document.getElementById('quizTopic').value.trim();if(!t)return;document.getElementById('quizArea').textContent='⏳ Generating...';try{const r=await fetch('/quiz',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({topic:t})});const d=await r.json();document.getElementById('quizArea').innerHTML=fmt(d.quiz||'Error.');}catch{document.getElementById('quizArea').textContent='❌ Error.';}}
let flashCards=[],flashIdx=0;
async function doFlash(){const t=document.getElementById('flashTopic').value.trim();if(!t)return;document.getElementById('cFront').textContent='⏳ Generating...';document.getElementById('cBack').textContent='';flashCards=[];flashIdx=0;try{const r=await fetch('/flashcards',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({topic:t})});const d=await r.json();if(d.cards&&d.cards.length){flashCards=d.cards;showCard();}else document.getElementById('cFront').textContent='❌ Failed.';}catch{document.getElementById('cFront').textContent='❌ Error.';}}
function showCard(){if(!flashCards.length)return;const c=flashCards[flashIdx];const card=document.getElementById('flashcard');card.classList.remove('flipped');setTimeout(()=>{document.getElementById('cFront').textContent=c.front;document.getElementById('cBack').textContent=c.back;document.getElementById('cardCount').textContent=`${flashIdx+1}/${flashCards.length}`;},250);}
function flipCard(){document.getElementById('flashcard').classList.toggle('flipped');}
function nextCard(){if(flashIdx<flashCards.length-1){flashIdx++;showCard();}}
function prevCard(){if(flashIdx>0){flashIdx--;showCard();}}
async function doSummarize(){const text=document.getElementById('sumInput').value.trim();if(!text)return;document.getElementById('sumResult').textContent='⏳ Summarizing...';try{const r=await fetch('/summarize',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({text})});const d=await r.json();document.getElementById('sumResult').innerHTML=fmt(d.summary||'Error.');}catch{document.getElementById('sumResult').textContent='❌ Error.';}}

// ── STOCKS ──
let watchlist=JSON.parse(localStorage.getItem('astra_wl')||'[]');
function loadWL(){renderWL();}
function renderWL(){const el=document.getElementById('wlEl'),hint=document.getElementById('wlHint');if(!watchlist.length){el.innerHTML='';hint.style.display='block';return;}hint.style.display='none';el.innerHTML='';watchlist.forEach((sym,i)=>{const d=document.createElement('div');d.className='wl-item';d.innerHTML=`<span class="wl-sym">${sym}</span><span id="wlp-${sym}" style="color:var(--text);font-size:0.75rem;">...</span><span class="wl-del" onclick="rmWL(${i})">✕</span>`;el.appendChild(d);fwlp(sym);});}
async function fwlp(sym){try{const r=await fetch('/stock',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({symbol:sym})});const d=await r.json();const el=document.getElementById('wlp-'+sym);if(el){const lines=d.result.split('\n');el.innerHTML=lines[1]||'—';}}catch{}}
function addWL(){const sym=document.getElementById('stockInput').value.trim().toUpperCase();if(!sym||watchlist.includes(sym))return;watchlist.push(sym);localStorage.setItem('astra_wl',JSON.stringify(watchlist));renderWL();}
function rmWL(i){watchlist.splice(i,1);localStorage.setItem('astra_wl',JSON.stringify(watchlist));renderWL();}
async function fetchStock(){const sym=document.getElementById('stockInput').value.trim().toUpperCase();if(!sym)return;const box=document.getElementById('stockResult');box.innerHTML='<span style="color:var(--dim)">⏳ Fetching...</span>';try{const r=await fetch('/stock',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({symbol:sym})});const d=await r.json();box.innerHTML=fmt(d.result||'No data.');}catch{box.innerHTML='❌ Error.';}}
function qs(s){document.getElementById('stockInput').value=s;fetchStock();}
async function fetchCrypto(){const coin=document.getElementById('cryptoInput').value.trim();if(!coin)return;const box=document.getElementById('cryptoResult');box.innerHTML='<span style="color:var(--dim)">⏳ Fetching...</span>';try{const r=await fetch('/crypto',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({coin})});const d=await r.json();box.innerHTML=fmt(d.result||'No data.');}catch{box.innerHTML='❌ Error.';}}
function qc(c){document.getElementById('cryptoInput').value=c;fetchCrypto();}

// ── MUSIC ──
async function playMusic(){const q=document.getElementById('musicInput').value.trim();if(!q)return;loadMusic(q);}
function playMood(m){document.getElementById('musicInput').value=m;loadMusic(m);}
async function loadMusic(q){document.getElementById('musicPH').style.display='flex';document.getElementById('musicPH').querySelector('div:last-child').textContent='⏳ Loading...';document.getElementById('playerWrap').style.display='none';document.getElementById('npLabel').style.display='none';try{const r=await fetch('/play-music',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({query:q})});const d=await r.json();if(d.url)toPlayer(d.url,d.title);else document.getElementById('musicPH').querySelector('div:last-child').textContent='❌ Not found.';}catch{document.getElementById('musicPH').querySelector('div:last-child').textContent='❌ Error.';}}
function toPlayer(url,title){document.getElementById('musicPH').style.display='none';document.getElementById('musicIframe').src=url;document.getElementById('playerWrap').style.display='block';if(title){const l=document.getElementById('npLabel');l.textContent='🎵 '+title;l.style.display='block';}document.querySelectorAll('.tab-btn')[3].click();}

// ── TOOLS ──
async function doExplain(){const code=document.getElementById('codeInput').value.trim();if(!code)return;document.getElementById('codeResult').textContent='⏳ Analyzing...';try{const r=await fetch('/explain-code',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({code})});const d=await r.json();document.getElementById('codeResult').innerHTML=fmt(d.result||'Error.');}catch{document.getElementById('codeResult').textContent='❌ Error.';}}
let lastPrompt='';
async function doImgPrompt(){const desc=document.getElementById('imgDesc').value.trim();if(!desc)return;document.getElementById('imgResult').textContent='⏳ Generating...';document.getElementById('copyBtn').style.display='none';try{const r=await fetch('/image-prompt',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({description:desc})});const d=await r.json();lastPrompt=d.prompt||'Error.';document.getElementById('imgResult').textContent=lastPrompt;document.getElementById('copyBtn').style.display='block';}catch{document.getElementById('imgResult').textContent='❌ Error.';}}
function copyPrompt(){navigator.clipboard.writeText(lastPrompt).then(()=>alert('✅ Copied!'));}
async function doTranslate(){const text=document.getElementById('transInput').value.trim(),lang=document.getElementById('transLang').value.trim();if(!text||!lang)return;document.getElementById('transResult').textContent='⏳ Translating...';try{const r=await fetch('/ask-stream',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({message:`Translate to ${lang}:\n\n${text}`})});const reader=r.body.getReader();const dec=new TextDecoder();let full='';while(true){const{done,value}=await reader.read();if(done)break;full+=dec.decode(value);}document.getElementById('transResult').innerHTML=fmt(full);}catch{document.getElementById('transResult').textContent='❌ Error.';}}
async function doMath(){const prob=document.getElementById('mathInput').value.trim();if(!prob)return;document.getElementById('mathResult').textContent='⏳ Solving...';try{const r=await fetch('/ask-stream',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({message:`Solve step by step:\n${prob}`})});const reader=r.body.getReader();const dec=new TextDecoder();let full='';while(true){const{done,value}=await reader.read();if(done)break;full+=dec.decode(value);}document.getElementById('mathResult').innerHTML=fmt(full);}catch{document.getElementById('mathResult').textContent='❌ Error.';}}

// ── MEMORY ──
async function loadMem(){try{const r=await fetch('/memory',{headers:{'Authorization':'Bearer '+token}});const d=await r.json();const list=document.getElementById('memList');if(!d.memory||!Object.keys(d.memory).length){list.innerHTML='<span style="color:var(--dim);font-size:0.74rem;">No memories yet.</span>';return;}list.innerHTML='';Object.entries(d.memory).forEach(([k,v])=>{const item=document.createElement('div');item.className='mem-item';const val=typeof v==='object'?JSON.stringify(v):String(v);item.innerHTML=`<span class="mem-key">${esc(k)}</span><span class="mem-val" title="${esc(val)}">${esc(val.slice(0,55))}${val.length>55?'...':''}</span>`;list.appendChild(item);});}catch{document.getElementById('memList').innerHTML='<span style="color:var(--dim)">Load failed.</span>';}}
async function saveMem(){const k=document.getElementById('memKey').value.trim(),v=document.getElementById('memVal').value.trim();if(!k||!v)return;try{await fetch('/memory/save',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({key:k,value:v})});document.getElementById('memKey').value='';document.getElementById('memVal').value='';loadMem();}catch{alert('Save failed.');}}

// ── TICKER ──
async function updateTicker(){try{const r=await fetch('/market-ticker',{headers:{'Authorization':'Bearer '+token}});const d=await r.json();document.getElementById('tickerInner').innerHTML=d.ticker||'Loading...';}catch{}}
setInterval(updateTicker,60000);

// ── PWA ──
let deferredPWA=null;
window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();deferredPWA=e;document.getElementById('installBanner').classList.add('show');});
function installPWA(){if(deferredPWA){deferredPWA.prompt();deferredPWA.userChoice.then(()=>{deferredPWA=null;document.getElementById('installBanner').classList.remove('show');});}}
if('serviceWorker'in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}
</script>
</body>
</html>"""

# ─── ROUTES ───
@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/manifest.json')
def manifest(): return jsonify({"name":"Astra Level 9","short_name":"Astra","start_url":"/","display":"standalone","background_color":"#020408","theme_color":"#00f0ff","description":"Astra AI by Akram","icons":[{"src":"https://img.icons8.com/fluency/192/artificial-intelligence.png","sizes":"192x192","type":"image/png"}]})

@app.route('/sw.js')
def sw(): return Response("const C='astra-v3';const U=['/'];self.addEventListener('install',e=>e.waitUntil(caches.open(C).then(c=>c.addAll(U))));self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;e.respondWith(fetch(e.request).catch(()=>caches.match(e.request)));});", mimetype='application/javascript')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    u = data.get('username','').strip().lower()
    p = data.get('password','').strip()
    stored = USERS.get(u)
    if stored and stored == hashlib.sha256(p.encode()).hexdigest():
        return jsonify({'token': generate_token(u), 'username': u})
    return jsonify({'error': 'Galat username ya password!'}), 401

@app.route('/verify-token')
@require_auth
def verify_token_route(): return jsonify({'valid': True})

@app.route('/ask-stream', methods=['POST'])
@require_auth
def ask_stream():
    data = request.get_json()
    input_text = data.get('message','').strip()
    if not input_text: return Response("Boliye...", mimetype='text/plain')
    username = get_username_from_request()
    def generate():
        low = input_text.lower()
        if any(x in low for x in ['stock','share price']):
            for sym in ['RELIANCE','TCS','INFY','WIPRO','HDFCBANK','TATAMOTORS','NVIDIA','AAPL','TSLA','GOOGL','MSFT','ICICIBANK','SBIN']:
                if sym.lower() in low: yield get_stock_price(sym); return
            yield get_stock_price('RELIANCE'); return
        if any(x in low for x in ['bitcoin','btc','ethereum','eth','solana','sol','crypto','doge','dogecoin']):
            coin = 'bitcoin'
            if 'ethereum' in low or ' eth' in low: coin = 'ethereum'
            elif 'solana' in low: coin = 'solana'
            elif 'doge' in low: coin = 'dogecoin'
            yield get_crypto_price(coin); return
        if low.startswith('play ') or 'gaana bajao' in low:
            q = re.sub(r'^play |gaana bajao', '', low).strip()
            if q:
                url, title, _ = get_youtube_embed_url(q + ' song')
                if url: yield json.dumps({"message":"🎵 Playing!","music_url":url,"title":title or q}); return
                yield f"🎵 Search: https://youtube.com/results?search_query={urllib.parse.quote(q)}"
            else: yield "Kaunsa gaana? 🎵"
            return
        if 'weather' in low or 'mausam' in low:
            city = re.sub(r'weather|mausam|in|ka|of|aaj|kya|hai', '', low).strip() or 'Delhi'
            yield get_weather(city); return
        if 'news' in low or 'khabar' in low:
            q = re.sub(r'news|khabar|aaj ki|latest|top|today', '', low).strip() or None
            yield get_news(query=q); return
        if 'start study' in low or 'pomodoro' in low:
            m = re.search(r'(\d+)\s*min', low); mins = int(m.group(1)) if m else 25
            threading.Thread(target=study_timer_thread, args=(mins,), daemon=True).start()
            yield f"🎓 **{mins} minute** study timer started! 💪"; return
        if 'yaad rakho' in low or 'remember that' in low:
            item = re.sub(r'yaad rakho|remember that', '', low).strip()
            if item:
                mem = get_memory_cloud(username, 'notes') or []
                if not isinstance(mem, list): mem = []
                mem.append(item); save_memory_cloud(username, 'notes', mem[-20:])
                yield f"💾 Yaad rakh liya: **{item}** ✅"; return
        for chunk in ask_nvidia_stream(input_text, username): yield chunk
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/memory')
@require_auth
def get_memory_route(): return jsonify({'memory': get_memory_cloud(get_username_from_request()) or {}})

@app.route('/memory/save', methods=['POST'])
@require_auth
def save_memory_route():
    data = request.get_json(); k, v = data.get('key','').strip(), data.get('value','').strip()
    if k and v: save_memory_cloud(get_username_from_request(), k, v); return jsonify({'status':'saved'})
    return jsonify({'error':'Missing key/value'}), 400

@app.route('/start-study', methods=['POST'])
@require_auth
def start_study():
    data = request.get_json(); mins = data.get('minutes', 25)
    threading.Thread(target=study_timer_thread, args=(mins,), daemon=True).start()
    return jsonify({"status":"started"})

@app.route('/stop-study', methods=['POST'])
@require_auth
def stop_study():
    study_state["active"] = False; study_state["remaining"] = 0
    return jsonify({"status":"stopped"})

@app.route('/stock', methods=['POST'])
@require_auth
def stock_route(): return jsonify({"result": get_stock_price(request.get_json().get('symbol','RELIANCE').strip().upper())})

@app.route('/crypto', methods=['POST'])
@require_auth
def crypto_route(): return jsonify({"result": get_crypto_price(request.get_json().get('coin','bitcoin').strip())})

@app.route('/play-music', methods=['POST'])
@require_auth
def play_music():
    q = request.get_json().get('query','').strip()
    if not q: return jsonify({"error":"No query"})
    url, title, dur = get_youtube_embed_url(q)
    if url: return jsonify({"url":url,"title":title or q,"duration":dur})
    return jsonify({"error":"Not found"})

@app.route('/quiz', methods=['POST'])
@require_auth
def quiz_route():
    topic = request.get_json().get('topic','').strip()
    return jsonify({"quiz": generate_quiz(topic) if topic else "Topic batao."})

@app.route('/flashcards', methods=['POST'])
@require_auth
def flashcards_route():
    topic = request.get_json().get('topic','').strip()
    return jsonify({"cards": generate_flashcards(topic) if topic else []})

@app.route('/summarize', methods=['POST'])
@require_auth
def summarize_route():
    text = request.get_json().get('text','').strip()
    return jsonify({"summary": summarize_text(text) if text else "Text dalo."})

@app.route('/explain-code', methods=['POST'])
@require_auth
def explain_code_route():
    code = request.get_json().get('code','').strip()
    return jsonify({"result": explain_code(code) if code else "Code paste karo."})

@app.route('/image-prompt', methods=['POST'])
@require_auth
def image_prompt_route():
    desc = request.get_json().get('description','').strip()
    return jsonify({"prompt": generate_image_prompt(desc) if desc else "Description do."})

@app.route('/market-ticker')
def market_ticker():
    try:
        s = get_portfolio_summary(); parts = s.split(' | ')
        html_p = [f'<span class="t-up">{p}</span>' if '▲' in p else f'<span class="t-dn">{p}</span>' if '▼' in p else p for p in parts]
        return jsonify({"ticker": ' &nbsp;·&nbsp; '.join(html_p) + ' &nbsp;·&nbsp; <span style="color:var(--dim)">ASTRA L9 · AKRAM ANSARI</span>'})
    except: return jsonify({"ticker":"Market loading..."})

@app.route('/health')
def health(): return "Astra Level 9 Online! 🚀", 200

VOICE = "hi-IN-SwaraNeural"

@app.route('/speak')
@require_auth
def speak_route():
    text = request.args.get('text', '').strip()
    if not text: return "No text provided", 400
    
    def generate():
        # Using a temporary file to ensure stable streaming in production
        temp_id = str(uuid.uuid4())
        tmp_path = os.path.join(tempfile.gettempdir(), f"astra_voice_{temp_id}.mp3")
        try:
            asyncio.run(edge_tts.Communicate(text, VOICE, rate="+5%", pitch="+2Hz").save(tmp_path))
            if os.path.exists(tmp_path):
                with open(tmp_path, "rb") as f:
                    while chunk := f.read(8192):
                        yield chunk
        finally:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
                
    return Response(stream_with_context(generate()), mimetype='audio/mpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
