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
from dateparser import parse
from functools import lru_cache, wraps

# ---------- Kill-Switch (Safety for Render) ----------
import sys
# Block any accidental Gemini/Claude/Anthropic calls if they exist in sub-dependencies
sys.modules['google.generativeai'] = None
sys.modules['anthropic'] = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'astra-level-9-super-secret')

# ---------- Configuration ----------
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

def ask_nvidia_stream(prompt, system_message=None):
    if not system_message:
        system_message = "You are Astra, a Level 9 AI assistant. You are smart, direct, and speak in Hinglish. You help Akram with stocks, studies, and news."
    try:
        response = client.chat.completions.create(
            model="meta/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"AI Error: {str(e)}"

# ---------- Financial Module ----------
@ttl_cache(300) # 5 min cache
def get_stock_price(symbol):
    try:
        # Check if user meant an Indian stock (usually needs .NS for Yahoo)
        if symbol.upper() in ['RELIANCE', 'TCS', 'INFY', 'WIPRO', 'HDFCBANK']:
            symbol = f"{symbol.upper()}.NS"
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        res = data['chart']['result'][0]['meta']
        price = res['regularMarketPrice']
        currency = res['currency']
        change = res.get('regularMarketChange', 0)
        return f"📈 **{symbol.replace('.NS','')}**\nPrice: {currency} {price:,.2f}\nChange: {change:+.2f}"
    except Exception as e:
        return f"Could not fetch stock {symbol}."

@ttl_cache(300)
def get_crypto_price(coin):
    try:
        coin_id = coin.lower().strip()
        # Mapping common names to CoinGecko IDs
        mapping = {'btc': 'bitcoin', 'eth': 'ethereum', 'doge': 'dogecoin', 'sol': 'solana'}
        coin_id = mapping.get(coin_id, coin_id)
        
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd,inr"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        prices = data[coin_id]
        return f"🪙 **{coin_id.upper()}**\n💰 ${prices['usd']:,.2f} USD\n🇮🇳 ₹{prices['inr']:,.2f} INR"
    except:
        return "Crypto price check failed."

# ---------- News Module (GNews) ----------
@ttl_cache(1800)
def get_news(query=None, country="in"):
    api_key = os.getenv('GNEWS_API_KEY') or os.getenv('NEWS_API_KEY')
    if not api_key: return "News API key missing."
    if query:
        url = f"https://gnews.io/api/v4/search?q={urllib.parse.quote(query)}&token={api_key}&lang=en&max=3"
    else:
        url = f"https://gnews.io/api/v4/top-headlines?country={country}&token={api_key}&max=3"
    try:
        data = requests.get(url, timeout=10).json()
        articles = data.get('articles', [])
        if not articles: return "No news found."
        return "\n".join([f"📰 **{a['title']}**\n🔗 [Link]({a['url']})\n" for a in articles])
    except: return "News error."

# ---------- Study Mode Module ----------
study_state = {"active": False, "remaining": 0, "total": 0}

def study_timer_thread(minutes):
    global study_state
    study_state["active"] = True
    study_state["total"] = minutes * 60
    study_state["remaining"] = study_state["total"]
    while study_state["remaining"] > 0 and study_state["active"]:
        time.sleep(1)
        study_state["remaining"] -= 1
    study_state["active"] = False

# ---------- Weather Module ----------
@ttl_cache(600)
def get_weather(city):
    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key: return "Weather API key missing."
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        data = requests.get(url, timeout=10).json()
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        return f"☁️ {city.title()}: {temp}°C, {desc}."
    except: return "Weather check failed."

# ---------- YouTube Mini-Player Helper ----------
def get_youtube_embed_url(query):
    """Get YouTube embed URL for a search query"""
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if 'entries' in info and info['entries']:
                video_id = info['entries'][0]['id']
                return f"https://www.youtube.com/embed/{video_id}?autoplay=1&controls=1&rel=0"
    except:
        pass
    return None

# ---------- HTML (Cinematic Level 9 HUD) ----------
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Astra HUD | Level 9</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --p: #00ff9d; --s: #ff00e5; --bg: #000; }
        body { background: var(--bg); color: #fff; font-family: 'Poppins', sans-serif; height: 100vh; overflow: hidden; display: flex; justify-content: center; align-items: center; }
        .stars { position: fixed; width: 100%; height: 100%; z-index: -1; }
        .container { width: 95%; max-width: 1000px; height: 85vh; background: rgba(10,15,25,0.7); backdrop-filter: blur(15px); border: 1px solid var(--p); border-radius: 30px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 0 30px rgba(0,255,157,0.2); }
        .header { padding: 15px 30px; border-bottom: 1px solid rgba(0,255,157,0.2); display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-family: 'Orbitron'; font-size: 1.2rem; letter-spacing: 2px; color: var(--p); text-shadow: 0 0 10px var(--p); }
        .chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; scroll-behavior: smooth; }
        .msg { max-width: 80%; padding: 12px 18px; border-radius: 18px; font-size: 0.95rem; line-height: 1.5; position: relative; animation: slideUp 0.3s ease; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .user { align-self: flex-end; background: linear-gradient(135deg, var(--p), var(--s)); color: #000; border-bottom-right-radius: 4px; }
        .bot { align-self: flex-start; background: rgba(255,255,255,0.05); border: 1px solid rgba(0,255,157,0.3); border-bottom-left-radius: 4px; }
        .input-area { padding: 20px; border-top: 1px solid rgba(0,255,157,0.2); display: flex; gap: 10px; }
        input { flex: 1; background: rgba(0,0,0,0.5); border: 1px solid var(--p); border-radius: 30px; padding: 12px 20px; color: #fff; outline: none; transition: 0.3s; }
        input:focus { box-shadow: 0 0 15px var(--p); }
        button { background: var(--p); color: #000; border: none; border-radius: 30px; padding: 0 25px; font-family: 'Orbitron'; font-weight: bold; cursor: pointer; transition: 0.3s; }
        button:hover { background: #fff; transform: scale(1.05); }

        /* Widgets */
        #studyWidget { position: fixed; top: 20px; right: 20px; background: rgba(255,0,229,0.15); border: 1px solid var(--s); padding: 15px; border-radius: 20px; backdrop-filter: blur(10px); display: none; text-align: center; width: 150px; z-index: 10; }
        #studyTimer { font-family: 'Orbitron'; font-size: 1.8rem; color: #fff; margin: 5px 0; }
        
        #musicPlayer { position: fixed; bottom: 20px; left: 20px; background: rgba(0,0,0,0.8); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid var(--p); display: none; z-index: 100; overflow: hidden; box-shadow: 0 0 20px var(--p); }
        #musicPlayer iframe { width: 300px; height: 170px; border: none; }
        .close-music { position: absolute; top: 5px; right: 5px; background: var(--s); border: none; border-radius: 50%; width: 22px; height: 22px; color: white; cursor: pointer; font-size: 12px; z-index: 101; }

        .ticker { position: fixed; bottom: 0; left: 0; width: 100%; background: rgba(0,255,157,0.05); font-family: 'Orbitron'; font-size: 0.7rem; padding: 5px; overflow: hidden; white-space: nowrap; border-top: 1px solid rgba(0,255,157,0.1); }
        .ticker-inner { display: inline-block; animation: ticker 30s linear infinite; }
        @keyframes ticker { from { transform: translateX(100%); } to { transform: translateX(-100%); } }
    </style>
</head>
<body>
    <div class="stars" id="stars"></div>
    <div id="studyWidget">
        <div style="font-size: 0.6rem; color: var(--s);">STUDY MODE</div>
        <div id="studyTimer">25:00</div>
        <button onclick="stopStudy()" style="font-size: 0.6rem; padding: 5px 10px; background: var(--s);">STOP</button>
    </div>
    <div id="musicPlayer">
        <button class="close-music" onclick="closeMusic()">✕</button>
        <iframe id="musicIframe" src="" allow="autoplay"></iframe>
    </div>
    <div class="container">
        <div class="header">
            <h1>ASTRA <span style="font-size: 0.6rem; opacity: 0.7;">LEVEL 9 HUD</span></h1>
            <div id="status" style="font-size: 0.6rem; font-family: 'Orbitron'; color: var(--p);">SYSTEM ACTIVE</div>
        </div>
        <div class="chat" id="chat"></div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Ask Astra anything..." autocomplete="off">
            <button onclick="send()">SEND</button>
        </div>
    </div>
    <div class="ticker">
        <div class="ticker-inner" id="tickerText">BTC: Loading... | ETH: Loading... | Reliance: Loading... | Nifty: Loading...</div>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');

        function addMsg(role, text) {
            const div = document.createElement('div');
            div.className = `msg ${role}`;
            div.innerHTML = text.replace(/\\n/g, '<br>');
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        async function send() {
            const val = input.value.trim();
            if (!val) return;
            addMsg('user', val);
            input.value = '';
            
            const botDiv = addMsg('bot', 'Processing...');
            try {
                const resp = await fetch('/ask-stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: val})
                });
                const reader = resp.body.getReader();
                const decoder = new TextDecoder();
                botDiv.innerHTML = '';
                let fullText = '';
                let musicUrl = null;

                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    if (chunk.startsWith('{') && chunk.includes('music_url')) {
                        try {
                            const data = JSON.parse(chunk);
                            if (data.music_url) musicUrl = data.music_url;
                            if (data.message) fullText += data.message;
                        } catch(e) { fullText += chunk; }
                    } else {
                        fullText += chunk;
                    }
                    botDiv.innerHTML = fullText.replace(/\\n/g, '<br>');
                    chat.scrollTop = chat.scrollHeight;
                }
                if (musicUrl) showMusicPlayer(musicUrl);
            } catch { botDiv.innerHTML = 'Connection Error.'; }
        }

        function showMusicPlayer(url) {
            const player = document.getElementById('musicPlayer');
            const iframe = document.getElementById('musicIframe');
            iframe.src = url;
            player.style.display = 'block';
        }

        function closeMusic() {
            const player = document.getElementById('musicPlayer');
            const iframe = document.getElementById('musicIframe');
            iframe.src = '';
            player.style.display = 'none';
        }

        function stopStudy() { fetch('/stop-study', {method: 'POST'}); }

        // Sync Logic
        setInterval(async () => {
            const resp = await fetch('/status');
            const data = await resp.json();
            const widget = document.getElementById('studyWidget');
            if (data.study.active) {
                widget.style.display = 'block';
                const m = Math.floor(data.study.remaining/60);
                const s = data.study.remaining%60;
                document.getElementById('studyTimer').innerText = `${m}:${s < 10 ? '0'+s : s}`;
            } else { widget.style.display = 'none'; }
            
            // Update Ticker (Sample logic)
            if (data.market) {
                document.getElementById('tickerText').innerText = data.market;
            }
        }, 1000);

        // Stars
        const stars = document.getElementById('stars');
        for(let i=0; i<100; i++) {
            const s = document.createElement('div');
            s.style.position = 'absolute';
            s.style.background = '#fff';
            s.style.width = Math.random()*3+'px';
            s.style.height = s.style.width;
            s.style.left = Math.random()*100+'%';
            s.style.top = Math.random()*100+'%';
            s.style.borderRadius = '50%';
            s.style.opacity = Math.random();
            stars.appendChild(s);
        }

        input.addEventListener('keypress', (e) => { if(e.key==='Enter') send(); });
        window.onload = () => addMsg('bot', 'Welcome back, Akram. Level 9 HUD online. How can I help you today?');
    </script>
</body>
</html>
"""

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/status')
def status():
    # Fetch ticker data (sample)
    market = "BTC: $75,340 | ETH: $2,890 | RELIANCE: ₹1,363 | NVIDIA: $145.20"
    return jsonify({"study": study_state, "market": market})

@app.route('/stop-study', methods=['POST'])
def stop_study():
    study_state["active"] = False
    return "OK"

@app.route('/ask-stream', methods=['POST'])
def ask_stream():
    data = request.get_json()
    input_text = data.get('message', '').strip()
    if not input_text: return Response("Boliye...", mimetype='text/plain')
    
    def generate():
        low = input_text.lower()
        
        # Financial Commands
        if 'stock' in low or 'share' in low:
            for symbol in ['RELIANCE', 'TCS', 'INFY', 'WIPRO', 'HDFCBANK', 'NVIDIA', 'AAPL', 'TSLA']:
                if symbol.lower() in low:
                    yield get_stock_price(symbol)
                    return
            yield get_stock_price('RELIANCE')
            return
            
        if 'crypto' in low or 'bitcoin' in low or 'btc' in low:
            coin = 'bitcoin'
            if 'ethereum' in low or 'eth' in low: coin = 'ethereum'
            elif 'solana' in low or 'sol' in low: coin = 'solana'
            yield get_crypto_price(coin)
            return

        # YouTube Music Command
        if low.startswith('play song ') or low.startswith('play '):
            song = low.replace('play song ', '').replace('play ', '').strip()
            if not song:
                yield "Kaunsa gaana chahiye? Song name batao."
            else:
                query = f"{song} song"
                embed_url = get_youtube_embed_url(query)
                if embed_url:
                    import json
                    yield json.dumps({
                        "message": f"🎵 **Playing: {song}**\nEnjoy the music! 🎧",
                        "music_url": embed_url
                    })
                else:
                    search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(song)}"
                    yield f"🎵 Could not play directly. [Click here to search on YouTube]({search_url})"
            return

        # Study Mode
        if 'start study' in low or 'focus mode' in low:
            mins = 25
            match = re.search(r'(\d+)\s*min', low)
            if match: mins = int(match.group(1))
            threading.Thread(target=study_timer_thread, args=(mins,)).start()
            yield f"🎓 **Study Mode Activated!**\nTimer set for {mins} minutes. Focus well, Akram! 📖"
            return
            
        if 'analyze syllabus' in low or 'analyze text' in low:
            text = input_text.replace('analyze syllabus','').replace('analyze text','').strip()
            if not text: yield "Pehle syllabus context dijiye."
            else:
                for chunk in ask_nvidia_stream(f"Analyze this syllabus and give me a study plan: {text}"):
                    yield chunk
            return

        # Regular AI
        for chunk in ask_nvidia_stream(input_text):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
