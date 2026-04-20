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
app.secret_key = os.getenv('SECRET_KEY', 'astra-secret-888-chhapra')

# ---------- Configuration ----------
PROFILES_DIR = 'profiles'
REMINDER_CHECK_INTERVAL = 3600
os.makedirs(PROFILES_DIR, exist_ok=True)

# ---------- Simple TTL Cache ----------
def ttl_cache(seconds):
    """Time-based cache decorator"""
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
    """Streaming version for anti-gravity feel"""
    if not system_message:
        system_message = "You are Astra, a highly intelligent and helpful AI assistant for Akram from Chhapra, Bihar. Respond in Hinglish. Be direct, concise, and smart. Greet only in the first turn if instructed."
    
    try:
        response = client.chat.completions.create(
            model="meta/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"AI error: {str(e)}"

# ---------- Cached Weather ----------
@ttl_cache(600)  # Cache for 10 minutes
def get_weather(city):
    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key:
        return "Weather API key missing."
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get('cod') != 200:
            return f"City '{city}' not found."
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        return f"Weather in {city.title()}: {temp}°C, {desc}."
    except:
        return "Weather service error."

# ---------- Cached News (GNews) ----------
@ttl_cache(1800)  # Cache for 30 minutes
def get_news(query=None, country="in"):
    api_key = os.getenv('GNEWS_API_KEY') or os.getenv('NEWS_API_KEY')
    if not api_key:
        return "News API key missing. Please set GNEWS_API_KEY or NEWS_API_KEY in Render."
    
    # Use GNews API if available, otherwise fallback logic can be added
    # For now, following the GNews URL structure provided
    if query:
        url = f"https://gnews.io/api/v4/search?q={urllib.parse.quote(query)}&token={api_key}&lang=en&max=5"
    else:
        url = f"https://gnews.io/api/v4/top-headlines?country={country}&token={api_key}&max=5"
    
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get('errors'):
            return f"News error: {data['errors'][0]}"
        articles = data.get('articles', [])
        if not articles:
            return "No news found. Try a different keyword."
        news_list = []
        for art in articles[:5]:
            title = art.get('title', 'No title')
            link = art.get('url', '#')
            news_list.append(f"📰 **{title}**\n🔗 [Read more]({link})\n")
        return "\n".join(news_list)
    except Exception as e:
        # Fallback to NewsAPI logic if GNews fails and it's a NewsAPI key
        if "newsapi.org" not in locals(): # Placeholder for fallback logic
             return f"News error: {e}"

# ---------- Smart Web Search ----------
def smart_search(query):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."
            context = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
            prompt = f"Summarize the following search results about '{query}' in 2-3 sentences in Hinglish:\n{context}"
            summary = ask_nvidia_stream(prompt, "You are a helpful summarizer.")
            summary_text = ""
            for chunk in summary:
                summary_text += chunk
            links = "\n".join([f"🔗 {r['title']}: {r['href']}" for r in results])
            return f"{summary_text}\n\nSource links:\n{links}"
    except Exception as e:
        return f"Search error: {e}"

# ---------- User Profile Management ----------
def get_profile_file(user):
    return os.path.join(PROFILES_DIR, f"{user}.json")

def load_profile(user="akram"):
    file = get_profile_file(user)
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)
    return {"name": user.capitalize(), "memory": {}, "graph": {}, "reminders": [], "theme": "default"}

def save_profile(user, data):
    with open(get_profile_file(user), 'w') as f:
        json.dump(data, f, indent=2)

# ---------- Reminders ----------
def add_reminder(user, time_str, message):
    parsed = parse(time_str, settings={'PREFER_DATES_FROM': 'future'})
    if not parsed:
        return False, "Sorry, I couldn't understand the time."
    profile = load_profile(user)
    profile["reminders"].append({"time": parsed.isoformat(), "message": message})
    save_profile(user, profile)
    return True, f"Reminder set for {parsed.strftime('%I:%M %p on %b %d')}: {message}"

def check_reminders(user):
    profile = load_profile(user)
    now = datetime.now()
    due = []
    new_reminders = []
    for r in profile["reminders"]:
        dt = datetime.fromisoformat(r["time"])
        if dt <= now:
            due.append(r)
        else:
            new_reminders.append(r)
    profile["reminders"] = new_reminders
    save_profile(user, profile)
    return due

# ---------- Morning Briefing ----------
def morning_briefing(user):
    profile = load_profile(user)
    weather = get_weather("Chhapra") if os.getenv('WEATHER_API_KEY') else "Weather not available."
    due = check_reminders(user)
    reminder_text = ""
    if due:
        reminder_text = "🔔 Reminders:\n" + "\n".join([f"- {r['message']} (at {datetime.fromisoformat(r['time']).strftime('%I:%M %p')})" for r in due])
    else:
        reminder_text = "No reminders for today."
    import random
    quotes = [
        "The only way to do great work is to love what you do. – Steve Jobs",
        "Believe you can and you're halfway there. – Theodore Roosevelt",
        "Start where you are. Use what you have. Do what you can. – Arthur Ashe"
    ]
    quote = random.choice(quotes)
    return f"🌞 Good morning, {profile['name']}!\n\n{weather}\n\n{reminder_text}\n\n💡 {quote}\n\n🎵 How about listening to 'Jawan'? 🎵"

# ---------- HTML (Cinematic UI with Streaming) ----------
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Astra | Anti-Gravity HUD</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #00ff9d;
            --secondary: #ff00e5;
            --bg-gradient: radial-gradient(circle at 30% 40%, #0d0b1a, #000000);
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            min-height: 100vh;
            background: var(--bg-gradient);
            font-family: 'Poppins', sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }
        .stars {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }
        .star {
            position: absolute;
            background: #fff;
            border-radius: 50%;
            opacity: 0;
            animation: twinkle 3s infinite alternate;
        }
        @keyframes twinkle {
            0% { opacity: 0; transform: scale(0.5); }
            100% { opacity: 0.8; transform: scale(1); }
        }
        .container {
            width: 100%;
            max-width: 900px;
            background: rgba(15, 20, 30, 0.5);
            backdrop-filter: blur(12px);
            border-radius: 32px;
            border: 1px solid var(--primary);
            box-shadow: 0 25px 45px rgba(0,0,0,0.3), 0 0 20px rgba(0,255,157,0.2);
            z-index: 2;
        }
        .header {
            padding: 20px 30px;
            border-bottom: 1px solid rgba(0,255,157,0.2);
            text-align: center;
        }
        .header h1 {
            font-family: 'Orbitron', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .badge {
            display: inline-block;
            margin-top: 8px;
            background: rgba(0,255,157,0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.7rem;
            color: var(--primary);
            font-family: 'Orbitron', monospace;
        }
        .chat {
            height: 450px;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            scroll-behavior: smooth;
        }
        .chat::-webkit-scrollbar {
            width: 5px;
        }
        .chat::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        .chat::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 10px;
        }
        .msg {
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 20px;
            font-size: 0.95rem;
            line-height: 1.4;
            word-wrap: break-word;
            animation: fadeInUp 0.3s ease-out;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user {
            align-self: flex-end;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: #0a0f1a;
            border-bottom-right-radius: 4px;
        }
        .bot {
            align-self: flex-start;
            background: rgba(30, 35, 50, 0.8);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(0,255,157,0.3);
            color: #e0e0e0;
            border-bottom-left-radius: 4px;
        }
        .typing {
            display: flex;
            gap: 6px;
            align-items: center;
            padding: 12px 18px;
            background: rgba(30, 35, 50, 0.6);
            border-radius: 20px;
            width: fit-content;
        }
        .typing span {
            width: 8px;
            height: 8px;
            background: var(--primary);
            border-radius: 50%;
            animation: bounce 1.2s infinite;
        }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
            30% { transform: translateY(-8px); opacity: 1; }
        }
        .input-area {
            padding: 20px;
            border-top: 1px solid rgba(0,255,157,0.2);
            display: flex;
            gap: 12px;
        }
        .input-area input {
            flex: 1;
            background: rgba(10, 15, 26, 0.6);
            border: 1px solid rgba(0,255,157,0.4);
            border-radius: 40px;
            padding: 14px 20px;
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
            color: #fff;
            outline: none;
            transition: all 0.3s;
        }
        .input-area input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 12px rgba(0,255,157,0.4);
        }
        .input-area button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border: none;
            border-radius: 40px;
            padding: 0 24px;
            font-family: 'Orbitron', monospace;
            font-weight: 600;
            font-size: 0.9rem;
            color: #0a0f1a;
            cursor: pointer;
            transition: all 0.2s;
        }
        .input-area button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,255,157,0.5);
        }
        @media (max-width: 600px) {
            .msg { max-width: 90%; font-size: 0.85rem; }
            .header h1 { font-size: 1.4rem; }
            .input-area input, .input-area button { padding: 12px 16px; }
        }
    </style>
</head>
<body>
    <div class="stars" id="stars"></div>
    <div class="container">
        <div class="header">
            <h1>▲ ASTRA ANTI-GRAVITY</h1>
            <div class="badge">STREAMING | CACHED | INSTANT</div>
        </div>
        <div class="chat" id="chat"></div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Ask Astra..." autocomplete="off">
            <button onclick="startVoice()">🎤</button>
            <button onclick="send()">SEND</button>
        </div>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');

        function addMessage(role, text, isTyping = false) {
            const div = document.createElement('div');
            div.className = `msg ${role}`;
            if (isTyping) {
                div.innerHTML = `<div class="typing"><span></span><span></span><span></span></div>`;
            } else {
                div.innerHTML = text.replace(/\\n/g, '<br>');
            }
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        window.addEventListener('load', () => {
            const starsContainer = document.getElementById('stars');
            for (let i = 0; i < 150; i++) {
                const star = document.createElement('div');
                star.classList.add('star');
                const size = Math.random() * 3 + 1;
                star.style.width = size + 'px';
                star.style.height = size + 'px';
                star.style.left = Math.random() * 100 + '%';
                star.style.top = Math.random() * 100 + '%';
                star.style.animationDelay = Math.random() * 5 + 's';
                star.style.animationDuration = Math.random() * 3 + 2 + 's';
                starsContainer.appendChild(star);
            }
            addMessage('bot', '🖖 Asalamlekuim Akram! How can I help you today? 😊');
        });

        async function send() {
            const text = input.value.trim();
            if (!text) return;
            addMessage('user', text);
            input.value = '';
            
            const typingDiv = addMessage('bot', '', true);
            
            try {
                const response = await fetch('/ask-stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                
                typingDiv.remove();
                
                const botDiv = addMessage('bot', '', false);
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = '';
                
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    fullText += chunk;
                    botDiv.innerHTML = fullText.replace(/\\n/g, '<br>');
                    chat.scrollTop = chat.scrollHeight;
                }
                
                if (!fullText) {
                    botDiv.innerHTML = 'Sorry, no response.';
                }
            } catch (err) {
                if (typingDiv) typingDiv.remove();
                addMessage('bot', 'Network error. Please try again.');
            }
        }

        let recognition = null;
        function startVoice() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                addMessage('bot', 'Sorry, your browser does not support voice input.');
                return;
            }
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.lang = 'hi-IN';
            recognition.interimResults = false;
            recognition.onresult = (event) => {
                const text = event.results[0][0].transcript;
                input.value = text;
                send();
            };
            recognition.onerror = () => addMessage('bot', 'Voice recognition error.');
            recognition.start();
        }

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') send();
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
    user_input = data.get('message', '').strip()
    if not user_input:
        return Response("Kuch boliye.", mimetype='text/plain')
    
    def generate():
        user_lower = user_input.lower()
        
        # Weather command
        if any(w in user_lower for w in ['weather', 'temp', 'mausam', 'taapmaan']):
            city = "Chhapra"
            known_cities = ['delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'patna', 'chhapra', 'london', 'new york']
            for c in known_cities:
                if c in user_lower:
                    city = c
                    break
            yield get_weather(city)
            return
        
        # News command
        if 'news' in user_lower or 'khabar' in user_lower:
            query = None
            if user_lower.startswith('news '):
                query = user_input[5:].strip()
            elif 'news about ' in user_lower:
                query = user_lower.split('news about ')[-1].strip()
            yield get_news(query=query)
            return
        
        # Morning briefing
        if any(w in user_lower for w in ['good morning', 'morning', 'subah']):
            yield morning_briefing("akram")
            return

        # General AI with streaming
        for chunk in ask_nvidia_stream(user_input):
            yield chunk
    
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/telegram', methods=['GET', 'POST'])
def telegram_webhook():
    if request.method == 'GET':
        return "🤖 Astra Anti-Gravity Webhook is ACTIVE", 200
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
