import os
import re
import json
import requests
import urllib.parse
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import yt_dlp
from dotenv import load_dotenv
from dateparser import parse

load_dotenv()

app = Flask(__name__)

# ---------- Configuration ----------
PROFILES_DIR = 'profiles'       # folder for user profiles
REMINDER_CHECK_INTERVAL = 3600  # seconds (1 hour)
os.makedirs(PROFILES_DIR, exist_ok=True)

# ---------- NVIDIA AI Client ----------
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def ask_nvidia(prompt, system_message=None):
    if not system_message:
        system_message = "You are Astra, a helpful AI assistant for Akram from Chhapra, Bihar. Respond in Hinglish."
    try:
        response = client.chat.completions.create(
            model="meta/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI error: {str(e)}"

# ---------- Web Search (DuckDuckGo) ----------
def web_search(query):
    """Perform a DuckDuckGo search and return a summary + link."""
    try:
        # Use DuckDuckGo's free, no-key instant answer API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("AbstractText"):
            summary = data["AbstractText"]
            link = data.get("AbstractURL", "")
            return f"🔍 {summary}\n\nRead more: {link}" if link else f"🔍 {summary}"
        elif data.get("RelatedTopics"):
            # fallback to first related topic
            first = data["RelatedTopics"][0]
            if "Text" in first:
                return f"🔍 {first['Text']}"
        return f"Sorry, I couldn't find detailed info on '{query}'. You can search directly at https://duckduckgo.com/?q={urllib.parse.quote(query)}"
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
    # Default profile
    return {
        "name": user.capitalize(),
        "memory": {},
        "graph": {},
        "reminders": [],
        "theme": "default",
        "conversation": []
    }

def save_profile(user, data):
    with open(get_profile_file(user), 'w') as f:
        json.dump(data, f, indent=2)

# ---------- Reminder Helpers ----------
def add_reminder(user, time_str, message):
    parsed = parse(time_str, settings={'PREFER_DATES_FROM': 'future'})
    if not parsed:
        return False, "Sorry, I couldn't understand the time."
    profile = load_profile(user)
    profile["reminders"].append({
        "time": parsed.isoformat(),
        "message": message
    })
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

# ---------- Proactive Reminder Thread ----------
reminder_notifications = []  # list of (user, message)
def reminder_monitor():
    while True:
        time.sleep(REMINDER_CHECK_INTERVAL)
        # In a real multi-user scenario, we would scan all profiles.
        # For simplicity, we'll scan all .json files in profiles/
        for f in os.listdir(PROFILES_DIR):
            if f.endswith('.json'):
                user = f[:-5]
                due = check_reminders(user)
                for r in due:
                    reminder_notifications.append((user, r["message"]))
        # Also check the default profile if not already covered
        default_profile = os.path.join(PROFILES_DIR, "akram.json")
        if not os.path.exists(default_profile):
            due = check_reminders("akram")
            for r in due:
                reminder_notifications.append(("akram", r["message"]))

# Start background thread
threading.Thread(target=reminder_monitor, daemon=True).start()

# ---------- Knowledge Graph Functions (per user) ----------
def update_graph(user, text):
    profile = load_profile(user)
    graph = profile["graph"]
    text_lower = text.lower()
    if 'meri sister' in text_lower or 'meri bahan' in text_lower:
        parts = text.split()
        for i, word in enumerate(parts):
            if word in ['hai', 'hain'] and i+1 < len(parts):
                name = parts[i+1].strip(' .!,?')
                graph['Akram'] = graph.get('Akram', {})
                graph['Akram']['sister'] = name
                save_profile(user, profile)
                return f"✅ Yaad rakha: Akram ki sister {name} hain."
    if 'mera jija' in text_lower or 'mera brother in law' in text_lower:
        parts = text.split()
        for i, word in enumerate(parts):
            if word in ['hai', 'hain'] and i+1 < len(parts):
                name = parts[i+1].strip(' .!,?')
                graph['Akram'] = graph.get('Akram', {})
                graph['Akram']['jija'] = name
                save_profile(user, profile)
                return f"✅ Yaad rakha: Akram ke jija {name} hain."
    return None

def query_graph(user, query):
    profile = load_profile(user)
    graph = profile["graph"]
    q = query.lower()
    if 'sister' in q:
        sister = graph.get('Akram', {}).get('sister')
        if sister:
            return f"Aapki sister {sister} hain."
    if 'jija' in q or 'brother in law' in q:
        jija = graph.get('Akram', {}).get('jija')
        if jija:
            return f"Aapke jija {jija} hain."
    return None

# ---------- Image Generation (Pollinations) ----------
def generate_image(prompt):
    import urllib.parse
    encoded = urllib.parse.quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded}"

# ---------- YouTube ----------
def get_youtube_metadata(song_name):
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
            info = ydl.extract_info(f"ytsearch1:{song_name}", download=False)
            if 'entries' in info and info['entries']:
                v = info['entries'][0]
                return {
                    'url': f"https://www.youtube.com/watch?v={v['id']}",
                    'title': v['title'],
                    'thumbnail': f"https://img.youtube.com/vi/{v['id']}/0.jpg"
                }
    except:
        return None

playlist = []  # per‑session queue, could be moved to profile later

# ---------- Weather ----------
def get_weather(city):
    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key:
        return "Weather API key missing."
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        resp = requests.get(url)
        data = resp.json()
        if data.get('cod') != 200:
            return f"City '{city}' not found."
        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        return f"Weather in {city.title()}: {temp}°C, {desc}."
    except:
        return "Weather service error."

# ---------- Emotion Detection (keyword) ----------
def detect_emotion(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ['sad', 'depressed', 'unhappy', 'upset']):
        return 'sad'
    if any(w in text_lower for w in ['angry', 'frustrated', 'annoyed']):
        return 'angry'
    if any(w in text_lower for w in ['happy', 'joy', 'excited', 'great']):
        return 'happy'
    return 'neutral'

# ---------- Morning Briefing ----------
def morning_briefing(user):
    profile = load_profile(user)
    # Weather for default city (Chhapra)
    weather = get_weather("Chhapra") if os.getenv('WEATHER_API_KEY') else "Weather not available."
    # Reminders
    due = check_reminders(user)
    reminder_text = ""
    if due:
        reminder_text = "🔔 Reminders:\n" + "\n".join([f"- {r['message']} (at {datetime.fromisoformat(r['time']).strftime('%I:%M %p')})" for r in due])
    else:
        reminder_text = "No reminders for today."
    # Quote
    import random
    quotes = [
        "The only way to do great work is to love what you do. – Steve Jobs",
        "Believe you can and you're halfway there. – Theodore Roosevelt",
        "Start where you are. Use what you have. Do what you can. – Arthur Ashe"
    ]
    quote = random.choice(quotes)
    # Song suggestion
    song_suggestion = "How about listening to 'Jawan'? 🎵"
    return f"🌞 Good morning, {profile['name']}!\n\n{weather}\n\n{reminder_text}\n\n💡 {quote}\n\n🎵 {song_suggestion}"

# ---------- Dynamic Themes ----------
def load_theme(user):
    profile = load_profile(user)
    theme_name = profile.get("theme", "default")
    themes = {
        'cyberpunk': {
            'primary': '#00ff9d',
            'secondary': '#ff00e5',
            'bg_gradient': 'radial-gradient(circle at 30% 40%, #0d0b1a, #000000)'
        },
        'sunset': {
            'primary': '#ff6b6b',
            'secondary': '#ff8e53',
            'bg_gradient': 'radial-gradient(circle at 70% 20%, #1e3c72, #2a5298)'
        },
        'default': {
            'primary': '#e6b91e',
            'secondary': '#ffaa33',
            'bg_gradient': 'radial-gradient(circle at 20% 30%, #0a0f1a, #03060c)'
        }
    }
    return themes.get(theme_name, themes['default'])

def set_theme(user, theme_name):
    profile = load_profile(user)
    profile["theme"] = theme_name
    save_profile(user, profile)
    return load_theme(user)

# ---------- HTML (same as before, but we'll embed it) ----------
HTML = \"\"\"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Astra | Cinematic HUD</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #e6b91e;
            --secondary: #ffaa33;
            --bg-gradient: radial-gradient(circle at 20% 30%, #0a0f1a, #03060c);
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
            box-shadow: 0 25px 45px rgba(0,0,0,0.3), 0 0 20px rgba(230,185,30,0.2);
            z-index: 2;
        }
        .header {
            padding: 20px 30px;
            border-bottom: 1px solid rgba(230,185,30,0.2);
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
            background: rgba(230,185,30,0.2);
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
            box-shadow: 0 2px 8px rgba(230,185,30,0.3);
        }
        .bot {
            align-self: flex-start;
            background: rgba(30, 35, 50, 0.8);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(230,185,30,0.3);
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
            border-top: 1px solid rgba(230,185,30,0.2);
            display: flex;
            gap: 12px;
        }
        .input-area input {
            flex: 1;
            background: rgba(10, 15, 26, 0.6);
            border: 1px solid rgba(230,185,30,0.4);
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
            box-shadow: 0 0 12px rgba(230,185,30,0.4);
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
            box-shadow: 0 4px 12px rgba(230,185,30,0.5);
        }
        @media (max-width: 600px) {
            .msg { max-width: 90%; font-size: 0.85rem; }
            .header h1 { font-size: 1.4rem; }
            .input-area input, .input-area button { padding: 12px 16px; }
        }
    </style>
    <script>
        // Stars generation
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
        });
    </script>
</head>
<body>
    <div class="stars" id="stars"></div>
    <div class="container">
        <div class="header">
            <h1>▲ ASTRA LEVEL 7</h1>
            <div class="badge">CINEMATIC INTERFACE | NVIDIA CORE</div>
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
                div.innerHTML = text;
            }
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        window.addEventListener('load', () => {
            addMessage('bot', '🖖 Asalamlekuim Akram! How can I help you today? 😊');
        });

        let typingDiv = null;
        async function send() {
            const text = input.value.trim();
            if (!text) return;
            addMessage('user', text);
            input.value = '';
            typingDiv = addMessage('bot', '', true);
            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                const data = await res.json();
                const reply = data.reply || 'No response.';
                if (typingDiv) typingDiv.remove();
                addMessage('bot', reply);
                // Apply theme if returned
                if (data.theme) {
                    document.documentElement.style.setProperty('--primary', data.theme.primary);
                    document.documentElement.style.setProperty('--secondary', data.theme.secondary);
                    document.documentElement.style.setProperty('--bg-gradient', data.theme.bg_gradient);
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
\"\"\"

@app.route('/')
def index():
    # default user = akram; we can later add cookie based selection
    theme = load_theme("akram")
    return render_template_string(HTML, theme=theme)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_input = data.get('message', '').strip()
    if not user_input:
        return jsonify({'reply': 'Kuch boliye.'})

    # Detect user from query param or cookie? For now, default "akram"
    # In a real multi-user system, you would pass ?user=raushan in URL
    # For simplicity, we'll accept "switch user <name>" command
    global current_user
    if not hasattr(app, 'current_user'):
        app.current_user = "akram"
    if user_input.startswith('switch user '):
        new_user = user_input[12:].strip().lower()
        app.current_user = new_user
        # Create profile if not exists
        load_profile(new_user)  # ensures file exists
        return jsonify({'reply': f"Switched to profile: {new_user.capitalize()}"})

    user = app.current_user

    # Check for any pending proactive reminders for this user
    global reminder_notifications
    pending = [msg for (u, msg) in reminder_notifications if u == user]
    reminder_notifications = [(u, msg) for (u, msg) in reminder_notifications if u != user]
    reminder_msg = ""
    if pending:
        reminder_msg = "🔔 **Proactive Reminder:**\\n" + "\\n".join(pending) + "\\n\\n"

    # 1. Theme change command
    if user_input.startswith('set theme '):
        theme_name = user_input[10:].strip()
        theme = set_theme(user, theme_name)
        return jsonify({'reply': f"Theme changed to {theme_name}.", 'theme': theme})

    # 2. Morning briefing
    if user_input.lower() in ['good morning', 'morning', 'subah']:
        reply = morning_briefing(user)
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    # 3. Web search
    if user_input.startswith('search '):
        query = user_input[7:].strip()
        if not query:
            reply = "What would you like to search?"
        else:
            reply = web_search(query)
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    # 4. Knowledge Graph update/query
    graph_update = update_graph(user, user_input)
    if graph_update:
        return jsonify({'reply': reminder_msg + graph_update if reminder_msg else graph_update})
    graph_answer = query_graph(user, user_input)
    if graph_answer:
        return jsonify({'reply': reminder_msg + graph_answer if reminder_msg else graph_answer})

    # 5. Reminders
    if user_input.startswith('remind me '):
        text = user_input[10:].strip()
        match = re.search(r'(?:at|on)?\\s*(\\d{1,2}(?::\\d{2})?\\s*(?:am|pm))', text, re.IGNORECASE)
        if match:
            time_str = match.group(1)
            message = text.replace(match.group(0), '').strip()
            if message.startswith('to '):
                message = message[3:].strip()
            if not message:
                message = "reminder"
            success, result = add_reminder(user, time_str, message)
            reply = result if success else f"Error: {result}"
        else:
            reply = "Please use format like 'remind me at 5 PM to call mom'."
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    # 6. Weather
    if user_input.startswith('weather in ') or user_input.startswith('weather '):
        city = user_input.replace('weather in ', '').replace('weather ', '').strip()
        reply = get_weather(city)
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    # 7. Image generation
    if user_input.startswith('draw ') or user_input.startswith('generate '):
        prompt = user_input.replace('draw ', '').replace('generate ', '').strip()
        if not prompt:
            reply = "What should I draw?"
        else:
            img_url = generate_image(prompt)
            reply = f"🎨 Here's an image for '{prompt}':<br><img src='{img_url}' style='max-width:100%; border-radius:12px;'>"
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    # 8. YouTube commands
    if user_input.startswith('play song ') or user_input.startswith('play '):
        song = re.sub(r'^(play song |play )', '', user_input).strip()
        if not song:
            return jsonify({'reply': 'Which song?'})
        meta = get_youtube_metadata(song)
        if meta:
            reply = f'🎵 <b>{meta["title"]}</b><br><img src="{meta["thumbnail"]}" width="200"><br><a href="{meta["url"]}" target="_blank" style="background:#ff0000;color:white;padding:8px 16px;text-decoration:none;border-radius:8px;">▶ Play on YouTube</a>'
        else:
            reply = f'<a href="https://www.youtube.com/results?search_query={song.replace(" ", "+")}" target="_blank">🔍 Search YouTube for "{song}"</a>'
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    # 9. Queue (global, not per‑user for simplicity)
    global playlist
    if user_input.startswith('add to queue '):
        song = user_input.replace('add to queue ', '').strip()
        playlist.append(song)
        reply = f'✅ Added "{song}". {len(playlist)} in queue.'
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})
    elif user_input == 'show queue':
        if not playlist:
            reply = 'Queue empty.'
        else:
            reply = '📋 Queue:<br>' + '<br>'.join(f'{i+1}. {s}' for i,s in enumerate(playlist))
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})
    elif user_input == 'play next':
        if playlist:
            song = playlist.pop(0)
            meta = get_youtube_metadata(song)
            if meta:
                reply = f'▶ Now playing: <b>{meta["title"]}</b><br><a href="{meta["url"]}" target="_blank">Play</a>'
            else:
                reply = f'Now playing: {song} (link not available)'
        else:
            reply = 'Queue empty.'
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    # 10. Emotion + General AI
    emotion = detect_emotion(user_input)
    emotion_prefix = ""
    if emotion == 'sad':
        emotion_prefix = "I'm here for you. "
    elif emotion == 'angry':
        emotion_prefix = "Let's calm down. "
    elif emotion == 'happy':
        emotion_prefix = "That's great! "

    ai_reply = ask_nvidia(user_input)
    final_reply = emotion_prefix + (reminder_msg + ai_reply if reminder_msg else ai_reply)
    return jsonify({'reply': final_reply})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
