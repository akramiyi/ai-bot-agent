import os
import re
import json
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import yt_dlp
from dotenv import load_dotenv
from dateparser import parse

load_dotenv()

app = Flask(__name__)

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

# ---------- Image Generation ----------
def generate_image(prompt):
    """Generate an image using NVIDIA's free flux-schnell model."""
    try:
        response = client.images.generate(
            model="black-forest-labs/flux-schnell",
            prompt=prompt,
            n=1,
            size="512x512"
        )
        return response.data[0].url
    except Exception as e:
        return None

# ---------- Emotion Detection ----------
def detect_emotion(text):
    """Return 'happy', 'sad', 'angry', or 'neutral' based on simple keywords."""
    text_lower = text.lower()
    if any(w in text_lower for w in ['sad', 'depressed', 'unhappy', 'upset']):
        return 'sad'
    if any(w in text_lower for w in ['angry', 'frustrated', 'annoyed']):
        return 'angry'
    if any(w in text_lower for w in ['happy', 'joy', 'excited', 'great']):
        return 'happy'
    return 'neutral'

# ---------- Morning Briefing ----------
def morning_briefing():
    """Return a morning greeting with weather, reminders, quote, and song suggestion."""
    weather = get_weather("Chhapra") if os.getenv('WEATHER_API_KEY') else "Weather not available."
    due = check_reminders()
    reminder_text = ""
    if due:
        reminder_text = "🔔 Reminders:\n" + "\n".join([f"- {r['message']} (at {r['time'].strftime('%I:%M %p')})" for r in due])
    else:
        reminder_text = "No reminders for today."
    quote = get_motivational_quote()
    song_suggestion = "How about listening to 'Jawan'? 🎵"
    return f"🌞 Good morning, Akram!\n\n{weather}\n\n{reminder_text}\n\n💡 {quote}\n\n🎵 {song_suggestion}"

def get_motivational_quote():
    quotes = [
        "The only way to do great work is to love what you do. – Steve Jobs",
        "Believe you can and you're halfway there. – Theodore Roosevelt",
        "Start where you are. Use what you have. Do what you can. – Arthur Ashe"
    ]
    import random
    return random.choice(quotes)

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

# ---------- Reminders ----------
reminders = []

def add_reminder(time_str, message):
    parsed = parse(time_str, settings={'PREFER_DATES_FROM': 'future'})
    if not parsed:
        return False, "Sorry, I couldn't understand the time."
    reminders.append({'time': parsed, 'message': message})
    return True, f"Reminder set for {parsed.strftime('%I:%M %p on %b %d')}: {message}"

def check_reminders():
    now = datetime.now()
    due = [r for r in reminders if r['time'] <= now]
    reminders[:] = [r for r in reminders if r['time'] > now]
    return due

# ---------- Knowledge Graph ----------
GRAPH_FILE = 'graph.json'
def load_graph():
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_graph(graph):
    with open(GRAPH_FILE, 'w') as f:
        json.dump(graph, f, indent=2)

def update_graph(user_input, graph):
    """Extract simple relationships like 'X is Y' or 'X's Y is Z'."""
    text = user_input.lower()
    if 'meri sister' in text or 'meri bahan' in text:
        parts = text.split()
        for i, word in enumerate(parts):
            if word in ['hai', 'hain'] and i+1 < len(parts):
                name = parts[i+1].strip(' .!,?')
                graph['Akram'] = graph.get('Akram', {})
                graph['Akram']['sister'] = name
                return f"✅ Yaad rakha: Akram ki sister {name} hain."
    if 'mera jija' in text or 'mera brother in law' in text:
        parts = text.split()
        for i, word in enumerate(parts):
            if word in ['hai', 'hain'] and i+1 < len(parts):
                name = parts[i+1].strip(' .!,?')
                graph['Akram'] = graph.get('Akram', {})
                graph['Akram']['jija'] = name
                return f"✅ Yaad rakha: Akram ke jija {name} hain."
    return None

def query_graph(query, graph):
    """Answer based on stored relationships."""
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

playlist = []

# ---------- Dynamic UI Themes ----------
THEME_FILE = 'theme.json'
def load_theme():
    if os.path.exists(THEME_FILE):
        with open(THEME_FILE, 'r') as f:
            return json.load(f)
    return {'primary': '#e6b91e', 'secondary': '#ffaa33', 'bg_gradient': 'radial-gradient(circle at 20% 30%, #0a0f1a, #03060c)'}

def save_theme(theme):
    with open(THEME_FILE, 'w') as f:
        json.dump(theme, f, indent=2)

def apply_theme(theme_name):
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
    theme = themes.get(theme_name, themes['default'])
    save_theme(theme)
    return theme

# ---------- Frontend HTML ----------
HTML = """
<!DOCTYPE html>
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
            <h1>▲ ASTRA LEVEL 6</h1>
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
"""

@app.route('/')
def index():
    theme = load_theme()
    return render_template_string(HTML, theme=theme)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_input = data.get('message', '').strip()
    if not user_input:
        return jsonify({'reply': 'Kuch boliye.'})

    # 1. Theme change command
    if user_input.lower().startswith('set theme '):
        theme_name = user_input.lower().replace('set theme ', '').strip()
        theme = apply_theme(theme_name)
        return jsonify({'reply': f"Theme changed to {theme_name}.", 'theme': theme})

    # 2. Morning briefing
    if user_input.lower() in ['good morning', 'morning', 'subah']:
        reply = morning_briefing()
        return jsonify({'reply': reply})

    # 3. Knowledge Graph: Update or query
    graph = load_graph()
    update_msg = update_graph(user_input, graph)
    if update_msg:
        save_graph(graph)
        return jsonify({'reply': update_msg})
    graph_answer = query_graph(user_input, graph)
    if graph_answer:
        return jsonify({'reply': graph_answer})

    # 4. Check for due reminders
    due = check_reminders()
    reminder_msg = ""
    if due:
        reminder_msg = "🔔 <b>Reminders:</b><br>" + "<br>".join([f"- {r['message']} (at {r['time'].strftime('%I:%M %p')})" for r in due]) + "<br><br>"

    # 5. Special commands
    if user_input.lower().startswith('weather in ') or user_input.lower().startswith('weather '):
        city = user_input.lower().replace('weather in ', '').replace('weather ', '').strip()
        reply = get_weather(city)
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    elif user_input.lower().startswith('remind me '):
        text = user_input[10:].strip()
        match = re.search(r'(?:at|on)?\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm))', text, re.IGNORECASE)
        if match:
            time_str = match.group(1)
            message = text.replace(match.group(0), '').strip()
            if message.startswith('to '):
                message = message[3:].strip()
            if not message:
                message = "reminder"
            success, result = add_reminder(time_str, message)
            reply = result if success else f"Error: {result}"
        else:
            reply = "Please use format like 'remind me at 5 PM to call mom'."
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    elif user_input.lower().startswith('draw ') or user_input.lower().startswith('generate '):
        prompt = user_input.lower().replace('draw ', '').replace('generate ', '').strip()
        if not prompt:
            reply = "What should I draw?"
        else:
            img_url = generate_image(prompt)
            if img_url:
                reply = f"Here's an image for '{prompt}':<br><img src='{img_url}' style='max-width:100%; border-radius:12px;'>"
            else:
                reply = "Sorry, I couldn't generate that image. Please try again."
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    # YouTube commands
    if user_input.lower().startswith('play song ') or user_input.lower().startswith('play '):
        song = re.sub(r'^(play song |play )', '', user_input.lower()).strip()
        if not song:
            return jsonify({'reply': 'Which song?'})
        meta = get_youtube_metadata(song)
        if meta:
            reply = f'🎵 <b>{meta["title"]}</b><br><img src="{meta["thumbnail"]}" width="200"><br><a href="{meta["url"]}" target="_blank" style="background:#ff0000;color:white;padding:8px 16px;text-decoration:none;border-radius:8px;">▶ Play on YouTube</a>'
        else:
            reply = f'<a href="https://www.youtube.com/results?search_query={song.replace(" ", "+")}" target="_blank">🔍 Search YouTube for "{song}"</a>'
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    elif user_input.lower().startswith('add to queue '):
        song = user_input.lower().replace('add to queue ', '').strip()
        playlist.append(song)
        reply = f'✅ Added "{song}". {len(playlist)} in queue.'
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    elif user_input.lower() == 'show queue':
        if not playlist:
            reply = 'Queue empty.'
        else:
            reply = '📋 Queue:<br>' + '<br>'.join(f'{i+1}. {s}' for i,s in enumerate(playlist))
        return jsonify({'reply': reminder_msg + reply if reminder_msg else reply})

    elif user_input.lower() == 'play next':
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

    # 6. Emotion detection
    emotion = detect_emotion(user_input)
    emotion_prefix = ""
    if emotion == 'sad':
        emotion_prefix = "I'm here for you. "
    elif emotion == 'angry':
        emotion_prefix = "Let's calm down. "
    elif emotion == 'happy':
        emotion_prefix = "That's great! "

    # 7. General AI
    ai_reply = ask_nvidia(user_input)
    final_reply = emotion_prefix + (reminder_msg + ai_reply if reminder_msg else ai_reply)
    return jsonify({'reply': final_reply})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
