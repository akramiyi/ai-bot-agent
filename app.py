import os
import re
import json
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
# import yt_dlp moved to function for memory optimization

from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

load_dotenv()
app = Flask(__name__)

# --- FIREBASE CLOUD MEMORY SETUP ---
FB_URL = os.getenv("FIREBASE_DB_URL") or 'https://astra-ai-2cc5a-default-rtdb.asia-southeast1.firebasedatabase.app'
try:
    if not firebase_admin._apps:
        if os.getenv("FIREBASE_CREDENTIALS"):
            firebase_data = json.loads(os.getenv("FIREBASE_CREDENTIALS"))
            cred = credentials.Certificate(firebase_data)
            firebase_admin.initialize_app(cred, {'databaseURL': FB_URL})
        elif os.path.exists("firebase.json"):
            cred = credentials.Certificate("firebase.json")
            firebase_admin.initialize_app(cred, {'databaseURL': FB_URL})
except Exception as e:
    print(f"Firebase Init Error: {e}")

MEMORY_FILE = 'memory.json'

# ---------- Memory Functions ----------
def load_memory():
    """Load memory from JSON file, return dict."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_memory(memory):
    """Save memory dict to JSON file."""
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)

def update_memory_from_text(text, memory):
    """Extract facts from user message and update memory."""
    text_lower = text.lower()
    # Example rules
    if 'meri sister' in text_lower or 'meri bahan' in text_lower:
        parts = text.split()
        for i, word in enumerate(parts):
            if word.lower() in ['hai', 'hain'] and i+1 < len(parts):
                name = " ".join(parts[i+1:]).strip(' .!,?')
                memory['sister'] = name
                return f"✅ Yaad rakha ki aapki sister {name} hain."
    
    if 'mera jija' in text_lower or 'mera brother in law' in text_lower:
        parts = text.split()
        for i, word in enumerate(parts):
            if word.lower() in ['hai', 'hain'] and i+1 < len(parts):
                name = " ".join(parts[i+1:]).strip(' .!,?')
                memory['jija'] = name
                return f"✅ Yaad rakha ki aapke jija {name} hain."
    
    if 'my name is' in text_lower or 'mera naam' in text_lower:
        parts = text.split()
        for i, word in enumerate(parts):
            if word.lower() in ['is', 'hai'] and i+1 < len(parts):
                name = " ".join(parts[i+1:]).strip(' .!,?')
                memory['user_name'] = name
                return f"✅ Yaad rakha ki aapka naam {name} hai."
    
    # Generic "yaad rakho" command
    if text_lower.startswith('yaad rakho '):
        fact = text[11:].strip()
        if '=' in fact:
            key, val = fact.split('=', 1)
            memory[key.strip()] = val.strip()
        else:
            memory['last_fact'] = fact
        return f"✅ Maine yaad rakh liya: {fact}"
    return None

def build_system_prompt(memory):
    """Create system prompt including all stored facts and personal profile."""
    prompt = """
You are Astra, an advanced AI assistant created for Akram.
Reply short, smart, and helpful. Maximum 2 lines.

User Profile:
- Name: Akram Ansari | Role: Aspiring Software Engineer 
- Location: Chhapra, Bihar, India
- Phone: +91 6204110766 | Email: meakramiyi@gmail.com
- LinkedIn: linkedin.com/in/akram-alii
- Education: B.Tech in CS (2024–2028), Brainware University

Family & Friends:
- Father: Ajmat Ali | Mother: Maimun Nisha
- Siblings: Raushan Khatoon (Sister), Ekram Ali (Brother)
- Friends: Rosidul Islam (Best Friend), Munshi Insiyat (Karate), Arjit Ghost (Rich), Aryan Raj (Editor), Kaif Ali, Nayan, Shahid, Wasim, Kunaal, Asif.

Always remember you are talking to Akram. Use Hindi-English mix (Hinglish). 
When the user says hello or greets you, always greet back with: 'Asalamlekuim Akram! How can I help you today?'
"""
    if memory:
        prompt += "\nUser Memory/Facts:\n"
        if isinstance(memory, dict):
            for key, value in memory.items():
                prompt += f"- {key}: {value}\n"
        else:
            prompt += f"- {memory}\n"
    return prompt

# ---------- NVIDIA AI ----------
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def ask_nvidia(prompt, system_message=None):
    if not system_message:
        system_message = (
            "You are Astra, a helpful AI assistant for Akram from Chhapra, Bihar. "
            "When the user says hello, greet with 'Asalamlekuim Akram! How can I help you today?' "
            "Respond in Hinglish."
        )
    try:
        response = client.chat.completions.create(
            model="minimaxai/minimax-m2.5",
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

# ---------- YouTube Pro ----------
def get_youtube_metadata(song_name):
    import yt_dlp
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
MAX_HISTORY = 20
conversation_history = []


# ---------- Web UI ----------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Astra | Cinematic HUD</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            min-height: 100vh;
            background: radial-gradient(circle at 20% 30%, #0a0f1a, #03060c);
            font-family: 'Poppins', sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            position: relative;
            overflow-x: hidden;
        }

        /* Animated stars background */
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

        /* Main container */
        .container {
            width: 100%;
            max-width: 900px;
            background: rgba(15, 20, 30, 0.5);
            backdrop-filter: blur(12px);
            border-radius: 32px;
            border: 1px solid rgba(230, 185, 30, 0.3);
            box-shadow: 0 25px 45px rgba(0,0,0,0.3), 0 0 20px rgba(230,185,30,0.2);
            z-index: 2;
            transition: all 0.3s ease;
        }

        /* Header */
        .header {
            padding: 20px 30px;
            border-bottom: 1px solid rgba(230,185,30,0.2);
            text-align: center;
        }
        .header h1 {
            font-family: 'Orbitron', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #e6b91e, #ffaa33);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            letter-spacing: 2px;
            text-shadow: 0 0 5px rgba(230,185,30,0.3);
        }
        .badge {
            display: inline-block;
            margin-top: 8px;
            background: rgba(230,185,30,0.2);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 500;
            color: #e6b91e;
            font-family: 'Orbitron', monospace;
            backdrop-filter: blur(4px);
        }

        /* Chat area */
        .chat {
            height: 450px;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            scroll-behavior: smooth;
        }
        /* Custom scrollbar */
        .chat::-webkit-scrollbar {
            width: 5px;
        }
        .chat::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
        }
        .chat::-webkit-scrollbar-thumb {
            background: #e6b91e;
            border-radius: 10px;
        }

        /* Message bubbles */
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
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .user {
            align-self: flex-end;
            background: linear-gradient(135deg, #e6b91e, #ffaa33);
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

        /* Typing indicator */
        .typing {
            display: flex;
            gap: 6px;
            align-items: center;
            padding: 12px 18px;
            background: rgba(30, 35, 50, 0.6);
            border-radius: 20px;
            width: fit-content;
            backdrop-filter: blur(4px);
        }
        .typing span {
            width: 8px;
            height: 8px;
            background: #e6b91e;
            border-radius: 50%;
            animation: bounce 1.2s infinite;
        }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
            30% { transform: translateY(-8px); opacity: 1; }
        }

        /* Input area */
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
            border-color: #e6b91e;
            box-shadow: 0 0 12px rgba(230,185,30,0.4);
            background: rgba(10, 15, 26, 0.8);
        }
        .input-area button {
            background: linear-gradient(135deg, #e6b91e, #ffaa33);
            border: none;
            border-radius: 40px;
            padding: 0 24px;
            font-family: 'Orbitron', monospace;
            font-weight: 600;
            font-size: 0.9rem;
            color: #0a0f1a;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 2px 8px rgba(230,185,30,0.3);
        }
        .input-area button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(230,185,30,0.5);
        }

        /* Responsive */
        @media (max-width: 600px) {
            .container {
                border-radius: 24px;
            }
            .msg {
                max-width: 90%;
                font-size: 0.85rem;
            }
            .header h1 {
                font-size: 1.4rem;
            }
            .input-area input, .input-area button {
                padding: 12px 16px;
            }
        }
    </style>
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
            <button onclick="send()">SEND</button>
        </div>
    </div>

    <script>
        // Generate stars background
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

        async function send() {
            const text = input.value.trim();
            if (!text) return;
            addMessage('user', text);
            input.value = '';
            
            let retries = 1;
            let success = false;
            let reply = null;
            
            while (retries >= 0 && !success) {
                const typingDiv = addMessage('bot', '', true);
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: text})
                    });
                    const data = await res.json();
                    reply = data.reply || 'No response.';
                    success = true;
                    typingDiv.remove();
                    addMessage('bot', reply);
                } catch (err) {
                    typingDiv.remove();
                    if (retries > 0) {
                        // Show retrying message
                        addMessage('bot', '⏳ Retrying...');
                        retries--;
                        await new Promise(r => setTimeout(r, 2000)); // wait 2 sec
                    } else {
                        addMessage('bot', 'Network error. Please try again.');
                    }
                }
            }
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
    return render_template_string(HTML)

@app.route('/ask', methods=['POST'])
def ask():
    global conversation_history
    data = request.get_json()
    user_input_raw = data.get('message', '').strip()
    user_input = user_input_raw.lower()
    if not user_input:
        return jsonify({'reply': 'Kuch boliye.'})

    # Memory handling (Cloud First)
    try:
        memory = db.reference("memory").get() or {}
    except:
        memory = load_memory() # Fallback to local JSON

    # Special command: kya yaad hai
    if user_input == 'kya yaad hai':
        if not memory:
            return jsonify({'reply': "Meri memory abhi khali hai. Aap kuch batao, main yaad rakhunga."})
        else:
            # If it's a dict, format it. If it's already a string, show it.
            if isinstance(memory, dict):
                reply = "Mujhe yeh yaad hai:<br>" + "<br>".join([f"- {k}: {v}" for k,v in memory.items()])
            else:
                reply = f"Mujhe yeh yaad hai: {memory}"
            return jsonify({'reply': reply})

    # Check if user wants to teach something
    update_msg = update_memory_from_text(user_input_raw, memory)
    if update_msg:
        save_memory(memory)
        return jsonify({'reply': update_msg})

    # YouTube commands
    if user_input.startswith('play song ') or user_input.startswith('play '):
        song = re.sub(r'^(play song |play )', '', user_input).strip()
        if not song:
            return jsonify({'reply': 'Which song?'})
        meta = get_youtube_metadata(song)
        if meta:
            reply = f'🎵 <b>{meta["title"]}</b><br><img src="{meta["thumbnail"]}" width="200"><br><a href="{meta["url"]}" target="_blank" style="background:#ff0000;color:white;padding:8px 16px;text-decoration:none;border-radius:8px;">▶ Play on YouTube</a>'
        else:
            reply = f'<a href="https://www.youtube.com/results?search_query={song.replace(" ", "+")}" target="_blank">🔍 Search YouTube for "{song}"</a>'
        return jsonify({'reply': reply})

    elif user_input.startswith('add to queue '):
        song = user_input.replace('add to queue ', '').strip()
        playlist.append(song)
        return jsonify({'reply': f'✅ Added "{song}". {len(playlist)} in queue.'})

    elif user_input == 'show queue':
        if not playlist:
            return jsonify({'reply': 'Queue empty.'})
        return jsonify({'reply': '📋 Queue:<br>' + '<br>'.join(f'{i+1}. {s}' for i,s in enumerate(playlist))})

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
        return jsonify({'reply': reply})

    # Build system prompt with memory (Injecting into user prompt as per "Final Fix")
    system_prompt = build_system_prompt(memory)
    
    # 🥇 FINAL FIX: PROMPT ME MEMORY + HISTORY INJECT KARO
    cloud_memory = memory # We already fetched it above
    
    # Format global history for context
    history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in conversation_history])
    
    full_user_prompt = f"""
Chat History:
{history_str}

User Memory (Facts):
{cloud_memory}

User Question:
{user_input_raw}
"""
    # General AI
    ai_reply = ask_nvidia(full_user_prompt, system_prompt)
    
    # Add to rolling history
    conversation_history.append({"role": "user", "content": user_input_raw})
    conversation_history.append({"role": "assistant", "content": ai_reply})
    
    # Trim history
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]
        
    return jsonify({'reply': ai_reply})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
