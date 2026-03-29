import os
import re
import json
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import yt_dlp
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
    """Create system prompt including all stored facts."""
    prompt = "You are Astra, a helpful AI assistant for Akram from Chhapra, Bihar. Respond in Hinglish.\n"
    if memory:
        prompt += "\nImportant facts you must remember:\n"
        for key, value in memory.items():
            prompt += f"- {key}: {value}\n"
    return prompt

# ---------- NVIDIA AI ----------
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def ask_nvidia(prompt, system_message=None):
    if not system_message:
        system_message = "You are Astra, a helpful AI assistant for Akram from Chhapra, Bihar. Respond in Hinglish."
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

# ---------- Web UI ----------
HTML = """
<!DOCTYPE html>
<html>
<head><title>Astra Level 6</title>
<style>
body { background: #0a0f1a; color: #e6b91e; font-family: monospace; padding: 20px; }
.container { max-width: 800px; margin: auto; }
.chat { background: #1a1f2e; border-radius: 12px; padding: 20px; height: 400px; overflow-y: auto; }
.msg { margin: 10px 0; padding: 8px 12px; border-radius: 8px; }
.user { background: #2a2f3e; text-align: right; }
.bot { background: #0f1420; border-left: 3px solid #e6b91e; }
input, button { background: #1a1f2e; border: 1px solid #e6b91e; color: #e6b91e; padding: 10px; border-radius: 8px; }
button { cursor: pointer; }
button:hover { background: #e6b91e; color: #0a0f1a; }
</style>
</head>
<body>
<div class="container">
<h1>🎙️ Astra Level 6</h1>
<div class="chat" id="chat"></div>
<div style="display: flex; gap: 10px; margin-top: 10px;">
<input type="text" id="input" placeholder="Ask Astra..." style="flex:1;">
<button onclick="send()">Send</button>
</div>
</div>
<script>
const chat = document.getElementById('chat');
function add(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = `<strong>${role === 'user' ? 'You' : 'Astra'}:</strong><br>${text}`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}
async function send() {
    const input = document.getElementById('input');
    const text = input.value.trim();
    if (!text) return;
    add('user', text);
    input.value = '';
    add('bot', '⌛ Thinking...');
    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text})
        });
        const data = await res.json();
        const last = chat.lastChild;
        chat.removeChild(last);
        add('bot', data.reply || 'Sorry, no response.');
    } catch(e) {
        chat.removeChild(chat.lastChild);
        add('bot', 'Network error.');
    }
}
document.getElementById('input').addEventListener('keypress', (e) => e.key === 'Enter' && send());
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/ask', methods=['POST'])
def ask():
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
    
    # 🥇 FINAL FIX: PROMPT ME MEMORY INJECT KARO
    cloud_memory = memory # We already fetched it above
    full_user_prompt = f"""
User Memory:
{cloud_memory}

User Question:
{user_input_raw}
"""
    # General AI
    ai_reply = ask_nvidia(full_user_prompt, system_prompt)
    return jsonify({'reply': ai_reply})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
