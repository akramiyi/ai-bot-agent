import os
import re
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import yt_dlp
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

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
    user_input = data.get('message', '').strip().lower()
    if not user_input:
        return jsonify({'reply': 'Please say something.'})

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

    # General AI
    ai_reply = ask_nvidia(user_input)
    return jsonify({'reply': ai_reply})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
