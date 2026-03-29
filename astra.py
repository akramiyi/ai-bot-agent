import datetime
import json
import os
import pytz
import time
import webbrowser
import requests
import xml.etree.ElementTree as ET
import warnings
import threading
warnings.filterwarnings("ignore")

from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, db
import urllib.parse
import yt_dlp
from bs4 import BeautifulSoup

# Optional Imports (GUI/System/Voice)
try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import pywhatkit
except ImportError:
    pywhatkit = None

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import ctypes
except ImportError:
    ctypes = None

try:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
    import pygame
except ImportError:
    pygame = None

# Global Flags
FACE_RECOGNITION_AVAILABLE = False

# -------- FIREBASE CLOUD MEMORY --------
FB_URL = os.getenv("FIREBASE_DB_URL") or 'https://astra-ai-2cc5a-default-rtdb.asia-southeast1.firebasedatabase.app'

try:
    # 1. Check for Environment Variable (JSON String)
    if os.getenv("FIREBASE_CREDENTIALS"):
        firebase_data = json.loads(os.getenv("FIREBASE_CREDENTIALS"))
        cred = credentials.Certificate(firebase_data)
        firebase_admin.initialize_app(cred, {'databaseURL': FB_URL})
        print("✅ Firebase Connected (Env Variable)!")

    # 2. Check for Secret File (Common in Render)
    elif os.path.exists("FIREBASE_CREDENTIALS"):
        cred = credentials.Certificate("FIREBASE_CREDENTIALS")
        firebase_admin.initialize_app(cred, {'databaseURL': FB_URL})
        print("✅ Firebase Connected (Secret File)!")

    # 3. Check for Secret File in /etc/secrets/ (Alternative Render path)
    elif os.path.exists("/etc/secrets/FIREBASE_CREDENTIALS"):
        cred = credentials.Certificate("/etc/secrets/FIREBASE_CREDENTIALS")
        firebase_admin.initialize_app(cred, {'databaseURL': FB_URL})
        print("✅ Firebase Connected (/etc/secrets/)!")

    # 4. Local Fallback
    elif os.path.exists("firebase.json"):
        cred = credentials.Certificate("firebase.json")
        firebase_admin.initialize_app(cred, {'databaseURL': FB_URL})
        print("✅ Firebase Connected (Local file)!")

    else:
        print("ℹ️ Firebase credentials not found. Cloud sync disabled.")
    
    # Initialize app with database URL if cred exists (User Fix)
    if 'cred' in locals():
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(cred, {'databaseURL': FB_URL})
            print(f"🚀 Firebase Fully Initialized with {FB_URL}")
except Exception as e:
    print(f"⚠️ Firebase Setup Info: {e}")
    print("💡 Proceeding with local memory fallback.")

# -------- LOGIN SYSTEM --------
USERS = {
    "akram": {"password": "1619"}
}

def login(username, password):
    user = USERS.get(username)
    if user and user["password"] == password:
        return True
    return False

# -------- EXTERNAL UI CALLBACK --------
ui_callback = None
ui_label = None # Global for old Tkinter compatibility

def set_ui_callback(func):
    global ui_callback
    ui_callback = func

def update_ui(text):
    if ui_callback:
        ui_callback(text)
    # Old Tkinter backward compatibility
    global ui_label
    if ui_label:
        try:
            ui_label.config(text=text)
        except:
            pass

# 🔥 NVIDIA AI ENGINE (NVIDIA Integrate)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)
# Using Minimax M2.5 (Free & Fast for 2026)
MODEL_NAME = "minimaxai/minimax-m2.5"

# -------- SPEAK --------
# Use gTTS (original Hindi voice) + pygame for crash-free playback
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
def speak(text):
    print(f"🔊 Astra Speaking: {text}")
    update_ui(f"Speaking: {text}")
    
    if gTTS is None or pygame is None:
        print("ℹ️ Voice output (speaker) is not available on this system.")
        return

    try:
        tts = gTTS(text=text, lang='hi')
        tts.save("voice.mp3")
        pygame.mixer.music.load("voice.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        if os.path.exists("voice.mp3"):
            os.remove("voice.mp3")
    except Exception as e:
        print(f"❌ Speech error: {e}")
        update_ui(f"Error: {e}")

# -------- NORMALIZE --------
def normalize_command(command):
    replacements = {
        "ओपन युटुब": "open youtube",
        "युटुब": "youtube",
        "खोलो": "open",
        "चलाo": "play",
        "चलाओ": "play",
        "गाना": "song",
        "समय": "time",
        "टाइम": "time",
        "मौसम": "weather",
        "खबर": "news"
    }

    for hindi, eng in replacements.items():
        command = command.replace(hindi, eng)

    return command

# -------- PRO YOUTUBE SYSTEM --------
def get_youtube_info(song_name):
    try:
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': False,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search for the video
            search_query = f"ytsearch1:{song_name}"
            info = ydl.extract_info(search_query, download=False)
            if 'entries' in info and len(info['entries']) > 0:
                video = info['entries'][0]
                video_id = video['id']
                return {
                    "title": video.get('title', song_name),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "thumb": f"https://img.youtube.com/vi/{video_id}/0.jpg",
                    "id": video_id
                }
    except Exception as e:
        print(f"yt-dlp error: {e}")
    
    # Fallback to search link
    query = urllib.parse.quote(song_name)
    return {
        "title": song_name,
        "url": f"https://www.youtube.com/results?search_query={query}",
        "thumb": "https://www.youtube.com/s/desktop/28169112/img/favicon_144x144.png",
        "id": None
    }

def get_suggestions(query):
    try:
        url = f"https://suggestqueries.google.com/complete/search?client=firefox&q={urllib.parse.quote(query)}"
        resp = requests.get(url, timeout=5).json()
        return resp[1][:5] if len(resp) > 1 else []
    except:
        return []

# -------- PLAYLIST QUEUE --------
playlist_queue = []
current_queue_index = 0

# -------- AI EXPERT (NVIDIA) --------
def ai_chat(prompt):
    try:
        system_prompt = f"""
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

Instruction:
Always remember you are talking to Akram. Use Hindi-English mix (Hinglish) if natural.
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ NVIDIA Error: {e}")
        return "AI system currently busy. ⚠️"

# -------- LISTEN --------
def listen():
    if sr is None:
        print("🎙️ Mic system is not available on this system.")
        return ""

    update_ui("Listening... 🎙️")
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎙️ Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=0.8)
            print("🎧 Ready! Speak now...")
            audio = r.listen(source, phrase_time_limit=10) # 10s for password

        command = r.recognize_google(audio, language="en-IN").lower()
        command = normalize_command(command)
        print("You:", command)
        return command
    except Exception as e:
        print(f"Mic error: {e}")
        return ""

# -------- MEMORY FILE --------
MEMORY_FILE = "memory.json"

def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

# --- NEW MEMORY SYSTEM (User Fix) ---
def save_memory(key, value):
    """Save a specific key-value pair to Firebase."""
    try:
        ref = db.reference("memory")
        ref.update({key: value})
        print(f"💾 Saved to Cloud: {key} = {value}")
    except Exception as e:
        print(f"❌ Cloud Save Error: {e}")

def get_memory(key):
    """Retrieve a specific key from Firebase."""
    try:
        ref = db.reference("memory")
        data = ref.get()
        return data.get(key) if data else None
    except Exception as e:
        print(f"❌ Cloud Load Error: {e}")
        return None

def save_full_memory(memory, username="akram"):
    """Existing full memory save (local + cloud sync)."""
    # Save Local
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)
    
    # Save Cloud (Firebase)
    def firebase_sync():
        try:
            ref = db.reference(f"users/{username}/history")
            ref.push(memory["history"][-1]) # Push latest command
            
            # Sync full memory if needed
            ref_all = db.reference(f"users/{username}/memory")
            ref_all.set(memory)
        except:
            pass

    # THREADING for memory boost
    threading.Thread(target=firebase_sync, daemon=True).start()

def load_cloud_memory():
    try:
        ref = db.reference("astra/memory")
        data = ref.get()
        if data:
            return data
    except:
        pass
    return None

memory = load_cloud_memory() or load_memory()

# -------- FACE RECOGNITION LOGIN --------
def face_login():
    if FACE_RECOGNITION_AVAILABLE:
        try:
            speak("Face scan ho raha hai... 👁️")
            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            cam.release()
            if ret and frame is not None:
                rgb = frame[:, :, ::-1]
                faces = face_recognition.face_encodings(rgb)
                if faces:
                    speak("Welcome Akram 😎")
                    return True
                else:
                    speak("Face not recognized, password try karo")
            else:
                speak("Camera not working, password login")
        except Exception as e:
            print("Camera/Face Login error:", e)
            speak("Face login failed, switching to password")
    
    # Fallback: Password login (3 attempts)
    valid_passwords = [
        "1619", "16 19", "one six one nine",
        "skip", "akram", "akram ansari",
        "start", "open", "hello",
        "password", "login", "yes",
        "access", "enter", "okay",
        "1619", "sixteen nineteen"
    ]
    
    for attempt in range(3):
        speak(f"Password bolo. Attempt {attempt + 1} of 3")
        command = listen()
        
        # Check if ANY valid password word is in the command
        if command != "" and any(pwd in command for pwd in valid_passwords):
            speak("Welcome Akram 😎")
            return True
        else:
            if attempt < 2:
                speak("Galat password, dobara bolo")
            else:
                speak("Access denied ❌")
                return False
    
    return False

# -------- SELF-LEARNING MEMORY --------
def learn_from_user(command):
    memory["history"] = memory.get("history", [])
    memory["history"].append(command)
    save_full_memory(memory)

# -------- MULTI-AGENT SYSTEM --------
def study_agent(command):
    if "study" in command:
        speak("Study mode activated 📚")

def system_agent(command):
    if "open" in command:
        speak("System control active 💻")

# -------- LAST COMMAND TRACKER --------
last_command = ""

# -------- STUDY PLAN --------
study_plan = {
    "monday": "Arrays",
    "tuesday": "Strings",
    "wednesday": "Linked List",
    "thursday": "Stack & Queue",
    "friday": "Tree",
    "saturday": "Graph",
    "sunday": "Revision"
}

# -------- SMART COMMAND ROUTER --------
def process_command(command):
    global last_command

    command = command.lower()

    # Wake word remove (all synonyms)
    for word in ["hey", "astra", "jarvis", "he", "stra"]:
        command = command.replace(word, "").strip()
    
    if command == "":
        speak("Yes, tell me your command")
        return

    if "repeat" not in command:
        last_command = command

    # -------- EXECUTING AGENTS AND HISTORY --------
    learn_from_user(command)
    study_agent(command)
    system_agent(command)

    # --- MEMORY COMMANDS (User Fix) ---
    if "mera bhai ka naam" in command:
        name = command.split("naam")[-1].strip()
        save_memory("bhai", name)
        speak("Theek hai, maine yaad rakh liya 👍")
        return

    elif "who is ekram" in command:
        bhai = get_memory("bhai")
        if bhai:
            speak(f"{bhai} aapka chhota bhai hai 😎")
        else:
            speak("Mujhe abhi tak nahi pata ki Ekram kaun hai.")
        return

    # -------- TIME --------
    if "time" in command:

        if "usa" in command or "america" in command:
            tz = pytz.timezone("America/New_York")
            country = "USA 🇺🇸"
        elif "london" in command or "uk" in command:
            tz = pytz.timezone("Europe/London")
            country = "UK 🇬🇧"
        elif "dubai" in command:
            tz = pytz.timezone("Asia/Dubai")
            country = "Dubai 🇦🇪"
        elif "saudi" in command or "dammam" in command:
            tz = pytz.timezone("Asia/Riyadh")
            country = "Saudi Arabia 🇸🇦"
        elif "japan" in command:
            tz = pytz.timezone("Asia/Tokyo")
            country = "Japan 🇯🇵"
        else:
            tz = pytz.timezone("Asia/Kolkata")
            country = "India 🇮🇳"

        current_time = datetime.datetime.now(tz).strftime("%I:%M %p")
        speak(f"Current time in {country} is {current_time} ⏰")

    # -------- PLAY MUSIC --------
    elif "play" in command:
        song = command.replace("play", "").replace("song", "").strip()
        if song == "":
            return "Which song would you like to play? 🎵"
        
        info = get_youtube_info(song)
        
        # PRO LEVEL HTML Response
        card = f'''
<div style="background: rgba(255, 0, 0, 0.1); border: 2px solid #ff0000; padding: 15px; border-radius: 15px; text-align: center; margin-top: 10px;">
    <h3 style="margin: 0 0 10px 0; color: #fff;">🎵 Now Playing</h3>
    <img src="{info['thumb']}" style="width: 100%; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 0 15px rgba(255,0,0,0.5);">
    <p style="color: #00d4ff; font-weight: bold;">{info['title']}</p>
    <a href="{info['url']}" target="_blank" style="display: inline-block; background: #ff0000; color: white; padding: 10px 25px; text-decoration: none; border-radius: 25px; font-weight: bold; font-size: 16px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); transition: 0.3s;">▶ Play on YouTube</a>
</div>
'''
        if IS_SERVER:
            return card
        else:
            try:
                if pywhatkit:
                    pywhatkit.playonyt(song)
                else:
                    webbrowser.open(info['url'])
                return f"Playing {info['title']} 🎵"
            except:
                return f"Auto-play failed. Click here: {info['url']}"

    elif "add to queue" in command or "queue me add" in command:
        song = command.replace("add to queue", "").replace("queue me add", "").strip()
        if not song: return "Song name bolo?"
        info = get_youtube_info(song)
        playlist_queue.append(info)
        return f"✅ '{info['title']}' added to queue. {len(playlist_queue)} songs in playlist."

    elif "show queue" in command or "list dikhao" in command:
        if not playlist_queue: return "Queue khali hai 📋"
        res = "<b>📋 Queue List:</b><br>"
        for i, s in enumerate(playlist_queue):
            res += f"{i+1}. {s['title']}<br>"
        return res

    elif "play next" in command or "agla gaana" in command:
        global current_queue_index
        if current_queue_index < len(playlist_queue) - 1:
            current_queue_index += 1
            info = playlist_queue[current_queue_index]
            return f"Playing next: {info['title']} 🎵<br><a href='{info['url']}' target='_blank'>▶ Play</a>"
        return "Queue khatam ho gayi! 📋"

    elif "suggest" in command or "did you mean" in command:
        query = command.replace("suggest", "").replace("did you mean", "").strip()
        suggestions = get_suggestions(query)
        if suggestions:
            res = "🔍 <b>Did you mean?</b><br>"
            for s in suggestions:
                res += f"• {s}<br>"
            return res
        return "No suggestions found."

    # -------- SAVE NOTES --------
    elif "yaad rakho" in command:
        data = command.replace("yaad rakho", "").strip()
        memory["notes"] = memory.get("notes", [])
        memory["notes"].append(data)
        save_full_memory(memory)
        speak("Maine yaad rakh liya 🧠")

    # -------- RECALL NOTES --------
    elif "kya yaad hai" in command:
        notes = memory.get("notes", [])
        if notes:
            for n in notes:
                speak(n)
        else:
            speak("Kuch yaad nahi hai")

    # -------- SAVE NAME --------
    elif "my name is" in command:
        name = command.replace("my name is", "").strip()
        memory["name"] = name
        save_full_memory(memory)
        speak(f"Got it, {name}. I'll remember that.")

    # -------- GET NAME --------
    elif "name" in command and ("my" in command or "me" in command):
        if "name" in memory:
            speak(f"Your name is {memory['name']} 😎")
        else:
            speak("I don't know your name yet")

    # -------- SAVE PREFERENCE --------
    elif "i like" in command:
        pref = command.replace("i like", "").strip()
        memory["preference"] = pref
        save_full_memory(memory)
        speak(f"Saved that you like {pref} 👍")

    # -------- GET PREFERENCE --------
    elif "what do i like" in command:
        if "preference" in memory:
            speak(f"You like {memory['preference']} 🎯")
        else:
            speak("No preference saved yet")

    # -------- SAVE FRIEND --------
    elif "mera dost ka naam" in command:
        name = command.replace("mera dost ka naam", "").strip()

        if name == "":
            speak("Naam batao kiska dost hai")
        else:
            if "friends" not in memory:
                memory["friends"] = []

            memory["friends"].append(name)
            save_full_memory(memory)

            speak(f"{name} ko yaad rakh liya 👍")

    # -------- SAVE CONTACT --------
    elif "number save" in command:
        parts = command.replace("number save", "").strip().split()

        if len(parts) >= 2:
            name = parts[0]
            number = parts[1]

            if "contacts" not in memory:
                memory["contacts"] = {}

            memory["contacts"][name] = number
            save_full_memory(memory)

            speak(f"{name} ka number save ho gaya 📱")
        else:
            speak("Naam aur number dono bolo")

    # -------- WHATSAPP SEND --------
    elif "send" in command:

        contacts = {
            "rashid": "+916296579165"
        }

        for name in contacts:
            if name in command:
                number = contacts[name]

                msg = command.replace("send", "").replace(name, "").strip()

                speak(f"{name} ko message bhej raha hoon 💬")

                try:
                    if pywhatkit and pyautogui:
                        # Step 1: Open WhatsApp + type message
                        pywhatkit.sendwhatmsg_instantly(number, msg, wait_time=15, tab_close=False)

                        # Step 2: EXTRA WAIT (very important 🔥)
                        time.sleep(10)

                        # Step 3: Make sure window active
                        pyautogui.click()

                        # Step 4: Press enter multiple times (guarantee send)
                        pyautogui.press("enter")
                        time.sleep(1)
                        pyautogui.press("enter")

                        speak("Message sent successfully ✅")
                    else:
                        speak("System integration (Automation) is not available on this system.")

                except Exception as e:
                    print(e)
                    speak("Message send nahi hua")

                return

        speak("Contact nahi mila")

    # -------- OPEN APPS --------
    elif "open" in command:

        if "youtube" in command:
            speak("Opening YouTube ▶️")
            webbrowser.open("https://www.youtube.com")

        elif "chrome" in command:
            speak("Opening Chrome 🌐")
            os.system("start chrome")

        elif "notepad" in command:
            speak("Opening Notepad 📝")
            os.system("notepad")

        elif "calculator" in command or "calc" in command:
            speak("Opening Calculator 🧮")
            os.system("calc")

        elif "whatsapp" in command:
            speak("Opening WhatsApp 💬")
            webbrowser.open("https://web.whatsapp.com")

        elif "folder" in command:
            speak("Opening Downloads folder 📂")
            os.startfile("C:\\Users\\Akram Ansari\\Downloads")

        elif "code" in command or "vs code" in command:
            speak("Opening VS Code 💻")
            os.system("code")

        else:
            speak("App not found")

    # -------- YOUTUBE SEARCH --------
    elif "youtube" in command:
        if "song" in command or "search" in command:
            song = command.replace("youtube", "").replace("search", "").replace("song", "").strip()
            if song != "":
                speak(f"Playing {song} on YouTube 🎵")
                try:
                    pywhatkit.playonyt(song)
                except Exception as e:
                    print("Pywhatkit error:", e)
                    speak("Playing via search fallback ⚠️")
                    webbrowser.open(f"https://www.youtube.com/results?search_query={song}")
            else:
                speak("What should I search on YouTube?")
        else:
            speak("Opening YouTube ▶️")
            webbrowser.open("https://www.youtube.com")

    # -------- SCREENSHOT --------
    elif "screenshot" in command:
        if pyautogui:
            speak("Taking screenshot 📸")
            img = pyautogui.screenshot()
            img.save("screenshot.png")
        else:
            speak("Screenshot feature is not available on this system.")

    # -------- VOLUME --------
    elif "volume up" in command:
        if pyautogui:
            pyautogui.press("volumeup")
            speak("Volume up 🔊")

    elif "volume down" in command:
        if pyautogui:
            pyautogui.press("volumedown")
            speak("Volume down 🔉")

    elif "mute" in command:
        if pyautogui:
            pyautogui.press("volumemute")
            speak("Muted 🔇")

    # -------- WEATHER --------
    elif "weather" in command or "mausam" in command:
        city = command.replace("weather", "").replace("mausam", "").replace("in", "").replace("of", "").strip() or "Delhi"
        speak(get_weather(city))

    # -------- NEWS --------
    elif "news" in command or "khabar" in command:
        speak("Top headlines la raha hoon 📰")
        news_list = get_news()
        for i, n in enumerate(news_list):
            speak(f"News {i+1}: {n}")

    # -------- SYSTEM CONTROL --------
    elif "shutdown" in command:
        speak("Shutting down system 🔒")
        os.system("shutdown /s /t 5")

    elif "restart" in command:
        speak("Restarting system 🔄")
        os.system("shutdown /r /t 5")

    elif "lock" in command:
        speak("Locking system 🔐")
        os.system("rundll32.exe user32.dll,LockWorkStation")

    # -------- STUDY SYSTEM --------
    elif "study plan" in command:
        speak("Tumhara weekly study plan 📚")
        for day, topic in study_plan.items():
            speak(f"{day}: {topic}")

    elif "aaj kya padhna hai" in command or "today study" in command:
        day = datetime.datetime.now().strftime("%A").lower()
        topic = study_plan.get(day, "Revision")
        speak(f"Aaj tumhe {topic} padhna hai 📚")

    # -------- FOCUS MODE --------
    elif "focus mode" in command:
        speak("Focus mode start ho gaya 🔥 25 minutes study")
        time.sleep(1500)  # 25 min
        speak("Break time! 5 minutes relax ☕")

    # -------- OPEN FILES --------
    elif "open file" in command or "file kholo" in command:
        folder = "C:\\Users\\Akram Ansari\\Downloads"
        for file in os.listdir(folder):
            if "resume" in command and ".pdf" in file.lower():
                speak("Resume open kar raha hoon 📄")
                os.startfile(os.path.join(folder, file))
                return
        speak("File nahi mili")

    # -------- SEARCH FILES --------
    elif "search file" in command:
        keyword = command.replace("search file", "").strip()
        folder = "C:\\Users\\Akram Ansari\\Downloads"
        found = False
        for file in os.listdir(folder):
            if keyword in file.lower():
                speak(f"{file} mil gaya")
                os.startfile(os.path.join(folder, file))
                found = True
                break
        if not found:
            speak("File nahi mili")

    # -------- SMART SUGGESTION & DASHBOARD --------
    elif "suggest" in command:
        history = memory.get("history", [])
        if history:
            speak(f"Tum aksar bolte ho: {history[-1]}")
        else:
            speak("Abhi data nahi hai")

    elif "dashboard" in command:
        speak("Tumhara system report 📊")
        speak(f"Total commands: {len(memory.get('history', []))}")
        speak("System running smoothly ⚡")

    # -------- EXIT --------
    elif "stop" in command or "exit" in command:
        speak("Shutting down Astra 🔒")
        exit()

    # -------- REPORT LAST COMMAND --------
    elif "repeat" in command:
        speak(f"Tumne bola tha: {last_command}")

    # -------- CONTROLLED AI SYSTEM --------
    elif any(word in command for word in [
        "kaise", "kyu", "kya", "kaun", "explain", "who", "what", "tell", "kaisa", "batao", 
        "dost", "friend", "father", "mother", "bhai", "sister", "papa", "mummy", 
        "ajmat", "maimun", "raushan", "ekram", "nadeem", "zidaan", "rosidul", "munshi",
        "russian", "arjit", "motu", "aryan", "kaif", "nayan", "shahid", "wasim", "kunaal", "asif"
    ]):
        speak("Thinking... 🧠")
        reply = ai_chat(command)

        if "quota" in reply.lower() or "error" in reply.lower():
            speak("AI unavailable hai ⚠️")
        else:
            speak(reply)

    # -------- SMART FALLBACK --------
    else:
        speak("Command samajh nahi aaya 🤔")

# -------- MAIN --------
def main():
    speak("Jarvis mode activated 🔥")

    # CONTINUOUS LISTENING (No wake word required after login)
    while True:
        command = listen()

        if command != "":
            # Still strip wake words if the user says them
            wake_words = ["hey astra", "he astra", "astra", "jarvis", "hello astra"]
            for word in wake_words:
                command = command.replace(word, "").strip()
            
            if command != "":
                process_command(command)

# Helper functions for UI and API
def get_weather(city):
    KEY = "YOUR_WEATHER_API_KEY" # Placeholder
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={KEY}&q={city}"
        data = requests.get(url).json()
        temp = data["current"]["temp_c"]
        cond = data["current"]["condition"]["text"]
        return f"{city} me filhal {cond} hai aur temperature {temp} degree hai 🌡️"
    except:
        return "Weather update fetch nahi ho paya"

def get_news():
    KEY = "YOUR_NEWS_API_KEY" # Placeholder
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={KEY}"
        data = requests.get(url).json()
        articles = data.get("articles", [])[:3]
        return [a["title"] for a in articles]
    except:
        return ["News unavailable..."]

if __name__ == "__main__":
    # Old start logic (replaced by ui.py)
    # threading.Thread(target=start_ui, daemon=True).start()
    
    # if face_login():
    #     main()
    # else:
    #     speak("Astra access denied. Goodbye.")
    print("Astra core initialized. Use ui.py or api.py to start.")
