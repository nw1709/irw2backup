import streamlit as st
import openai
from anthropic import Anthropic
from PIL import Image
import google.generativeai as genai
import logging
import hashlib

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Validation ---
def validate_keys():
    required_keys = {
        'gemini_key': ('AIza', "Gemini"),
        'claude_key': ('sk-ant', "Claude"),
        'openai_key': ('sk-', "OpenAI")
    }
    missing = []
    invalid = []

    for key, (prefix, name) in required_keys.items():
        if key not in st.secrets:
            missing.append(name)
        elif not st.secrets[key].startswith(prefix):
            invalid.append(name)

    if missing or invalid:
        st.error(f"API Key Problem: Missing {', '.join(missing)} | Invalid {', '.join(invalid)}")
        st.stop()

validate_keys()

# --- UI-Einstellungen ---
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦧")
st.title("🦧 Koifox-Bot")
st.markdown("*Made with coffee, deep minimal and tiny gummy bears*")

# --- Cache Management ---
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Cache leeren", type="secondary", help="Löscht gespeicherte OCR-Ergebnisse"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# --- OCR mit Caching ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    try:
        logger.info(f"Starting OCR for file hash: {file_hash}")
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written. Include all question numbers, text, and answer options (A, B, C, D, E). Do NOT interpret or solve.",
                _image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 4000
            }
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- Prompt-Vorlage laden ---
@st.cache_data
def load_prompt_header():
    with open("prompt_header.txt", "r", encoding="utf-8") as f:
        return f.read()

# --- GPT-4 Fallback Funktion ---
def call_gpt4_fallback(prompt):
    try:
        openai.api_key = st.secrets["openai_key"]
        logger.info("Calling OpenAI GPT-4 fallback...")
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0
        )
        logger.info("GPT-4 fallback successful")
        return response.choices[0].message.content.strip(), True
    except Exception as e:
        logger.error(f"OpenAI GPT-4 Error: {str(e)}")
        st.error(f"Backup-Fallback fehlgeschlagen: {str(e)}")
        return "❌ Fehler bei beiden Systemen.", True

# --- Claude oder Fallback verwenden ---
def call_claude_or_fallback(prompt):
    try:
        logger.info("Calling Claude API...")
        client = Anthropic(api_key=st.secrets["claude_key"])
        response = client.messages.create(
            model="claude-4-opus-20250514",
            max_tokens=4000,
            temperature=0,
            top_p=1.0,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        logger.info("Claude API call successful")
        return response.content[0].text, False
    except Exception as e:
        logger.error(f"Claude API Error: {str(e)} – trying fallback")
        return call_gpt4_fallback(prompt)

# --- UI Optionen ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False, help="Zeigt OCR-Ergebnis und Details")
usermode_gpt = st.checkbox("🧠 GPT-4 Turbo stattdessen verwenden", valu_
