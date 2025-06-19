import streamlit as st
from anthropic import Anthropic
from PIL import Image
import google.generativeai as genai
from openai
import logging
import hashlib
import base64
import io

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Validation ---
def validate_keys():
    required_keys = {
        'gemini_key': ('AIza', "Gemini"),
        'claude_key': ('sk-ant', "Claude"),
        'openai_key': ('sk-', "OpenAI")  # NEU
    }
    missing = []
    invalid = []
    
    for key, (prefix, name) in required_keys.items():
        if key not in st.secrets:
            missing.append(name)
        elif not st.secrets[key].startswith(prefix):
            invalid.append(name)
    
    # Mindestens 2 von 3 APIs müssen verfügbar sein
    if len(missing) >= 2:
        st.error(f"Zu viele fehlende API Keys: {', '.join(missing)}")
        st.stop()
    elif missing:
        st.warning(f"Fehlende API Keys: {', '.join(missing)} - Fallback aktiv")

validate_keys()

# --- UI-Einstellungen ---
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot")
st.markdown("*Made with coffee, deep minimal and tiny gummy bears*")

# API Status anzeigen
st.sidebar.markdown("### 🔌 API Status")
api_status = {
    "Gemini": "gemini_key" in st.secrets,
    "Claude": "claude_key" in st.secrets,
    "GPT-4": "openai_key" in st.secrets
}
for api, available in api_status.items():
    st.sidebar.markdown(f"{'✅' if available else '❌'} {api}")

# --- Cache Management ---
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Cache leeren", type="secondary", help="Löscht gespeicherte OCR-Ergebnisse"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --- API Konfiguration ---
if "gemini_key" in st.secrets:
    genai.configure(api_key=st.secrets["gemini_key"])
    vision_model = genai.GenerativeModel("gemini-1.5-flash")

if "openai_key" in st.secrets:
    openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- OCR mit Fallback ---
@st.cache_data(ttl=3600)
def extract_text_with_fallback(_image, file_hash):
    """OCR mit automatischem Fallback: Gemini → GPT-4 Vision"""
    
    # Versuch 1: Gemini
    if "gemini_key" in st.secrets:
        try:
            logger.info(f"Trying Gemini OCR for hash: {file_hash}")
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
            st.success("✅ OCR mit Gemini erfolgreich")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini failed: {e}")
            st.warning("⚠️ Gemini nicht verfügbar, verwende Fallback...")
    
    # Versuch 2: GPT-4 Vision
    if "openai_key" in st.secrets:
        try:
            logger.info("Trying GPT-4 Vision OCR")
            
            # Bild zu base64
            buffered = io.BytesIO()
            _image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract ALL text from this exam image EXACTLY as written. Include all question numbers, text, and answer options. Do NOT interpret or solve."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }],
                max_tokens=4000,
                temperature=0
            )
            st.info("ℹ️ OCR mit GPT-4 Vision (Fallback)")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT-4 Vision failed: {e}")
    
    raise Exception("Alle OCR-Dienste nicht verfügbar")

# --- Solving mit Fallback ---
def solve_with_fallback(prompt):
    """Lösung mit automatischem Fallback: Claude → GPT-4"""
    
    # Versuch 1: Claude 4 Opus
    if "claude_key" in st.secrets:
        try:
