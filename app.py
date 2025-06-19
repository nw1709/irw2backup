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
usermode_gpt = st.checkbox("🧠 GPT-4 Turbo stattdessen verwenden", value=False, help="Nutzt GPT-4 direkt, z. B. für zweite Meinung oder wenn Claude down ist")

# --- Datei-Upload ---
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"],
    key="file_uploader"
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)

        with st.spinner("Lese Text mit Gemini Flash..."):
            ocr_text = extract_text_with_gemini(image, file_hash)

        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis", expanded=False):
                st.code(ocr_text)
                st.info(f"File Hash: {file_hash[:8]}... (für Caching)")

        if st.button("🧮 Aufgaben lösen", type="primary"):
            header = load_prompt_header()
            prompt = f"""{header}

OCR-TEXT START:
{ocr_text}
OCR-TEXT ENDE

KRITISCHE ANWEISUNGEN:
1. Lies die Aufgabe SEHR sorgfältig
2. Bei Rechenaufgaben:
   - Zeige JEDEN Rechenschritt
   - Prüfe dein Ergebnis nochmal
3. Bei Multiple Choice: Prüfe jede Option einzeln
4. VERIFIZIERE deine Antwort bevor du antwortest
5. Stelle SICHER, dass deine Antwort mit deiner Analyse übereinstimmt!

FORMAT - WICHTIG:
Aufgabe [Nr]: [NUR die finale Antwort - Zahl oder Buchstabe(n)]
Begründung: [1 Satz auf Deutsch]
"""

            if debug_mode:
                with st.expander("🔍 Claude Prompt", expanded=False):
                    st.code(prompt)

            with st.spinner("Löse Aufgabe..."):
                if usermode_gpt:
                    result, claude_failed = call_gpt4_fallback(prompt)
                else:
                    result, claude_failed = call_claude_or_fallback(prompt)

            st.markdown("---")
            st.markdown("### Lösung:")

            lines = result.split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('Aufgabe'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            st.markdown(f"### {parts[0]}: **{parts[1].strip()}**")
                        else:
                            st.markdown(f"### {line}")
                    elif line.startswith('Begründung:'):
                        st.markdown(f"*{line}*")
                    else:
                        st.markdown(line)

            if claude_failed and not usermode_gpt:
                st.warning("Claude war aktuell nicht erreichbar. Die Antwort wurde stattdessen von GPT-4 Turbo erstellt. Versuche es später erneut, wenn du Claude bevorzugst.")

            st.info("💡 OCR-Ergebnisse werden gecached, Lösungen werden immer neu berechnet.")

    except Exception as e:
        logger.error(f"General error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

st.markdown("---")
st.caption("Made by Fox | OCR cached, Solutions always fresh")
