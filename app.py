import streamlit as st
from anthropic import Anthropic
from PIL import Image
import google.generativeai as genai
from openai import OpenAI
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
            logger.info("Trying Claude 4 Opus")
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
            st.success("✅ Gelöst mit Claude 4 Opus")
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude failed: {e}")
            st.warning("⚠️ Claude nicht verfügbar, verwende GPT-4...")
    
    # Versuch 2: GPT-4 Turbo
    if "openai_key" in st.secrets:
        try:
            logger.info("Trying GPT-4 Turbo")
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für Internes Rechnungswesen an der Fernuni Hagen."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4000,
                temperature=0,
                top_p=1.0
            )
            st.info("ℹ️ Gelöst mit GPT-4 Turbo (Fallback)")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-4 failed: {e}")
    
    raise Exception("Alle Solving-Dienste nicht verfügbar")

# --- UI Optionen ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False, help="Zeigt OCR-Ergebnis und Details")

# --- Datei-Upload ---
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"],
    key="file_uploader"
)

if uploaded_file is not None:
    try:
        # Eindeutiger Hash für die Datei
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        # Bild laden und anzeigen
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        
        # OCR mit Fallback
        with st.spinner("Lese Text..."):
            try:
                ocr_text = extract_text_with_fallback(image, file_hash)
            except Exception as e:
                st.error(f"❌ OCR fehlgeschlagen: {str(e)}")
                st.stop()
            
        # Debug: OCR-Ergebnis anzeigen
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis", expanded=False):
                st.code(ocr_text)
                st.info(f"File Hash: {file_hash[:8]}...")
        
        # Button zum Lösen
        if st.button("🧮 Aufgaben lösen", type="primary"):
            
            # Prompt (gleich für alle Modelle)
            prompt = f"""You are a highly qualified accounting expert with PhD-level 
knowledge of the university course "Internes Rechnungswesen (31031)" at Fernuniversität Hagen. 
Your task is to answer exam questions with 100% accuracy.

THEORETICAL SCOPE
Use only the decision-oriented German managerial-accounting (Controlling) framework:
- Cost-type, cost-center and cost-unit accounting (Kostenarten-, Kostenstellen-, Kostenträgerrechnung)
- Full, variable, marginal, standard (Plankosten-) and process/ABC costing systems
- Flexible and Grenzplankostenrechnung variance analysis
- Single- and multi-level contribution-margin accounting and break-even logic
- Causality & allocation (Verursachungs- und Zurechnungsprinzip)
- Business-economics MRS convention (MRS = MP₂ / MP₁ unless stated otherwise)
- Activity-analysis production & logistics models (LP, Standort- & Transportprobleme)
- Marketing segmentation, price-elasticity, contribution-based pricing & mix planning

WICHTIG: Analysiere NUR den folgenden OCR-Text. Erfinde KEINE anderen Aufgaben! 
Sei extrem präzise und verwende die Lösungswege und die Terminologie der Fernuni Hagen. Es gibt absolut keinen Raum für Fehler!

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
                with st.expander("🔍 Prompt", expanded=False):
                    st.code(prompt)
            
            # Solving mit Fallback
            with st.spinner("Löse Aufgabe..."):
                try:
                    result = solve_with_fallback(prompt)
                except Exception as e:
                    st.error(f"❌ Alle Dienste nicht verfügbar: {str(e)}")
                    st.stop()
            
            # Ergebnisse anzeigen
            st.markdown("---")
            st.markdown("### Lösung:")
            
            # Formatierte Ausgabe
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
                        
            # Info über verwendete APIs
            st.info("💡 OCR-Ergebnisse werden gecached, Lösungen werden immer neu berechnet.")
            
            # Kostenwarnung bei GPT-4
            if "⚠️" in st.session_state.get("last_warning", ""):
                st.warning("💰 Hinweis: GPT-4 Turbo ist teurer als Claude/Gemini")
                    
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Made by Fox | Multi-API mit automatischem Fallback")
