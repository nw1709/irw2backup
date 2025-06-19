import streamlit as st
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
        'claude_key': ('sk-ant', "Claude")
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
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot")
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
@st.cache_data(ttl=3600)  # Cache für 1 Stunde
def extract_text_with_gemini(_image, file_hash):
    """Extrahiert Text aus Bild - gecached basierend auf file_hash"""
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
        logger.info("OCR completed successfully")
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

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
        
        # OCR (gecached)
        with st.spinner("Lese Text mit Gemini Flash..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
            
        # Debug: OCR-Ergebnis anzeigen
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis", expanded=False):
                st.code(ocr_text)
                st.info(f"File Hash: {file_hash[:8]}... (für Caching)")
        
        # Button zum Lösen - KEINE Caching für Lösungen!
        if st.button("🧮 Aufgaben lösen", type="primary"):
            
            # Verbesserter Prompt mit Chain-of-Thought
            prompt = f"""You are a highly qualified accounting expert with PhD-level 
knowledge of the university course "Internes Rechnungswesen (31031)" at Fernuniversität Hagen. 
Your task is to answer exam questions with 100% accuracy.

THEORETICAL SCOPE
Use only the decision-oriented German managerial-accounting (Controlling) framework:
• Cost-type, cost-center and cost-unit accounting (Kostenarten-, Kostenstellen-, Kostenträgerrechnung)
• Full, variable, marginal, standard (Plankosten-) and process/ABC costing systems
• Flexible and Grenzplankostenrechnung variance analysis
• Single- and multi-level contribution-margin accounting and break-even logic
• Causality & allocation (Verursachungs- und Zurechnungsprinzip)
• Business-economics MRS convention (MRS = MP₂ / MP₁ unless stated otherwise)
• Activity-analysis production & logistics models (LP, Standort- & Transportprobleme)
• Marketing segmentation, price-elasticity, contribution-based pricing & mix planning

OCR-TEXT START:
{ocr_text}
OCR-TEXT ENDE

CRITICAL THINKING PROCESS:
1. Read the question COMPLETELY before making any assumptions
2. Identify what is GIVEN and what is ASKED
3. Check if all necessary conditions are met before drawing conclusions
4. For each answer option: Test it rigorously against the given conditions
5. Never assume additional constraints that are not explicitly stated
6. If a mathematical property requires specific conditions, verify they are met

STEP-BY-STEP APPROACH:
- First: State what you need to check
- Second: Perform the analysis systematically
- Third: Draw conclusions based ONLY on your analysis
- Fourth: Verify your answer matches your reasoning

FORMAT:
Aufgabe [Nr]: [Final answer only]
Begründung: [Concise explanation in German with your reasoning]

Remember: Logical rigor is paramount. Do not jump to conclusions."""
            
            if debug_mode:
                with st.expander("🔍 Claude Prompt", expanded=False):
                    st.code(prompt)
            
            # Claude API-Aufruf mit optimierten Parametern für besseres Reasoning
            with st.spinner("Löse Aufgabe..."):
                try:
                    logger.info("Calling Claude API...")
                    client = Anthropic(api_key=st.secrets["claude_key"])
                    
                    # System Message für bessere Grundlogik
                    system_message = """You are a precise academic expert. Always:
1. Question your assumptions
2. Check if conditions for mathematical properties are actually met
3. Consider all possibilities before concluding
4. Be especially careful with terms like 'homogeneous', 'linear', etc. - they have precise mathematical definitions
5. If something is true only under specific conditions, those conditions must be verified"""
                    
                    response = client.messages.create(
                        model="claude-4-opus-20250514",
                        max_tokens=4000,
                        temperature=0.1,      # Leicht erhöht für besseres Reasoning
                        top_p=1.0,
                        top_k=40,            # NEU: Begrenzt Auswahl für konsistenteres Reasoning
                        system=system_message,  # NEU: System message für Grundlogik
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    
                    result = response.content[0].text
                    logger.info("Claude API call successful")
                    
                except Exception as e:
                    logger.error(f"Claude API Error: {str(e)}")
                    st.error(f"API Fehler: {str(e)}")
                    raise e
            
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
                        
            # Info über Caching
            st.info("💡 OCR-Ergebnisse werden gecached, Lösungen werden immer neu berechnet.")
                    
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Made by Fox | Enhanced reasoning with systematic thinking")
