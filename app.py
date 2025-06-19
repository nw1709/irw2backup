import streamlit as st
from anthropic import Anthropic
from PIL import Image
import google.generativeai as genai
import logging
import hashlib
import json

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
@st.cache_data(ttl=3600)
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

# --- NEUE FUNKTION: Doppelte Verifizierung ---
def verify_solution(ocr_text, first_solution):
    """Verifiziert die erste Lösung durch eine zweite, unabhängige Analyse"""
    verify_prompt = f"""Du bist ein ZWEITER Experte der die Lösung eines Kollegen überprüft.

AUFGABENTEXT:
{ocr_text}

ERSTE LÖSUNG:
{first_solution}

DEINE AUFGABE:
1. Löse die Aufgabe KOMPLETT NEU und UNABHÄNGIG
2. Vergleiche deine Lösung mit der ersten Lösung
3. Bei Unstimmigkeiten: Erkläre genau wo der Fehler liegt

WICHTIGE PRÜFPUNKTE:
- Bei Homogenität: Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Wenn nur α + β gegeben ist (nicht α = β), ist die Funktion i.A. NICHT homogen
- Prüfe JEDE mathematische Schlussfolgerung kritisch
- Hinterfrage Annahmen die nicht explizit gegeben sind

Antworte im Format:
VERIFIKATION: [BESTÄTIGT/FEHLER GEFUNDEN]
KORREKTE LÖSUNG: [Aufgabe X: Antwort]
ERKLÄRUNG: [Warum die Lösung richtig/falsch ist]"""

    client = Anthropic(api_key=st.secrets["claude_key"])
    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=2000,
        temperature=0.3,  # Etwas höher für kritisches Denken
        messages=[{"role": "user", "content": verify_prompt}]
    )
    
    return response.content[0].text

# --- UI Optionen ---
col1, col2 = st.columns([1, 1])
with col1:
    debug_mode = st.checkbox("🔍 Debug-Modus", value=False)
with col2:
    double_check = st.checkbox("✅ Doppelte Verifizierung", value=True, help="Empfohlen für maximale Genauigkeit")

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
        
        # Button zum Lösen
        if st.button("🧮 Aufgaben lösen", type="primary"):
            
            # NEUER PROMPT mit expliziten Fallstricken
            prompt = f"""You are an expert in "Internes Rechnungswesen (31031)" at Fernuniversität Hagen.

MATHEMATICAL RIGOR RULES:
1. A function is homogeneous of degree k if f(λr) = λ^k·f(r) for ALL λ and ALL valid inputs
2. For f(r₁,r₂) = (r₁^α + r₂^β)^γ to be homogeneous, you MUST be able to factor out λ completely
3. This is ONLY possible if α = β. If α ≠ β, the function is NOT homogeneous
4. NEVER assume α = β unless explicitly stated
5. "α + β = 3" does NOT imply α = β

ANALYZE THIS TEXT:
{ocr_text}

SYSTEMATIC APPROACH:
Step 1: Identify what is GIVEN (write it down)
Step 2: Identify what is ASKED (write it down)
Step 3: Apply definitions RIGOROUSLY
Step 4: Check each answer option against your analysis
Step 5: Select ONLY options that are mathematically correct

Think step by step. Show your work. Question your assumptions.

FORMAT:
Aufgabe [Nr]: [Answer]
Begründung: [Explanation in German]"""
            
            # Erste Lösung
            with st.spinner("Löse Aufgabe (Schritt 1/2)..."):
                client = Anthropic(api_key=st.secrets["claude_key"])
                
                response = client.messages.create(
                    model="claude-4-opus-20250514",
                    max_tokens=4000,
                    temperature=0.2,
                    top_p=0.95,
                    system="You are a rigorous mathematician. Never make unjustified assumptions. If a property requires specific conditions, verify they are met.",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                first_solution = response.content[0].text
            
            # Doppelte Verifizierung wenn aktiviert
            if double_check:
                with st.spinner("Verifiziere Lösung (Schritt 2/2)..."):
                    verification = verify_solution(ocr_text, first_solution)
                    
                    if debug_mode:
                        with st.expander("🔍 Verifizierungsergebnis"):
                            st.code(verification)
                    
                    # Finale Lösung basierend auf Verifizierung
                    if "FEHLER GEFUNDEN" in verification:
                        st.warning("⚠️ Inkonsistenz entdeckt - verwende verifizierte Lösung")
                        # Extrahiere korrekte Lösung aus Verifizierung
                        final_solution = verification
                    else:
                        final_solution = first_solution
            else:
                final_solution = first_solution
            
            # Ergebnisse anzeigen
            st.markdown("---")
            st.markdown("### Lösung:")
            
            # Parse die finale Lösung
            lines = final_solution.split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('Aufgabe') or line.startswith('KORREKTE LÖSUNG:'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            st.markdown(f"### {parts[0]}: **{parts[1].strip()}**")
                    elif line.startswith('Begründung:') or line.startswith('ERKLÄRUNG:'):
                        st.markdown(f"*{line}*")
                    elif line.startswith('VERIFIKATION:'):
                        if "BESTÄTIGT" in line:
                            st.success("✅ Lösung wurde verifiziert")
                        else:
                            st.info("ℹ️ Lösung wurde korrigiert")
                    
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Made by Fox | Double-verification system for maximum accuracy")
