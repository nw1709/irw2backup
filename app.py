import streamlit as st
from anthropic import Anthropic
from PIL import Image
import google.generativeai as genai
import logging
import hashlib
import re
import time
import io  # FEHLTE!

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- API Key Validation ---
def validate_keys():
    required_keys = {
        'gemini_key': ('AIza', "Gemini"),
        'claude_key': ('sk-ant', "Claude")
    }
    
    for key, (prefix, name) in required_keys.items():
        if key not in st.secrets:
            st.error(f"❌ {name} API Key fehlt in secrets.toml")
            st.stop()
        elif not st.secrets[key].startswith(prefix):
            st.error(f"❌ {name} API Key ist ungültig")
            st.stop()
    
    st.sidebar.success("✅ Alle API Keys validiert")

# --- UI Setup ---
st.set_page_config(
    page_title="Koifox-Bot", 
    page_icon="🦊",
    layout="centered"
)

st.title("🦊 Koifox-Bot")
st.markdown("*Optimierte Version - Gemini OCR + Claude Solver*")

# Validate keys
validate_keys()

# --- Initialize APIs ---
@st.cache_resource
def init_apis():
    """Initialisiert APIs einmalig"""
    genai.configure(api_key=st.secrets["gemini_key"])
    vision_model = genai.GenerativeModel("gemini-1.5-flash")
    claude_client = Anthropic(api_key=st.secrets["claude_key"])
    logger.info("APIs initialisiert")
    return vision_model, claude_client

vision_model, claude_client = init_apis()

# --- Stats in Sidebar ---
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'ocr_calls': 0,
        'solve_calls': 0,
        'errors': 0
    }

st.sidebar.markdown("### 📊 Session Stats")
st.sidebar.metric("OCR Calls", st.session_state.stats['ocr_calls'])
st.sidebar.metric("Solve Calls", st.session_state.stats['solve_calls'])
st.sidebar.metric("Errors", st.session_state.stats['errors'])

if st.sidebar.button("🗑️ Reset Session"):
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- OCR Function ---
@st.cache_data(ttl=3600, show_spinner=False)
def extract_text_with_gemini(_image_bytes, file_hash):
    """OCR mit Gemini - gecached"""
    try:
        logger.info(f"Starting Gemini OCR for hash: {file_hash[:8]}...")
        st.session_state.stats['ocr_calls'] += 1
        
        # Image from bytes
        image = Image.open(io.BytesIO(_image_bytes))
        
        # OCR prompt
        ocr_prompt = """
        Extract ALL text from this exam image:
        1. Start from the very top
        2. Read line by line until the bottom
        3. Include ALL text: questions, formulas, values, options (A,B,C,D,E)
        4. Include question numbers (e.g. "Aufgabe 45 (5 RP)")
        5. Preserve mathematical notation exactly
        6. DO NOT skip anything
        7. DO NOT interpret or solve
        
        Output the text EXACTLY as written.
        """
        
        # Call Gemini
        response = vision_model.generate_content(
            [ocr_prompt, image],
            generation_config={
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 8192
            }
        )
        
        ocr_text = response.text.strip()
        logger.info(f"Gemini OCR completed: {len(ocr_text)} characters")
        
        # Validate OCR
        if len(ocr_text) < 100:
            raise ValueError(f"OCR zu kurz: nur {len(ocr_text)} Zeichen")
        
        return ocr_text
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        st.session_state.stats['errors'] += 1
        raise e

# --- Claude Solver ---
def solve_with_claude(ocr_text):
    """Claude löst basierend auf OCR Text"""
    try:
        logger.info("Starting Claude solver...")
        st.session_state.stats['solve_calls'] += 1
        
        # Find all task numbers in OCR text
        task_numbers = re.findall(r'Aufgabe\s+(\d+)', ocr_text, re.IGNORECASE)
        logger.info(f"Gefundene Aufgaben: {task_numbers}")
        
        solve_prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

WICHTIGE REGELN:
- Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Wenn nur "α + β = k" gegeben ist, kann α ≠ β sein → NICHT homogen
- Bei Multiple Choice: Prüfe JEDE Option einzeln
- Verwende präzise Fachterminologie

AUFGABENTEXT (via OCR):
{ocr_text}

ANWEISUNGEN:
1. Beantworte ALLE Aufgaben die du im Text findest
2. Überspringe KEINE Aufgabe
3. Format für JEDE Aufgabe:

Aufgabe [Nummer]: [Antwort]
Begründung: [Kurze Erklärung auf Deutsch]

Gefundene Aufgaben im Text: {', '.join(task_numbers) if task_numbers else 'Bitte selbst identifizieren'}

WICHTIG: Stelle sicher, dass du ALLE Aufgaben beantwortest!"""

        # Call Claude with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = claude_client.messages.create(
                    model="claude-4-opus-20250514",
                    max_tokens=4096,
                    temperature=0.1,
                    top_p=1.0,
                    system="Du bist ein präziser Experte für deutsches Controlling. Beantworte ALLE Aufgaben im Text.",
                    messages=[{
                        "role": "user",
                        "content": solve_prompt
                    }]
                )
                
                solution = response.content[0].text
                logger.info("Claude solver completed successfully")
                return solution
                
            except Exception as e:
                if "rate_limit" in str(e) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
                    
    except Exception as e:
        logger.error(f"Claude Solver Error: {str(e)}")
        st.session_state.stats['errors'] += 1
        raise e

# --- Solution Parser ---
def parse_and_display_solution(solution_text, expected_tasks):
    """Parse und zeige Lösung strukturiert"""
    
    # Extract all answers
    pattern = r'Aufgabe\s+(\d+)\s*:\s*([^\n]+)'
    matches = re.findall(pattern, solution_text, re.IGNORECASE)
    
    if not matches:
        st.error("❌ Keine Lösungen im erwarteten Format gefunden")
        st.code(solution_text)
        return
    
    # Display solutions
    found_tasks = []
    for task_num, answer in matches:
        found_tasks.append(task_num)
        st.markdown(f"### Aufgabe {task_num}: **{answer.strip()}**")
        
        # Find reasoning
        begr_pattern = rf'Aufgabe\s+{task_num}.*?\nBegründung:\s*([^\n]+(?:\n(?!Aufgabe)[^\n]+)*)'
        begr_match = re.search(begr_pattern, solution_text, re.IGNORECASE | re.DOTALL)
        
        if begr_match:
            st.markdown(f"*Begründung: {begr_match.group(1).strip()}*")
        
        st.markdown("---")
    
    # Check completeness
    if expected_tasks:
        missing = set(expected_tasks) - set(found_tasks)
        if missing:
            st.warning(f"⚠️ Fehlende Aufgaben: {', '.join(sorted(missing))}")

# --- MAIN UI ---
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen**",
    type=["png", "jpg", "jpeg"],
    help="PNG/JPG/JPEG Bilder"
)

if uploaded_file is not None:
    # Process upload
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
    
    # Info
    st.info(f"📄 Datei: {uploaded_file.name} | Größe: {len(file_bytes)/1024:.1f} KB")
    
    # OCR Section
    st.markdown("### 1️⃣ Texterkennung (OCR)")
    
    try:
        with st.spinner("🔍 Lese Text mit Gemini..."):
            # Use bytes for caching
            ocr_text = extract_text_with_gemini(file_bytes, file_hash)
        
        # Show OCR result
        with st.expander(f"📝 OCR Ergebnis ({len(ocr_text)} Zeichen)", expanded=False):
            st.code(ocr_text)
            
            # Find tasks
            found_tasks = re.findall(r'Aufgabe\s+(\d+)', ocr_text, re.IGNORECASE)
            if found_tasks:
                st.success(f"✅ Gefundene Aufgaben: {', '.join(set(found_tasks))}")
            else:
                st.warning("⚠️ Keine Aufgaben-Nummern gefunden")
    
    except Exception as e:
        st.error(f"❌ OCR Fehler: {str(e)}")
        st.stop()
    
    # Solve Section
    st.markdown("### 2️⃣ Aufgaben lösen")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        solve_button = st.button(
            "🧮 Alle Aufgaben lösen",
            type="primary",
            disabled=not ocr_text
        )
    with col2:
        show_debug = st.checkbox("Debug", value=False)
    
    if solve_button:
        try:
            with st.spinner("💭 Claude löst die Aufgaben..."):
                solution = solve_with_claude(ocr_text)
            
            if show_debug:
                with st.expander("🔍 Rohe Claude-Antwort"):
                    st.code(solution)
            
            # Display solutions
            st.markdown("### 3️⃣ Lösungen")
            parse_and_display_solution(solution, found_tasks)
            
            # Success message
            st.success("✅ Fertig!")
            
        except Exception as e:
            st.error(f"❌ Fehler beim Lösen: {str(e)}")

else:
    # Instructions
    st.markdown("""
    ### 📖 Anleitung:
    1. Lade ein Foto der Klausuraufgabe hoch
    2. Der Bot liest den Text automatisch (OCR)
    3. Klicke auf "Aufgaben lösen"
    4. Erhalte präzise Lösungen mit Begründungen
    
    **Unterstützte Aufgabentypen:**
    - Multiple Choice (x aus 5)
    - Rechenaufgaben
    - Theoretische Fragen
    - Alle Themen aus "Internes Rechnungswesen"
    """)

# Footer
st.markdown("---")
st.caption("Koifox-Bot v4.0 | Optimized for reliability")
