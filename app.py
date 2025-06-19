import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image
import google.generativeai as genai
import logging
import hashlib
import re

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- UI-Einstellungen ---
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot")
st.markdown("*Multi-Model Consensus System*")

# --- API Key Validation ---
def validate_keys():
    required_keys = {
        'gemini_key': ('AIza', "Gemini"),
        'claude_key': ('sk-ant', "Claude"),
        'openai_key': ('sk-', "OpenAI")
    }
    missing = []
    
    for key, (prefix, name) in required_keys.items():
        if key not in st.secrets:
            missing.append(name)
        elif not st.secrets[key].startswith(prefix):
            missing.append(f"{name} (invalid)")
    
    if missing:
        st.error(f"Fehlende API Keys: {', '.join(missing)}")
        st.stop()

validate_keys()

# --- API Clients initialisieren ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")
claude_client = Anthropic(api_key=st.secrets["claude_key"])
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- Test welches GPT Modell verfügbar ist ---
@st.cache_data
def get_available_gpt_model():
    """Testet welches GPT Modell tatsächlich funktioniert"""
    test_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
    
    for model in test_models:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            logger.info(f"Model {model} ist verfügbar")
            return model
        except Exception as e:
            logger.warning(f"Model {model} nicht verfügbar: {str(e)}")
            continue
    
    return "gpt-3.5-turbo"

GPT_MODEL = get_available_gpt_model()
st.sidebar.info(f"🤖 Verwende: Claude + {GPT_MODEL}")

# --- GEMINI OCR FUNKTION ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    """Extrahiert Text aus Bild mit Gemini"""
    try:
        logger.info(f"Starting Gemini OCR for hash: {file_hash}")
        
        # OCR mit Gemini
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written. "
                "Include all question numbers, text, formulas, and answer options (A, B, C, D, E). "
                "Do NOT interpret or solve anything, just extract the text.",
                _image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 8000
            }
        )
        
        ocr_text = response.text.strip()
        logger.info(f"Gemini OCR completed: {len(ocr_text)} characters")
        return ocr_text
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- Lösungsextraktion ---
def extract_answers(solution_text):
    """Extrahiert strukturierte Antworten aus Lösungstext"""
    answers = {}
    lines = solution_text.split('\n')
    
    for line in lines:
        match = re.search(r'Aufgabe\s*(\d+)\s*:\s*([A-E,\s]+|\d+|[\d,]+)', line, re.IGNORECASE)
        if match:
            task_num = match.group(1)
            answer = match.group(2).strip()
            if any(letter in answer.upper() for letter in 'ABCDE'):
                answer = ''.join(sorted(c for c in answer.upper() if c in 'ABCDE'))
            answers[f"Aufgabe {task_num}"] = answer
    
    return answers

# --- Claude Solver ---
def solve_with_claude(ocr_text):
    """Claude löst die Aufgabe"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

KRITISCHE REGELN:
- Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Wenn nur α + β gegeben ist (ohne α = β), ist die Funktion NICHT homogen
- Bei Multiple Choice: Prüfe JEDE Option einzeln

ANALYSIERE DIESEN TEXT:
{ocr_text}

FORMAT:
Aufgabe [Nr]: [Antwort]
Begründung: [Kurze Erklärung auf Deutsch]"""

    response = claude_client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=2000,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# --- GPT Solver ---
def solve_with_gpt(ocr_text):
    """GPT löst die Aufgabe"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

KRITISCHE REGELN:
- Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Wenn nur α + β gegeben ist (ohne α = β), ist die Funktion NICHT homogen
- Bei Multiple Choice: Prüfe JEDE Option einzeln

ANALYSIERE:
{ocr_text}

FORMAT:
Aufgabe [Nr]: [Antwort]
Begründung: [Kurze Erklärung auf Deutsch]"""

    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "Du bist ein präziser Mathematik-Experte."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.1
    )
    
    return response.choices[0].message.content

# --- Consensus System ---
def achieve_consensus_multi(ocr_text):
    """Consensus zwischen Claude und GPT"""
    
    # Lösungen generieren
    with st.spinner("Claude löst..."):
        claude_solution = solve_with_claude(ocr_text)
    
    with st.spinner(f"{GPT_MODEL} löst..."):
        gpt_solution = solve_with_gpt(ocr_text)
    
    # Antworten extrahieren
    claude_answers = extract_answers(claude_solution)
    gpt_answers = extract_answers(gpt_solution)
    
    # Debug Info
    with st.expander("🔍 Debug: Antworten"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Claude:**")
            st.json(claude_answers)
        with col2:
            st.write(f"**{GPT_MODEL}:**")
            st.json(gpt_answers)
    
    # Vergleiche
    all_tasks = set(claude_answers.keys()) | set(gpt_answers.keys())
    consensus = True
    
    for task in sorted(all_tasks):
        claude_ans = claude_answers.get(task, "?")
        gpt_ans = gpt_answers.get(task, "?")
        
        if claude_ans != gpt_ans:
            consensus = False
            st.warning(f"Diskrepanz bei {task}: Claude={claude_ans}, GPT={gpt_ans}")
    
    if consensus and claude_answers:
        st.success("✅ Modelle sind sich einig!")
        return claude_solution
    else:
        st.info("🔄 Verwende Claude mit Verifikation")
        # Selbst-Verifikation
        verify_prompt = (
            "Prüfe diese Lösung kritisch:\n\n"
            f"AUFGABE:\n{ocr_text}\n\n"
            f"LÖSUNG:\n{claude_solution}\n\n"
            "Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β!\n\n"
            "Gib die FINALE KORREKTE Lösung im gleichen Format."
        )

        response = claude_client.messages.create(
            model="claude-4-opus-20250514",
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": verify_prompt}]
        )
        
        return response.content[0].text

# --- MAIN UI ---
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
    
    # GEMINI OCR
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    with st.spinner("📖 Lese Text mit Gemini..."):
        try:
            ocr_text = extract_text_with_gemini(image, file_hash)
            st.success(f"✅ OCR erfolgreich: {len(ocr_text)} Zeichen extrahiert")
        except Exception as e:
            st.error(f"❌ OCR Fehler: {str(e)}")
            st.stop()
    
    with st.expander("📝 OCR-Text (Gemini)"):
        st.code(ocr_text)
    
    # Solve Button
    if st.button("🧮 Mit Multi-Model Consensus lösen", type="primary"):
        st.markdown("---")
        st.markdown("### 🤝 Consensus-Prozess:")
        
        try:
            # Consensus erreichen
            final_solution = achieve_consensus_multi(ocr_text)
            
            # Ergebnisse anzeigen
            st.markdown("---")
            st.markdown("### 📊 Lösung:")
            
            lines = final_solution.split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('Aufgabe'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            st.markdown(f"### {parts[0]}: **{parts[1].strip()}**")
                    elif line.startswith('Begründung:'):
                        st.markdown(f"*{line}*")
                        
        except Exception as e:
            st.error(f"Fehler: {str(e)}")
            logger.error(f"Processing error: {str(e)}")

# Footer
st.markdown("---")
st.caption(f"Multi-Model Consensus System | Gemini OCR + Claude 4 Opus + {GPT_MODEL}")
