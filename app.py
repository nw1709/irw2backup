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

# --- API Key Validation ---
def validate_keys():
    required_keys = {
        'gemini_key': ('AIza', "Gemini"),
        'claude_key': ('sk-ant', "Claude")
    }
    optional_keys = {
        'openai_key': ('sk-', "OpenAI")
    }
    
    missing_required = []
    for key, (prefix, name) in required_keys.items():
        if key not in st.secrets:
            missing_required.append(name)
        elif not st.secrets[key].startswith(prefix):
            missing_required.append(f"{name} (invalid)")
    
    if missing_required:
        st.error(f"Erforderliche API Keys fehlen: {', '.join(missing_required)}")
        st.stop()
    
    # Check optional
    has_openai = 'openai_key' in st.secrets and st.secrets['openai_key'].startswith('sk-')
    return has_openai

HAS_OPENAI = validate_keys()

# --- UI-Einstellungen ---
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot")

# --- API Clients ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")
claude_client = Anthropic(api_key=st.secrets["claude_key"])

if HAS_OPENAI:
    openai_client = OpenAI(api_key=st.secrets["openai_key"])
    st.markdown("*Multi-Model Consensus System (Claude + GPT)*")
else:
    st.markdown("*Single-Model System (nur Claude verfügbar)*")

# --- Test welches GPT Modell verfügbar ist ---
@st.cache_data
def get_available_gpt_model():
    """Testet welches GPT Modell tatsächlich funktioniert"""
    if not HAS_OPENAI:
        return None
        
    test_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
    
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
    
    return None

GPT_MODEL = get_available_gpt_model() if HAS_OPENAI else None

if GPT_MODEL:
    st.sidebar.success(f"✅ Verwende: Claude + {GPT_MODEL}")
else:
    st.sidebar.warning("⚠️ Nur Claude verfügbar")

# --- OCR mit Caching ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    """Extrahiert Text aus Bild"""
    try:
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written. Include all question numbers, text, and answer options. Do NOT interpret.",
                _image
            ],
            generation_config={"temperature": 0, "max_output_tokens": 4000}
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- Lösungsextraktion ---
def extract_answers(solution_text):
    """Extrahiert strukturierte Antworten aus Lösungstext"""
    answers = {}
    lines = solution_text.split('\n')
    
    for line in lines:
        # Flexibleres Pattern für verschiedene Formate
        patterns = [
            r'Aufgabe\s*(\d+)\s*:\s*([A-E,\s]+|\d+[\d,]*)',
            r'AUFGABE\s*(\d+)\s*:\s*([A-E,\s]+|\d+[\d,]*)',
            r'Task\s*(\d+)\s*:\s*([A-E,\s]+|\d+[\d,]*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                task_num = match.group(1)
                answer = match.group(2).strip()
                if any(letter in answer.upper() for letter in 'ABCDE'):
                    # Normalisiere Multiple-Choice
                    answer = ''.join(sorted(c for c in answer.upper() if c in 'ABCDE'))
                answers[f"Aufgabe {task_num}"] = answer
                break
    
    return answers

# --- Claude Solver ---
def solve_with_claude(ocr_text, iteration=1):
    """Claude löst die Aufgabe"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

KRITISCHE REGELN:
- Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Wenn nur α + β gegeben ist (ohne α = β), ist die Funktion NICHT homogen
- Prüfe ALLE Bedingungen bevor du Schlüsse ziehst

ANALYSIERE DIESEN TEXT:
{ocr_text}

FORMAT (WICHTIG):
Aufgabe [Nr]: [Antwort - NUR Buchstabe(n) oder Zahl]
Begründung: [Kurze Erklärung auf Deutsch]

Beispiel:
Aufgabe 1: BD
Begründung: Optionen B und D sind korrekt, weil...

DENKE SCHRITT FÜR SCHRITT!"""

    response = claude_client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=2000,
        temperature=0.1 if iteration == 1 else 0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# --- Single Model Solution ---
def solve_single_model(ocr_text):
    """Lösung nur mit Claude (wenn kein GPT verfügbar oder bei Uneinigkeit)"""
    st.info("🔄 Verwende erweiterte Claude-Analyse...")
    
    # Erste Lösung
    with st.spinner("Claude analysiert (Durchgang 1/2)..."):
        solution1 = solve_with_claude(ocr_text, iteration=1)
    
    # Selbst-Verifikation
    with st.spinner("Claude verifiziert eigene Lösung..."):
        verify_prompt = f"""Du bist ein ZWEITER Experte. Prüfe diese Lösung kritisch:

AUFGABE:
{ocr_text}

ZU PRÜFENDE LÖSUNG:
{solution1}

WICHTIG: Bei Homogenitätsprüfung - f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β!

Gib die FINALE KORREKTE Lösung im Format:
Aufgabe [Nr]: [Antwort]
Begründung: [Erklärung]"""

        response = claude_client.messages.create(
            model="claude-4-opus-20250514",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": verify_prompt}]
        )
        
        solution2 = response.content[0].text
    
    return solution2

# --- Multi Model Consensus ---
def achieve_consensus_multi(ocr_text, gpt_model):
    """Consensus zwischen Claude und GPT"""
    st.info(f"🤝 Verwende Consensus-System: Claude + {gpt_model}")
    
    # Erste Lösungen
    with st.spinner("Claude löst..."):
        claude_solution = solve_with_claude(ocr_text)
    
    with st.spinner(f"{gpt_model} löst..."):
        gpt_prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

KRITISCHE REGELN:
- Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Wenn nur α + β gegeben ist (ohne α = β), ist die Funktion NICHT homogen

ANALYSIERE:
{ocr_text}

FORMAT:
Aufgabe [Nr]: [Antwort]
Begründung: [Erklärung auf Deutsch]"""

        response = openai_client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "Du bist ein präziser Mathematik-Experte. Bei Homogenität: Eine Funktion ist NUR homogen wenn alle Bedingungen erfüllt sind."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        gpt_solution = response.choices[0].message.content
    
    # Debug Info
    with st.expander("🔍 Debug: Rohe Antworten"):
        st.markdown("**Claude:**")
        st.code(claude_solution)
        st.markdown("**GPT:**")
        st.code(gpt_solution)
    
    # Vergleiche
    claude_answers = extract_answers(claude_solution)
    gpt_answers = extract_answers(gpt_solution)
    
    # Debug Info
    st.write("Extrahierte Antworten:")
    st.write(f"Claude: {claude_answers}")
    st.write(f"GPT: {gpt_answers}")
    
    consensus = True
    all_tasks = set(claude_answers.keys()) | set(gpt_answers.keys())
    
    for task in sorted(all_tasks):
        claude_ans = claude_answers.get(task, "?")
        gpt_ans = gpt_answers.get(task, "?")
        
        if claude_ans != gpt_ans:
            consensus = False
            st.warning(f"Diskrepanz bei {task}: Claude={claude_ans}, GPT={gpt_ans}")
        else:
            st.success(f"✅ Einigkeit bei {task}: {claude_ans}")
    
    if consensus and claude_answers:  # Prüfe ob überhaupt Antworten gefunden wurden
        st.success("✅ Modelle sind sich einig!")
        return claude_solution
    else:
        st.error("❌ Modelle sind sich uneinig - verwende Claude mit Verifikation")
        return solve_single_model(ocr_text)

# --- UI ---
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
    
    # OCR
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    with st.spinner("📖 Lese Text..."):
        ocr_text = extract_text_with_gemini(image, file_hash)
    
    # Debug OCR
    with st.expander("🔍 OCR-Text"):
        st.code(ocr_text)
    
    # Solve Button
    if st.button("🧮 Aufgaben lösen", type="primary"):
        st.markdown("---")
        
        try:
            if GPT_MODEL:
                # Multi-Model Consensus
                final_solution = achieve_consensus_multi(ocr_text, GPT_MODEL)
            else:
                # Single Model mit Verifikation
                final_solution = solve_single_model(ocr_text)
            
            # Zeige finale Lösung
            st.markdown("---")
            st.markdown("### 📊 Lösung:")
            
            # Parse und zeige Lösung
            if final_solution:
                lines = final_solution.split('\n')
                for line in lines:
                    if line.strip():
                        if any(x in line for x in ['Aufgabe', 'AUFGABE', 'Task']):
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                st.markdown(f"### {parts[0].strip()}: **{parts[1].strip()}**")
                            else:
                                st.markdown(f"### {line}")
                        elif 'Begründung:' in line or 'BEGRÜNDUNG:' in line:
                            st.markdown(f"*{line.strip()}*")
                        elif line.strip() and not line.startswith('---'):
                            st.markdown(line.strip())
                            
        except Exception as e:
            st.error(f"Fehler: {str(e)}")
            logger.error(f"Processing error: {str(e)}", exc_info=True)

# Footer
model_info = f"Claude 4 Opus + {GPT_MODEL}" if GPT_MODEL else "Claude 4 Opus (Single Model)"
st.markdown("---")
st.caption(f"Koifox-Bot | {model_info}")
