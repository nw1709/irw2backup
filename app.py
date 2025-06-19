import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image
import google.generativeai as genai
import logging
import hashlib
import re
import io

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- UI-Einstellungen ---
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot")
st.markdown("*Multi-Model Consensus System mit optimierter OCR*")

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

# --- VERBESSERTE GEMINI OCR ---
@st.cache_data(ttl=3600, show_spinner=False)
def extract_text_with_gemini(_image_bytes, file_hash):
    """Extrahiert KOMPLETTEN Text aus Bild mit Gemini"""
    try:
        logger.info(f"Starting Gemini OCR for hash: {file_hash[:8]}...")
        
        # Image aus bytes laden
        image = Image.open(io.BytesIO(_image_bytes))
        
        # Bild nicht zu stark verkleinern für bessere OCR
        max_size = 4096
        if max(image.width, image.height) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Verbesserter OCR Prompt
        ocr_prompt = """
        Extrahiere WIRKLICH ALLEN Text aus diesem Prüfungsbild:
        
        1. Beginne GANZ OBEN und lies bis GANZ UNTEN
        2. Erfasse JEDE ZEILE Text
        3. Inkludiere:
           - Alle Aufgabennummern (z.B. "Aufgabe 45 (5 RP)")
           - Komplette Aufgabentexte
           - ALLE Formeln und mathematische Ausdrücke
           - ALLE Antwortoptionen (A, B, C, D, E) mit vollem Text
           - Alle Zahlen, Werte, Parameter
        4. Behalte die exakte Formatierung bei
        5. Überspringe NICHTS
        6. Interpretiere oder löse NICHTS
        
        Gib den KOMPLETTEN Text wieder, vom ersten bis zum letzten Zeichen!
        """
        
        # OCR mit höheren Limits
        response = vision_model.generate_content(
            [ocr_prompt, image],
            generation_config={
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 16384  # Maximale Token!
            }
        )
        
        ocr_text = response.text.strip()
        logger.info(f"Gemini OCR completed: {len(ocr_text)} characters")
        
        # Validierung
        if len(ocr_text) < 500:
            logger.warning(f"OCR möglicherweise unvollständig: nur {len(ocr_text)} Zeichen")
            # Zweiter Versuch mit anderem Prompt
            response2 = vision_model.generate_content(
                ["Read and transcribe EVERY SINGLE word from this exam image. Start from the very top and continue to the very bottom. Include all text, numbers, formulas, and answer options.", image],
                generation_config={"temperature": 0, "max_output_tokens": 16384}
            )
            if len(response2.text) > len(ocr_text):
                ocr_text = response2.text.strip()
                logger.info(f"Zweiter OCR-Versuch erfolgreicher: {len(ocr_text)} Zeichen")
        
        return ocr_text
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- Lösungsextraktion ---
def extract_answers(solution_text):
    """Extrahiert strukturierte Antworten aus Lösungstext"""
    answers = {}
    
    # Verschiedene Patterns probieren
    patterns = [
        r'Aufgabe\s*(\d+)\s*:\s*([A-E,\s]+|\d+[\d,\.]*)',
        r'Task\s*(\d+)\s*:\s*([A-E,\s]+|\d+[\d,\.]*)',
        r'(\d+)\.\s*([A-E,\s]+|\d+[\d,\.]*)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, solution_text, re.IGNORECASE | re.MULTILINE)
        for task_num, answer in matches:
            answer = answer.strip()
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
- Beantworte ALLE Aufgaben die du im Text findest

ANALYSIERE DIESEN TEXT:
{ocr_text}

FORMAT für JEDE gefundene Aufgabe:
Aufgabe [Nr]: [Antwort]
Begründung: [Kurze Erklärung auf Deutsch]

WICHTIG: Überspringe keine Aufgabe!"""

    response = claude_client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=3000,
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
- Beantworte ALLE Aufgaben die du im Text findest

ANALYSIERE:
{ocr_text}

FORMAT für JEDE gefundene Aufgabe:
Aufgabe [Nr]: [Antwort]
Begründung: [Kurze Erklärung auf Deutsch]"""

    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "Du bist ein präziser Mathematik-Experte. Beantworte ALLE Aufgaben im Text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
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
    with st.expander("🔍 Debug: Rohe Antworten"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Claude:**")
            st.code(claude_solution)
        with col2:
            st.write(f"**{GPT_MODEL}:**")
            st.code(gpt_solution)
    
    with st.expander("🔍 Extrahierte Antworten"):
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
    discrepancies = []
    
    for task in sorted(all_tasks):
        claude_ans = claude_answers.get(task, "?")
        gpt_ans = gpt_answers.get(task, "?")
        
        if claude_ans != gpt_ans:
            consensus = False
            discrepancies.append(f"{task}: Claude={claude_ans}, GPT={gpt_ans}")
            st.warning(f"❌ Diskrepanz bei {task}: Claude={claude_ans}, GPT={gpt_ans}")
        else:
            st.success(f"✅ Einigkeit bei {task}: {claude_ans}")
    
    if consensus and claude_answers:
        st.success("🎉 Modelle sind sich einig!")
        return claude_solution
    else:
        st.info("🔄 Verwende Claude mit Verifikation")
        # Selbst-Verifikation mit Diskrepanz-Info
        verify_prompt = (
            "Prüfe diese Lösung kritisch:\n\n"
            f"AUFGABE:\n{ocr_text}\n\n"
            f"LÖSUNG:\n{claude_solution}\n\n"
            f"DISKREPANZEN MIT ANDEREM MODELL:\n" + "\n".join(discrepancies) + "\n\n"
            "Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β!\n\n"
            "Gib die FINALE KORREKTE Lösung im gleichen Format."
        )

        response = claude_client.messages.create(
            model="claude-4-opus-20250514",
            max_tokens=3000,
            temperature=0.2,
            messages=[{"role": "user", "content": verify_prompt}]
        )
        
        return response.content[0].text

# --- MAIN UI ---
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"],
    help="Lade ein klares Bild der Klausuraufgabe hoch"
)

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Hochgeladene Klausuraufgabe ({image.width}x{image.height}px)", use_container_width=True)
    
    # Datei-Info
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    st.info(f"📄 Datei: {uploaded_file.name} | Größe: {len(file_bytes)/1024:.1f} KB | Hash: {file_hash[:8]}...")
    
    # GEMINI OCR mit Status
    with st.spinner("📖 Lese Text mit Gemini Flash (kann 10-20 Sekunden dauern)..."):
        try:
            ocr_text = extract_text_with_gemini(file_bytes, file_hash)
            
            # OCR Erfolgsmeldung
            char_count = len(ocr_text)
            if char_count < 500:
                st.warning(f"⚠️ Nur {char_count} Zeichen extrahiert - möglicherweise unvollständig!")
            else:
                st.success(f"✅ OCR erfolgreich: {char_count} Zeichen extrahiert")
                
        except Exception as e:
            st.error(f"❌ OCR Fehler: {str(e)}")
            st.stop()
    
    # OCR Text anzeigen
    with st.expander(f"📝 OCR-Text (Gemini) - {len(ocr_text)} Zeichen"):
        st.code(ocr_text)
        # Aufgaben-Erkennung
        found_tasks = re.findall(r'Aufgabe\s+(\d+)', ocr_text, re.IGNORECASE)
        if found_tasks:
            st.info(f"Erkannte Aufgaben: {', '.join(set(found_tasks))}")
    
    # Solve Button
    if st.button("🧮 Mit Multi-Model Consensus lösen", type="primary", disabled=(len(ocr_text) < 100)):
        st.markdown("---")
        st.markdown("### 🤝 Consensus-Prozess:")
        
        try:
            # Consensus erreichen
            final_solution = achieve_consensus_multi(ocr_text)
            
            # Ergebnisse anzeigen
            st.markdown("---")
            st.markdown("### 📊 Finale Lösung:")
            
            lines = final_solution.split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('Aufgabe'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            st.markdown(f"### {parts[0]}: **{parts[1].strip()}**")
                    elif line.startswith('Begründung:'):
                        st.markdown(f"*{line}*")
                    elif not line.startswith('---'):
                        st.markdown(line)
                        
        except Exception as e:
            st.error(f"Fehler beim Lösen: {str(e)}")
            logger.error(f"Processing error: {str(e)}", exc_info=True)

# Footer
st.markdown("---")
st.caption(f"Multi-Model Consensus System | Gemini 1.5 Flash OCR + Claude 4 Opus + {GPT_MODEL}")
