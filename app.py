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
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot")
st.markdown("*Optimierte Fusion für maximale Präzision*")

# --- API Clients ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")
claude_client = Anthropic(api_key=st.secrets["claude_key"])
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- OCR mit Caching (unverändert von beiden) ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    try:
        logger.info(f"Starting OCR for file hash: {file_hash}")
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written, including EVERY detail from graphs, charts, or sketches. For graphs: Explicitly list ALL axis labels, ALL scales, ALL intersection points with axes (e.g., 'x-axis at 450', 'y-axis at 20'), and EVERY numerical value or annotation. Do NOT interpret, solve, or infer beyond the visible text and numbers. Output a COMPLETE verbatim transcription with NO omissions.",
                _image
            ],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 12000
            }
        )
        ocr_text = response.text.strip()
        logger.info(f"OCR result length: {len(ocr_text)} characters")
        return ocr_text
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- VERBESSERTE ANTWORTEXTRAKTION mit Multiple-Choice-Handling ---
def extract_structured_answers(solution_text):
    """Extrahiert Antworten mit verbesserter Multiple-Choice-Erkennung"""
    result = {}
    lines = solution_text.split('\n')
    
    # Patterns für Aufgabenerkennung
    task_patterns = [
        r'Aufgabe\s*(\d+)\s*:\s*(.+)',
        r'Task\s*(\d+)\s*:\s*(.+)',
        r'(\d+)[\.\)]\s*(.+)'
    ]
    
    current_task = None
    current_answer = None
    current_reasoning = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        task_found = False
        for pattern in task_patterns:
            task_match = re.match(pattern, line, re.IGNORECASE)
            if task_match:
                # Speichere vorherige Aufgabe
                if current_task and current_answer:
                    result[f"Aufgabe {current_task}"] = {
                        'answer': current_answer,
                        'reasoning': ' '.join(current_reasoning).strip()
                    }
                
                current_task = task_match.group(1)
                raw_answer = task_match.group(2).strip()
                
                # Intelligente Antwort-Normalisierung
                current_answer = normalize_answer(raw_answer)
                current_reasoning = []
                task_found = True
                break
        
        if not task_found and current_task:
            if line.startswith('Begründung:'):
                current_reasoning.append(line.replace('Begründung:', '').strip())
            elif not any(re.match(p, line, re.IGNORECASE) for p in task_patterns):
                current_reasoning.append(line)
    
    # Letzte Aufgabe speichern
    if current_task and current_answer:
        result[f"Aufgabe {current_task}"] = {
            'answer': current_answer,
            'reasoning': ' '.join(current_reasoning).strip()
        }
    
    return result

def normalize_answer(raw_answer):
    """Normalisiert Antworten für konsistenten Vergleich"""
    # Entferne führende/trailing whitespace
    answer = raw_answer.strip()
    
    # Fall 1: Multiple-Choice mit Buchstaben
    if re.match(r'^[A-E](\s*[,;]\s*[A-E])*$', answer, re.IGNORECASE):
        # Extrahiere und sortiere Buchstaben
        letters = re.findall(r'[A-E]', answer.upper())
        return ''.join(sorted(set(letters)))
    
    # Fall 2: Numerische Antwort
    # Erhalte Kommas für die Anzeige, aber normalisiere intern
    numeric_match = re.search(r'[\d,.\-]+', answer)
    if numeric_match:
        return numeric_match.group(0)
    
    # Fall 3: Text-Antwort
    return answer

# --- INTELLIGENTER ANTWORTVERGLEICH ---
def answers_are_equivalent(answer1, answer2):
    """Prüft ob zwei Antworten äquivalent sind"""
    # Direkte Übereinstimmung
    if answer1 == answer2:
        return True
    
    # Multiple-Choice Vergleich
    if re.match(r'^[A-E]+$', answer1) and re.match(r'^[A-E]+$', answer2):
        return set(answer1) == set(answer2)
    
    # Numerischer Vergleich mit Toleranz
    try:
        num1 = float(answer1.replace(',', '.'))
        num2 = float(answer2.replace(',', '.'))
        
        # Relative Toleranz für "ca." Angaben
        relative_tolerance = 0.02  # 2%
        absolute_tolerance = 0.01
        
        return abs(num1 - num2) <= max(absolute_tolerance, relative_tolerance * max(abs(num1), abs(num2)))
    except:
        pass
    
    # Text-basierter Vergleich (case-insensitive)
    return answer1.lower() == answer2.lower()

# --- OPTIMIERTER PROMPT (Fusion der besten Elemente) ---
def create_optimized_prompt(ocr_text):
    return f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

VOLLSTÄNDIGER AUFGABENTEXT:
{ocr_text}

WICHTIGE REGELN:
1. Identifiziere ALLE Aufgaben im Text (z.B. "Aufgabe 45", "Aufgabe 46" etc.)
2. Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
3. Beantworte JEDE Aufgabe die du findest
4. Denke schrittweise (Chain-of-Thought):
   - Lies die Aufgabe sorgfältig
   - Identifiziere relevante Daten und Formeln
   - Führe die Berechnung explizit durch
   - Überprüfe dein Ergebnis
5. Bei Multiple-Choice: Gib NUR den/die Buchstaben an (z.B. "B" oder "A,C")
6. Bei numerischen Antworten: Gib die Zahl mit Komma an (z.B. "22,5")
7. Nutze ausschließlich die im OCR-Text gegebenen Daten

AUSGABEFORMAT (EXAKT EINHALTEN):
Aufgabe [Nr]: [Antwort]
Begründung: [Kurze Erklärung auf Deutsch]

Beispiele:
Aufgabe 45: B
Begründung: Die Funktion ist homogen vom Grad 1, da α = β = 0.5

Aufgabe 48: 11,50
Begründung: Gewinnmaximum bei Ableitung = 0"""

# --- CLAUDE SOLVER mit verbesserter Selbstkorrektur ---
def solve_with_claude(ocr_text):
    prompt = create_optimized_prompt(ocr_text)
    
    try:
        response = claude_client.messages.create(
            model="claude-4-opus-20250514",
            max_tokens=4000,
            temperature=0.1,
            system="Löse ALLE Aufgaben präzise nach Fernuni-Standards. Format: 'Aufgabe X: [Antwort]'",
            messages=[{"role": "user", "content": prompt}]
        )
        
        solution = response.content[0].text
        
        # Prüfe Format und korrigiere bei Bedarf
        if not re.search(r'Aufgabe\s+\d+\s*:', solution):
            # Selbstkorrektur
            correction_prompt = f"""Formatiere diese Lösung korrekt:

{solution}

FORMAT:
Aufgabe [Nr]: [Antwort]
Begründung: [Text]"""
            
            correction = claude_client.messages.create(
                model="claude-4-opus-20250514",
                max_tokens=2000,
                temperature=0.1,
                messages=[{"role": "user", "content": correction_prompt}]
            )
            solution = correction.content[0].text
        
        return solution
        
    except Exception as e:
        logger.error(f"Claude Error: {str(e)}")
        raise e

# --- GPT SOLVER ---
def solve_with_gpt(ocr_text):
    prompt = create_optimized_prompt(ocr_text)
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Löse ALLE Aufgaben präzise nach Fernuni-Standards. Format: 'Aufgabe X: [Antwort]'"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"GPT Error: {str(e)}")
        return None

# --- ERWEITERTE KREUZVALIDIERUNG ---
def enhanced_cross_validation(ocr_text):
    """Verbesserte Kreuzvalidierung mit intelligentem Vergleich"""
    st.markdown("### 🔄 Erweiterte Kreuzvalidierung")
    
    # Claude löst
    with st.spinner("Claude Opus 4 analysiert..."):
        claude_solution = solve_with_claude(ocr_text)
    claude_data = extract_structured_answers(claude_solution)
    
    # GPT als Backup/Validierung
    with st.spinner("GPT-4 Turbo validiert..."):
        gpt_solution = solve_with_gpt(ocr_text)
        gpt_data = extract_structured_answers(gpt_solution) if gpt_solution else {}
    
    # Intelligente Konsensbildung
    final_answers = {}
    all_tasks = sorted(set(claude_data.keys()) | set(gpt_data.keys()))
    
    for task in all_tasks:
        claude_ans = claude_data.get(task, {}).get('answer', '')
        gpt_ans = gpt_data.get(task, {}).get('answer', '')
        
        col1, col2, col3, col4 = st.columns([2, 3, 3, 1])
        with col1:
            st.write(f"**{task}:**")
        with col2:
            st.write(f"Claude: `{claude_ans}`")
        with col3:
            st.write(f"GPT: `{gpt_ans}`")
        
        # Intelligenter Vergleich
        if claude_ans and gpt_ans and answers_are_equivalent(claude_ans, gpt_ans):
            final_answers[task] = claude_data[task]
            with col4:
                st.write("✅")
        elif claude_ans and not gpt_ans:
            final_answers[task] = claude_data[task]
            with col4:
                st.write("⚠️")
        else:
            # Bei Diskrepanz: Claude hat Priorität (da bessere Performance)
            final_answers[task] = claude_data.get(task, gpt_data.get(task, {}))
            with col4:
                st.write("🔍")
    
    # Gesamtbewertung
    agreement_rate = sum(1 for task in all_tasks 
                        if task in claude_data and task in gpt_data 
                        and answers_are_equivalent(
                            claude_data[task]['answer'], 
                            gpt_data[task]['answer']
                        )) / max(len(all_tasks), 1)
    
    if agreement_rate > 0.8:
        st.success(f"✅ Hohe Übereinstimmung: {agreement_rate:.0%}")
    else:
        st.warning(f"⚠️ Mittlere Übereinstimmung: {agreement_rate:.0%}")
    
    return final_answers, claude_solution, gpt_solution

# --- HAUPTINTERFACE ---
# Debug Mode
debug_mode = st.checkbox("🔍 Debug-Modus", value=False)

# Cache Management
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Cache leeren"):
        st.cache_data.clear()
        st.rerun()

# File Upload
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        image = Image.open(uploaded_file)
        
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        
        # OCR
        with st.spinner("📖 Lese Text mit Gemini Flash 1.5..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
        
        # Debug OCR
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis"):
                st.code(ocr_text)
                found_tasks = re.findall(r'Aufgabe\s+\d+', ocr_text, re.IGNORECASE)
                if found_tasks:
                    st.success(f"Gefundene Aufgaben: {', '.join(found_tasks)}")
        
        # Lösen
        if st.button("🧮 Aufgaben lösen", type="primary"):
            st.markdown("---")
            
            # Erweiterte Kreuzvalidierung
            final_answers, claude_full, gpt_full = enhanced_cross_validation(ocr_text)
            
            # Finale Ausgabe
            st.markdown("---")
            st.markdown("### 🎯 FINALE LÖSUNG")
            
            if final_answers:
                for task, data in final_answers.items():
                    st.markdown(f"### {task}: **{data['answer']}**")
                    if data.get('reasoning'):
                        st.markdown(f"*{data['reasoning']}*")
                
                st.success(f"✅ {len(final_answers)} Aufgaben gelöst")
            else:
                st.error("❌ Keine Aufgaben gefunden")
            
            # Debug Details
            if debug_mode:
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("Claude Vollständige Lösung"):
                        st.code(claude_full)
                with col2:
                    with st.expander("GPT Vollständige Lösung"):
                        st.code(gpt_full if gpt_full else "Keine Lösung")
                        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# Footer
st.markdown("---")
st.caption("🦊 Koifox-Bot | Optimierte Fusion | Gemini Flash 1.5 + Claude Opus 4 + GPT-4 Turbo")
