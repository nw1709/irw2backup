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
st.markdown("*Hi der Kai ❤️*")

# --- API Clients ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")
claude_client = Anthropic(api_key=st.secrets["claude_key"])
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- OCR mit Caching ---
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

# --- KORRIGIERTE Aufgaben-Extraktion ---
def extract_tasks_from_ocr(ocr_text):
    """Extrahiert NUR echte Aufgabennummern - keine zufälligen Zahlen!"""
    # NUR nach expliziten Aufgaben-Markierungen suchen
    task_patterns = [
        r'Aufgabe\s+(\d+)',           # "Aufgabe 7"
        r'Task\s+(\d+)',              # "Task 7"
        r'Frage\s+(\d+)',             # "Frage 7"
        r'Question\s+(\d+)',          # "Question 7"
        r'Problem\s+(\d+)',           # "Problem 7"
        r'Exercise\s+(\d+)',          # "Exercise 7"
        r'Übung\s+(\d+)',             # "Übung 7"
        # Sehr restriktiv: Nur wenn Zeile mit Nummer beginnt UND danach spezifische Keywords
        r'^(\d+)\s*[.:)]\s*(?:Aufgabe|Task|Frage)',
    ]
    
    tasks = set()
    
    # Zeile für Zeile prüfen für das letzte Pattern
    lines = ocr_text.split('\n')
    for line in lines:
        line = line.strip()
        # Prüfe ob Zeile mit Nummer + Aufgaben-Keyword beginnt
        if re.match(r'^(\d+)\s*[.:)]\s*(?:Aufgabe|Task|Frage)', line, re.IGNORECASE):
            match = re.match(r'^(\d+)', line)
            if match:
                tasks.add(match.group(1))
    
    # Normale Pattern-Suche
    for pattern in task_patterns[:-1]:  # Alle außer dem letzten
        matches = re.findall(pattern, ocr_text, re.IGNORECASE)
        tasks.update(matches)
    
    # Sortiere numerisch und gib zurück
    task_numbers = sorted([int(t) for t in tasks])
    task_strings = [str(t) for t in task_numbers]
    
    logger.info(f"Found actual tasks in OCR: {task_strings}")
    
    return task_strings

# --- ROBUSTE Antwort-Extraktion ---
def extract_structured_answers(solution_text, valid_tasks):
    """Extrahiert Antworten mit besserer Fehlerbehandlung"""
    result = {}
    
    if not solution_text:
        return result
        
    lines = solution_text.split('\n')
    
    # Patterns für Aufgabenerkennung
    task_patterns = [
        r'Aufgabe\s*(\d+)\s*:\s*(.+)',
        r'Task\s*(\d+)\s*:\s*(.+)',
        r'Frage\s*(\d+)\s*:\s*(.+)',
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
                task_num = task_match.group(1)
                # Nur valide Aufgaben verarbeiten
                if task_num in valid_tasks:
                    # Speichere vorherige Aufgabe
                    if current_task and current_answer:
                        result[f"Aufgabe {current_task}"] = {
                            'answer': current_answer,
                            'reasoning': ' '.join(current_reasoning).strip()
                        }
                    
                    current_task = task_num
                    raw_answer = task_match.group(2).strip()
                    current_answer = normalize_answer(raw_answer)
                    current_reasoning = []
                    task_found = True
                break
        
        if not task_found and current_task:
            if line.startswith(('Begründung:', 'Erklärung:', 'Reasoning:')):
                reasoning_text = re.sub(r'^(Begründung|Erklärung|Reasoning):\s*', '', line, flags=re.IGNORECASE)
                current_reasoning.append(reasoning_text)
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
    """Normalisiert Antworten"""
    answer = raw_answer.strip()
    
    # Multiple-Choice
    if re.match(r'^[A-E](\s*[,;]\s*[A-E])*$', answer, re.IGNORECASE):
        letters = re.findall(r'[A-E]', answer.upper())
        return ''.join(sorted(set(letters)))
    
    # Numerisch
    numeric_match = re.search(r'[\d,.\-]+', answer)
    if numeric_match:
        return numeric_match.group(0)
    
    return answer

# --- PRÄZISER PROMPT ---
def create_precise_prompt(ocr_text, tasks):
    if not tasks:
        return ""
        
    task_str = tasks[0] if len(tasks) == 1 else f"{', '.join(tasks[:-1])} und {tasks[-1]}"
    
    return f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

AUFGABENTEXT (OCR):
{ocr_text}

INSTRUKTIONEN:
- Es gibt genau {len(tasks)} Aufgabe(n) zu lösen: Aufgabe {task_str}
- Löse NUR diese Aufgabe(n), keine anderen!
- Bei Multiple-Choice: Wähle die richtigen Buchstaben (A-E)
- Bei Berechnungen: Zeige alle Schritte

AUSGABEFORMAT - EXAKT SO:
Aufgabe {tasks[0]}: [Deine Antwort hier]
Begründung: [Kurze Erklärung]

{f'Aufgabe {tasks[1]}: [Deine Antwort hier]' if len(tasks) > 1 else ''}
{f'Begründung: [Kurze Erklärung]' if len(tasks) > 1 else ''}

KEINE ANDEREN AUFGABEN ERWÄHNEN ODER LÖSEN!"""

# --- Claude Solver (vereinfacht) ---
def solve_with_claude(ocr_text, tasks):
    if not tasks:
        return ""
        
    prompt = create_precise_prompt(ocr_text, tasks)
    
    try:
        response = claude_client.messages.create(
            model="claude-4-1-opus-20250805",
            max_tokens=8000,
            temperature=0.1,
            system=f"Löse NUR Aufgabe(n) {', '.join(tasks)}. Keine anderen Nummern erwähnen!",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        
    except Exception as e:
        logger.error(f"Claude Error: {str(e)}")
        raise e

# --- GPT Solver (vereinfacht) ---
def solve_with_gpt(ocr_text, tasks):
    if not tasks:
        return ""
        
    prompt = create_precise_prompt(ocr_text, tasks)
    
    try:
        response = openai_client.chat.completions.create(
            model="o3",
            messages=[
                {
                    "role": "system", 
                    "content": f"Du bist Experte für Internes Rechnungswesen. Löse NUR Aufgabe(n) {', '.join(tasks)} und liefere eine detaillierte Begründung."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"GPT Error: {str(e)}")
        return None

# --- Antwortvergleich ---
def answers_are_equivalent(answer1, answer2):
    """Vergleicht zwei Antworten intelligent"""
    if answer1 == answer2:
        return True
    
    # Multiple-Choice
    if re.match(r'^[A-E]+$', answer1) and re.match(r'^[A-E]+$', answer2):
        return set(answer1) == set(answer2)
    
    # Numerisch
    try:
        num1 = float(answer1.replace(',', '.'))
        num2 = float(answer2.replace(',', '.'))
        relative_tolerance = 0.02
        absolute_tolerance = 0.01
        return abs(num1 - num2) <= max(absolute_tolerance, relative_tolerance * max(abs(num1), abs(num2)))
    except:
        pass
    
    return answer1.lower() == answer2.lower()

# --- Kreuzvalidierung ---
def cross_validation(ocr_text, tasks):
    """Einfache, robuste Kreuzvalidierung"""
    st.markdown("### 🔄 Kreuzvalidierung")
    
    # Claude
    with st.spinner("Claude Opus 4 analysiert..."):
        claude_solution = solve_with_claude(ocr_text, tasks)
    claude_data = extract_structured_answers(claude_solution, tasks)
    
    # GPT
    with st.spinner("GPT-o3 validiert..."):
        gpt_solution = solve_with_gpt(ocr_text, tasks)
        gpt_data = extract_structured_answers(gpt_solution, tasks) if gpt_solution else {}
    
    # Ergebnisse zusammenführen
    final_answers = {}
    
    for task in tasks:
        task_key = f"Aufgabe {task}"
        claude_ans = claude_data.get(task_key, {}).get('answer', '')
        gpt_ans = gpt_data.get(task_key, {}).get('answer', '')
        
        col1, col2, col3, col4 = st.columns([2, 3, 3, 1])
        with col1:
            st.write(f"**Aufgabe {task}:**")
        with col2:
            st.write(f"Claude: `{claude_ans}`" if claude_ans else "Claude: -")
        with col3:
            st.write(f"GPT: `{gpt_ans}`" if gpt_ans else "GPT: -")
        
        # Claude hat Priorität
        if claude_ans:
            final_answers[task_key] = claude_data[task_key]
            if gpt_ans and answers_are_equivalent(claude_ans, gpt_ans):
                with col4:
                    st.write("✅")
            else:
                with col4:
                    st.write("⚠️")
        elif gpt_ans:
            final_answers[task_key] = gpt_data[task_key]
            with col4:
                st.write("🔄")
        else:
            with col4:
                st.write("❌")
    
    return final_answers, claude_solution, gpt_solution

# --- HAUPTINTERFACE ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False)

# Cache Management
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Cache leeren"):
        st.cache_data.clear()
        st.rerun()

uploaded_file = st.file_uploader("**Klausuraufgabe hochladen...**", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        image = Image.open(uploaded_file)
        
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        
        # OCR
        with st.spinner("📖 Lese Text mit Gemini Flash 1.5..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
        
        # Aufgaben finden
        tasks = extract_tasks_from_ocr(ocr_text)
        
        if not tasks:
            st.error("❌ Keine Aufgaben im Bild gefunden!")
            if debug_mode:
                with st.expander("🔍 OCR-Text"):
                    st.code(ocr_text)
            st.stop()
        
        st.success(f"✅ Gefunden: Aufgabe {', '.join(tasks)}")
        
        # Debug
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis"):
                st.code(ocr_text)
        
        # Lösen
        if st.button("🧮 Aufgabe(n) lösen", type="primary"):
            st.markdown("---")
            
            # Kreuzvalidierung
            final_answers, claude_full, gpt_full = cross_validation(ocr_text, tasks)
            
            # Finale Ausgabe
            st.markdown("---")
            st.markdown("### 🎯 FINALE LÖSUNG")
            
            if final_answers:
                for task in tasks:
                    task_key = f"Aufgabe {task}"
                    if task_key in final_answers:
                        data = final_answers[task_key]
                        st.markdown(f"### {task_key}: **{data.get('answer', 'Keine Antwort')}**")
                        if data.get('reasoning'):
                            st.markdown(f"*{data['reasoning']}*")
                
                st.success(f"✅ {len(final_answers)} von {len(tasks)} Aufgaben gelöst")
            else:
                st.error("❌ Keine Lösungen generiert")
            
            # Debug
            if debug_mode:
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("Claude Rohausgabe"):
                        st.code(claude_full)
                with col2:
                    with st.expander("GPT Rohausgabe"):
                        st.code(gpt_full if gpt_full else "Keine Lösung")
                        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made by Fox & Koi-9 ❤️ | Gemini Flash 1.5 | Claude Opus 4 | GPT-o3")
