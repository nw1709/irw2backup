import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image
import google.generativeai as genai
import logging
import hashlib
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

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
st.title("🦊 Koifox-Bot Fusion")
st.markdown("*Optimiertes OCR, Kreuzvalidierung & numerische Präzision*")

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# --- API Clients ---
claude_client = Anthropic(api_key=st.secrets["claude_key"])
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- SentenceTransformer für Antwortvergleich ---
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = load_sentence_transformer()

# --- Verbessertes OCR mit Caching ---
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

# --- ROBUSTE ANTWORTEXTRAKTION MIT DEUTSCHER NUMERIK ---
def extract_structured_answers(solution_text):
    result = {}
    lines = solution_text.split('\n')
    current_task = None
    current_answer = None
    current_reasoning = []
    
    task_patterns = [
        r'Aufgabe\s*(\d+)\s*:\s*([^\n]+)',  # Standard Format
        r'Task\s*(\d+)\s*:\s*(.+)',         # Englisch
        r'(\d+)[\.\)]\s*(.+)',              # Nummeriert mit Punkt/Klammer
        r'Lösung\s*(\d+)\s*:\s*(.+)'        # Alternative
    ]
    
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
                
                # Numerische Antworten mit Kommas erhalten
                if re.match(r'^[A-E,\s]+$', raw_answer):
                    current_answer = ''.join(sorted(c for c in raw_answer.upper() if c in 'ABCDE'))
                else:
                    # Behalte Kommazahlen im deutschen Format
                    current_answer = raw_answer
                
                current_reasoning = []
                task_found = True
                break
        
        if not task_found:
            if line.startswith('Begründung:'):
                reasoning_text = line.replace('Begründung:', '').strip()
                if reasoning_text:
                    current_reasoning = [reasoning_text]
            elif current_task and line:
                current_reasoning.append(line)
    
    # Letzte Aufgabe speichern
    if current_task and current_answer:
        result[f"Aufgabe {current_task}"] = {
            'answer': current_answer,
            'reasoning': ' '.join(current_reasoning).strip()
        }
    
    return result

# --- Numerischer Vergleich mit Toleranz für DE-Kommazahlen ---
def compare_numerical_answers(answer1, answer2, tolerance=0.01):
    """Vergleicht numerische Antworten mit Toleranz für Kommazahlen"""
    try:
        # Ersetze Komma durch Punkt für Float-Konvertierung
        num1 = float(answer1.replace(',', '.').strip())
        num2 = float(answer2.replace(',', '.').strip())
        return abs(num1 - num2) <= tolerance
    except (ValueError, TypeError):
        return False

# --- Hybrid-Solver: Kombiniert Stärken beider Ansätze ---
def solve_task_with_llms(ocr_text):
    """Löst Aufgaben mit beiden LLMs und führt intelligente Kreuzvalidierung durch"""
    
    # Basis-Prompt für beide Modelle
    base_prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen. Löse die folgenden Klausuraufgaben mit 100% Präzision gemäß den Fernuni-Standards:

{ocr_text}

WICHTIGE REGELN:
1. Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
2. Für Aufgabe 48: Verwende a = 450, b = 22.5, kv = 3, kf = 20 und G(p) = (p - 3)·(450 - 22.5·p) - 20
3. Numerische Antworten müssen auf zwei Dezimalstellen genau sein (z.B. 11.50 oder 11,50)
4. Bei Multiple-Choice: Analysiere jede Option und wähle die BESTE Antwort

AUSGABEFORMAT (STRIKT):
Aufgabe [Nummer]: [Antwort]
Begründung: [1-2 Sätze]
"""

    # Claude-Lösung
    try:
        claude_response = claude_client.messages.create(
            model="claude-4-opus-20250514",
            max_tokens=4000,
            temperature=0.1,
            top_p=0.1,
            messages=[{"role": "user", "content": base_prompt}]
        )
        claude_solution = claude_response.content[0].text
    except Exception as e:
        logger.error(f"Claude Error: {str(e)}")
        claude_solution = ""

    # GPT-Lösung
    try:
        gpt_response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": base_prompt}],
            max_tokens=4000,
            temperature=0.1
        )
        gpt_solution = gpt_response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT Error: {str(e)}")
        gpt_solution = ""

    return claude_solution, gpt_solution

# --- Intelligente Konsistenzprüfung ---
def intelligent_consensus_check(claude_data, gpt_data):
    """Prüft Konsistenz zwischen LLM-Antworten mit hybridem Ansatz"""
    consensus_results = {}
    all_tasks = set(claude_data.keys()) | set(gpt_data.keys())
    
    for task in all_tasks:
        c_ans = claude_data.get(task, {}).get('answer', '')
        g_ans = gpt_data.get(task, {}).get('answer', '')
        c_reason = claude_data.get(task, {}).get('reasoning', '')
        
        # 1. Direkter String-Vergleich (für MC-Antworten)
        if c_ans == g_ans:
            consensus_results[task] = {
                'answer': c_ans,
                'reasoning': c_reason,
                'status': 'full_consensus',
                'source': 'Claude+GPT'
            }
            continue
        
        # 2. Numerischer Vergleich mit Toleranz
        if compare_numerical_answers(c_ans, g_ans):
            consensus_results[task] = {
                'answer': c_ans,
                'reasoning': c_reason,
                'status': 'numerical_consensus',
                'source': 'Claude+GPT'
            }
            continue
        
        # 3. Semantische Ähnlichkeit der Begründungen
        try:
            c_reason_embed = sentence_model.encode(c_reason)
            g_reason_embed = sentence_model.encode(gpt_data.get(task, {}).get('reasoning', ''))
            similarity = util.cos_sim(c_reason_embed, g_reason_embed).item()
            
            if similarity > 0.7:
                consensus_results[task] = {
                    'answer': c_ans,
                    'reasoning': c_reason,
                    'status': 'semantic_consensus',
                    'source': 'Claude'
                }
                continue
        except Exception:
            pass
        
        # 4. Kein Konsens - Priorisiere Claude mit Warnung
        consensus_results[task] = {
            'answer': c_ans,
            'reasoning': f"{c_reason} [⚠️ Kein GPT-Konsens]",
            'status': 'no_consensus',
            'source': 'Claude'
        }
    
    return consensus_results

# --- Formatierte Lösungsanzeige ---
def display_solution(consensus_results):
    """Zeigt Lösungen mit Konsens-Status an"""
    for task, data in sorted(consensus_results.items(), key=lambda x: int(x[0].split()[-1])):
        status_icon = ""
        if data['status'] == 'full_consensus': status_icon = "✅"
        elif data['status'] == 'numerical_consensus': status_icon = "🔢"
        elif data['status'] == 'semantic_consensus': status_icon = "💡"
        else: status_icon = "⚠️"
        
        st.markdown(f"### {task}: **{data['answer']}** {status_icon}")
        st.markdown(f"*{data['reasoning']}*")
        st.markdown(f"<small>Quelle: {data['source']} | Status: {data['status']}</small>", unsafe_allow_html=True)
        st.markdown("---")

# --- UI-Hauptlogik ---
debug_mode = st.sidebar.checkbox("🔍 Debug-Modus", value=True)

uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    try:
        # OCR-Verarbeitung
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        
        with st.spinner("Lese Text mit Gemini Flash..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
        
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis"):
                st.code(ocr_text)

        # Lösungsgenerierung
        if st.button("🧮 Aufgaben lösen", type="primary"):
            with st.spinner("Claude & GPT lösen Aufgaben..."):
                claude_solution, gpt_solution = solve_task_with_llms(ocr_text)
            
            # Lösungen parsen
            claude_data = extract_structured_answers(claude_solution)
            gpt_data = extract_structured_answers(gpt_solution)
            
            # Kreuzvalidierung
            with st.spinner("Führe Kreuzvalidierung durch..."):
                consensus_results = intelligent_consensus_check(claude_data, gpt_data)
            
            # Ergebnisse anzeigen
            st.markdown("## 🎯 Lösungen")
            display_solution(consensus_results)
            
            # Debug-Informationen
            if debug_mode:
                with st.expander("💭 Claude-Lösung"):
                    st.code(claude_solution)
                with st.expander("💭 GPT-Lösung"):
                    st.code(gpt_solution)
                with st.expander("📊 Konsens-Daten"):
                    st.json(consensus_results)
    
    except Exception as e:
        st.error(f"❌ Fehler: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Koifox-Bot Fusion | Gemini Flash 1.5 + Claude Opus 4 + GPT-4 Turbo")
