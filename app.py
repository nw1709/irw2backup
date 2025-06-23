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
st.markdown("*Made with coffee, deep minimal and tiny gummy bears*")

# --- Cache Management ---
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Cache leeren", type="secondary", help="Löscht gespeicherte OCR-Ergebnisse"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --- API Clients ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")
claude_client = Anthropic(api_key=st.secrets["claude_key"])
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- GPT Model Detection ---
@st.cache_data
def get_available_gpt_model():
    test_models = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    for model in test_models:
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            return model
        except:
            continue
    return "gpt-3.5-turbo"

GPT_MODEL = get_available_gpt_model()

# --- OCR mit Caching ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini(_image, file_hash):
    try:
        logger.info(f"Starting OCR for file hash: {file_hash}")
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written. Include all question numbers, text, and answer options (A, B, C, D, E). Do NOT interpret or solve.",
                _image
            ],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 4000  # Reduziert für Kompatibilität
            }
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- ANTWORTEXTRAKTION ---
def extract_structured_answers(solution_text):
    """Extrahiert Antworten und Begründungen strukturiert"""
    result = {}
    lines = solution_text.split('\n')
    current_task = None
    current_answer = None
    current_reasoning = []
    
    for line in lines:
        line = line.strip()
        
        # Erkenne Aufgabe
        task_match = re.match(r'Aufgabe\s*(\d+)\s*:\s*(.+)', line, re.IGNORECASE)
        if task_match:
            # Speichere vorherige Aufgabe
            if current_task and current_answer:
                result[f"Aufgabe {current_task}"] = {
                    'answer': current_answer,
                    'reasoning': ' '.join(current_reasoning).strip()
                }
            
            # Neue Aufgabe
            current_task = task_match.group(1)
            raw_answer = task_match.group(2).strip()
            
            # Normalisiere Antwort (Buchstaben sortieren)
            if re.match(r'^[A-E,\s]+$', raw_answer):
                current_answer = ''.join(sorted(c for c in raw_answer.upper() if c in 'ABCDE'))
            else:
                current_answer = raw_answer
            
            current_reasoning = []
        
        # Erkenne Begründung
        elif line.startswith('Begründung:'):
            reasoning_text = line.replace('Begründung:', '').strip()
            if reasoning_text:
                current_reasoning = [reasoning_text]
        
        # Fortsetzung der Begründung
        elif current_task and line and not line.startswith('Aufgabe'):
            current_reasoning.append(line)
    
    # Letzte Aufgabe speichern
    if current_task and current_answer:
        result[f"Aufgabe {current_task}"] = {
            'answer': current_answer,
            'reasoning': ' '.join(current_reasoning).strip()
        }
    
    return result

# --- DEIN URSPRÜNGLICHER PROMPT (UNVERÄNDERT) ---
def create_base_prompt(ocr_text, cross_check_info=None):
    """Dein ursprünglicher, flexibler Prompt"""
    
    cross_check_section = ""
    if cross_check_info:
        cross_check_section = f"""
CROSS-VALIDATION CONTEXT:
Another expert has provided this analysis: {cross_check_info}
Please validate this against your own analysis and provide your definitive answer.
"""
    
    return f"""You are a highly qualified accounting expert with PhD-level 
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

{cross_check_section}

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

# --- OPTIMIERTE SOLVER MIT KORRIGIERTEN TOKEN-LIMITS ---
def solve_with_claude(ocr_text, cross_check_info=None):
    """Claude mit optimierten Parametern"""
    prompt = create_base_prompt(ocr_text, cross_check_info)
    
    try:
        response = claude_client.messages.create(
            model="claude-4-opus-20250514",
            max_tokens=4000,           # Sicher für Claude
            temperature=0.2,           # Absolut deterministisch
            top_p=0.1,                # Sehr fokussiert auf wahrscheinlichste Tokens
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude API Error: {str(e)}")
        raise e

def solve_with_gpt(ocr_text, cross_check_info=None):
    """GPT mit korrigierten Token-Limits"""
    prompt = create_base_prompt(ocr_text, cross_check_info)
    
    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,           # KORRIGIERT: Unter 4096 Limit
            temperature=0.2,           # Absolut deterministisch
            top_p=0.1,                # Sehr fokussiert
            frequency_penalty=0.0,     # Keine Wiederholungsbestrafung
            presence_penalty=0.0,      # Keine Präsenzbestrafung
            seed=42                    # Reproduzierbare Ergebnisse
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT API Error: {str(e)}")
        raise e

# --- INTELLIGENTE KREUZVALIDIERUNG ---
def cross_validation_consensus(ocr_text, max_rounds=3):
    """Intelligente Kreuzvalidierung ohne Voreingenommenheit"""
    
    st.markdown("### 🔄 Kreuzvalidierung")
    
    # Runde 1: Unabhängige Analyse
    with st.spinner("Runde 1: Unabhängige Expertenanalyse..."):
        try:
            claude_solution = solve_with_claude(ocr_text)
            gpt_solution = solve_with_gpt(ocr_text)
        except Exception as e:
            st.error(f"API-Fehler in Runde 1: {str(e)}")
            return False, None
    
    # Strukturiere Antworten
    claude_data = extract_structured_answers(claude_solution)
    gpt_data = extract_structured_answers(gpt_solution)
    
    # Vergleiche Antworten
    all_tasks = set(claude_data.keys()) | set(gpt_data.keys())
    
    for round_num in range(max_rounds):
        st.markdown(f"#### Runde {round_num + 1} Analyse:")
        
        differences = []
        agreement_count = 0
        
        for task in sorted(all_tasks):
            claude_ans = claude_data.get(task, {}).get('answer', '')
            gpt_ans = gpt_data.get(task, {}).get('answer', '')
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                st.write(f"**{task}:**")
            with col2:
                st.write(f"Claude: `{claude_ans}`")
            with col3:
                st.write(f"GPT: `{gpt_ans}`")
            with col4:
                if claude_ans == gpt_ans:
                    st.write("✅")
                    agreement_count += 1
                else:
                    st.write("❌")
                    differences.append({
                        'task': task,
                        'claude': claude_ans,
                        'gpt': gpt_ans,
                        'claude_reasoning': claude_data.get(task, {}).get('reasoning', ''),
                        'gpt_reasoning': gpt_data.get(task, {}).get('reasoning', '')
                    })
        
        consensus_rate = (agreement_count / len(all_tasks)) * 100 if all_tasks else 0
        st.metric("Konsens-Rate", f"{consensus_rate:.0f}%", f"{agreement_count}/{len(all_tasks)}")
        
        if not differences:
            st.success("✅ Vollständiger Konsens erreicht!")
            return True, claude_data
        
        if round_num < max_rounds - 1:  # Nicht in letzter Runde
            st.warning(f"⚠️ {len(differences)} Diskrepanzen gefunden - Kreuzvalidierung...")
            
            # Kreuzvalidierung ohne Voreingenommenheit
            with st.spinner(f"Kreuzvalidierung Runde {round_num + 2}..."):
                try:
                    # Sammle nur die Diskrepanzen als neutralen Context
                    discrepancy_summary = f"Diskrepanzen gefunden bei: {[d['task'] for d in differences]}. Bitte nochmalige sorgfältige Prüfung."
                    
                    claude_solution = solve_with_claude(ocr_text, discrepancy_summary)
                    gpt_solution = solve_with_gpt(ocr_text, discrepancy_summary)
                except Exception as e:
                    st.error(f"API-Fehler in Runde {round_num + 2}: {str(e)}")
                    return False, (claude_data, gpt_data)
            
            claude_data = extract_structured_answers(claude_solution)
            gpt_data = extract_structured_answers(gpt_solution)
    
    # Finale Bewertung
    st.error(f"❌ Nach {max_rounds} Runden noch {len(differences)} Diskrepanzen")
    return False, (claude_data, gpt_data)

# --- UI ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False)

uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"],
    key="file_uploader"
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        
        with st.spinner("Lese Text mit Gemini Flash..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
            
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis"):
                st.code(ocr_text)
                st.info(f"File Hash: {file_hash[:8]}...")
        
        if st.button("🎯 Lösung mit Kreuzvalidierung", type="primary"):
            
            consensus, result = cross_validation_consensus(ocr_text)
            
            st.markdown("---")
            st.markdown("### 🏆 FINALE LÖSUNG:")
            
            if consensus:
                for task, data in result.items():
                    st.markdown(f"### {task}: **{data['answer']}**")
                    if data['reasoning']:
                        st.markdown(f"*Begründung: {data['reasoning']}*")
                    st.markdown("")
                    
                st.success("✅ Lösung durch Kreuzvalidierung bestätigt!")
                
            else:
                if result:  # Sicherstellen, dass result nicht None ist
                    st.error("❌ Experten uneinig - Beide Lösungen anzeigen:")
                    
                    claude_final, gpt_final = result
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Claude Finale Antworten:**")
                        for task, data in claude_final.items():
                            st.markdown(f"**{task}: {data['answer']}**")
                            st.caption(data['reasoning'])
                    
                    with col2:
                        st.markdown(f"**{GPT_MODEL} Finale Antworten:**")
                        for task, data in gpt_final.items():
                            st.markdown(f"**{task}: {data['answer']}**")
                            st.caption(data['reasoning'])
                else:
                    st.error("❌ Schwerwiegender API-Fehler - bitte erneut versuchen")
            
            st.info("💡 OCR gecacht | Token-Limits optimiert | Intelligente Kreuzvalidierung")
                    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

st.markdown("---")
st.caption(f"🦊 Token-Optimized System | Claude-4 Opus + {GPT_MODEL} | Max Performance")
