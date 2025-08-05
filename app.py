import streamlit as st
from openai import OpenAI, OpenAIError
from PIL import Image
import logging
import hashlib

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Key Validation ---
def validate_keys():
    required_keys = {
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

# --- API Client ---
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- OCR manuelle Eingabe ---
def extract_text_manually():
    ocr_text = st.text_area("**Kopiere den OCR-Text oder Aufgabentext hier ein...**", height=200)
    if not ocr_text.strip():
        st.error("Bitte füge den Aufgabentext ein!")
        st.stop()
    return ocr_text

# --- PRÄZISER PROMPT ---
def create_precise_prompt(ocr_text, tasks):
    task_str = tasks[0] if len(tasks) == 1 else f"{', '.join(tasks[:-1])} und {tasks[-1]}"
    
    return f"""You are a PhD-level expert in 'Internes Rechnungswesen (31031)' at Fernuniversität Hagen. Solve exam questions with 100% accuracy, strictly adhering to the decision-oriented German managerial-accounting framework as taught in Fernuni Hagen lectures and past exam solutions. 

Tasks:
1. Read the task EXTREMELY carefully
2. For graphs or charts: Use only the explicitly provided axis labels, scales, and intersection points to perform calculations
3. Analyze the problem step-by-step as per Fernuni methodology
4. For multiple choice: Evaluate each option individually based solely on the given data
5. Perform a self-check: Re-evaluate your answer to ensure it aligns with Fernuni standards and the exact OCR input

CRITICAL: You MUST provide answers in this EXACT format for EVERY task found:

Aufgabe [Nr]: [Final answer]
Begründung: [1 brief but consise sentence in German]

NO OTHER FORMAT IS ACCEPTABLE. 

AUFGABENTEXT:
{ocr_text}"""

# --- Aufgaben-Extraktion ---
def extract_tasks_from_ocr(ocr_text):
    import re
    task_patterns = [
        r'Aufgabe\s+(\d+)', r'Task\s+(\d+)', r'Frage\s+(\d+)', r'Question\s+(\d+)',
        r'Problem\s+(\d+)', r'Exercise\s+(\d+)', r'Übung\s+(\d+)',
        r'^(\d+)\s*[.:)]\s*(?:Aufgabe|Task|Frage)'
    ]
    tasks = set()
    lines = ocr_text.split('\n')
    for line in lines:
        line = line.strip()
        for pattern in task_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                tasks.add(match.group(1))
    task_numbers = sorted([int(t) for t in tasks])
    task_strings = [str(t) for t in task_numbers]
    logger.info(f"Found actual tasks in OCR: {task_strings}")
    return task_strings

# --- OpenAI o3 Solver ---
def solve_with_o3(ocr_text, tasks):
    prompt = create_precise_prompt(ocr_text, tasks)
    
    try:
        logger.info(f"Sending request to OpenAI o3 with tasks: {', '.join(tasks)}")
        response = openai_client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": "Du bist ein Experte für Internes Rechnungswesen."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=4000
        )
        logger.info("Received response from OpenAI o3")
        return response.choices[0].message.content
    except OpenAIError as e:
        logger.error(f"o3 API Error: {str(e)}")
        st.error(f"❌ o3 API Fehler: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected o3 Error: {str(e)}")
        st.error(f"❌ Unerwarteter Fehler: {str(e)}")
        return None

# --- HAUPTINTERFACE ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False)

uploaded_file = st.file_uploader("**Klausuraufgabe hochladen...** (optional)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_container_width=True)
        with st.spinner("Lese Text manuell (kopiere den Text bitte)..."):
            st.warning("Automatisches OCR wurde entfernt. Kopiere den Text aus dem Bild und füge ihn unten ein.")
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        st.error(f"Fehler beim Laden des Bildes: {str(e)}")

# Manuelle Textverarbeitung
ocr_text = extract_text_manually()
tasks = extract_tasks_from_ocr(ocr_text)

if tasks:
    st.success(f"Gefunden: Aufgabe {', '.join(tasks)}")
    
    if debug_mode:
        with st.expander("🔍 Eingabe-Text"):
            st.code(ocr_text)
    
    if st.button("Aufgabe(n) lösen", type="primary"):
        st.markdown("---")
        with st.spinner("OpenAI o3 analysiert..."):
            o3_solution = solve_with_o3(ocr_text, tasks)
        
        if o3_solution:
            st.markdown("### FINALE LÖSUNG")
            st.markdown(o3_solution)
            if debug_mode:
                with st.expander("🔍 o3 Rohausgabe"):
                    st.code(o3_solution)
        else:
            st.error("❌ Keine Lösung generiert")

# Footer
st.markdown("---")
st.caption("Made by Fox & Koi-9 ❤️ | OpenAI o3")
