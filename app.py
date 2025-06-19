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
    
    for key, (prefix, name) in required_keys.items():
        if key not in st.secrets:
            missing.append(name)
        elif not st.secrets[key].startswith(prefix):
            missing.append(f"{name} (invalid)")
    
    if missing:
        st.error(f"Fehlende API Keys: {', '.join(missing)}")
        st.stop()

validate_keys()

# --- UI-Einstellungen ---
st.set_page_config(layout="centered", page_title="Koifox-Bot", page_icon="🦊")
st.title("🦊 Koifox-Bot")
st.markdown("*Multi-Model Consensus System für maximale Genauigkeit*")

# --- API Clients ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")
claude_client = Anthropic(api_key=st.secrets["claude_key"])
openai_client = OpenAI(api_key=st.secrets["openai_key"])

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
    
    for i, line in enumerate(lines):
        # Suche nach "Aufgabe X: Y" Pattern
        match = re.search(r'Aufgabe\s*(\d+)\s*:\s*([A-E,\s]+|\d+)', line, re.IGNORECASE)
        if match:
            task_num = match.group(1)
            answer = match.group(2).strip()
            # Normalisiere Multiple-Choice Antworten
            if any(letter in answer for letter in 'ABCDE'):
                answer = ''.join(sorted(c for c in answer.upper() if c in 'ABCDE'))
            answers[f"Aufgabe {task_num}"] = answer
    
    return answers

# --- Claude Solver ---
def solve_with_claude(ocr_text, previous_solution=None):
    """Claude löst die Aufgabe"""
    
    base_prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

WICHTIGE DEFINITIONEN:
- Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Homogenitätsgrad k bedeutet: f(λr) = λ^k·f(r) für ALLE λ
- "α + β = 3" impliziert NICHT α = β

ANALYSIERE DIESEN TEXT:
{ocr_text}

FORMAT:
Aufgabe [Nr]: [Antwort - nur Buchstaben oder Zahl]
Begründung: [Kurze Erklärung auf Deutsch]"""

    if previous_solution:
        base_prompt += f"\n\nEIN ANDERES MODELL HAT FOLGENDE LÖSUNG:\n{previous_solution}\n\nPRÜFE DIESE KRITISCH und gib DEINE EIGENE LÖSUNG."

    response = claude_client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=2000,
        temperature=0.1,
        messages=[{"role": "user", "content": base_prompt}]
    )
    
    return response.content[0].text

# --- GPT-4 Solver ---
def solve_with_gpt4(ocr_text, previous_solution=None):
    """GPT-4 löst die Aufgabe"""
    
    base_prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

WICHTIGE DEFINITIONEN:
- Eine Funktion f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
- Homogenitätsgrad k bedeutet: f(λr) = λ^k·f(r) für ALLE λ
- "α + β = 3" impliziert NICHT α = β

ANALYSIERE DIESEN TEXT:
{ocr_text}

FORMAT:
Aufgabe [Nr]: [Antwort - nur Buchstaben oder Zahl]
Begründung: [Kurze Erklärung auf Deutsch]"""

    if previous_solution:
        base_prompt += f"\n\nEIN ANDERES MODELL HAT FOLGENDE LÖSUNG:\n{previous_solution}\n\nPRÜFE DIESE KRITISCH und gib DEINE EIGENE LÖSUNG."

    response = openai_client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "Du bist ein präziser Mathematik-Experte. Mache keine unbegründeten Annahmen."},
            {"role": "user", "content": base_prompt}
        ],
        max_tokens=2000,
        temperature=0.1
    )
    
    return response.choices[0].message.content

# --- Consensus System ---
def achieve_consensus(ocr_text, max_iterations=3):
    """Iteratives Consensus-System zwischen Claude und GPT-4"""
    
    iteration_data = []
    
    for iteration in range(max_iterations):
        st.write(f"🔄 Iteration {iteration + 1}/{max_iterations}")
        
        # Erste Lösungen oder mit Feedback
        if iteration == 0:
            with st.spinner("Claude löst..."):
                claude_solution = solve_with_claude(ocr_text)
            with st.spinner("GPT-4 löst..."):
                gpt4_solution = solve_with_gpt4(ocr_text)
        else:
            # Mit gegenseitigem Feedback
            with st.spinner("Claude überprüft GPT-4's Lösung..."):
                claude_solution = solve_with_claude(ocr_text, gpt4_solution)
            with st.spinner("GPT-4 überprüft Claude's Lösung..."):
                gpt4_solution = solve_with_gpt4(ocr_text, claude_solution)
        
        # Extrahiere Antworten
        claude_answers = extract_answers(claude_solution)
        gpt4_answers = extract_answers(gpt4_solution)
        
        # Speichere Iteration
        iteration_data.append({
            'iteration': iteration + 1,
            'claude': {'full': claude_solution, 'answers': claude_answers},
            'gpt4': {'full': gpt4_solution, 'answers': gpt4_answers}
        })
        
        # Vergleiche Antworten
        all_tasks = set(claude_answers.keys()) | set(gpt4_answers.keys())
        consensus = True
        
        for task in all_tasks:
            claude_ans = claude_answers.get(task, "N/A")
            gpt4_ans = gpt4_answers.get(task, "N/A")
            
            if claude_ans != gpt4_ans:
                consensus = False
                st.warning(f"❌ Diskrepanz bei {task}: Claude={claude_ans}, GPT-4={gpt4_ans}")
            else:
                st.success(f"✅ Konsens bei {task}: {claude_ans}")
        
        if consensus:
            st.success(f"🎉 Konsens erreicht nach {iteration + 1} Iteration(en)!")
            return True, iteration_data
    
    st.error("❌ Kein Konsens nach maximalen Iterationen erreicht")
    return False, iteration_data

# --- UI ---
debug_mode = st.checkbox("🔍 Debug-Modus", value=False)
show_all_iterations = st.checkbox("📊 Alle Iterationen anzeigen", value=False)

# Datei-Upload
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
    
    if debug_mode:
        with st.expander("OCR-Text"):
            st.code(ocr_text)
    
    # Solve Button
    if st.button("🧮 Mit Multi-Model Consensus lösen", type="primary"):
        st.markdown("---")
        st.markdown("### 🤝 Consensus-Prozess:")
        
        # Consensus erreichen
        consensus_reached, iterations = achieve_consensus(ocr_text)
        
        # Ergebnisse anzeigen
        st.markdown("---")
        st.markdown("### 📊 Finale Lösung:")
        
        if consensus_reached:
            # Zeige finale übereinstimmende Lösung
            final_iteration = iterations[-1]
            final_answers = final_iteration['claude']['answers']
            
            for task, answer in sorted(final_answers.items()):
                st.markdown(f"### {task}: **{answer}**")
            
            # Zeige Begründungen
            with st.expander("Begründungen"):
                st.markdown("**Claude's Begründung:**")
                st.code(final_iteration['claude']['full'])
                st.markdown("**GPT-4's Begründung:**")
                st.code(final_iteration['gpt4']['full'])
        else:
            st.error("Keine eindeutige Lösung - bitte manuell prüfen!")
            
            # Zeige beide finalen Lösungen
            final_iteration = iterations[-1]
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Claude's finale Lösung:**")
                for task, answer in final_iteration['claude']['answers'].items():
                    st.markdown(f"{task}: **{answer}**")
                    
            with col2:
                st.markdown("**GPT-4's finale Lösung:**")
                for task, answer in final_iteration['gpt4']['answers'].items():
                    st.markdown(f"{task}: **{answer}**")
        
        # Zeige alle Iterationen wenn gewünscht
        if show_all_iterations:
            with st.expander("🔄 Alle Iterationen"):
                for iter_data in iterations:
                    st.markdown(f"**Iteration {iter_data['iteration']}:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("Claude:")
                        st.json(iter_data['claude']['answers'])
                    with col2:
                        st.markdown("GPT-4:")
                        st.json(iter_data['gpt4']['answers'])

# Footer
st.markdown("---")
st.caption("Multi-Model Consensus System | Claude 4 Opus + GPT-4 Turbo")
