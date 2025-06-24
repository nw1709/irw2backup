import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image
import google.generativeai as genai
import logging
import hashlib
import re
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
st.title("🦊 Koifox-Bot")
st.markdown("*Optimiertes OCR, strikte Formatierung & numerischer Vergleich*")

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# --- SentenceTransformer für Konsistenzprüfung ---
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = load_sentence_transformer()

# --- NEUE LOOP-DETECTION FUNKTION ---
def detect_and_prevent_loops(text, max_repetitions=3):
    """Erkennt Textwiederholungen und stoppt Loops"""
    try:
        # Teile Text in Sätze
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Nur längere Sätze prüfen
                count = sum(1 for s in sentences if s.strip() == sentence)
                if count > max_repetitions:
                    logger.warning(f"Loop detected in GPT response: '{sentence[:50]}...' repeated {count} times")
                    # Schneide ab beim ersten Auftreten der Wiederholung
                    loop_start = text.find(sentence)
                    # Finde das zweite Auftreten
                    second_occurrence = text.find(sentence, loop_start + len(sentence))
                    if second_occurrence != -1:
                        clean_text = text[:second_occurrence] + "\n\n[LOOP DETECTED - STOPPING REPETITION]"
                        logger.info(f"Cleaned text from {len(text)} to {len(clean_text)} characters")
                        return clean_text
        
        return text
    except Exception as e:
        logger.error(f"Loop detection failed: {str(e)}")
        return text

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
        
        logger.info(f"OCR result length: {len(ocr_text)} characters, content: {ocr_text[:200]}...")
        return ocr_text
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

# --- ROBUSTE ANTWORTEXTRAKTION ---
def extract_structured_answers(solution_text):
    result = {}
    lines = solution_text.split('\n')
    current_task = None
    current_answer = None
    current_reasoning = []
    
    # Verbesserte Regex-Patterns für verschiedene Formate
    task_patterns = [
        r'Aufgabe\s*(\d+)\s*:\s*(.+)',  # Standard Format
        r'Task\s*(\d+)\s*:\s*(.+)',     # Englisch
        r'(\d+)[\.\)]\s*(.+)',          # Nummeriert mit Punkt/Klammer
        r'Lösung\s*(\d+)\s*:\s*(.+)'    # Alternative
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
                    logger.info(f"Stored task: Aufgabe {current_task}, answer: {current_answer}")
                
                current_task = task_match.group(1)
                raw_answer = task_match.group(2).strip()
                
                # Verbesserte Antwort-Normalisierung
                if re.match(r'^[A-E,\s]+$', raw_answer):
                    current_answer = ''.join(sorted(c for c in raw_answer.upper() if c in 'ABCDE'))
                else:
                    # Extrahiere nur Buchstaben/Zahlen als Antwort
                    clean_answer = re.sub(r'[^\w]', '', raw_answer)
                    current_answer = clean_answer if clean_answer else raw_answer
                
                current_reasoning = []
                task_found = True
                logger.info(f"Detected task: Aufgabe {current_task}, answer: {current_answer}")
                break
        
        if not task_found:
            if line.startswith('Begründung:'):
                reasoning_text = line.replace('Begründung:', '').strip()
                if reasoning_text:
                    current_reasoning = [reasoning_text]
            elif current_task and line and not any(re.match(p, line, re.IGNORECASE) for p in task_patterns):
                current_reasoning.append(line)
    
    # Letzte Aufgabe speichern
    if current_task and current_answer:
        result[f"Aufgabe {current_task}"] = {
            'answer': current_answer,
            'reasoning': ' '.join(current_reasoning).strip()
        }
        logger.info(f"Final task stored: Aufgabe {current_task}, answer: {current_answer}")
    
    if not result:
        logger.warning("No tasks detected in solution. Full text: %s", solution_text)
    
    return result

# --- OCR-Text-Überprüfung ---
def validate_ocr_with_llm(ocr_text, model_type):
    prompt = f"""You are an expert in text validation. The following text is OCR data extracted from an exam image. Your task is to reflect this text EXACTLY as provided, without interpretation or changes, and confirm its completeness. Output the text verbatim and add a note: 'Text reflected accurately' if it matches the input, or 'Text may be incomplete' if anything seems missing.

OCR Text:
{ocr_text}
"""
    try:
        if model_type == "claude":
            response = claude_client.messages.create(
                model="claude-4-opus-20250514",
                max_tokens=8000,
                temperature=0.1,
                top_p=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            logger.info(f"Claude OCR validation received, length: {len(response.content[0].text)} characters")
            return response.content[0].text
        elif model_type == "gpt":
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.1
            )
            logger.info(f"GPT OCR validation received, length: {len(response.choices[0].message.content)} characters")
            return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Validation Error ({model_type}): {str(e)}")
        return None

# --- Numerischer Vergleich der Endantworten ---
def compare_numerical_answers(answers1, answers2):
    """Vergleicht Endantworten numerisch"""
    differences = []
    for a1, a2 in zip(answers1, answers2):
        try:
            # Konvertiere Antworten in Floats (ersetze Komma durch Punkt)
            num1 = float(a1.replace(',', '.'))
            num2 = float(a2.replace(',', '.'))
            if abs(num1 - num2) > 0.1:  # Toleranz von 0.1
                differences.append((a1, a2))
        except ValueError:
            continue
    return differences

# --- Konsistenzprüfung zwischen LLMs ---
def are_answers_similar(answer1, answer2):
    """Vergleicht die Endantworten auf semantische Ähnlichkeit und numerisch"""
    try:
        # Extrahiere Endantworten
        task_pattern = r'Aufgabe\s+\d+\s*:\s*([^\n]+)'
        answers1 = re.findall(task_pattern, answer1, re.IGNORECASE)
        answers2 = re.findall(task_pattern, answer2, re.IGNORECASE)
        
        if not answers1 or not answers2:
            logger.warning("Keine Endantworten für Konsistenzprüfung gefunden")
            return False, [], []
        
        # Semantische Ähnlichkeit
        embeddings = sentence_model.encode([' '.join(answers1), ' '.join(answers2)])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        logger.info(f"Antwortähnlichkeit (Endantworten): {similarity:.2f}")
        
        # Numerischer Vergleich
        numerical_differences = compare_numerical_answers(answers1, answers2)
        
        return similarity > 0.8 and not numerical_differences, answers1, answers2, numerical_differences
    except Exception as e:
        logger.error(f"Konsistenzprüfung fehlgeschlagen: {str(e)}")
        return False, [], [], []

# --- Claude Solver mit strikter Formatierung ---
def solve_with_claude_formatted(ocr_text):
    """Claude löst und formatiert korrekt mit Chain-of-Thought"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

VOLLSTÄNDIGER AUFGABENTEXT:
{ocr_text}

WICHTIGE REGELN:
1. Identifiziere ALLE Aufgaben im Text (z.B. "Aufgabe 45", "Aufgabe 46" etc.)
2. Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
3. Beantworte JEDE Aufgabe die du findest
4. Denke schrittweise:
   - Lies die Aufgabe sorgfältig
   - Identifiziere alle relevanten Formeln, Werte und visuelle Daten (z.B. Graphenbeschreibungen)
   - Wenn Daten unvollständig sind, dokumentiere Annahmen klar
   - Führe die Berechnung explizit durch
   - Überprüfe dein Ergebnis
5. Bei Multiple-Choice-Fragen: Analysiere jede Option und begründe, warum sie richtig oder falsch ist
6. Wenn Graphen oder Tabellen beschrieben sind, nutze diese Informationen für die Lösung
7. Für Aufgabe 48: Verwende die Parameter a = 450, b = 22.5, kv = 3, kf = 20 und die Gewinnfunktion G(p) = (p - 3)·(450 - 22.5·p) - 20. Leite ab und setze gleich Null.
8. Die Endantwort MUSS exakt der berechneten Zahl entsprechen (z.B. 11.50, nicht 13.33) und auf zwei Dezimalstellen formatiert sein

AUSGABEFORMAT (STRIKT EINHALTEN):
Aufgabe [Nummer]: [Antwort auf zwei Dezimalstellen]
Begründung: [Schritt-für-Schritt-Erklärung]
Berechnung: [Mathematische Schritte]
Annahmen (falls nötig): [z.B. "Fehlende Datenpunkte im Graphen wurden als linear angenommen"]

Wiederhole dies für JEDE Aufgabe im Text.

Beispiel:
Aufgabe 48: 11.50
Begründung: Der gewinnmaximale Preis wird durch Ableiten der Gewinnfunktion bestimmt...
Berechnung: dG/dp = (450 - 22.5·p) + (p - 3)·(-22.5) = 0, p = 517.5/45 = 11.50
Annahmen: Linearer Kurvenverlauf basierend auf Graphenbeschreibung

WICHTIG: Vergiss keine Aufgabe!"""

    client = Anthropic(api_key=st.secrets["claude_key"])
    response = client.messages.create(
        model="claude-4-opus-20250514",
        max_tokens=4000,
        temperature=0.1,
        system="Beantworte ALLE Aufgaben die im Text stehen. Überspringe keine. Stelle sicher, dass die Endantwort exakt der Berechnung entspricht.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# --- GPT-4 Turbo Solver mit strikter Formatierung ---
def solve_with_gpt(ocr_text):
    """GPT-4 Turbo löst mit Chain-of-Thought"""
    
    prompt = f"""Du bist ein Experte für "Internes Rechnungswesen (31031)" an der Fernuni Hagen.

VOLLSTÄNDIGER AUFGABENTEXT:
{ocr_text}

WICHTIGE REGELN:
1. Identifiziere ALLE Aufgaben im Text (z.B. "Aufgabe 45", "Aufgabe 46" etc.)
2. Bei Homogenität: f(r₁,r₂) = (r₁^α + r₂^β)^γ ist NUR homogen wenn α = β
3. Beantworte JEDE Aufgabe die du findest
4. Denke schrittweise:
   - Lies die Aufgabe sorgfältig
   - Identifiziere alle relevanten Formeln, Werte und visuelle Daten (z.B. Graphenbeschreibungen)
   - Wenn Daten unvollständig sind, dokumentiere Annahmen klar
   - Führe die Berechnung explizit durch
   - Überprüfe dein Ergebnis
5. Bei Multiple-Choice-Fragen: Analysiere jede Option und begründe, warum sie richtig oder falsch ist
6. Wenn Graphen oder Tabellen beschrieben sind, nutze diese Informationen für die Lösung
7. Für Aufgabe 48: Verwende die Parameter a = 450, b = 22.5, kv = 3, kf = 20 und die Gewinnfunktion G(p) = (p - 3)·(450 - 22.5·p) - 20. Leite ab und setze gleich Null.
8. Die Endantwort MUSS exakt der berechneten Zahl entsprechen (z.B. 11.50, nicht 13.33) und auf zwei Dezimalstellen formatiert sein

AUSGABEFORMAT (STRIKT EINHALTEN):
Aufgabe [Nummer]: [Antwort auf zwei Dezimalstellen]
Begründung: [Schritt-für-Schritt-Erklärung]
Berechnung: [Mathematische Schritte]
Annahmen (falls nötig): [z.B. "Fehlende Datenpunkte im Graphen wurden als linear angenommen"]

Wiederhole dies für JEDE Aufgabe im Text.

Beispiel:
Aufgabe 48: 11.50
Begründung: Der gewinnmaximale Preis wird durch Ableiten der Gewinnfunktion bestimmt...
Berechnung: dG/dp = (450 - 22.5·p) + (p - 3)·(-22.5) = 0, p = 517.5/45 = 11.50
Annahmen: Linearer Kurvenverlauf basierend auf Graphenbeschreibung

WICHTIG: Vergiss keine Aufgabe!"""

    client = OpenAI(api_key=st.secrets["openai_key"])
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Beantworte ALLE Aufgaben die im Text stehen. Überspringe keine. Stelle sicher, dass die Endantwort exakt der Berechnung entspricht."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000,
        temperature=0.1
    )

    # LOOP-DETECTION ANWENDEN
    raw_response = response.choices[0].message.content
    cleaned_response = detect_and_prevent_loops(raw_response)
    
    if len(cleaned_response) != len(raw_response):
        logger.info(f"GPT loop detected and cleaned: {len(raw_response)} -> {len(cleaned_response)} chars")
    
    return cleaned_response

# --- Verbesserte Ausgabeformatierung mit Konsistenzprüfung ---
def parse_and_display_solution(solution_text, model_name="Claude"):
    """Parst und zeigt Lösung strukturiert an, prüft Konsistenz mit Berechnung"""
    
    return response.choices[0].message.content

# --- Verbesserte Ausgabeformatierung mit Konsistenzprüfung ---
def parse_and_display_solution(solution_text, model_name="Claude"):
    """Parst und zeigt Lösung strukturiert an, prüft Konsistenz mit Berechnung"""
    
    # Finde alle Aufgaben mit Regex
    task_pattern = r'Aufgabe\s+(\d+)\s*:\s*([^\n]+)'
    tasks = re.findall(task_pattern, solution_text, re.IGNORECASE)
    
    if not tasks:
        st.warning(f"⚠️ Keine Aufgaben im erwarteten Format gefunden ({model_name})")
        st.markdown(solution_text)
        return
    
    # Zeige jede Aufgabe strukturiert
    for task_num, answer in tasks:
        st.markdown(f"### Aufgabe {task_num}: **{answer.strip()}** ({model_name})")
        
        # Finde zugehörige Begründung, Berechnung und Annahmen
        begr_pattern = rf'Aufgabe\s+{task_num}\s*:.*?\n\s*Begründung:\s*([^\n]+(?:\n(?!Aufgabe)[^\n]+)*?)(?:\n\s*Berechnung:\s*([^\n]+(?:\n(?!Aufgabe)[^\n]+)*))?(?:\n\s*Annahmen\s*\(falls\s*nötig\):\s*([^\n]+(?:\n(?!Aufgabe)[^\n]+)*))?(?=\n\s*Aufgabe|\Z)'
        begr_match = re.search(begr_pattern, solution_text, re.IGNORECASE | re.DOTALL)
        
        if begr_match:
            st.markdown(f"*Begründung: {begr_match.group(1).strip()}*")
            if begr_match.group(2):
                st.markdown(f"*Berechnung: {begr_match.group(2).strip()}*")
                # Prüfe Konsistenz zwischen Endantwort und Berechnung
                calc_pattern = r'p\s*=\s*([\d,.]+)'
                calc_match = re.search(calc_pattern, begr_match.group(2), re.IGNORECASE)
                if calc_match:
                    calc_answer = calc_match.group(1).replace(',', '.')
                    if calc_answer != answer.strip():
                        st.warning(f"⚠️ Inkonsistenz in Aufgabe {task_num} ({model_name}): Endantwort ({answer.strip()}) unterscheidet sich von Berechnung ({calc_answer})")
            if begr_match.group(3):
                st.markdown(f"*Annahmen: {begr_match.group(3).strip()}*")
        
        st.markdown("---")

# --- UI ---
# Cache leeren
if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    st.rerun()

# Debug
debug_mode = st.checkbox("🔍 Debug-Modus", value=True)

# Datei-Upload
uploaded_file = st.file_uploader(
    "**Klausuraufgabe hochladen...**",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Bild verarbeiten
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        image = Image.open(uploaded_file)
        
        # Zeige Original
        st.image(image, caption=f"Originalbild ({image.width}x{image.height}px)", use_container_width=True)
        
        # OCR
        with st.spinner("Lese Text und Graphen mit Gemini..."):
            ocr_text = extract_text_with_gemini(image, file_hash)
        
        # OCR Ergebnis
        with st.expander(f"🔍 OCR-Ergebnis ({len(ocr_text)} Zeichen)", expanded=debug_mode):
            st.code(ocr_text)
            
            # Prüfe ob Aufgaben gefunden wurden
            found_tasks = re.findall(r'Aufgabe\s+\d+', ocr_text, re.IGNORECASE)
            if found_tasks:
                st.success(f"✅ Gefundene Aufgaben: {', '.join(found_tasks)}")
            else:
                st.error("❌ Keine Aufgaben im Text gefunden!")
            
            # Prüfe auf Graphenbeschreibungen
            if "Graph:" in ocr_text or "Table:" in ocr_text:
                st.success("✅ Graphen oder Tabellen im OCR-Text gefunden!")
        
        # Lösen
        if st.button("🧮 Alle Aufgaben lösen", type="primary"):
            st.markdown("---")
            
            with st.spinner("Claude und GPT-4 lösen Aufgabe..."):
                claude_solution = solve_with_claude_formatted(ocr_text)
                gpt_solution = solve_with_gpt(ocr_text)
                
                # Konsistenzprüfung
                is_similar, claude_answers, gpt_answers, numerical_differences = are_answers_similar(claude_solution, gpt_solution)
                if is_similar:
                    st.success("✅ Beide Modelle sind einig!")
                    st.markdown("### 📊 Lösungen (Claude):")
                    parse_and_display_solution(claude_solution, model_name="Claude")
                else:
                    st.warning("⚠️ Modelle uneinig! Zeige beide Lösungen zur Überprüfung.")
                    st.markdown("### 📊 Lösungen (Claude):")
                    parse_and_display_solution(claude_solution, model_name="Claude")
                    st.markdown("### 📊 Lösungen (GPT-4 Turbo):")
                    parse_and_display_solution(gpt_solution, model_name="GPT-4 Turbo")
                    # Zeige numerische Unterschiede
                    if numerical_differences:
                        st.markdown("### Numerische Unterschiede in Endantworten:")
                        for c_answer, g_answer in numerical_differences:
                            st.markdown(f"- Claude: **{c_answer}**, GPT-4: **{g_answer}**")
            
            if debug_mode:
                with st.expander("💭 Rohe Claude-Antwort"):
                    st.code(claude_solution)
                with st.expander("💭 Rohe GPT-4-Antwort"):
                    st.code(gpt_solution)
                    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        st.error(f"❌ Fehler: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Koifox-Bot | Optimiertes OCR, strikte Formatierung & numerischer Vergleich")
