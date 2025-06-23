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

# --- Verbesserte OCR mit detaillierter Graphenbeschreibung ---
@st.cache_data(ttl=3600)
def extract_text_with_gemini_improved(_image, file_hash):
    """Extrahiert KOMPLETTEN Text und beschreibt Graphen detailliert"""
    try:
        logger.info(f"Starting GEMINI OCR for file hash: {file_hash}")
        
        # Bildskalierung beibehalten
        max_size = 3000
        if _image.width > max_size or _image.height > max_size:
            _image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            st.sidebar.warning(f"Bild wurde auf {max_size}px skaliert")
        
        # Strukturierter OCR Prompt
        response = vision_model.generate_content(
            [
                """Extract ALL content from this exam image in a structured format, including text, formulas, and visual elements like graphs, diagrams, or tables.

                IMPORTANT:
                - Read ALL text from top to bottom, including:
                  - Question numbers (e.g., "Aufgabe 45 (5 RP)")
                  - All questions, formulas, values, and answer options (A, B, C, D, E) with their complete text
                  - Mathematical symbols and formulas exactly as shown (e.g., f(r₁,r₂) = (r₁^α + r₂^β)^γ)
                - For graphs, diagrams, or tables:
                  - Describe ALL visual elements in detail, including:
                    - Axes labels (e.g., x-axis: "Quantity", y-axis: "Cost")
                    - ALL data points or values shown (e.g., "Point at (10, 500)")
                    - Curve shapes or trends (e.g., "Linear increasing curve")
                    - ALL annotations or labels in the graph
                    - If a table is present, extract it as a structured table (e.g., rows and columns)
                - Structure the output as follows:
                  - Aufgabe [Nummer]: [Fragetext]
                    - Option A: [Text]
                    - Option B: [Text]
                    - ...
                    - Formeln: [z. B. f(r₁,r₂) = (r₁^α + r₂^β)^γ]
                    - Graph/Table: [Detailed description, including ALL data points]
                - DO NOT summarize or skip any part
                - DO NOT solve anything
                - Ensure numerical parameters (e.g., a = 450, b = 22.5) are explicitly listed if present
                
                Start from the very top and continue to the very bottom of the image.""",
                _image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 10000
            }
        )
        
        ocr_result = response.text.strip()
        
        # Prüfe ob genug Text extrahiert wurde
        if len(ocr_result) < 500:
            st.warning(f"⚠️ Nur {len(ocr_result)} Zeichen extrahiert - möglicherweise unvollständig! Überprüfe, ob Graphen/Diagramme erkannt wurden.")
        
        # Prüfe auf Graphenbeschreibungen
        if "Graph:" not in ocr_result and "Table:" not in ocr_result:
            st.warning("⚠️ Keine Graphen oder Tabellen im OCR-Text gefunden. Möglicherweise wurden visuelle Elemente nicht erkannt.")
        
        logger.info(f"GEMINI OCR completed: {len(ocr_result)} characters")
        return ocr_result
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        raise e

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
            ocr_text = extract_text_with_gemini_improved(image, file_hash)
        
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
