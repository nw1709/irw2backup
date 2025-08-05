import streamlit as st
from openai import OpenAI, OpenAIError
from PIL import Image
import logging
import io

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
st.markdown("*Hi der Kai ❤️*")

# --- API Client ---
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- OpenAI o3 Solver mit Bildverarbeitung ---
def solve_with_o3(image):
    try:
        logger.info("Sending image to OpenAI o3 for processing")
        # Konvertiere Bild in Bytes für die API
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()

        response = openai_client.chat.completions.create(
            model="o3",
            messages=[
                {
                    "role": "system",
                    "content": """You are a PhD-level expert in 'Internes Rechnungswesen (31031)' at Fernuniversität Hagen. Solve exam questions with 100% accuracy, strictly adhering to the decision-oriented German managerial-accounting framework as taught in Fernuni Hagen lectures and past exam solutions. 

Tasks:
1. Read the task EXTREMELY carefully
2. For graphs or charts: Use only the explicitly provided axis labels, scales, and intersection points to perform calculations
3. Analyze the problem step-by-step as per Fernuni methodology
4. For multiple choice: Evaluate each option individually based solely on the given data
5. Perform a self-check: Re-evaluate your answer to ensure it aligns with Fernuni standards and the exact OCR input

CRITICAL: You MUST provide answers in this EXACT format for EVERY task found:

Aufgabe [Nr]: [Final answer]
Begründung: [1 brief but consise sentence in German]

NO OTHER FORMAT IS ACCEPTABLE."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from the provided exam image EXACTLY as written, including every detail from graphs, charts, or sketches. For graphs: Explicitly list ALL axis labels, ALL scales, ALL intersection points with axes (e.g., 'x-axis at 450', 'y-axis at 20'), and EVERY numerical value or annotation. Then, solve ONLY the tasks identified (e.g., Aufgabe 1). Use the following format: Aufgabe [number]: [Your answer here] Begründung: [Short explanation]. Do NOT mention or solve other tasks!"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_byte_arr}"}}  # Annahme: JPEG, anpassen bei Bedarf
                    ]
                }
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

uploaded_file = st.file_uploader("**Klausuraufgabe hochladen...**", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Hochgeladene Klausuraufgabe", use_column_width=True)
        
        if st.button("Aufgabe(n) lösen", type="primary"):
            st.markdown("---")
            with st.spinner("OpenAI o3 analysiert..."):
                o3_solution = solve_with_o3(image)
            
            if o3_solution:
                st.markdown("### FINALE LÖSUNG")
                st.markdown(o3_solution)
                if debug_mode:
                    with st.expander("🔍 o3 Rohausgabe"):
                        st.code(o3_solution)
            else:
                st.error("❌ Keine Lösung generiert")
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        st.error(f"❌ Fehler beim Laden des Bildes: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made by Fox & Koi-9 ❤️ | OpenAI o3")
