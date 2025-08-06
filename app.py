import streamlit as st
from openai import OpenAI, OpenAIError
from PIL import Image
import logging
import io
import pdf2image
import os
import base64

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

# --- API Client ---
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# --- Datei in Bild konvertieren ---
def convert_to_image(uploaded_file):
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        logger.info(f"Processing file with extension: {file_extension}")
        
        if file_extension in ['.png', '.jpeg', '.jpg', '.gif', '.webp']:
            image = Image.open(uploaded_file)
            if not image.format:
                image = image.convert('RGB')  # Zwangs-Konvertierung, falls Format nicht erkannt
            logger.info(f"Loaded image with format: {image.format}")
            return image
        
        
        else:
            st.error(f"❌ Nicht unterstütztes Format: {file_extension}. Bitte lade PNG, JPEG, GIF, WebP oder PDF hoch.")
            st.stop()
            
    except Exception as e:
        logger.error(f"Error converting file to image: {str(e)}")
        st.error(f"❌ Fehler bei der Konvertierung: {str(e)}")
        return None

# --- OpenAI o3 Solver mit Bildverarbeitung ---
def solve_with_o3(image):
    try:
        logger.info("Preparing image for OpenAI o3")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)  # Qualität anpassen
        img_bytes = img_byte_arr.getvalue()
        logger.info(f"Image size in bytes: {len(img_bytes)}")

        # Base64 kodieren und überprüfen
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        logger.info(f"Base64 encoded length: {len(img_base64)}")

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

NO OTHER FORMAT IS ACCEPTABLE. """
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from the provided exam image EXACTLY as written, including every detail from graphs, charts, or sketches. For graphs: Explicitly list ALL axis labels, ALL scales, ALL intersection points with axes (e.g., 'x-axis at 450', 'y-axis at 20'), and EVERY numerical value or annotation. Then, solve ONLY the tasks identified (e.g., Aufgabe 1). Use the following format: Aufgabe [number]: [Your answer here] Begründung: [Short explanation]. Do NOT mention or solve other tasks!"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
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

uploaded_file = st.file_uploader("**Klausuraufgabe hochladen...**", type=["png", "jpg", "jpeg", "gif", "webp", "pdf"])

if uploaded_file is not None:
    try:
        image = convert_to_image(uploaded_file)
        if image:
            st.image(image, caption="Verarbeitetes Bild", use_container_width=True)  # Deprecation behoben
            logger.info(f"Image format after conversion: {image.format}")
            
            if st.button("🧮 Aufgabe(n) lösen", type="primary"):
                st.markdown("---")
                with st.spinner("OpenAI o3 analysiert..."):
                    o3_solution = solve_with_o3(image)
                
                if o3_solution:
                    st.markdown("### 🎯 FINALE LÖSUNG")
                    st.markdown(o3_solution)
                    if debug_mode:
                        with st.expander("🔍 o3 Rohausgabe"):
                            st.code(o3_solution)
                else:
                    st.error("❌ Keine Lösung generiert")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"❌ Fehler bei der Verarbeitung: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made by Fox & Koi-9 ❤️ | OpenAI o3")
