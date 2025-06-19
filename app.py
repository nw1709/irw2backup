import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from anthropic import Anthropic
from PIL import Image, ImageEnhance, ImageOps
import google.generativeai as genai
import numpy as np
import cv2
import io
import zipfile
import logging

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Session State Reset ---
if 'last_upload' not in st.session_state:
    st.session_state.last_upload = None

# --- API Key Validation ---
def validate_keys():
    required_keys = {
        'gemini_key': ('AIza', "Gemini"),
        'claude_key': ('sk-ant', "Claude")
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
st.markdown("*Advanced exam solver with scanner-quality processing*")

# --- Google Drive-Verbindung ---
@st.cache_resource
def load_knowledge_from_drive():
    knowledge_base = ""
    if "gdrive_creds" not in st.secrets:
        return knowledge_base
        
    try:
        creds = service_account.Credentials.from_service_account_info(st.secrets["gdrive_creds"])
        drive_service = build("drive", "v3", credentials=creds)
        
        folder_response = drive_service.files().list(
            q="name='IRW_Bot_Gehirn' and mimeType='application/vnd.google-apps.folder'",
            pageSize=1,
            fields="files(id)"
        ).execute()
        
        folder = folder_response.get('files', [{}])[0]
        if not folder.get('id'):
            return knowledge_base
            
        zip_response = drive_service.files().list(
            q=f"'{folder['id']}' in parents and mimeType='application/zip'",
            pageSize=1,
            fields="files(id)"
        ).execute()
        
        zip_file = zip_response.get('files', [{}])[0]
        if not zip_file.get('id'):
            return knowledge_base
            
        downloaded = drive_service.files().get_media(fileId=zip_file['id']).execute()
        with zipfile.ZipFile(io.BytesIO(downloaded)) as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(('.txt', '.md')):
                    try:
                        content = zip_ref.read(file_name).decode('utf-8', errors='ignore')
                        knowledge_base += f"\n\n--- {file_name} ---\n{content}"
                    except Exception as e:
                        logger.warning(f"Konnte {file_name} nicht lesen: {e}")
                        
        logger.info(f"Knowledge Base geladen: {len(knowledge_base)} Zeichen")
        return knowledge_base
        
    except Exception as e:
        logger.error(f"Drive-Fehler: {str(e)}")
        return knowledge_base

# --- Bildvorverarbeitung wie Scanner-App ---
def preprocess_image_like_scanner(image):
    """Simuliert Scanner-App Vorverarbeitung für bessere Lesbarkeit und kleinere Dateien"""
    try:
        # PIL zu OpenCV
        img_array = np.array(image)
        
        # Zu Graustufen konvertieren
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Kontrast erhöhen
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Rauschen entfernen
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Adaptive Schwellwertbildung für Text
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Zurück zu PIL
        processed = Image.fromarray(binary)
        
        # Größe optimieren (max 2048px bei Beibehaltung des Seitenverhältnisses)
        processed.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        
        return processed
        
    except Exception as e:
        logger.warning(f"Bildvorverarbeitung fehlgeschlagen: {e}")
        return image

# --- Bilder zusammenfügen ---
def combine_images(images):
    """Fügt mehrere Bilder vertikal zusammen"""
    if len(images) == 1:
        return images[0]
    
    # Alle auf gleiche Breite bringen
    max_width = max(img.width for img in images)
    resized_images = []
    
    for img in images:
        if img.width != max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(img)
    
    # Gesamthöhe berechnen
    total_height = sum(img.height for img in resized_images) + (len(images) - 1) * 20  # 20px Abstand
    
    # Neues Bild erstellen
    combined = Image.new('RGB', (max_width, total_height), 'white')
    
    # Bilder einfügen
    y_offset = 0
    for i, img in enumerate(resized_images):
        combined.paste(img, (0, y_offset))
        y_offset += img.height + 20  # 20px Abstand zwischen Bildern
    
    return combined

# --- Gemini Flash Konfiguration ---
genai.configure(api_key=st.secrets["gemini_key"])
vision_model = genai.GenerativeModel("gemini-1.5-flash")

# --- OCR mit Failover ---
def extract_text_with_failover(image):
    """Extrahiert Text mit Gemini, falls fehlschlägt mit Claude"""
    try:
        # Primär: Gemini
        response = vision_model.generate_content(
            [
                "Extract ALL text from this exam image EXACTLY as written. Include all question numbers, text, and answer options (A, B, C, D, E). Do NOT interpret or solve.",
                image
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 4000
            }
        )
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Gemini OCR Error: {str(e)}")
        st.warning("⚠️ Gemini nicht verfügbar, verwende Claude für OCR")
        
        # Backup: Claude Vision
        try:
            # Bild für Claude vorbereiten
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            import base64
            image_data = base64.b64encode(buffered.getvalue()).decode()
            
            client = Anthropic(api_key=st.secrets["claude_key"])
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Günstiger für OCR
                max_tokens=4000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extract ALL text from this exam image EXACTLY as written. Do NOT solve, just OCR."
                        }
                    ]
                }]
            )
            return response.content[0].text
            
        except Exception as e2:
            logger.error(f"Claude OCR Error: {str(e2)}")
            raise e2

# --- Lösungen mit Failover ---
MODELS = {
    "primary": "claude-4-opus-20250514",
    "backup1": "claude-3-5-sonnet-20241022",
    "backup2": "claude-3-opus-20240229"
}

def solve_with_failover(prompt):
    """Versucht verschiedene Claude-Modelle bei Fehlern"""
    
    for model_key, model_name in MODELS.items():
        try:
            client = Anthropic(api_key=st.secrets["claude_key"])
            response = client.messages.create(
                model=model_name,
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            if model_key != "primary":
                st.info(f"ℹ️ Backup-Modell {model_name} verwendet")
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"{model_key} failed: {str(e)}")
            if model_key == list(MODELS.keys())[-1]:  # Letzter Versuch
                st.error(f"❌ Alle Modelle fehlgeschlagen: {str(e)}")
                return None
            continue

# --- Ergebnisformatierung ---
def format_minimal_results(full_response):
    """Extrahiert nur die Endergebnisse"""
    results = []
    lines = full_response.split('\n')
    
    for line in lines:
        if line.strip().startswith('Aufgabe'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                task_num = parts[0].replace('Aufgabe', '').strip()
                answer = parts[1].strip()
                results.append((task_num, answer))
    
    return results

# --- UI Optionen ---
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    use_knowledge = st.checkbox("📚 Kursmaterial", value=False)
with col2:
    minimal_output = st.checkbox("🎯 Nur Ergebnisse", value=True)
with col3:
    debug_mode = st.checkbox("🔍 Debug", value=False)

# --- Datei-Upload (MEHRERE DATEIEN) ---
uploaded_files = st.file_uploader(
    "**Klausuraufgabe(n) hochladen...**",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Mehrere Seiten werden automatisch zusammengefügt"
)

if uploaded_files:
    # Reset bei neuen Dateien
    file_ids = [f.file_id for f in uploaded_files]
    if st.session_state.last_upload != file_ids:
        st.session_state.last_upload = file_ids
        if 'ocr_result' in st.session_state:
            del st.session_state.ocr_result
    
    try:
        # Bilder laden
        images = []
        for file in uploaded_files:
            images.append(Image.open(file))
        
        # Zusammenfügen wenn mehrere
        if len(images) > 1:
            st.info(f"📄 {len(images)} Bilder werden zusammengefügt...")
            combined_image = combine_images(images)
        else:
            combined_image = images[0]
        
        # Vorverarbeitung
        with st.spinner("🔧 Optimiere Bildqualität (wie Scanner-App)..."):
            processed_image = preprocess_image_like_scanner(combined_image)
            
        # Anzeigen
        col1, col2 = st.columns(2)
        with col1:
            st.image(combined_image, caption="Original", use_container_width=True)
        with col2:
            st.image(processed_image, caption="Verarbeitet", use_container_width=True)
        
        # OCR
        if 'ocr_result' not in st.session_state:
            with st.spinner("📖 Lese Text..."):
                st.session_state.ocr_result = extract_text_with_failover(processed_image)
        
        ocr_text = st.session_state.ocr_result
        
        # Debug
        if debug_mode:
            with st.expander("🔍 OCR-Ergebnis"):
                st.code(ocr_text)
        
        # Lösen Button
        if st.button("🧮 Aufgaben lösen", type="primary"):
            
            # Knowledge Base
            knowledge_section = ""
            if use_knowledge:
                with st.spinner("📚 Lade Kursmaterial..."):
                    knowledge_base = load_knowledge_from_drive()
                    if knowledge_base:
                        knowledge_section = f"\n\nKURSMATERIAL:\n{knowledge_base[:10000]}"
            
            # Prompt
            prompt = f"""You are an accounting expert for "Internes Rechnungswesen (31031)" at Fernuniversität Hagen.

THEORETICAL FRAMEWORK:
- Kostenarten-, Kostenstellen-, Kostenträgerrechnung
- Voll-/Teilkostenrechnung, Grenzplankostenrechnung
- Deckungsbeitragsrechnung, Break-Even-Analyse
- Verursachungsprinzip, Zurechnungsprinzip

OCR-TEXT START:
{ocr_text}
OCR-TEXT ENDE
{knowledge_section}

ANWEISUNGEN:
1. Bei Multiple Choice "(x aus 5)": Prüfe JEDE Option einzeln
2. Format: 
   Aufgabe [Nr]: [Lösung]
   Begründung: [Kurze Erklärung auf Deutsch]

Antworte auf DEUTSCH!"""

            if debug_mode:
                with st.expander("🔍 Claude Prompt"):
                    st.code(prompt)
            
            # Lösen mit Failover
            with st.spinner("🧮 Löse Aufgaben..."):
                result = solve_with_failover(prompt)
            
            if result:
                st.markdown("---")
                
                if minimal_output:
                    # Nur Endergebnisse
                    st.markdown("### 📊 Lösungen:")
                    results = format_minimal_results(result)
                    
                    for task_num, answer in results:
                        st.markdown(f"# Aufgabe {task_num}: **{answer}**")
                else:
                    # Vollständige Ausgabe
                    st.markdown("### 📊 Detaillierte Lösungen:")
                    lines = result.split('\n')
                    for line in lines:
                        if line.strip():
                            if line.startswith('Aufgabe'):
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    st.markdown(f"### {parts[0]}: **{parts[1].strip()}**")
                            elif line.startswith('Begründung:'):
                                st.markdown(f"_{line}_")
                            else:
                                st.markdown(line)
                    
    except Exception as e:
        st.error(f"❌ Fehler: {str(e)}")
        st.info("Stelle sicher, dass die Bilder lesbar sind.")

# --- Footer ---
st.markdown("---")
st.caption("Koifox-Bot v3.0 | Multi-Image + Scanner Processing + Failover")
