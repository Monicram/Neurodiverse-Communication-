# app.py
import os
import uuid
import subprocess
from flask import Flask, request, render_template, jsonify, send_from_directory
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pyttsx3

# CONFIG
UPLOAD_DIR = "uploads"
AUDIO_OUT = "static/audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_OUT, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")

print("Loading models (this may take a while the first run)...")
# Whisper STT model (choose "tiny", "small", "base", "medium", "large" per resources)
stt_model = whisper.load_model("small")  # change to "tiny" for much faster CPU runs

# T5-small for a quick grammar-correction prompt (no fine-tune)
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# simple translation model map (English->X)
TRANSLATION_MODEL_MAP = {
    "es": "Helsinki-NLP/opus-mt-en-es",  # English -> Spanish
    "fr": "Helsinki-NLP/opus-mt-en-fr",  # English -> French
    "hi": "Helsinki-NLP/opus-mt-en-hi",  # English -> Hindi (if available)
}
_translation_cache = {}

def correct_text_t5(raw_text):
    """Use T5-small with a 'correct: ' prefix to improve grammar (quick demo)."""
    prompt = "correct: " + raw_text.strip()
    input_ids = t5_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = t5_model.generate(input_ids, max_length=256)
    corrected = t5_tokenizer.decode(out[0], skip_special_tokens=True)
    return corrected

def translate_text(text, target_code):
    """Translate using Helsinki-NLP models (lazy-loaded and cached)."""
    if target_code in (None, "", "en"):
        return text
    if target_code not in TRANSLATION_MODEL_MAP:
        return text
    if target_code not in _translation_cache:
        model_name = TRANSLATION_MODEL_MAP[target_code]
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _translation_cache[target_code] = (tok, model)
    tok, model = _translation_cache[target_code]
    inputs = tok.encode(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=512)
    return tok.decode(outputs[0], skip_special_tokens=True)

def webm_to_wav(in_path, out_path):
    """Use ffmpeg to convert an uploaded webm or other container to WAV suitable for whisper."""
    # Ensure ffmpeg installed and accessible on PATH
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-ar", "16000", "-ac", "1", "-loglevel", "quiet",
        out_path
    ]
    subprocess.run(cmd, check=True)

def tts_save(text, filename):
    """Save TTS audio (mp3) using pyttsx3 (offline)."""
    full = os.path.join(AUDIO_OUT, filename)
    engine = pyttsx3.init()
    # adjust voice/rate if you want:
    engine.setProperty('rate', 150)
    engine.save_to_file(text, full)
    engine.runAndWait()
    return full

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts 'audio_data' multipart form file (webm/wav).
    Returns JSON:
      { "raw": "...", "corrected": "...", "translated": "...", "audio_url": "/static/audio/..." }
    """
    file = request.files.get("audio_data")
    target_lang = request.form.get("target_language", "en")
    if not file:
        return jsonify({"error": "No audio uploaded"}), 400

    uid = uuid.uuid4().hex
    in_path = os.path.join(UPLOAD_DIR, f"{uid}_{file.filename}")
    file.save(in_path)

    # convert to wav for whisper
    wav_path = os.path.join(UPLOAD_DIR, f"{uid}.wav")
    try:
        webm_to_wav(in_path, wav_path)
    except Exception as e:
        # if conversion fails try to assume the file is already wav
        wav_path = in_path

    # STT using whisper
    result = stt_model.transcribe(wav_path, language="en")
    raw_text = result.get("text", "").strip()

    # grammar correction (quick demo)
    corrected = correct_text_t5(raw_text)

    # translation
    translated = translate_text(corrected, target_lang if target_lang else "en")

    # TTS: synthesize the translated text
    audio_filename = f"{uid}.mp3"
    audio_path = tts_save(translated, audio_filename)

    audio_url = f"/{audio_path}" if audio_path.startswith("static/") else f"/{audio_path}"
    # make route accessible: static folder is already served by Flask

    return jsonify({
        "raw": raw_text,
        "corrected": corrected,
        "translated": translated,
        "audio_url": f"/{audio_path}",
    })

@app.route("/static/audio/<path:fn>")
def audio(fn):
    return send_from_directory(AUDIO_OUT, fn)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
