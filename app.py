import os
import json
import csv
import re
import numpy as np
from flask import Flask, request, render_template, session, redirect, url_for
import joblib

from nlp import extract_symptoms, load_vn_lexicon, load_symptom_list

MODEL_PATH = "artifacts/model.pkl"
META_PATH = "artifacts/meta.json"

app = Flask(__name__)
app.secret_key = "your_secret_key"

MAX_HISTORY = 30

# ===== Load resources =====
SYMPTOM_LIST = []
VN_LEXICON = load_vn_lexicon("data/vn_lexicon.json")

DESC_MAP = {}
PRECAUTION_MAP = {}
DISEASE_MAP = {}

# Load disease vi map
if os.path.exists("data/disease_vi.json"):
    try:
        with open("data/disease_vi.json", "r", encoding="utf-8") as f:
            DISEASE_MAP = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file ánh xạ tên bệnh: {e}")

# Load descriptions
DESC_FILE = "data/symptom_Description.csv"
if os.path.exists(DESC_FILE):
    try:
        with open(DESC_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            disease_idx = desc_idx = None

            if headers:
                for i, h in enumerate(headers):
                    h_low = str(h).strip().lower()
                    if h_low in ("disease", "prognosis"):
                        disease_idx = i
                    if h_low in ("description", "symptom_description", "details"):
                        desc_idx = i

            if disease_idx is None:
                disease_idx = 0
            if desc_idx is None:
                desc_idx = 1

            for row in reader:
                if not row or len(row) <= desc_idx:
                    continue
                disease_name = str(row[disease_idx]).strip()
                description = str(row[desc_idx]).strip()
                if disease_name:
                    DESC_MAP[disease_name] = description
    except Exception as e:
        print(f"Lỗi khi đọc file mô tả bệnh: {e}")

# Load precautions
PRE_FILE = "data/symptom_precaution.csv"
if os.path.exists(PRE_FILE):
    try:
        with open(PRE_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                disease_name = str(row[0]).strip()
                precautions = [str(x).strip() for x in row[1:] if x and str(x).strip()]
                if disease_name:
                    PRECAUTION_MAP[disease_name] = precautions
    except Exception as e:
        print(f"Lỗi khi đọc file biện pháp: {e}")

# Load model & meta
MODEL = None
META = {}
if os.path.exists(MODEL_PATH):
    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        MODEL = None

if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            META = json.load(f)
    except Exception as e:
        print(f"Lỗi khi load meta: {e}")
        META = {}

if META.get("symptom_cols"):
    SYMPTOM_LIST = META["symptom_cols"]
else:
    SYMPTOM_LIST = load_symptom_list("data")

def _trim_history(history):
    return history[-MAX_HISTORY:] if len(history) > MAX_HISTORY else history

@app.route("/", methods=["GET", "POST"])
def index():
    # Model chưa sẵn sàng
    if MODEL is None or not SYMPTOM_LIST:
        warning = "Model chưa được huấn luyện. Hãy huấn luyện mô hình trước khi sử dụng chatbot."
        return render_template("index.html", error=warning, disable_form=True, history=[])

    if request.method == "GET":
        if "history" not in session or not session.get("history"):
            session["history"] = [
                {
                    "sender": "bot",
                    "text": (
                        "Xin chào bạn 👋\n"
                        "Mình là chatbot tư vấn sức khỏe. Bạn hãy mô tả các triệu chứng bạn đang gặp nhé.\n"
                    ),
                },
                {
                    "sender": "bot",
                    "text": (
                        "Ví dụ bạn có thể nhập:\n"
                        "• \"Sổ mũi, hắt hơi, nghẹt mũi\"\n"
                        "• \"Sốt nhẹ, ho, đau họng, có đờm\"\n"
                        "• \"Đau bụng, buồn nôn, tiêu chảy\"\n\n"
                    ),
                },
            ]
        return render_template("index.html", history=session["history"])

    # ===== POST =====
    user_message = request.form.get("message", "").strip()
    if not user_message:
        return render_template("index.html", history=session.get("history", []))

    chat_history = session.get("history", [])
    chat_history.append({"sender": "user", "text": user_message})

    recognized_symptoms = extract_symptoms(user_message, SYMPTOM_LIST, VN_LEXICON)
    recognized_symptoms = [s for s in recognized_symptoms if s in SYMPTOM_LIST]

    if not recognized_symptoms:
        chat_history.append({
            "sender": "bot",
            "text": "Xin lỗi, tôi không nhận ra triệu chứng nào từ mô tả của bạn.",
            "error": True
        })
        session["history"] = _trim_history(chat_history)
        return redirect(url_for("index"))

    # Vector đặc trưng
    features = [1 if s in recognized_symptoms else 0 for s in SYMPTOM_LIST]

    # ===== Top 3 =====
    top3_vi = []
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba([features])[0]
        classes = MODEL.classes_
        top_idx = np.argsort(proba)[-3:][::-1]
        top3_en = [(str(classes[i]).strip(), float(proba[i])) for i in top_idx]
        predicted_disease = top3_en[0][0]
        for disease_en, p in top3_en:
            top3_vi.append({"disease": DISEASE_MAP.get(disease_en, disease_en), "prob": p})
    else:
        predicted_disease = str(MODEL.predict([features])[0]).strip()

    predicted_disease = str(predicted_disease).strip()
    disease_vi = DISEASE_MAP.get(predicted_disease, predicted_disease)

    # Mô tả + triệu chứng thường gặp
    description_full = DESC_MAP.get(predicted_disease, "Chưa có thông tin mô tả cho bệnh này.")
    common_symptoms = []
    description = description_full
    splitter = re.split(r"Triệu\s*chứng\s*thường\s*gặp\s*:\s*", description_full, maxsplit=1, flags=re.IGNORECASE)
    if len(splitter) == 2:
        description = splitter[0].strip() or description_full
        common_part = splitter[1].strip()
        common_symptoms = [s.strip() for s in common_part.split(",") if s.strip()]

    precautions = PRECAUTION_MAP.get(predicted_disease, [])

    # Triệu chứng tiếng Việt
    recognized_symptoms_vi = []
    for sym in recognized_symptoms:
        if sym in VN_LEXICON and VN_LEXICON[sym]:
            recognized_symptoms_vi.append(VN_LEXICON[sym][0])
        else:
            recognized_symptoms_vi.append(sym)

    chat_history.append({
        "sender": "bot",
        "symptoms": recognized_symptoms_vi,
        "disease": disease_vi,
        "top3": top3_vi,
        "description": description,
        "common_symptoms": common_symptoms,
        "precautions": precautions
    })

    session["history"] = _trim_history(chat_history)
    return redirect(url_for("index"))

@app.route("/reset")
def reset():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
