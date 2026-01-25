import csv
import json
import os
import re
import string
import unicodedata


def _strip_accents(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.replace("đ", "d").replace("Đ", "D")


def _normalize(text: str) -> str:
    if not text:
        return ""
    t = str(text).lower().replace("_", " ")
    t = re.sub(rf"[{re.escape(string.punctuation)}]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_vn_lexicon(path: str = "data/vn_lexicon.json") -> dict:
    """canonical -> list synonyms (đã normalize)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        lex = {}
        for canon, syns in raw.items():
            if not isinstance(syns, list):
                continue
            out = []
            for s in syns:
                s_norm = _normalize(s)
                if s_norm:
                    out.append(s_norm)
            if out:
                lex[canon] = out
        return lex
    except Exception as e:
        print(f"Lỗi khi tải từ điển đồng nghĩa: {e}")
        return {}


def load_symptom_list(data_dir: str = "data") -> list:
    train_path = os.path.join(data_dir, "Training.csv")
    if not os.path.exists(train_path):
        return []

    try:
        with open(train_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None) or []
        cols = [str(c).strip() for c in header if str(c).strip()]
        return [c for c in cols if c != "prognosis"]
    except Exception as e:
        print(f"Lỗi khi đọc danh sách triệu chứng: {e}")
        return []


def extract_symptoms(message: str, symptom_list: list, vn_lexicon: dict = None) -> list:
    if not message:
        return []

    text = _normalize(message)
    text_no = _strip_accents(text)

    found = []
    found_set = set()

    # 1) Match theo từ điển đồng nghĩa
    if vn_lexicon:
        syn_map = {}
        for canon, syns in vn_lexicon.items():
            for syn in syns:
                syn_norm = _normalize(syn)
                if not syn_norm:
                    continue
                syn_no = _strip_accents(syn_norm)
                if syn_no:
                    syn_map[syn_no] = canon

        for syn_no in sorted(syn_map.keys(), key=len, reverse=True):
            if syn_no and syn_no in text_no:
                canon = syn_map[syn_no]
                if canon not in found_set:
                    found.append(canon)
                    found_set.add(canon)
                text_no = text_no.replace(syn_no, " ")

    # 2) Match trực tiếp theo danh sách triệu chứng của model
    if symptom_list:
        phrases = [(canon, _normalize(str(canon).replace("_", " "))) for canon in symptom_list]
        for canon, phrase in sorted(phrases, key=lambda x: len(x[1]), reverse=True):
            if not phrase:
                continue
            if phrase in text or _strip_accents(phrase) in text_no:
                if canon not in found_set:
                    found.append(canon)
                    found_set.add(canon)
                text = text.replace(phrase, " ")
                text_no = text_no.replace(_strip_accents(phrase), " ")

    return found
