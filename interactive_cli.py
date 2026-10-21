#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive CLI: Multi-Stage Conversational Passage Retrieval
Domain: Pharma / Medical QA (openFDA Drug Labeling)
Metodologi: Lin et al. (ACM TOIS 2021)
"""

import os
import sys
import re
import json
import math
import torch
import spacy
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder

def sigmoid(score: float) -> float:
    """Konversi logit score Cross-Encoder ke probabilitas relevansi (0.0 - 1.0)."""
    try:
        return 1.0 / (1.0 + math.exp(-score))
    except OverflowError:
        return 0.0 if score < 0 else 1.0

# Path Konfigurasi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASSAGES_FILE = os.path.join(BASE_DIR, "data", "processed", "passages.jsonl")

# Hyperparameter Algoritma 1 (HQE)
R_TOPIC = 4.5
R_SUB = 3.5
ETA_AMBIGUITY = 14.0
M_WINDOW = 2
RRF_K = 60

def print_banner():
    banner = """
================================================================================
   PHARMA CONVERSATIONAL PASSAGE RETRIEVAL (Lin et al., TOIS 2021)
   Domain: openFDA Drug Product Labeling | Multi-Stage Early Fusion
================================================================================
Perintah Khusus:
  - 'reset' : Bersihkan riwayat percakapan dan mulai topik obat baru.
  - 'exit'  : Keluar dari program.
================================================================================
"""
    print(banner)

def load_system():
    print("[1/4] Memuat spaCy (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")

    print(f"[2/4] Memuat korpus pasase dari {PASSAGES_FILE}...")
    if not os.path.exists(PASSAGES_FILE):
        print(f"[Error] File {PASSAGES_FILE} tidak ditemukan! Jalankan Notebook 01 terlebih dahulu.")
        sys.exit(1)

    passages = []
    passage_dict = {}
    corpus_tokens = []
    with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            passages.append(p)
            pid = p["passage_id"]
            passage_dict[pid] = p
            corpus_tokens.append([w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", p["passage_text"])])

    print(f"      -> {len(passages)} pasase berhasil dimuat. Membangun indeks BM25Okapi (k1=0.82, b=0.68)...")
    bm25 = BM25Okapi(corpus_tokens, k1=0.82, b=0.68)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3/4] Memuat model NTR (castorini/t5-base-canard) pada [{device.upper()}]...")
    t5_model_name = "castorini/t5-base-canard"
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(device)
    t5_model.eval()

    print(f"[4/4] Memuat model Neural Re-ranker (cross-encoder/ms-marco-MiniLM-L-6-v2) pada [{device.upper()}]...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    print("\n[OK] Seluruh komponen berhasil dimuat! Sistem siap digunakan.\n")
    return nlp, passages, passage_dict, bm25, t5_tokenizer, t5_model, cross_encoder, device

# ==================== FUNGSI MODUL HQE ====================
def keyword_extractor(bm25, token: str) -> float:
    scores = bm25.get_scores([token.lower()])
    return float(max(scores)) if len(scores) > 0 else 0.0

def query_performance_predictor(bm25, query_tokens: list) -> float:
    if not query_tokens:
        return 0.0
    scores = bm25.get_scores(query_tokens)
    return float(max(scores)) if len(scores) > 0 else 0.0

def get_pos_filtered_tokens(nlp, text: str) -> list:
    doc = nlp(text)
    filtered = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and len(token.text) > 1:
            filtered.append(token.text.lower())
    return filtered

def run_hqe(nlp, bm25, history: list, current_utterance: str):
    all_utterances = history + [current_utterance]
    i = len(all_utterances) - 1

    w_topic = set()
    w_sub = set()

    for j, utt in enumerate(all_utterances):
        cand_tokens = get_pos_filtered_tokens(nlp, utt)
        for t in cand_tokens:
            r_val = keyword_extractor(bm25, t)
            if r_val > R_TOPIC:
                w_topic.add(t)
            if (r_val > R_SUB) and (j >= (i - M_WINDOW)):
                w_sub.add(t)

    reformulated = []
    curr_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", current_utterance)]
    a_i = query_performance_predictor(bm25, curr_tokens)

    if i > 0:
        for t in sorted(list(w_topic)):
            reformulated.append(t)
        if a_i < ETA_AMBIGUITY:
            for t in sorted(list(w_sub)):
                if t not in reformulated:
                    reformulated.append(t)
        reformulated.append(current_utterance)
        return " ".join(reformulated), a_i
    else:
        return current_utterance, a_i

# ==================== FUNGSI MODUL NTR ====================
def run_ntr(t5_tokenizer, t5_model, device, history: list, current_utterance: str) -> str:
    if not history:
        return current_utterance

    context_text = " ||| ".join(history + [current_utterance])
    inputs = t5_tokenizer(context_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = t5_model.generate(**inputs, max_new_tokens=64, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==================== FUNGSI RETRIEVAL & FUSION ====================
def retrieve_bm25(bm25, passages, query_str: str, top_n: int = 50) -> list:
    toks = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", query_str)]
    scores = bm25.get_scores(toks)
    ranked_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:top_n]
    return [passages[k]["passage_id"] for k in ranked_idx]

def reciprocal_rank_fusion(list_hqe: list, list_ntr: list, k_const: int = 60) -> list:
    scores = {}
    for rank, pid in enumerate(list_hqe):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_const + rank + 1)
    for rank, pid in enumerate(list_ntr):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_const + rank + 1)
    return sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

# ==================== LOOP UTAMA CLI ====================
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    print_banner()
    nlp, passages, passage_dict, bm25, t5_tokenizer, t5_model, cross_encoder, device = load_system()

    history = []
    turn_idx = 1

    while True:
        try:
            prompt_str = f"User (Turn {turn_idx}) > "
            user_input = input(prompt_str).strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                print("\nTerima kasih telah menggunakan Pharma Conversational Passage Retrieval CLI. Sampai jumpa!")
                break

            if user_input.lower() == "reset":
                history = []
                turn_idx = 1
                print("\n[INFO] Riwayat percakapan telah dibersihkan. Memulai sesi baru.\n")
                continue

            # 1. Jalankan CQR: HQE (+POS)
            q_hqe, a_i = run_hqe(nlp, bm25, history, user_input)

            # 2. Jalankan CQR: NTR (T5)
            q_ntr = run_ntr(t5_tokenizer, t5_model, device, history, user_input)

            # 3. First-stage BM25 retrieval
            cand_hqe = retrieve_bm25(bm25, passages, q_hqe, top_n=50)
            cand_ntr = retrieve_bm25(bm25, passages, q_ntr, top_n=50)

            # 4. Early Fusion via Reciprocal Rank Fusion (RRF)
            fused_candidates = reciprocal_rank_fusion(cand_hqe, cand_ntr, k_const=RRF_K)[:50]

            # 5. Second-stage Neural Re-ranking via Cross-Encoder
            ce_pairs = [[q_ntr, passage_dict[pid]["passage_text"]] for pid in fused_candidates]
            ce_scores = cross_encoder.predict(ce_pairs)
            sorted_order = sorted(range(len(ce_scores)), key=lambda k: ce_scores[k], reverse=True)

            top1_score = float(ce_scores[sorted_order[0]])
            top1_prob = sigmoid(top1_score)
            top1_pid = fused_candidates[sorted_order[0]]
            top1_passage = passage_dict[top1_pid]

            # ==================== DETEKSI OOT DETERMINISTIK ====================
            # Turn 1: BM25 mentah (a_i) < 4.0 ATAU probabilitas Cross-Encoder < 18%
            # Turn > 1: Probabilitas Cross-Encoder < 18% (murni semantik, abaikan a_i)
            if turn_idx == 1:
                is_oot = (a_i < 4.0) or (top1_prob < 0.18)
                oot_reason = f"Skor BM25 awal ({a_i:.2f} < 4.00) atau probabilitas relevansi ({top1_prob:.2%} < 18.00%) terlalu rendah."
            else:
                is_oot = (top1_prob < 0.18)
                oot_reason = f"Probabilitas relevansi ({top1_prob:.2%} < 18.00%) berada di bawah ambang batas minimum."

            # ==================== DEBUG OUTPUT TRANSPARAN ====================
            print("-" * 80)
            print(f"[DEBUG - CQR & Retrieval Pipeline]")
            print(f"  * Turn                        : {turn_idx} ({'Kueri Pembuka' if turn_idx == 1 else 'Kueri Lanjutan Multi-Turn'})")
            print(f"  * QPP Ambiguity Score (A_i)   : {a_i:.2f} (Threshold: {ETA_AMBIGUITY})")
            print(f"  * [HQE Query]                 : {q_hqe}")
            print(f"  * [NTR Query]                 : {q_ntr}")
            print(f"  * [RRF Top Fused Pool]        : {len(fused_candidates)} kandidat pasase")
            print(f"  * Cross-Encoder Logit Score   : {top1_score:.4f}")
            print(f"  * Relevansi Probabilitas (P)  : {top1_prob:.2%} (Threshold OOT: 18.00%)")
            print(f"  * Status Domain               : {'[OUT OF TOPIC / DI LUAR DOMAIN]' if is_oot else '[IN DOMAIN / VALID]'}")
            print("-" * 80)

            if is_oot:
                print("\n" + "!" * 80)
                print("  [WARNING: OUT OF TOPIC / DI LUAR BASIS PENGETAHUAN OBAT]")
                print(f"  Alasan: {oot_reason}")
                if turn_idx > 1:
                    print("  [Tips] Jika Anda ingin beralih menanyakan obat yang berbeda, ketik 'reset'.")
                print("!" * 80)

                print(f"\n[HASIL RETRIEVAL TERDEKAT (Kurang Relevan - Skor: {top1_score:.4f} | Prob: {top1_prob:.2%})]")
                print(f"  * Passage ID   : {top1_pid}")
                print(f"  * Obat         : {top1_passage['brand_name']} ({top1_passage['generic_name']})")
                print(f"  * Seksi Medis  : {top1_passage['section']}")
                print(f"\n[Kutipan Pasase]:")
                print(f"  \"{top1_passage['passage_text'][:250]}...\"")
                print("=" * 80 + "\n")
                # Tidak menambahkan kueri OOT ke riwayat percakapan agar tidak meracuni konteks turn berikutnya
            else:
                print(f"\n[HASIL RETRIEVAL TERBAIK (TOP-1)]")
                print(f"  * Passage ID   : {top1_pid}")
                print(f"  * Obat         : {top1_passage['brand_name']} ({top1_passage['generic_name']})")
                print(f"  * Seksi Medis  : {top1_passage['section']}")
                print(f"  * Re-rank Score: {top1_score:.4f} (Probabilitas Relevansi: {top1_prob:.2%})")
                print(f"\n[Isi Pasase]:")
                print(f"  \"{top1_passage['passage_text']}\"")
                print("=" * 80 + "\n")

                # Update riwayat percakapan hanya jika valid
                history.append(user_input)
                turn_idx += 1

        except (KeyboardInterrupt, EOFError):
            print("\nProgram dihentikan oleh pengguna. Sampai jumpa!")
            break
        except Exception as e:
            print(f"\n[Error] Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
