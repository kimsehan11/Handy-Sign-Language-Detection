from sentence_transformers import SentenceTransformer, util
import csv



def load_common_sentences(filepath):
    with open(filepath, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_unigram_words(filepath):
    # unigram_freq.csv: word,freq (header 있음)
    words = []
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                word, freq = row[0], int(row[1])
                words.append((word, freq))
    return words

def get_common_word_recommendations(prefix, common_words, max_results=3):
    prefix = prefix.lower()
    filtered = [w for w in common_words if w.startswith(prefix)] if prefix else []
    return filtered[:max_results]

def get_sentence_recommendation(prefix, common_sentences, sbert_model, min_prefix_len=3, sim_threshold=0.6):
    if not prefix or len(prefix) < min_prefix_len:
        return None
    # 1. prefix로 시작하는 문장 있으면 tail 바로 반환
    for s in common_sentences:
        if s.startswith(prefix) and len(s) > len(prefix):
            tail = s[len(prefix):]
            return tail.lstrip()  # 앞 공백 제거
    # 2. SBERT fallback
    candidates = [s for s in common_sentences if len(s) > len(prefix)]
    if not candidates:
        return None
    prefix_emb = sbert_model.encode(prefix, convert_to_tensor=True)
    cand_embs = sbert_model.encode(candidates, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(prefix_emb, cand_embs)[0]
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])
    if best_sim < sim_threshold:
        return None
    best_sent = candidates[best_idx]
    tail = best_sent[len(prefix):]
    return tail.lower() if tail else None

def get_unigram_word_recommendations(prefix, unigram_words, max_results=3):
    prefix = prefix.lower()
    filtered = [(w, f) for w, f in unigram_words if w.startswith(prefix)] if prefix else []
    filtered.sort(key=lambda x: -x[1])  # freq 내림차순
    return [w for w, _ in filtered[:max_results]]
