import re
import math
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Tuple, List

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
CSV_FILE = "/Users/valiyevmurad/VSCodeProjects/NLP/az_wikipedia_plaintext_with3.csv"
CORPUS_FILE = "az_wiki_corpus.txt"     # concatenated clean text (will be created)
OUTPUT_FOLDER = "results"
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# ────────────────────────────────────────────────
#  0. Load CSV → concatenate texts → basic cleaning
# ────────────────────────────────────────────────
def load_and_prepare_corpus():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}\nPlease place it in the same folder as this script.")

    print(f"Reading CSV: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    expected_columns = {'title', 'text'}
    if not expected_columns.issubset(df.columns):
        print("Warning: CSV should contain at least 'title' and 'text' columns.")
        print("Available columns:", df.columns.tolist())

    # Combine all article texts
    full_text = "\n\n".join(df['text'].astype(str).tolist())

    # Basic cleaning (Wikipedia-style artifacts)
    full_text = re.sub(r"==.*?==", "", full_text)               # headings
    full_text = re.sub(r"\[\[.*?\]\]", "", full_text)           # wiki links
    full_text = re.sub(r"\{\{.*?\}\}", "", full_text)           # templates
    full_text = re.sub(r"<.*?>", "", full_text)                 # html
    full_text = re.sub(r"''+", "", full_text)                   # bold/italic
    full_text = re.sub(r"\s*\n\s*\n+", "\n\n", full_text)       # normalize paragraphs
    full_text = re.sub(r"\s{2,}", " ", full_text)               # multiple spaces

    full_text = full_text.strip()

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Corpus created: {CORPUS_FILE}")
    print(f"Total characters: {len(full_text):,}")
    print(f"Number of articles used: {len(df):,}")

    return full_text, df


# ────────────────────────────────────────────────
#  Azerbaijani-friendly tokenizer
# ────────────────────────────────────────────────
def tokenize(text: str, lowercase=True):
    if lowercase:
        text = text.replace('I', 'ı').replace('İ', 'i')
        text = text.lower()

    tokens = re.findall(r"\d{1,3}(?:[.,]\d{1,3})*(?:[.,]\d+)?[%$€₼]| \d{1,2}:\d{2}(?::\d{2})? | \d+-[a-zə]{1,3} | [a-zəçğıöşüA-ZƏÇĞIÖŞÜ]+(?:-[a-zəçğıöşüA-ZƏÇĞIÖŞÜ]+)*", text)
    return [t[1:] for t in tokens if t.strip() and len(t) >= 1]

# ────────────────────────────────────────────────
#  Task 1: Tokens, types, frequencies
# ────────────────────────────────────────────────
def task1_tokens_types(tokens):
    counter = Counter(tokens)
    n_tokens = len(tokens)
    n_types = len(counter)

    print("\n" + "="*50)
    print("TASK 1 — Basic statistics")
    print(f"Tokens (N)     : {n_tokens:,}")
    print(f"Types (V)      : {n_types:,}")
    print(f"Type-Token Ratio: {n_tokens / n_types if n_types else 0}")
    print("\nTop 25 most frequent tokens:")
    for word, cnt in counter.most_common(25):
        print(f"{cnt:6,d}   {word}")

    return counter, n_tokens, n_types


# ────────────────────────────────────────────────
#  Task 2: Heaps' law  V ≈ k * N^β
# ────────────────────────────────────────────────
def heaps_func(N, k, beta):
    return k * N ** beta


def task2_heaps(tokens):
    vocab_sizes = []
    seen = set()
    step = max(300, len(tokens)//200)  # sample points

    for i, tok in enumerate(tokens, 1):
        seen.add(tok)
        if i % step == 0 or i == len(tokens):
            vocab_sizes.append((i, len(seen)))

    if len(vocab_sizes) < 5:
        print("Warning: too few points for reliable fit")
        return None, None

    Nv = np.array([x[0] for x in vocab_sizes])
    Vv = np.array([x[1] for x in vocab_sizes])

    try:
        popt, _ = curve_fit(heaps_func, Nv, Vv, p0=(5, 0.5), bounds=(0, [1000, 1]))
        k, beta = popt
        print("\n" + "="*50)
        print("TASK 2 — Heaps' law")
        print(f"k    ≈ {k:.2f}")
        print(f"β    ≈ {beta:.3f}")
        return k, beta
    except:
        print("Could not fit Heaps' law reliably.")
        return None, None


# ────────────────────────────────────────────────
#  Task 3: BPE
# ────────────────────────────────────────────────
def simple_bpe_demo(tokens, num_merges=25):
    print("\n" + "="*50)
    print("TASK 3 — BPE demonstration (first 25 merges)")
    vocab = Counter(' '.join(list(t)) + ' </w>' for t in tokens[:15000])  # limit for speed

    for i in range(num_merges):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            sym = word.split()
            for j in range(len(sym)-1):
                pairs[(sym[j], sym[j+1])] += freq

        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        new_token = ''.join(best_pair)
        print(f"merge {i+1:2d}: {best_pair[0]:<8} + {best_pair[1]:<8} → {new_token:<10}  (freq {pairs[best_pair]:,})")

        new_vocab = {}
        for word, freq in vocab.items():
            new_vocab[word.replace(' '.join(best_pair), new_token)] = freq
        vocab = new_vocab

    print("(stopped after 25 merges)")



# ────────────────────────────────────────────────
#  Task 5
# ────────────────────────────────────────────────

def get_sentences(text: str) -> list[str]:
    """
    Sentence splitter for Azerbaijani text.
    Protects common abbreviations so we don't split inside them.
    """
    # Step 1: List of common abbreviations
    # Format: full abbreviation with dot → temporary placeholder
    abbreviation_map = {
        'Dr.':      'DrDOT',
        'Prof.':    'ProfDOT',
        'Cən.':     'CənDOT',
        'b.e.':     'beDOT',
        'və s.':    'vəsDOT',
        'və s.':    'vesDOT',
        'etc.':     'etcDOT',
        'səh.':     'səhDOT',
        'sm.':      'smDOT',
        'km.':      'kmDOT',
        'mm.':      'mmDOT',
        'kg.':      'kgDOT',
        # Add more: 't.': 'tDOT', 'min.': 'minDOT', 'saat.': 'saatDOT', etc.
    }

    # Protect abbreviations
    protected_text = text
    for abbr, placeholder in abbreviation_map.items():
        protected_text = protected_text.replace(abbr, placeholder)

    # Step 2: Split on sentence boundaries + whitespace
    # We split keeping the delimiter (.!?), then recombine
    parts = re.split(r'([.!?])(\s+)', protected_text)

    sentences = []
    current = ""

    for i in range(0, len(parts), 3):  # groups of (text before, delimiter, space after)
        if i + 2 < len(parts):
            before = parts[i]
            delim  = parts[i+1]
            after  = parts[i+2]

            current += before + delim + after

            # Heuristic: if next part starts with lowercase or number → probably continuation
            if i + 3 < len(parts) and parts[i+3].strip() and parts[i+3][0].islower():
                continue

            sentences.append(current.strip())
            current = ""
        else:
            # leftover
            if parts[i].strip():
                current += parts[i]
                sentences.append(current.strip())

    # Step 3: Restore abbreviations
    final_sentences = []
    for sent in sentences:
        for placeholder, orig in {v: k for k, v in abbreviation_map.items()}.items():
            sent = sent.replace(placeholder, orig)
        if sent.strip():
            final_sentences.append(sent.strip())

    # Optional: filter very short fragments
    final_sentences = [s for s in final_sentences if len(s) > 15]

    return final_sentences


# ────────────────────────────────────────────────
#  Task 5
# ────────────────────────────────────────────────
def levenshtein(s1, s2):
    if len(s1) < len(s2): return levenshtein(s2, s1)
    if len(s2) == 0: return len(s1)
    prev = list(range(len(s2)+1))
    for i, c1 in enumerate(s1):
        curr = [i+1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+cost))
        prev = curr
    return prev[-1]


def simple_spellchecker_demo(vocab_counter, test_words):
    print("\n" + "="*50)
    print("TASK 5 — Simple Levenshtein spell-checker demo")
    for word in test_words:
        cands = []
        for v in list(vocab_counter.keys())[:15000]:  # limit for speed
            d = levenshtein(word, v)
            if d <= 3:
                cands.append((d, v, vocab_counter[v]))
        cands.sort(key=lambda x: (x[0], -x[2]))
        best = cands[:5]
        print(f"{word:14} →  " + ", ".join(f"{t[1]} (d={t[0]})" for t in best) if best else "no close matches")



# ────────────────────────────────────────────────
#  Extra Task: Confusion Matrix + Weighted Edit Distance
# ────────────────────────────────────────────────

# Typical error costs for Azerbaijani keyboard / common typos
# Values are relative costs (higher = more expensive / less likely error)
# 1.0 = standard cost, <1.0 = common error (cheaper), >1.0 = rare error
CONFUSION_MATRIX: Dict[Tuple[str, str], float] = {
    # Substitutions (common adjacent keys or phonetic confusions)
    ('a', 'ə'): 0.4,    ('ə', 'a'): 0.4,
    ('o', 'ö'): 0.5,    ('ö', 'o'): 0.5,
    ('u', 'ü'): 0.5,    ('ü', 'u'): 0.5,
    ('i', 'ı'): 0.4,    ('ı', 'i'): 0.4,
    ('ç', 'c'): 0.6,    ('c', 'ç'): 0.6,
    ('ş', 's'): 0.6,    ('s', 'ş'): 0.6,
    ('ğ', 'g'): 0.7,    ('g', 'ğ'): 0.7,
    ('ı', 'i'): 0.4,    # very frequent in typos
    ('e', 'ə'): 0.8,
    # Transpositions (common: adjacent letter swaps)
    ('iy', 'yı'): 0.6,  ('yı', 'iy'): 0.6,
    ('in', 'ni'): 0.7,
    # Insertions / Deletions of frequent letters
    # We give slightly lower cost to inserting/deleting vowels or frequent suffixes
    (' ', ' '): 1.0,    # no-op
}

# Default cost if pair not in matrix
DEFAULT_SUB_COST = 1.0
DEFAULT_INS_DEL_COST = 1.0

def weighted_levenshtein(s1: str, s2: str) -> float:
    """
    Weighted Levenshtein distance using confusion matrix for substitution costs.
    Insertion/deletion cost = 1.0 by default (can be made character-dependent later).
    """
    if len(s1) < len(s2):
        return weighted_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1) * DEFAULT_INS_DEL_COST

    # Use float for weighted costs
    prev = [float(j) * DEFAULT_INS_DEL_COST for j in range(len(s2) + 1)]
    for i, c1 in enumerate(s1):
        curr = [float(i + 1) * DEFAULT_INS_DEL_COST]
        for j, c2 in enumerate(s2):
            # Substitution cost from confusion matrix (or default)
            sub_cost = CONFUSION_MATRIX.get((c1, c2), DEFAULT_SUB_COST)
            if c1 == c2:
                sub_cost = 0.0

            cost_sub = prev[j] + sub_cost
            cost_del = curr[j] + DEFAULT_INS_DEL_COST
            cost_ins = prev[j + 1] + DEFAULT_INS_DEL_COST

            curr.append(min(cost_sub, cost_del, cost_ins))
        prev = curr
    return prev[-1]


def weighted_spellchecker_demo(vocab_counter: Counter, test_words: List[str], max_dist: float = 3.5, top_n: int = 5):
    print("\n" + "="*60)
    print("EXTRA TASK — Weighted Edit Distance Spell Checker Demo")
    print("Using confusion matrix with Azerbaijani-specific common errors\n")

    for word in test_words:
        candidates = []
        # Limit vocab scan for speed
        for v in list(vocab_counter.keys())[:20000]:
            if v == word:  # skip exact match
                continue
            dist = weighted_levenshtein(word, v)
            if dist <= max_dist:
                candidates.append((dist, v, vocab_counter[v]))

        candidates.sort(key=lambda x: (x[0], -x[2]))  # dist first, then frequency
        best = candidates[:top_n]

        print(f"{word:16} → ", end="")
        if best:
            suggestions = [f"{t[1]} (w-dist={t[0]:.2f}, freq={t[2]})" for t in best]
            print(", ".join(suggestions))
        else:
            print("no close candidates (weighted dist ≤ {:.1f})".format(max_dist))

    print("\n" + "-"*60 + "\n")

# ────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────
def main():
    try:
        full_text, df = load_and_prepare_corpus()
    except Exception as e:
        print("Error loading CSV:", e)
        return

    tokens = tokenize(full_text)

    vocab_counter, N, V = task1_tokens_types(tokens)
    k, beta = task2_heaps(tokens)

    simple_bpe_demo(tokens)

    sentences = get_sentences(full_text)

    print(f"\nNaive sentence count: {len(sentences):,}")
    print(f"First sentence example: {sentences[0]}")
    print(f"Second sentence example: {sentences[1]}")
    print(f"Third sentence example: {sentences[2]}")


    # Spell checker example words (common typos in Azerbaijani)
    test_typos = [
        "azərbaycan", "azəbaycan", "azarbaycan",
        "qarabağ", "qarabag", "qarabaq",
        "məsələ", "mesele", "məsələn", "meselen",
        "səhifə", "sehife", "səhife"
    ]
    simple_spellchecker_demo(vocab_counter, test_typos)

    print("\n" + "="*60)


    # ── Extra task demo ──
    # Reuse the same test typos
    weighted_spellchecker_demo(vocab_counter, test_typos, max_dist=3.5, top_n=5)

    print("Extra task completed: confusion matrix + weighted edit distance added.")
    print("Compare results with the uniform-cost Levenshtein above.")

if __name__ == "__main__":
    main()