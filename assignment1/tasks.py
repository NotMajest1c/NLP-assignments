import re
import math
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

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
    print("\nTop 20 most frequent tokens:")
    for word, cnt in counter.most_common(20):
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

        plt.figure(figsize=(8, 6))
        plt.loglog(Nv, Vv, 'o', label='Observed data', markersize=6, alpha=0.7)
        plt.loglog(Nv, heaps_func(Nv, k, beta), '-', label=f'Fit: k={k:.2f}, β={beta:.3f}', linewidth=2, color='red')

        plt.xlabel('Number of tokens (N)', fontsize=12)
        plt.ylabel('Vocabulary size (V)', fontsize=12)
        plt.title("Heaps' Law: Vocabulary Growth (log-log scale)", fontsize=14)
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        plt.savefig('heaps_law_plot.png', dpi=300, bbox_inches='tight')
        print("Heaps' law plot saved as 'heaps_law_plot.png'")
        plt.show()
        return k, beta
    except Exception as e:
        print(f"Could not fit Heaps' law reliably: {e}")
        return None, None


def simple_bpe_demo(tokens, num_merges=500, output_file="results/bpe_vocab.txt"):
    """
    Simple BPE demonstration:
    - Performs `num_merges` merges on the most frequent character pairs.
    - Writes final vocabulary to `output_file`.
    """
    print("\n" + "="*50)
    print("TASK 3 — BPE demonstration")
    
    # Take a subset of tokens for demonstration (for speed)
    subset_tokens = tokens[:50000] if len(tokens) > 50000 else tokens
    print(f"Using {len(subset_tokens):,} tokens for BPE demonstration")
    
    # Step 1: Initialize vocabulary with characters
    # Represent each word as a list of characters with </w> marker at the end
    word_freqs = Counter(subset_tokens)
    
    # Build initial vocabulary: all unique characters
    vocab = set()
    for word in word_freqs:
        vocab.update(list(word))
    vocab.add('</w>')
    vocab = sorted(list(vocab))
    
    # Step 2: Split words into characters for BPE processing
    # Each word is represented as tuple of characters + </w>
    splits = {}
    for word, freq in word_freqs.items():
        # Add word boundary marker
        chars = list(word) + ['</w>']
        splits[word] = (tuple(chars), freq)
    
    # Step 3: BPE merging
    merges = []
    merge_log = []
    
    for i in range(num_merges):
        # Count all pairs
        pair_freqs = Counter()
        for word, (splitted, freq) in splits.items():
            # Count pairs in this word
            for j in range(len(splitted) - 1):
                pair = (splitted[j], splitted[j + 1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            print(f"No more pairs to merge after {i} merges")
            break
        
        # Find most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_freq = pair_freqs[best_pair]
        
        # Update vocabulary with new merged symbol
        new_symbol = best_pair[0] + best_pair[1]
        vocab.append(new_symbol)
        
        # Apply merge to all words
        new_splits = {}
        for word, (splitted, freq) in splits.items():
            # Merge the pair in this word
            new_split = []
            j = 0
            while j < len(splitted):
                if j < len(splitted) - 1 and splitted[j] == best_pair[0] and splitted[j + 1] == best_pair[1]:
                    new_split.append(new_symbol)
                    j += 2
                else:
                    new_split.append(splitted[j])
                    j += 1
            new_splits[word] = (tuple(new_split), freq)
        
        splits = new_splits
        merges.append((best_pair, new_symbol))
        
        # Log the merge
        merge_log.append(f"Merge {i+1:3d}: {best_pair[0]} + {best_pair[1]} → {new_symbol:<10} (freq: {best_freq:,})")
        
        # Print progress
        if (i + 1) % 50 == 0 or i == 0 or i == num_merges - 1:
            print(f"Merges completed: {i+1}, Vocabulary size: {len(vocab)}")
    
    # Step 4: Generate final BPE tokens
    final_vocab = Counter()
    for word, (splitted, freq) in splits.items():
        # Join symbols without spaces (remove word boundary for display)
        for symbol in splitted:
            if symbol != '</w>':
                display_symbol = symbol.replace('</w>', '')
                final_vocab[display_symbol] += freq
    
    # Also count full words that resulted from merges
    for word, (splitted, _) in splits.items():
        # Construct BPE tokenization of the word
        bpe_tokens = []
        for symbol in splitted:
            if symbol != '</w>':
                bpe_tokens.append(symbol)
        
        # Add the full BPE tokenization as a potential token
        if len(bpe_tokens) == 1:
            final_vocab[bpe_tokens[0]] += word_freqs.get(word, 1)
    
    # Step 5: Save results
    Path(output_file).parent.mkdir(exist_ok=True)
    
    # Save vocabulary
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("BPE Vocabulary (sorted by frequency):\n")
        f.write("="*60 + "\n")
        for token, freq in final_vocab.most_common():
            f.write(f"{token:<20} {freq:,}\n")
    
    # Save merge log
    merge_log_file = output_file.replace("_vocab.txt", "_merges.txt")
    with open(merge_log_file, "w", encoding="utf-8") as f:
        f.write("BPE Merge Operations:\n")
        f.write("="*60 + "\n")
        for log_entry in merge_log:  # Save first 100 merges
            f.write(log_entry + "\n")
    
    # Step 6: Demonstrate BPE on example words
    print("\n" + "-"*40)
    print("BPE Tokenization Examples:")
    print("-"*40)
    
    example_words = ["azərbaycan", "qarabağ", "məsələ", "universitet", "telefon"]
    for word in example_words:
        # Apply BPE merges to the word
        splitted = list(word) + ['</w>']
        for best_pair, new_symbol in merges[:50]:  # Apply first 50 merges
            new_split = []
            j = 0
            while j < len(splitted):
                if j < len(splitted) - 1 and splitted[j] == best_pair[0] and splitted[j + 1] == best_pair[1]:
                    new_split.append(new_symbol)
                    j += 2
                else:
                    new_split.append(splitted[j])
                    j += 1
            splitted = new_split
        
        # Remove word boundary for display
        bpe_tokens = [token.replace('</w>', '') for token in splitted if token != '</w>']
        print(f"{word:15} → {' '.join(bpe_tokens)}")
    
    print(f"\nBPE merges completed: {len(merges)}")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Vocabulary saved to: {output_file}")
    print(f"Merge log saved to: {merge_log_file}")
    
    return final_vocab, merges




# ────────────────────────────────────────────────
#  Task 4
# ────────────────────────────────────────────────

def get_sentences(text: str) -> list[str]:
    """
    Sentence splitter for Azerbaijani text.
    Protects common abbreviations so we don't split inside them.
    """
    # Step 1: List of common abbreviations
    # Format: full abbreviation with dot → temporary placeholder
    abbreviation_map = {
    # Original multi-letter abbreviations
    'Dr.':      'DrDOT',
    'Prof.':    'ProfDOT',
    'Cən.':     'CənDOT',
    'b.e.':     'beDOT',
    'və s.':    'vəsDOT',
    'və s.':    'vesDOT',  # duplicate is harmless
    'etc.':     'etcDOT',
    'səh.':     'səhDOT',
    'sm.':      'smDOT',
    'km.':      'kmDOT',
    'mm.':      'mmDOT',
    'kg.':      'kgDOT',

    # ── Single-letter initials (all 32 Azerbaijani uppercase letters) ──
    'A.':       'ADOT',
    'B.':       'BDOT',
    'C.':       'CDOT',
    'Ç.':       'ÇDOT',
    'D.':       'DDOT',
    'E.':       'EDOT',
    'Ə.':       'ƏDOT',
    'F.':       'FDOT',
    'G.':       'GDOT',
    'Ğ.':       'ĞDOT',
    'H.':       'HDOT',
    'X.':       'XDOT',
    'I.':       'IDOT',
    'İ.':       'İDOT',
    'J.':       'JDOT',
    'K.':       'KDOT',
    'Q.':       'QDOT',
    'L.':       'LDOT',
    'M.':       'MDOT',
    'N.':       'NDOT',
    'O.':       'ODOT',
    'Ö.':       'ÖDOT',
    'P.':       'PDOT',
    'R.':       'RDOT',
    'S.':       'SDOT',
    'Ş.':       'ŞDOT',
    'T.':       'TDOT',
    'U.':       'UDOT',
    'Ü.':       'ÜDOT',
    'V.':       'VDOT',
    'Y.':       'YDOT',
    'Z.':       'ZDOT',
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
            if d <= 6:
                cands.append((d, v, vocab_counter[v]))
        cands.sort(key=lambda x: (x[0], -x[2]))
        best = cands[:5]
        print(f"{word:14} →  " + ", ".join(f"{t[1]} (d={t[0]})" for t in best) if best else "no close matches")



# ────────────────────────────────────────────────
#  Extra Task: Confusion Matrix + Weighted Edit Distance
# ────────────────────────────────────────────────

def weighted_levenshtein2(
    s1: str,
    s2: str,
    confusion_matrix: Dict[Tuple[str, str], float],
    default_sub_cost: float = 1.0,
    default_ins_del_cost: float = 1.0
) -> float:
    """
    Weighted Levenshtein distance using a learned confusion matrix.
    Lower cost = more likely / cheaper edit.
    """

    if len(s1) < len(s2):
        return weighted_levenshtein2(
            s2, s1, confusion_matrix,
            default_sub_cost, default_ins_del_cost
        )

    if len(s2) == 0:
        return len(s1) * default_ins_del_cost

    prev = [j * default_ins_del_cost for j in range(len(s2) + 1)]

    for i, c1 in enumerate(s1):
        curr = [(i + 1) * default_ins_del_cost]
        for j, c2 in enumerate(s2):

            if c1 == c2:
                sub_cost = 0.0
            else:
                sub_cost = confusion_matrix.get(
                    (c1, c2),
                    default_sub_cost
                )

            cost_sub = prev[j] + sub_cost
            cost_del = curr[j] + default_ins_del_cost
            cost_ins = prev[j + 1] + default_ins_del_cost

            curr.append(min(cost_sub, cost_del, cost_ins))

        prev = curr

    return prev[-1]

def weighted_spellchecker_demo2(
    vocab_counter: Counter,
    confusion_matrix: Dict[Tuple[str, str], float],
    test_words: List[str],
    max_dist: float = 3.5,
    top_n: int = 5,
    vocab_limit: int = 20000
):
    """
    Spell checker demo using weighted Levenshtein distance
    learned from the corpus.
    """

    print("\n" + "="*65)
    print("WEIGHTED SPELL CHECKER (Corpus-Learned Confusion Matrix)")
    print("="*65)

    vocab_items = list(vocab_counter.items())[:vocab_limit]

    for word in test_words:
        candidates = []

        for vocab_word, freq in vocab_items:
            if vocab_word == word:
                continue

            dist = weighted_levenshtein2(
                word,
                vocab_word,
                confusion_matrix
            )

            if dist <= max_dist:
                candidates.append((dist, vocab_word, freq))

        candidates.sort(key=lambda x: (x[0], -x[2]))
        best = candidates[:top_n]

        print(f"\n{word:18} → ", end="")
        if best:
            print(", ".join(
                f"{w} (d={d:.2f}, freq={f})"
                for d, w, f in best
            ))
        else:
            print("no candidates within threshold")

    print("\n" + "-"*65)


def generate_letter_confusion_matrix_from_corpus(tokens):
    """
    Automatically generate letter confusion matrix by analyzing character substitutions
    in similar words in the corpus.
    """
    print("Generating letter confusion matrix from corpus data...")
    
    # Get most frequent words
    vocab_counter = Counter(tokens)
    common_words = [word for word, freq in vocab_counter.most_common(5000)]
    
    # Initialize counts for all letter pairs
    letter_pairs_count = Counter()
    total_substitutions = 0
    
    # Analyze word pairs that are similar
    for i in range(min(2000, len(common_words))):
        word1 = common_words[i]
        
        # Compare with other words of similar length
        for j in range(i+1, min(i+100, len(common_words))):
            word2 = common_words[j]
            
            # Only compare words with similar length
            if abs(len(word1) - len(word2)) > 1:
                continue
            
            # Use Levenshtein to find if they're similar
            dist = levenshtein(word1, word2)
            
            if dist == 1:  # Single substitution
                # Find which character is different
                sub_pair = find_single_substitution(word1, word2)
                if sub_pair:
                    c1, c2 = sub_pair
                    letter_pairs_count[(c1, c2)] += 1
                    total_substitutions += 1
    
    # Also analyze common prefix/suffix patterns
    analyze_prefix_suffix_patterns(common_words, letter_pairs_count)
    
    print(f"Found {total_substitutions} character substitution examples")
    
    # Convert counts to probabilities/weights
    confusion_matrix = {}
    azerbaijani_letters = 'abcçdeəfgğhxıijklmnoöpqrsştuüvyz'
    
    # Initialize with default high cost
    for c1 in azerbaijani_letters:
        for c2 in azerbaijani_letters:
            if c1 == c2:
                confusion_matrix[(c1, c2)] = 0.0
            else:
                confusion_matrix[(c1, c2)] = 1.0
    
    # Apply learned weights from corpus
    if total_substitutions > 0:
        for (c1, c2), count in letter_pairs_count.items():
            # Convert count to probability (more count = lower cost)
            probability = count / total_substitutions
            
            # Convert probability to cost: more frequent = cheaper substitution
            # Range: 0.2 (very common) to 0.8 (rare but observed)
            cost = max(0.2, min(0.8, 1.0 - (probability * 3)))
            
            confusion_matrix[(c1, c2)] = cost
    
    # Add symmetric pairs for observed substitutions
    observed_pairs = list(letter_pairs_count.keys())
    for (c1, c2) in observed_pairs:
        if (c2, c1) not in letter_pairs_count:
            # If we only saw one direction, assume symmetric with slightly higher cost
            confusion_matrix[(c2, c1)] = min(1.0, confusion_matrix.get((c1, c2), 1.0) * 1.2)
    
    # Add common Azerbaijani linguistic patterns if not observed
    add_linguistic_patterns(confusion_matrix, letter_pairs_count)
    
    # Visualize the matrix
    visualize_generated_confusion_matrix(confusion_matrix, letter_pairs_count)
    
    # Save to file
    save_confusion_matrix_to_file(confusion_matrix, letter_pairs_count)
    
    print(f"Generated confusion matrix with weights for {len(confusion_matrix)} letter pairs")
    return confusion_matrix


def find_single_substitution(word1, word2):
    """
    Find which single character differs between two words of same length.
    Returns (char_in_word1, char_in_word2) or None.
    """
    if len(word1) != len(word2):
        return None
    
    diffs = []
    for c1, c2 in zip(word1, word2):
        if c1 != c2:
            diffs.append((c1, c2))
    
    if len(diffs) == 1:
        return diffs[0]
    return None


def analyze_prefix_suffix_patterns(words, letter_pairs_count):
    """
    Analyze common prefix/suffix variations to find letter substitutions.
    """
    # Group words by length
    length_groups = {}
    for word in words[:1000]:
        length = len(word)
        if length >= 4:  # Only analyze words of reasonable length
            length_groups.setdefault(length, []).append(word)
    
    # Look for words that differ only in one position
    for length, word_list in length_groups.items():
        for i in range(len(word_list)):
            for j in range(i+1, min(i+50, len(word_list))):
                word1 = word_list[i]
                word2 = word_list[j]
                
                # Count differences
                diffs = 0
                diff_positions = []
                for pos in range(length):
                    if word1[pos] != word2[pos]:
                        diffs += 1
                        diff_positions.append((word1[pos], word2[pos]))
                
                if diffs == 1:
                    c1, c2 = diff_positions[0]
                    letter_pairs_count[(c1, c2)] += 1


def add_linguistic_patterns(confusion_matrix, observed_pairs):
    """
    Add common Azerbaijani linguistic patterns to the matrix.
    """
    # Common vowel confusions
    vowel_pairs = [
        ('ə', 'e'), ('e', 'ə'),
        ('ı', 'i'), ('i', 'ı'),
        ('ö', 'o'), ('o', 'ö'),
        ('ü', 'u'), ('u', 'ü'),
        ('a', 'ə'), ('ə', 'a'),
    ]
    
    for c1, c2 in vowel_pairs:
        if (c1, c2) not in confusion_matrix or confusion_matrix[(c1, c2)] == 1.0:
            confusion_matrix[(c1, c2)] = 0.4  # Common vowel substitution
    
    # Common diacritic confusions
    diacritic_pairs = [
        ('ç', 'c'), ('c', 'ç'),
        ('ş', 's'), ('s', 'ş'),
        ('ğ', 'g'), ('g', 'ğ'),
    ]
    
    for c1, c2 in diacritic_pairs:
        if (c1, c2) not in confusion_matrix or confusion_matrix[(c1, c2)] == 1.0:
            confusion_matrix[(c1, c2)] = 0.5


def visualize_generated_confusion_matrix(confusion_matrix, observed_counts):
    """
    Visualize the automatically generated confusion matrix.
    """
    # Get Azerbaijani letters in order
    letters = sorted(set('abcçdeəfgğhxıijklmnoöpqrsştuüvyz'))
    
    # Create heatmap data
    size = len(letters)
    heatmap_data = np.zeros((size, size))
    letter_to_idx = {letter: i for i, letter in enumerate(letters)}
    
    for (c1, c2), weight in confusion_matrix.items():
        if c1 in letter_to_idx and c2 in letter_to_idx:
            i = letter_to_idx[c1]
            j = letter_to_idx[c2]
            # Invert weight for visualization: lower cost = higher intensity
            heatmap_data[i, j] = weight
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Substitution Weight (cost)', fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels(letters, fontsize=9)
    ax.set_yticklabels(letters, fontsize=9)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add annotations for observed substitutions
    for (c1, c2), count in observed_counts.most_common(20):
        if c1 in letter_to_idx and c2 in letter_to_idx:
            i = letter_to_idx[c1]
            j = letter_to_idx[c2]
            if count > 0:
                ax.text(j, i, str(count), ha='center', va='center', 
                       color='black', fontsize=8, fontweight='bold')
    
    ax.set_title("Letter Confusion Matrix\n", 
                fontsize=14, pad=20)
    ax.set_xlabel("Substitute Letter", fontsize=12)
    ax.set_ylabel("Original Letter", fontsize=12)
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(OUTPUT_FOLDER, "auto_generated_confusion_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix visualization saved to: {output_file}")
    
    plt.show()


def save_confusion_matrix_to_file(confusion_matrix, observed_counts):
    """
    Save the generated confusion matrix to a file.
    """
    output_file = os.path.join(OUTPUT_FOLDER, "auto_generated_confusion_weights.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("LETTER CONFUSION MATRIX\n")
        f.write("="*60 + "\n")
        f.write("Format: (original, substitute) -> weight (lower = more likely)\n")
        f.write("="*60 + "\n\n")
        
        # Sort by weight (lowest first = most likely substitutions)
        sorted_pairs = sorted(confusion_matrix.items(), key=lambda x: x[1])
        
        for (c1, c2), weight in sorted_pairs:
            if weight < 1.0:  # Only save non-default values
                count = observed_counts.get((c1, c2), 0)
                f.write(f"('{c1}', '{c2}') -> {weight:.3f} (observed: {count} times)\n")
    
    print(f"Confusion matrix weights saved to: {output_file}")


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
    sentences_file = os.path.join(OUTPUT_FOLDER, "sentences.txt")
    with open(sentences_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    print(f"\nNaive sentence count: {len(sentences):,}")
    print(f"First sentence example: {sentences[0]}")
    print(f"Second sentence example: {sentences[1]}")
    print(f"Third sentence example: {sentences[2]}")


    # Spell checker example words (common typos in Azerbaijani)
    test_typos = [
        "azərbaycan", "azəbaycan", "azarbaycan",
        "qarabağ", "qarabag", "qarabaq",
        "məsələ", "məsələn", "meselen",
        "səhifə", "sehife", "telebe", "nomrelerde"
    ]
    simple_spellchecker_demo(vocab_counter, test_typos)

    print("\n" + "="*60)



    confusion_matrix = generate_letter_confusion_matrix_from_corpus(tokens)

    weighted_spellchecker_demo2(
    vocab_counter,
    confusion_matrix,
    test_typos,
    max_dist=3.5,
    top_n=5
)


    user_word = input("\nEnter a word to spell check: ").strip()
    
    if user_word:
        print(f"\nChecking word: '{user_word}'")
        print("-" * 40)
        
        # First check if it's already correct
        if user_word in vocab_counter:
            print(f"✓ Word '{user_word}' is in vocabulary (frequency: {vocab_counter[user_word]:,})")
        else:
            print(f"✗ Word '{user_word}' is not in vocabulary")
        
        # Run both spell checkers on the single word
        test_typos = [user_word]
        
        print("\n[Simple Levenshtein Spell Checker]")
        simple_spellchecker_demo(vocab_counter, test_typos)
        
        print("\n" + "="*40)
        print("\n[Weighted Edit Distance Spell Checker]")
        weighted_spellchecker_demo(vocab_counter, test_typos, max_dist=3.5, top_n=5)
        
    else:
        print("No word entered, skipping spell check.")

    print("Extra task completed: confusion matrix + weighted edit distance added.")
    print("Compare results with the uniform-cost Levenshtein above.")

if __name__ == "__main__":
    main()