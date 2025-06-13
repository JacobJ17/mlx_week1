import argparse
import html
import json
import os
import pickle
import random
import re
import string
from collections import Counter
from itertools import chain

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from tqdm import tqdm


# Load spaCy English tokenizer with language detection
@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe("sentencizer")  # Required to support language detector that uses doc.sents
nlp.add_pipe("language_detector", last=True)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def clean_text(text):
    # Unescape HTML entities like &amp; to &
    text = html.unescape(text)
    # Replace ampersands with "and"
    text = re.sub(r"&", "and", text)
    # Remove text in parentheses or brackets
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    # Remove remaining punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_segments_from_file(path, delimiter):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    segments = re.split(delimiter, raw)
    # segments = segments[:50000]
    return [seg.strip() for seg in segments if seg.strip()]

def tokenize_segments(segments, lang="en"):
    tokenised = []
    cleaned_segments = [clean_text(seg) for seg in segments]
    total = len(cleaned_segments)
    for doc in tqdm(nlp.pipe(cleaned_segments, batch_size=1024), total=total, desc="Tokenizing segments"):
        # if doc._.language["language"] != lang:
        #     continue
        tokens = [tok.text.lower() for tok in doc if not tok.is_space]
        if len(tokens) > 2:
            tokenised.append(tokens)
    return tokenised


def build_vocab(tokenised_segments, vocab_size, min_freq=5):
    tokens = chain.from_iterable(tokenised_segments)
    counter = Counter(tokens)
    most_common = [(w, c) for w, c in counter.items() if c >= min_freq]
    most_common = sorted(most_common, key=lambda x: -x[1])[:vocab_size - 1]
    vocab = {word: i + 1 for i, (word, _) in enumerate(most_common)}
    vocab["<UNK>"] = 0
    idx_to_word = {i: w for w, i in vocab.items()}
    return vocab, idx_to_word

def generate_cbow_data(segments, vocab, window_size, drop_unknown_targets=True):
    data = []
    for segment in tqdm(segments, desc="Generating CBOW pairs"):
        token_ids = [vocab.get(w, 0) for w in segment]
        for i in range(window_size, len(token_ids) - window_size):
            context = token_ids[i - window_size:i] + token_ids[i+1:i + window_size + 1]
            target = token_ids[i]
            if drop_unknown_targets and target == 0:
                continue
            data.append((context, target))
    return data

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main():
    input_paths = ["hn_comments_gt5_replies.txt"]
    # delimiters = ["<TITLE>"]
    delimiters = [r"<COMMENT>\d+\n"]
    output_dir = f"output_{input_paths[0].split('.')[0]}"
    vocab_size = 9000
    window_size = 3
    min_freq = 5

    os.makedirs(output_dir, exist_ok=True)

    all_tokenised = []

    # vocab = load_pickle(os.path.join(output_dir, "vocab.pkl"))
    # idx_to_word = load_pickle(os.path.join(output_dir, "idx_to_word.pkl"))

    print("Loading and tokenising...")
    for path, delim in zip(input_paths, delimiters):
        segments = load_segments_from_file(path, delim)
        tokenised = tokenize_segments(segments, lang="en")
        print(f"{path}: {len(tokenised)} English segments")
        all_tokenised.extend(tokenised)
    save_pickle(all_tokenised, os.path.join(output_dir, "all_tokenised.pkl"))

    print("Building vocabulary...")
    vocab, idx_to_word = build_vocab(all_tokenised, vocab_size, min_freq)
    save_pickle(vocab, os.path.join(output_dir, "vocab.pkl"))
    save_pickle(idx_to_word, os.path.join(output_dir, "idx_to_word.pkl"))

    print("Generating CBOW pairs...")
    cbow_data = generate_cbow_data(all_tokenised, vocab, window_size)
    save_pickle(cbow_data, os.path.join(output_dir, "cbow_data.pkl"))

    print(f"Saved {len(cbow_data)} CBOW examples to {output_dir}")
    # Print 10 random CBOW examples using words instead of indices
    print("Random CBOW pairs (context -> target):")
    for context, target in random.sample(cbow_data, 10):
        context_words = [idx_to_word.get(idx, "<UNK>") for idx in context]
        target_word = idx_to_word.get(target, "<UNK>")
        print(f"{context_words} -> {target_word}")
if __name__ == "__main__":
    main()


# You may need to run "python -m spacy download en_core_web_sm" to download the English model for spaCy.

# python create_cbow_pairs.py \
#   --input hn_titles_delimited.txt \
#   --delimiters '<TITLE>' \
#   --output_dir hn_titles_output \
#   --vocab_size 10000 \
#   --window_size 3 \
#   --min_freq 5
