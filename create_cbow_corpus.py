import html
import os
import pickle
import re
import string
from tqdm import tqdm

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def tokenise(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"&", "and", text)
    text = text.lower()
    text = text.replace("(", "").replace(")", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("<", "").replace(">", "")
    text = text.replace("'", "")
    # text = text.replace('.',  ' <PERIOD> ')
    # text = text.replace(',',  ' <COMMA> ')
    # text = text.replace('"',  ' <QUOTATION_MARK> ')
    # text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace("-", " ")
    text = text.replace(':',  ' <COLON> ')
    text = re.sub(r'[^\w\s<>]', '', text)
    tokens = text.strip().split()
    return " ".join(tokens)

def load_segments_from_file(path, delimiter):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    segments = re.split(delimiter, raw)
    return [seg.strip() for seg in segments if seg.strip()]


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
    input_path = "hn_titles_delimited.txt"
    # delimiter = r"<COMMENT>\d+\n"
    delimiter = r"<TITLE>"
    output_dir = f"output_{input_path.split('.')[0]}"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading and tokenising...")
    segments = load_segments_from_file(input_path, delimiter)
    output_strings = []
    
    for seg in tqdm(segments, desc="Tokenizing segments"):
        if "<COMMENT>" in delimiter:
            tokens = tokenise(seg)
            if tokens:
                output_strings.append("<COMMENT> " + tokens)
        elif "TITLE" in delimiter:
            tokens = tokenise(seg)
            if tokens:
                output_strings.append("<TITLE> " + tokens)

    # Join everything into one long string, separating by spaces
    long_string = " ".join(output_strings)
    with open(os.path.join(output_dir, "tokenized.txt"), "w", encoding="utf-8") as f:
        f.write(long_string)
    print(f"Saved tokenized text to {os.path.join(output_dir, 'tokenized.txt')}")

if __name__ == "__main__":
    main()