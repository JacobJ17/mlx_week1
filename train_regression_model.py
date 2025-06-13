import html
import io
import random
import re
import string
import pickle
from collections import Counter
from cbow_training import CBOW
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import wandb

class CBOWEmbedder(torch.nn.Module):
    def __init__(self, cbow_model_path):
        super().__init__()
        checkpoint = torch.load(cbow_model_path, map_location='cpu')
        self.model = CBOW(vocab_size=checkpoint['vocab_size'], embedding_dim=checkpoint['embedding_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def embed_sentence(self, token_ids):
        # token_ids: tensor of shape (seq_len,)
        embs = self.model.embeddings(token_ids)  # (seq_len, emb_dim)
        return torch.mean(embs, dim=0)  # (emb_dim,)
    

class HNRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, cbow_vocab, embedder):
        self.df = df
        self.tokenizer = tokenizer  # maps text -> list of tokens
        self.vocab = cbow_vocab     # maps token -> index
        self.embedder = embedder    # CBOWEmbedder class

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Title embedding
        tokens = self.tokenizer(row['clean_title'])  # list of str
        token_ids = [self.vocab.get(tok, self.vocab.get('<unk>', 0)) for tok in tokens]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        title_emb = self.embedder.embed_sentence(token_tensor)

        # Other features (numerical, categorical)
        domain_idx = torch.tensor(row['domain_idx'], dtype=torch.long)
        user_age = torch.tensor(row['user_age_scaled'], dtype=torch.float32)
        karma = torch.tensor(row['user_karma_log'], dtype=torch.float32)
        activity = torch.tensor(row['user_submitted_log'], dtype=torch.float32)
        tod_sin = torch.tensor(row['tod_sin'], dtype=torch.float32)
        tod_cos = torch.tensor(row['tod_cos'], dtype=torch.float32)
        weekday_ohe = torch.tensor([row['is_weekday'], row['is_saturday'], row['is_sunday']], dtype=torch.float32)

        features = torch.cat([
            title_emb,
            weekday_ohe,
            tod_sin.unsqueeze(0),
            tod_cos.unsqueeze(0),
            user_age.unsqueeze(0),
            karma.unsqueeze(0),
            activity.unsqueeze(0)
        ])  # shape: (N,)

        return features, row['score']


class HNRegressor(torch.nn.Module):
    def __init__(self, input_dim, domain_vocab_size, domain_emb_dim=4):
        super().__init__()
        self.domain_emb = torch.nn.Embedding(domain_vocab_size, domain_emb_dim)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim + domain_emb_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, features, domain_idx):
        dom_emb = self.domain_emb(domain_idx)  # (B, emb_dim)
        x = torch.cat([features, dom_emb], dim=1)
        return self.regressor(x).squeeze(1)


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for features, targets in dataloader:
        domain_idx = features[:, -1].long()  # assume domain_idx is last
        x = features[:, :-1].float().to(device)
        y = targets.float().to(device)

        optimizer.zero_grad()
        preds = model(x, domain_idx.to(device))
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)

def fast_clean_and_tokenize(text):
    if not isinstance(text, str):
        return None
    text = html.unescape(text)
    text = re.sub(r"&", "and", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("-", " ")  # <-- NEW: break up compound words
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    tokens = text.lower().strip().split()
    tokens = [t for t in tokens if t.isalnum()]
    return tokens if tokens else None

def main():
    # Load CBOWEmbedder and vocab
    embedder = CBOWEmbedder('cbow_model.pth')
    
    with open("output_titles/vocab.pkl", "rb") as f:
        cbow_vocab = pickle.load(f)
    
    tokenizer = fast_clean_and_tokenize  # or your actual tokeniser

    # Build dataset + dataloader
    dataset = HNRegressionDataset(df, tokenizer, cbow_vocab, embedder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Instantiate model
    input_dim = 100  # adjust based on features
    domain_vocab_size = 101
    model = HNRegressor(input_dim=input_dim, domain_vocab_size=domain_vocab_size)

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(10):
        loss = train(model, dataloader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}: loss = {loss:.4f}")
