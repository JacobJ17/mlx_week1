import re
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from cbow_training import CBOW
from load_df import load_and_preprocess_data
from tqdm import tqdm
import wandb


class CBOWEmbedder(torch.nn.Module):
    def __init__(self, cbow_model_path):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(cbow_model_path, map_location=device)
        self.vocab_size = checkpoint['vocab_size']
        self.embedding_dim = checkpoint['embedding_dim']
        self.vocab = checkpoint['vocab']  # dict: token -> idx
        self.idx_to_word = checkpoint.get('idx_to_word', None)  # optional reverse mapping

        self.model = CBOW(vocab_size=self.vocab_size, embedding_dim=self.embedding_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(device)
        self.device = device

    def embed_sentence(self, token_ids):
        token_ids = token_ids.to(self.device)
        with torch.no_grad():
            embs = self.model.embeddings(token_ids)
            return embs.mean(dim=0)


def tokeniser(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r"&", "and", text)
    text = text.lower()
    text = text.replace("(", "").replace(")", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("<", "").replace(">", "")
    text = text.replace("'", "")
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace("-", " ")
    text = text.replace(':', ' <COLON> ')
    text = re.sub(r'[^\w\s<>]', '', text)
    tokens = text.strip().split()
    return tokens


class HNRegressionDataset(Dataset):
    def __init__(self, df, tokenizer, cbow_vocab, embedder):
        self.df = df
        self.tokenizer = tokenizer
        self.vocab = cbow_vocab
        self.embedder = embedder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tokens = self.tokenizer(row['title'])
        token_ids = [self.vocab.get(tok, self.vocab.get('<unk>', 0)) for tok in tokens]
        if not token_ids:
            token_ids = [self.vocab.get('<unk>', 0)]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        title_emb = self.embedder.embed_sentence(token_tensor)

        domain_idx = torch.tensor(row['domain_idx'], dtype=torch.long)
        user_age = torch.tensor(row['log_user_age'], dtype=torch.float32)
        karma = torch.tensor(row['log_karma'], dtype=torch.float32)
        activity = torch.tensor(row['log_submitted_count'], dtype=torch.float32)
        tod_sin = torch.tensor(row['time_sin'], dtype=torch.float32)
        tod_cos = torch.tensor(row['time_cos'], dtype=torch.float32)
        weekday_ohe = torch.tensor([
            row['day_group_weekday'],
            row['day_group_saturday'],
            row['day_group_sunday']
        ], dtype=torch.float32)

        features = torch.cat([
            title_emb,
            weekday_ohe,
            tod_sin.unsqueeze(0),
            tod_cos.unsqueeze(0),
            user_age.unsqueeze(0),
            karma.unsqueeze(0),
            activity.unsqueeze(0)
        ])  # shape: (108,)

        return features, domain_idx, torch.tensor(row['log_score'], dtype=torch.float32)


class HNRegressor(nn.Module):
    def __init__(self, input_dim, domain_vocab_size, domain_emb_dim=8, dropout_p=0.3):
        super().__init__()
        self.domain_emb = nn.Embedding(domain_vocab_size, domain_emb_dim)

        self.bn1 = nn.BatchNorm1d(input_dim + domain_emb_dim)

        self.model = nn.Sequential(
            nn.Linear(input_dim + domain_emb_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features, domain_idx):
        dom_emb = self.domain_emb(domain_idx)
        x = torch.cat([features, dom_emb], dim=1)
        x = self.bn1(x)
        return self.model(x).squeeze(1)


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for features, domain_idx, targets in dataloader:
        features = features.to(device)
        domain_idx = domain_idx.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(features, domain_idx)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)

    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, targets in dataloader:
            domain_idx = features[:, -1].long().to(device)  # domain index assumed last in features
            x = features[:, :-1].float().to(device)
            y = targets.float().to(device)

            preds = model(x, domain_idx)
            loss = loss_fn(preds, y)
            total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)


def main(
    cbow_model_path,
    parquet_path,
    batch_size,
    lr,
    num_epochs,
    domain_vocab_size,
    domain_emb_dim,
    project_name,
    run_name
):
    # Init wandb
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "loss": "SmoothL1Loss",
            "domain_emb_dim": domain_emb_dim,
            "log_target": True
        }
    )

    # Load CBOW embedder and vocab
    embedder = CBOWEmbedder(cbow_model_path)
    cbow_vocab = embedder.vocab  # get vocab dict directly from embedder

    # Load and preprocess dataset
    df = load_and_preprocess_data(parquet_path)

    # Build dataset and dataloader
    dataset = HNRegressionDataset(df, tokeniser, cbow_vocab, embedder)
    val_fraction = 0.2
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Infer input dimension
    sample_features, _ = dataset[0]
    input_dim = sample_features.shape[0]

    # Create model
    model = HNRegressor(input_dim=input_dim, domain_vocab_size=domain_vocab_size, domain_emb_dim=domain_emb_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")        wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch + 1
    })

    wandb.finish()


if __name__ == "__main__":
    main(
        cbow_model_path="cbow_model.pth",
        parquet_path="hacker_news_ml_ready.parquet",
        batch_size=64,
        lr=1e-3,
        num_epochs=10,
        domain_vocab_size=101,
        domain_emb_dim=4,
        project_name="hacker-news-regression",
        run_name="run_" + wandb.util.generate_id()
    )
