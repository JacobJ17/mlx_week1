import random
from collections import Counter
import io
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, random_split

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    words = text.split()
    return words

def load_segmented_corpus(path, delimiter="\n"):
    with open(path, "r", encoding="utf-8") as f:
        segments = f.read().strip().split(delimiter)
    return [segment.strip().split() for segment in segments if segment.strip()]


def build_vocab_and_tokens(corpus, vocab_size=10000):
    word_counts = Counter(corpus)
    most_common = word_counts.most_common(vocab_size - 1)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
    vocab["<UNK>"] = 0
    idx_to_word = {idx: word for word, idx in vocab.items()}
    tokens = [vocab.get(word, 0) for word in corpus]
    return tokens, vocab, idx_to_word

def generate_cbow_data(tokens, window_size, drop_unknown_targets=True):
    data = []
    for i in range(window_size, len(tokens) - window_size):
        context = tokens[i - window_size : i] + tokens[i + 1 : i + window_size + 1]
        target = tokens[i]
        if drop_unknown_targets and target == 0:  # Skip unknown tokens
            continue
        data.append((context, target))
    return data

def generate_segmented_cbow_data(token_segments, vocab, window_size, drop_unknown_targets=True):
    data = []
    for segment in token_segments:
        tokens = [vocab.get(word, 0) for word in segment]
        if len(tokens) < 2 * window_size + 1:
            continue  # Skip too-short segments
        for i in range(window_size, len(tokens) - window_size):
            context = tokens[i - window_size : i] + tokens[i + 1 : i + window_size + 1]
            target = tokens[i]
            if drop_unknown_targets and target == 0:
                continue
            data.append((context, target))
    return data


class CBOWDataset(Dataset):
    def __init__(self, cbow_data):
        self.data = cbow_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)
    def forward(self, inputs):
        embs = self.embeddings(inputs)
        embs = torch.mean(embs, dim=1)
        out = self.linear(embs)
        return torch.nn.functional.log_softmax(out, dim=1)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts, targets = contexts.to(device), targets.to(device)
            output = model(contexts)
            loss = F.nll_loss(output, targets, reduction="sum")
            pred = output.argmax(dim=1)
            total_correct += (pred == targets).sum().item()
            total_loss += loss.item()
            total_count += targets.size(0)
    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count
    return avg_loss, accuracy

def train_cbow(
    model, train_data, val_data, idx_to_word, batch_size=128, num_epochs=5, lr=0.01, print_every=1000, device="cpu"
):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, (contexts, targets) in enumerate(train_loader):
            contexts, targets = contexts.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(contexts)
            loss = torch.nn.functional.nll_loss(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (step + 1) % print_every == 0:
                with torch.no_grad():
                    pred_idx = output.argmax(dim=1)[0].item()
                    pred_word = idx_to_word.get(pred_idx, str(pred_idx))
                    context_words = [idx_to_word.get(c.item(), str(c.item())) for c in contexts[0]]
                    target_word = idx_to_word.get(targets[0].item(), str(targets[0].item()))
                    print(f"Epoch {epoch+1} Step {step+1}: Context: {context_words} -> Target: {target_word} | "
                          f"Predicted: {pred_word} | Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1} complete. Avg loss: {total_loss / len(train_loader):.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

def plot_embeddings(model, idx_to_word, method="pca", num_points=200):
    embeddings = model.embeddings.weight.data.cpu().numpy()
    words = [idx_to_word[i] for i in range(len(idx_to_word)) if i != 0][:num_points]
    vectors = embeddings[1:num_points+1]  # skip <UNK>

    if method == "pca":
        projector = PCA(n_components=2)
    elif method == "tsne":
        projector = TSNE(n_components=2, perplexity=30, init='random', random_state=42)
    else:
        raise ValueError("Unknown method")

    reduced = projector.fit_transform(vectors)

    plt.figure(figsize=(12, 10))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)

    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced[i, 0], reduced[i, 1]), fontsize=9, alpha=0.75)

    plt.title(f"{method.upper()} Projection of Word Embeddings")
    plt.grid(True)
    plt.tight_layout()
    # Save plot to a buffer and log to W&B
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    wandb.log({f"{method}_embedding_plot": wandb.Image(image)})
    plt.close()  # Close the plot so it doesn't show in notebooks etc.

def main(data_path, vocab_size, embedding_dim, batch_size, num_epochs, lr, device="cpu", save_path="cbow_model.pth"):
    set_seed(42)
    # Initialize Weights & Biases
    wandb.init(
        project="cbow-word2vec",
        config={
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            # "window_size": window_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr
        }
    )

    # Load and preprocess the corpus
    words1 = load_corpus("text8")
    segments2 = load_segmented_corpus("output_hn_titles_delimited/tokenized.txt", delimiter="<TITLE>")
    words2 = [word for segment in segments2 for word in segment]  # flatten segments
    segments3 = load_segmented_corpus("output_hn_comments_gt5_replies/tokenized.txt", delimiter="<COMMENT>")
    words3 = [word for segment in segments3 for word in segment]  # flatten segments

    all_words = words1 + words2 + words3
    tokens, vocab, idx_to_word = build_vocab_and_tokens(all_words, vocab_size=vocab_size)
    tokens1 = [vocab.get(w, 0) for w in words1]
    cbow_data1 = generate_cbow_data(tokens1, window_size=4)
    cbow_data2 = generate_segmented_cbow_data(segments2, vocab, window_size=3)
    cbow_data3 = generate_segmented_cbow_data(segments3, vocab, window_size=4)

    cbow_data = cbow_data1 + cbow_data2 + cbow_data3


    # print("Loading corpus...")
    # words = load_corpus(data_path)
    # tokens, vocab, idx_to_word = build_vocab_and_tokens(words, vocab_size=vocab_size)
    # print(f"Vocabulary size: {len(vocab)}")
    # # Generate CBOW data
    # cbow_data = generate_cbow_data(tokens, window_size, drop_unknown_targets=True)
    full_dataset = CBOWDataset(cbow_data)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Ensures reproducible split
    )
    print(f"Generated {len(cbow_data)} CBOW training examples.")
    model = CBOW(vocab_size, embedding_dim)
    print("Training CBOW model...")
    # Train the CBOW model
    train_cbow(
        model, train_dataset, 
        val_dataset, idx_to_word,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        device=device
    )
    # Save the trained model
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "vocab": vocab,
        "idx_to_word": idx_to_word
    }, save_path)
    print(f"Model saved to {save_path}")

    # Visualize embeddings:
    plot_embeddings(model, idx_to_word, method="tsne", num_points=300)
    

if __name__ == "__main__":
        main(data_path="text8", 
             vocab_size=30000,
             embedding_dim=150, # how many dimensions to use to represent each word.
             batch_size=1024,
             num_epochs=10, 
             lr=0.001,
             device = "cuda" if torch.cuda.is_available() else "cpu"
             )