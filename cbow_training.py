import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    words = text.split()
    return words

def build_vocab_and_tokens(corpus, vocab_size=10000):
    word_counts = Counter(corpus)
    most_common = word_counts.most_common(vocab_size - 1)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
    vocab["<UNK>"] = 0
    idx_to_word = {idx: word for word, idx in vocab.items()}
    tokens = [vocab.get(word, 0) for word in corpus]
    return tokens, vocab, idx_to_word

def generate_cbow_data(tokens, window_size):
    data = []
    for i in range(window_size, len(tokens) - window_size):
        context = tokens[i - window_size : i] + tokens[i + 1 : i + window_size + 1]
        target = tokens[i]
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

def train_cbow(
    model, cbow_data, idx_to_word, batch_size=128, num_epochs=5, lr=0.01, print_every=1000, device="cpu"
):
    dataset = CBOWDataset(cbow_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, (contexts, targets) in enumerate(dataloader):
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
        print(f"Epoch {epoch+1} complete. Avg loss: {total_loss / len(dataloader):.4f}")

def main(data_path, vocab_size, embedding_dim, window_size, batch_size, num_epochs, lr, device="cpu"):
    words = load_corpus(data_path)
    tokens, vocab, idx_to_word = build_vocab_and_tokens(words, vocab_size=vocab_size)
    cbow_data = generate_cbow_data(tokens, window_size)
    model = CBOW(vocab_size, embedding_dim)
    train_cbow(
        model, cbow_data, idx_to_word,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        device=device
    )

if __name__ == "__main__":
        main(data_path="text8", 
             vocab_size=5000, 
             embedding_dim=20, # how many dimensions to use to represent each word.
             window_size=4, # how many words to include either side of the target.
             batch_size=128, 
             num_epochs=5, 
             lr=0.01
             )