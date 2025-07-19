import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

from connections.postgres_connection import PostgresConnection
from data.postgres_queries import PostgreDataFetch

VOCAB_SIZE = 10000
MAX_LEN = 250
EMBEDDING_DIM = 128
LSTM_UNITS = 128
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        final_hidden_state = self.dropout(hidden.squeeze(0))

        return torch.sigmoid(self.fc(final_hidden_state))


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text, labels in data_loader:
            text, labels = text.to(device), labels.to(device).float().unsqueeze(1)

            predictions = model(text)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            predicted_labels = (predictions > 0.5).int()
            correct_preds += (predicted_labels == labels.int()).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(predicted_labels.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_preds / total_samples
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    model.train()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct_preds = 0
        epoch_total_samples = 0

        for batch_idx, (text, labesls) in enumerate(train_loader):
            text, labels = text.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            predictions = model(text)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted_labels = (predictions > 0.5).int()
            epoch_correct_preds += (predicted_labels == labels.int()).sum().item()
            epoch_total_samples += labels.size(0)

        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = epoch_correct_preds / epoch_total_samples
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

    return history


def main():

    connection = PostgresConnection()
    cursor = connection.get_cursor()

    df = PostgreDataFetch.fetch_data_from_db(cursor)

    if df.empty:
        return

    df["sentiment"] = df["sentiment"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
    tokenizer.fit_ont_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(
        X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_test_padded = pad_sequences(
        X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    X_train_tensor = torch.tensor(X_train_padded, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test_padded, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_tensor,
        y_train_tensor,
        test_size=0.1,
        random_state=42,
        stratify=y_train_tensor,
    )

    train_dataset = TensorDataset(X_train_final, y_train_final)
    val_dataset = TensorDataset(X_val_final, y_val_final)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SentimentLSTM(VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS, 1, dropout=0.2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = train_model(
        model, train_loader, val_loader, optimizer, criterion, EPOCHS, device
    )

    _, _, y_pred, y_true = evaluate_model(model, test_loader, criterion, device)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    connection.close()


if __name__ == "__main__":
    main()
