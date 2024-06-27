import streamlit as st
import numpy as np
import torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import re

# Define the LyricsDataset class
class LyricsDataset(Dataset):
    def __init__(self, text, seq_length):
        chars = sorted(list(set(text)))
        self.char_to_int = {ch: i for i, ch in enumerate(chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(chars)}
        self.data = [self.char_to_int[ch] for ch in text]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (torch.tensor(self.data[index:index+self.seq_length]),
                torch.tensor(self.data[index+1:index+self.seq_length+1]))

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, states):
        x = self.embedding(x)
        x, states = self.lstm(x, states)
        x = self.fc(x)
        return x, states

    def init_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())

# Load vocabulary
@st.cache_data(allow_output_mutation=True)
def load_vocab(file_path):
    with open(file_path, 'r') as file:
        vocab_words = file.read().splitlines()
    return set(vocab_words)

# Load and prepare lyrics
@st.cache_data(allow_output_mutation=True)
def load_lyrics(file_path):
    lyrics_df = pd.read_parquet(file_path)
    return ' '.join(lyrics_df['Lyrics_clean'].values)

# Filter lyrics based on the vocabulary set
def lyrics_filter(lyrics, vocab_set):
    return ' '.join(word if word.lower() in vocab_set else '' for word in re.findall(r'\b\w+\b', lyrics)).strip()

# Load model
@st.cache_data(allow_output_mutation=True)
def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_layers):
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Generate lyrics
def generate_lyrics(model, start_str, int_to_char, char_to_int, length):
    model.eval()
    states = model.init_states(1)
    input_indices = [char_to_int[ch] for ch in start_str.lower() if ch in char_to_int]

    input_tensor = torch.tensor([input_indices]).cuda()
    text = start_str

    for _ in range(length):
        output, states = model(input_tensor, states)
        output = output[:, -1, :]
        probabilities = torch.softmax(output, dim=1)
        char_id = torch.multinomial(probabilities, 1).item()
        char = int_to_char[char_id]
        text += char
        input_tensor = torch.tensor([[char_id]]).cuda()

    return text

# Streamlit interface
def main():
    st.title("One Direction Song Lyrics Generator")
    user_input = st.text_input("Enter the starting words of the song:", "You are")

    if st.button('Generate Lyrics'):
        vocab_path = "C:\\Users\\Ananya\\anaconda3\\Dissertation - UL\\Music-Generation-with-AI-1\\Project\\data\\The_Oxford_3000.txt"
        lyrics_path = "C:\\Users\\Ananya\\anaconda3\\Dissertation - UL\\Music-Generation-with-AI-1\\Project\\data\\One_Direction_cleaned_lyrics.parquet"
        model_path = "C:\\Users\\Ananya\\anaconda3\\Dissertation - UL\\Music-Generation-with-AI-1\\Project\lstm_model_lyrics.pth"
        vocab_set = load_vocab(vocab_path)
        all_lyrics = load_lyrics(lyrics_path)
        all_lyrics_filt = lyrics_filter(all_lyrics, vocab_set)
        dataset = LyricsDataset(all_lyrics_filt, 100)
        model = load_model(model_path, len(dataset.char_to_int), 256, 512, 2)
        final_lyrics = generate_lyrics(model, user_input, dataset.int_to_char, dataset.char_to_int, 1000)
        st.text_area("Generated Lyrics", value=final_lyrics, height=300)

if __name__ == '__main__':
    st.run(main())
