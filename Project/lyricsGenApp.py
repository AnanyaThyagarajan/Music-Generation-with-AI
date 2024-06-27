import streamlit as st
import numpy as np
import torch
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import re
# Assuming model and other functions are properly imported or included here

@st.cache(allow_output_mutation=True)
#loading the vocabulary from a text 

def load_vocab(file_path):
    with open(file_path, 'r') as file:
        vocab_words = file.read().splitlines()
    return set(vocab_words)

#loading and preparing the lyrics

def loading_lyrics(file_path2):
    lyrics_df2 = pd.read_parquet(file_path2)
    return ' '.join(lyrics_df2['Lyrics_clean'].values)
#filtering lyrics based on the vocab set

def lyrics_filter(lyrics, vocab_set):
    filtered = ' '.join([word if word.lower() in vocab_set else '' for word in re.findall(r'\b\w+\b', lyrics)])
    return re.sub(r'\s+', ' ', filtered).strip()


# Lyrics Dataset
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

vocab_path = "C:\\Users\\Ananya\\anaconda3\\Dissertation - UL\\Music-Generation-with-AI-1\\Project\\data\\The_Oxford_3000.txt"
lyrics_path = "C:\\Users\\Ananya\\anaconda3\\Dissertation - UL\\Music-Generation-with-AI-1\\Project\\data\\One_Direction_cleaned_lyrics.parquet"
model_save_path = "C:\\Users\\Ananya\\anaconda3\\Dissertation - UL\\Music-Generation-with-AI-1\\Project\lstm_model_lyrics.pth"

# loading vocab and lyrics

vocab_set = load_vocab(vocab_path)
all_lyrics = loading_lyrics(lyrics_path)
all_lyrics_filt = lyrics_filter(all_lyrics, vocab_set)

# Dataset and DataLoader
seq_length = 100
batch_size = 64
dataset = LyricsDataset(all_lyrics_filt, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def generate_lyrics(model, start_str, int_to_char, char_to_int, length):
    model.eval()
    states = model.init_states(1)
    input_indices = []

    # Convert start string to lowercase to match training data
    start_str = start_str.lower()

    # Convert characters to indices safely
    for ch in start_str:
        if ch in char_to_int:
            input_indices.append(char_to_int[ch])
        else:
            print(f'Character "{ch}" not in dictionary, skipping.')
            continue  # Or handle unknown character

    if not input_indices:
        print("No valid characters to process.")
        return ""

    input_tensor = torch.tensor([input_indices]).cuda()
    text = start_str

    for _ in range(length):
        output, states = model(input_tensor, states)
        output = output[:, -1, :]  # Get the last time step
        probabilities = torch.softmax(output, dim=1)
        char_id = torch.multinomial(probabilities, 1).item()
        char = int_to_char[char_id]
        text += char
        input_tensor = torch.tensor([[char_id]]).cuda()

    return text
def save_lyrics(lyrics, filename):
    with open(filename, 'w') as file:
        file.write(lyrics)

def load_model():
    # Dummy function to load and return the trained model, replace with your actual model loading
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers).cuda()
    model.load_state_dict(torch.load('model_save_path.pt'))  # Ensure model is saved and path is correct
    model.eval()
    return model

# Function to generate text and insert names into the lyrics
def insert_names_and_generate(model, start_str):
    generated_lyrics = generate_lyrics(model, start_str, dataset.int_to_char, dataset.char_to_int, 1000)
    names = ['Harry', 'Niall', 'Louis', 'Zayn', 'Liam']
    np.random.shuffle(names)  # Shuffle names to distribute randomly
    parts = np.array_split(generated_lyrics.split(), len(names))
    lyrics_with_names = []
    for name, part in zip(names, parts):
        part.insert(0, name + ":")
        lyrics_with_names.append(" ".join(part))
    return "\n\n".join(lyrics_with_names)

# Streamlit interface
def main():
    st.title("One Direction Song Lyrics Generator")
    user_input = st.text_input("Enter the starting words of the song:", "You are")

    if st.button('Generate Lyrics'):
        model = load_model()
        final_lyrics = insert_names_and_generate(model, user_input)
        st.text_area("Generated Lyrics", value=final_lyrics, height=300, help="Generated lyrics with names distributed.")

if __name__ == '__main__':
    main()
