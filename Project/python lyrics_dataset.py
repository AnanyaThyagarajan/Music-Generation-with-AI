import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset class definition
class LyricsDataset(Dataset):
    def __init__(self, text, seq_length):
        """
        Initializes the dataset with given text and sequence length.
        Args:
            text (str): The complete string of text data.
            seq_length (int): The length of each input sequence.
        """
        self.chars = sorted(list(set(text)))  # Unique characters sorted
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char_to_int[ch] for ch in text]  # Convert all chars to integers
        self.seq_length = seq_length

    def __len__(self):
        # The number of sequences that can be generated
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        """
        Fetches the input and target sequences from the dataset.
        Args:
            index (int): The index position to start the sequence.
        Returns:
            tuple: containing the input and target sequences
        """
        input_seq = torch.tensor(self.data[index:index+self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.data[index+1:index+self.seq_length+1], dtype=torch.long)
        return input_seq, target_seq

# Example usage of LyricsDataset
def main():
    text = "Here is a sample of example text that could represent lyrics."
    seq_length = 10  # Length of each sequence
    dataset = LyricsDataset(text, seq_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Using a small batch size for demonstration

    # Output some sample data from the dataloader
    for input_seq, target_seq in dataloader:
        print("Input sequence (indices):", input_seq)
        print("Target sequence (indices):", target_seq)
        # Decode back to characters to visualize what's being processed
        input_chars = ''.join(dataset.int_to_char[i] for i in input_seq[0])
        target_chars = ''.join(dataset.int_to_char[i] for i in target_seq[0])
        print("Decoded Input:", input_chars)
        print("Decoded Target:", target_chars)
        break  # Only display the first batch for brevity

if __name__ == '__main__':
    main()
