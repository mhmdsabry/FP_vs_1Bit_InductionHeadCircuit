import torch
import random
from random import choice, choices

from torch.utils.data import Dataset

seq_len = 28 # total sequence
prefix_len = 4 # region where first induction token occurs
vocab = "abcdefghijklmnopqrstuvwxyz"
induction_token = "&"
assert prefix_len < seq_len - 4

def generate_sentence():
    # prefix is the region where the special token first occurs
    memory_tok = choice(vocab)
    induction_pos = choice(range(prefix_len))
    # Section 3.1 from https://arxiv.org/pdf/2212.14052.pdf the 'special token'
    induction = [induction_token, memory_tok]
    pre = choices(vocab, k=induction_pos)
    noise = choices(vocab, k=seq_len-induction_pos-2)
    seq = pre + induction + noise + induction
    induction_index = len(pre) + len(induction) - 1
    return seq, induction_index

# Generate 6000 sentences (5000 for train, 500 for eval, 500 for test)
sentences = []
induction_indices = []
for _ in range(6000):
    sentence, induction_index = generate_sentence()
    induction_indices.append(induction_index)
    sentences.append(sentence)


class IHCDataset(Dataset):
    def __init__(self, data, data_induction_indices):
        chars = vocab+"&"

        data_size, vocab_size = len(data), len(chars)

        self.tokenizer = {ch:i for i, ch in enumerate(chars)}
        self.decoder = {i:ch for i,ch in enumerate(chars)}

        self.vocab_size = vocab_size
        self.data_induction_indices = data_induction_indices
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        induction_index = self.data_induction_indices[idx]
        encoding = torch.tensor([self.tokenizer[c] for c in chunk], dtype=torch.long)

        input_ids = encoding[:-1]
        labels = encoding[1:]
        return input_ids.long(), labels.long(), induction_index  #x,y, ind_idx


def get_induction_data():
    train_set = sentences[:5500]
    train_induction_indices = induction_indices[:5500]

    eval_set = sentences[5500:]
    eval_induction_indices = induction_indices[5500:]

    train_dataset = IHCDataset(train_set, train_induction_indices)
    eval_dataset = IHCDataset(eval_set, eval_induction_indices)

    block_size = len(sentences[0])
    return train_dataset, eval_dataset, block_size


if __name__ == "__main__":
    dataset = IHCDataset([sentences[0], sentences[1]], [induction_indices[0], induction_indices[1]])
    print("len", len(dataset))
    print(dataset.get_vocab_size())
    print("input_ids", dataset[0][0].shape)
    print("input ids", dataset[0])
    print("labels", dataset[0][1].shape)
    print(" text sample\n", ''.join([dataset.decoder[c.item()] for c in dataset[0][0][:30]]))
    print(" label sample\n", ''.join([dataset.decoder[c.item()] for c in dataset[0][1][:30]]))
    print(f"Induction index: {dataset[0][2]}")
    print(" text sample\n", ''.join([dataset.decoder[c.item()] for c in dataset[1][0][:30]]))
    print(" label sample\n", ''.join([dataset.decoder[c.item()] for c in dataset[1][1][:30]]))
    print(f"Induction index: {dataset[1][2]}")