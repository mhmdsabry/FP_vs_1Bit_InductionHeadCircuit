import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def plot_losses(train_losses, eval_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True)
    if not os.path.exists("./imgs"):
        os.makedirs("./imgs")
    plt.savefig(f'./imgs/learning_curve_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
    plt.show()

def extract_last_token_attention(attention_scores):
    """
    Extracts the attention scores for the last token across all layers and heads.
    :param attention_scores: Nested list [batch][layer][head][seq_length, seq_length]
    :return: A tensor of shape [num_layers, num_heads] representing the average
             attention score of the last token across all batches.
    """
    num_batches = len(attention_scores)

    num_layers = len(attention_scores[0][0])

    num_samples = len(attention_scores[0][0][0])

    num_heads = len(attention_scores[0][0][0][0])

  
    layer_scores = []
    for layer in range(num_layers):
        head_scores = []
        for head in range(num_heads):
            batch_scores = []
            for batch in range(num_batches):
                induction_indices = attention_scores[batch][1]
                sample_scores = []
                for sample in range(num_samples):
                    # Assuming attention_scores[batch][layer][head] is a tensor
                    last_token_score = attention_scores[batch][0][layer][sample][head][-1, induction_indices[sample]].mean().item()
                    sample_scores.append(last_token_score)
                batch_scores.append(sum(sample_scores)/len(sample_scores))
            head_scores.append(sum(batch_scores)/len(batch_scores))
        layer_scores.append(head_scores)

    return torch.tensor(layer_scores)


def old_plot_attention_scores(attention_scores1, attention_scores2, id=""):
    """
    Plots the attention scores for the last token across all layers and heads for two epochs side by side.
    
    attention_scores1: A tensor of shape [num_layers, num_heads] for the first condition.
    attention_scores2: A tensor of shape [num_layers, num_heads] for the second condition.
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns
    # Find the overall minimum value across both tensors
    min_score = 0
    max_score = 1

    # Plot for the first condition
    cax1 = axs[0].matshow(attention_scores1.numpy(), cmap='Blues', origin='lower', aspect='auto', vmin=min_score, vmax=max_score)
    #fig.colorbar(cax1, ax=axs[0])
    axs[0].set_xlabel('Head')
    axs[0].set_ylabel('Layer')
    axs[0].set_title(f'Induction Heads Scores-FP Model')

    # Plot for the second condition
    cax2 = axs[1].matshow(attention_scores2.numpy(), cmap='Blues', origin='lower', aspect='auto', vmin=min_score, vmax=max_score)
    axs[1].set_title(f'Induction Heads Scores-Bit Model')
    fig.colorbar(cax2, ax=axs[1])
  
    
    plt.tight_layout()
    plt.savefig(f'./imgs/induction_scores_{id}.png')
    plt.show()



def plot_attention_scores(*attention_scores_tensors, ids=None):
    """
    Plots the attention scores for the last token across all layers and heads for multiple conditions side by side.
    
    *attention_scores_tensors: A series of tensors of shape [num_layers, num_heads], each representing a different condition.
    ids: Optional list of IDs corresponding to each tensor for labeling purposes. If provided, should match the number of tensors.
    """
    num_tensors = len(attention_scores_tensors)
    fig, axs = plt.subplots(1, num_tensors, figsize=(20, 6))  # 1 row, N columns based on the number of tensors

    # If only one tensor, axs may not be an array. This ensures consistency.
    if num_tensors == 1:
        axs = [axs]

    # Find the overall minimum and maximum value across all tensors for consistent coloring
    min_score = min(tensor.numpy().min() for tensor in attention_scores_tensors)
    max_score = max(tensor.numpy().max() for tensor in attention_scores_tensors)

    for i, attention_scores in enumerate(attention_scores_tensors):
        # Plot for the current condition
        cax = axs[i].matshow(attention_scores.numpy(), cmap='Blues', origin='lower', aspect='auto', vmin=min_score, vmax=max_score)
    

        # Set labels and titles
        axs[i].set_xlabel('Head')
        axs[i].set_ylabel('Layer')
        title = f'Induction Heads Scores-Model {ids[i]}' if ids and len(ids) > i else f'Induction Heads Scores-Model {i+1}'
        axs[i].set_title(title)
    
    fig.colorbar(cax, ax=axs[i])
  
    plt.tight_layout()
    plt.show()


def save_tensor_to_file(tensor, filename, directory="induction_scores"):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    torch.save(tensor, filepath)

def load_tensor_from_file(filename, directory="induction_scores"):
    filepath = os.path.join(directory, filename)
    return torch.load(filepath)

def load_tensors_from_files(filenames, directory="induction_scores"):
    tensors = []
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        tensor = torch.load(filepath)
        tensors.append(tensor)
    return tensors
