import gc
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm

import configparser
import argparse 

from Trainer import TrainerConfig, Trainer

from prepare_dataset import get_induction_data
from induction_circuit import IHCModel
from utils import *

#command line parser for config file
config = configparser.ConfigParser()
parser = argparse.ArgumentParser(prog="IHC Tests")
parser.add_argument("-c","--config",dest="filename", help="Pass a training config file",metavar="FILE")
parser.add_argument("--embed_size", help="Embedding/model dimension", type=int, default=768)
parser.add_argument("--num_head", help="Number of attention heads per layer", type=int, default=8)
parser.add_argument("--pos_embed", help="Select a position encoding strategy: no_pos, learnable or trigonometric", type=str, default="trigonometric")
parser.add_argument("--norm_mode", help="Select a normalisation strategy: no_norm, pre_norm or post_norm", type=str, default="no_norm")
parser.add_argument("--softmax_mode", help="Select a softmax strategy: vanilla, or off1 (off by 1)", type=str, default="vanilla")
parser.add_argument("--seed", help=" set seeds for reproducability", type=int, default=42)

args = parser.parse_args()
config.read(args.filename)

##########################################################
#                  Set up Configs and seeds Libs         #
##########################################################
embed_dim = args.embed_size
n_head = args.num_head
pos_embed = args.pos_embed
norm_mode  = args.norm_mode
softmax_mode = args.softmax_mode

n_layer = int(config['model_config']['n_layer'])

n_epochs = int(config['train_config']['n_epochs'])
learning_rate = float(config['train_config']['learning_rate'])
warmup_ratio = float(config['train_config']['warmup_ratio'])
train_batch_size = int(config['train_config']['train_batch_size'])
eval_batch_size = int(config['train_config']['eval_batch_size'])

wandb_run_name = config['train_config']['wandb_run_name']


#From utils:
seed = args.seed
seed_everything(seed)

if softmax_mode == "off1":
  wandb_model_name = f"{embed_dim}_{n_head}_{pos_embed}_{norm_mode}_softmax1_{seed}"
else:
  wandb_model_name = f"{embed_dim}_{n_head}_{pos_embed}_{norm_mode}_{seed}"

#create folder for artifact:
if not os.path.exists("./imgs"):
  os.makedirs("./imgs")

##########################################################
#                  Get Datasets                          #
##########################################################
train_dataset, eval_dataset, block_size = get_induction_data()
vocab_size = train_dataset.get_vocab_size()


##########################################################
#                 Setup Train Config                     #
##########################################################

train_config = TrainerConfig(
                            n_epochs = n_epochs,
                            learning_rate = learning_rate,
                            warmup_ratio = warmup_ratio,
                            train_batch_size = train_batch_size,
                            eval_batch_size = eval_batch_size,
                            wandb_run_name = wandb_run_name,
                            wandb_model_name = wandb_model_name,
)


##########################################################
#                     Train FP Model                     #
##########################################################

#FP Weights Model
FP_model = IHCModel(embed_dim, n_head, vocab_size, n_layer, block_size, Position_mode= pos_embed, norm_mode=norm_mode, softmax_mode=softmax_mode, P_Mode="FP")

FP_trainer = Trainer(FP_model, train_dataset, eval_dataset,
                    train_config,
                    model_P_mode="FP",
                  )

FP_trainer.train()

FP_epoch_train_attn_scores, FP_epoch_eval_attn_scores  = FP_trainer.get_attn_scores()

for name, param in FP_model.named_parameters():
  print(f"{name} has dtype: {param.dtype}")


#clean and collect
del FP_model
FP_trainer.delete_model()
if torch.cuda.is_available():
  torch.cuda.empty_cache()
  gc.collect()

##########################################################
#                     Train 1Bit Model                   #
##########################################################
#1 bit weights Model
bit_model = IHCModel(embed_dim, n_head, vocab_size, n_layer, block_size, Position_mode=pos_embed, norm_mode=norm_mode, softmax_mode=softmax_mode, P_Mode="1bit")

Bit_trainer = Trainer(bit_model, train_dataset, eval_dataset,
                  train_config,
                  model_P_mode="1bit"
                  )

Bit_trainer.train()
Bit_epoch_train_attn_scores, Bit_epoch_eval_attn_scores  = Bit_trainer.get_attn_scores()

for name, param in bit_model.named_parameters():
  weights = param.data
  alpha = weights.mean()
  print(f"{name} has dtype: {param.dtype}: {torch.sign(weights - alpha)}\n")

#clean and collect
del bit_model
if torch.cuda.is_available():
  torch.cuda.empty_cache()
  gc.collect()
##########################################################
#                     Plot Induction Maps                #
##########################################################
  
#We use plot functions in utils:
  
#Epoch to Plot its attention score:
epoch_id = n_epochs #last epoch

#Plot Train
FP_train_induction_token_scores = extract_last_token_attention(FP_epoch_train_attn_scores[epoch_id])
Bit_train_induction_token_scores = extract_last_token_attention(Bit_epoch_train_attn_scores[epoch_id])

#uncomment if you want to plot the currect run induction scores
#plot_attention_scores(FP_train_induction_token_scores, Bit_train_induction_token_scores, id=f"train_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_seed_{seed}")

if softmax_mode == "off1":
  save_tensor_to_file(FP_train_induction_token_scores, f"train_FP_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_norm_{norm_mode}_softmax1_seed_{seed}.pt")
  save_tensor_to_file(Bit_train_induction_token_scores, f"train_1Bit_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_norm_{norm_mode}_softmax1_seed_{seed}.pt")
else:
  save_tensor_to_file(FP_train_induction_token_scores, f"train_FP_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_norm_{norm_mode}_seed_{seed}.pt")
  save_tensor_to_file(Bit_train_induction_token_scores, f"train_1Bit_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_norm_{norm_mode}_seed_{seed}.pt")

#Plot Eval
FP_eval_induction_token_scores = extract_last_token_attention(FP_epoch_eval_attn_scores[epoch_id])
Bit_eval_induction_token_scores = extract_last_token_attention(Bit_epoch_eval_attn_scores[epoch_id])

#uncomment if you want to plot the currect run induction scores
#plot_attention_scores(FP_eval_induction_token_scores, Bit_eval_induction_token_scores, id=f"eval_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_seed_{seed}")

if softmax_mode == "off1":
  save_tensor_to_file(FP_eval_induction_token_scores, f"eval_FP_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_norm_{norm_mode}_softmax1_seed_{seed}.pt")
  save_tensor_to_file(Bit_eval_induction_token_scores, f"eval_1Bit_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_norm_{norm_mode}_softmax1_seed_{seed}.pt")
else:
  save_tensor_to_file(FP_eval_induction_token_scores, f"eval_FP_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_norm_{norm_mode}_seed_{seed}.pt")
  save_tensor_to_file(Bit_eval_induction_token_scores, f"eval_1Bit_embed_{embed_dim}_head_{n_head}_pos_{pos_embed}_norm_{norm_mode}_seed_{seed}.pt")