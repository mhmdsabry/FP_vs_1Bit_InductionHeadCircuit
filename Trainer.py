import logging
from tqdm import tqdm
import wandb

import math 

import torch
from torch.profiler import profile

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from typing import Type

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


class TrainerConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
  def __init__ (self, model: nn.Module, train_dataset: Type[Dataset], eval_dataset: Type[Dataset],
                    train_config: TrainerConfig, model_P_mode:str = "FP"):
      
      self.train_dataset = train_dataset
      self.eval_dataset = eval_dataset
      self.config = train_config
      wandb.init(project=self.config.wandb_run_name, name=f"{model_P_mode}_{self.config.wandb_model_name}")

      self.model = model
      self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
      self.model.to(self.device)
      

  def delete_model(self):
    # no need to save the model
    del self.model

  def train(self):
    model = self.model
    device = self.device
    config = self.config
    train_batch_size = config.train_batch_size
    eval_batch_size = config.eval_batch_size
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98))

    num_training_steps = int(len(self.train_dataset) / train_batch_size * config.n_epochs)
    num_warmup_steps = int(config.warmup_ratio * num_training_steps)
    lambda_func = lambda step: (
                                step / max(1, num_warmup_steps)
                                if step < num_warmup_steps
                                else max(0.0 , 0.5 * (1.0 + math.cos(math.pi  *  ((step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)))))
                                )

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    train_dataloader = DataLoader(self.train_dataset, batch_size=train_batch_size, drop_last=True)
    eval_dataloader = DataLoader(self.eval_dataset, batch_size=eval_batch_size, drop_last=True)

    def train_loop(train_dataloader: DataLoader):
        model.train()
        train_attn_scores = []

        for train_batch in tqdm(train_dataloader, total=len(train_dataloader), desc="Training.."):
                x_train, y_train = train_batch[0].to(device), train_batch[1].to(device)
                batch_induction_idx = train_batch[2]

                _, loss = model(x_train, y_train)

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
                scheduler.step()
                train_attn_scores.append((model.attention_scores, batch_induction_idx))
                train_metrics = {"train/train_loss": loss, "train/train_lr": scheduler.get_last_lr()[0]}
                wandb.log(train_metrics)


        return train_attn_scores

    def eval_loop(eval_dataloader:DataLoader):
        model.eval()
        eval_attn_scores = []
        for eval_batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval.."):
            x_eval, y_eval = eval_batch[0].to(device), eval_batch[1].to(device)
            batch_induction_idx = eval_batch[2]

            _, loss = model(x_eval, y_eval)

            eval_attn_scores.append((model.attention_scores, batch_induction_idx))

            val_metrics = {"val/val_loss": loss}
            wandb.log(val_metrics) 
        return eval_attn_scores


    self.epoch_train_attn_scores = {}
    self.epoch_eval_attn_scores = {}

    for i, epoch in enumerate(range(config.n_epochs)):
      logger.info(f"===========Epoch {i+1}/{config.n_epochs}===========\n")
      train_attn_scores = train_loop(train_dataloader)
      eval_attn_scores = eval_loop(eval_dataloader)

      self.epoch_train_attn_scores[i+1] = train_attn_scores
      self.epoch_eval_attn_scores[i+1] = eval_attn_scores

    wandb.finish()

  def get_attn_scores(self):
    if not self.epoch_train_attn_scores:
        logger.warning("Empty Attention Scores! You need to train the model to create the trained attention scores.")
        return 
    else:
        return self.epoch_train_attn_scores, self.epoch_eval_attn_scores 
    