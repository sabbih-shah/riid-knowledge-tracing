import gc
from abc import ABC
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# from optimizer import ScheduledOptim, NoamLR
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import Transformer
from model import SaintPlus
from config import CONFIG

# train_loader = torch.load("train_loader_0.8.pth")
# val_loader = torch.load("val_loader_0.2.pth")

train_loader, val_loader = get_dataloaders()

model = SaintPlus(CONFIG.total_questions, CONFIG.total_categories, 3, emb_dim=CONFIG.emb_dim,
                  num_layers=CONFIG.num_layers, num_heads=CONFIG.heads,
                  drop_val=0.4, pretrained_ex_path=None)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='lightning_logs/',
    filename='SaintPlus-{epoch:02d}-{val_loss:.2f}',
    save_top_k=-1,
    mode='min',

)

# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

trainer = pl.Trainer(gpus=-1, max_epochs=25, progress_bar_refresh_rate=21, callbacks=[checkpoint_callback],
                     num_sanity_val_steps=0)

trainer.fit(model=model,
            train_dataloader=train_loader,
            val_dataloaders=[val_loader, ])
trainer.save_checkpoint("model.pt")

print("done")
