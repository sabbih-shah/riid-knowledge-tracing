import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model import SaintPlus
from config import CONFIG
from dataset import get_dataloaders

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
