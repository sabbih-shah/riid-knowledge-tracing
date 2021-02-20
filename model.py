import copy
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# from optimizer import ScheduledOptim, NoamLR
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import Transformer
from config import CONFIG


def get_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len),
                      diagonal=1).to(dtype=torch.bool)
    mask = mask.to(CONFIG.device)
    return mask


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module, ABC):
    def __init__(self, seq_len, emb_dim):
        super(PositionalEncoding, self).__init__()
        self.position_embed = nn.Embedding(seq_len, emb_dim)

    def forward(self, seq):
        pos = self.position_embed(seq)
        return pos


class FeedForwardNetwork(nn.Module, ABC):
    def __init__(self, in_feat, drop_val=0.2):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(in_feat, in_feat)
        self.linear2 = nn.Linear(in_feat, in_feat)
        self.dropout = nn.Dropout(p=drop_val)

    def forward(self, x):
        out = F.relu(self.dropout(self.linear1(x)))
        out = self.linear2(out)
        return out


class Encoder(nn.Module, ABC):
    def __init__(self, num_dim, num_heads, drop_val=0.2):
        super(Encoder, self).__init__()

        self.multihead_attention = nn.MultiheadAttention(embed_dim=num_dim,
                                                         num_heads=num_heads,
                                                         dropout=drop_val)
        self.norm_layer1 = nn.LayerNorm(num_dim)
        self.norm_layer2 = nn.LayerNorm(num_dim)
        self.feed_forward = FeedForwardNetwork(num_dim)

    def forward(self, ex_emb):
        ex_emb = self.norm_layer1(ex_emb)
        ex_emb = ex_emb.permute(1, 0, 2)
        attention_out, _ = self.multihead_attention(ex_emb, ex_emb, ex_emb, attn_mask=get_mask(CONFIG.seq_len))

        ff_in = ex_emb + attention_out
        ff_in = ff_in.permute(1, 0, 2)
        ff_in = self.norm_layer2(ff_in)
        ff_out = self.feed_forward(ff_in)
        out = ff_in + ff_out
        return out


class DecoderInput(nn.Module, ABC):
    def __init__(self, total_response, emb_dim, positional_encoding):
        super(DecoderInput, self).__init__()

        self.response_emb = nn.Embedding(total_response, emb_dim)

        # saint plus stuff
        self.elapsed_time_embedding = nn.Linear(1, emb_dim, bias=False)  # excercise1 to response1  continous
        self.lag_time_embedding = ""  # response1 to exercise2   categorical

        self.positional_encoding = positional_encoding

    def forward(self, response, elapsed_time):  # elapsed_time, lag_time
        out = self.response_emb(response)
        seq = torch.arange(CONFIG.seq_len, device=CONFIG.device).unsqueeze(0)
        pos = self.positional_encoding(seq)
        elapsed_time = self.elapsed_time_embedding(elapsed_time)
        res_emb = out + pos + elapsed_time
        return res_emb


class EncoderInput(nn.Module, ABC):
    def __init__(self, total_questions, total_categories, emb_dim, positional_encoding, pretrained_ex_path=None):
        super(EncoderInput, self).__init__()

        if pretrained_ex_path is not None:
            embed_data = np.load(pretrained_ex_path)
            _, _, weights = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
            embedding_weights = torch.tensor(weights)
            self.question_embed = nn.Embedding.from_pretrained(embedding_weights)
            self.question_embed.requires_grad_(requires_grad=False)
            print("using_pretrained embedding")
        else:
            self.question_embed = nn.Embedding(total_questions, emb_dim)

        self.category_embed = nn.Embedding(total_categories, emb_dim)
        self.positional_encoding = positional_encoding

    def forward(self, questions, category):
        qu_out = self.question_embed(questions)
        cat_out = self.category_embed(category)

        seq = torch.arange(CONFIG.seq_len, device=CONFIG.device).unsqueeze(0)
        pos = self.positional_encoding(seq)

        ex_embed = qu_out + cat_out + pos
        return ex_embed


class Decoder(nn.Module, ABC):
    def __init__(self, num_dim, num_heads, drop_val=0.2):
        super(Decoder, self).__init__()

        self.multihead_attention1 = nn.MultiheadAttention(embed_dim=num_dim,
                                                          num_heads=num_heads,
                                                          dropout=drop_val)

        self.multihead_attention2 = nn.MultiheadAttention(embed_dim=num_dim,
                                                          num_heads=num_heads,
                                                          dropout=drop_val)

        self.norm_layer1 = nn.LayerNorm(num_dim)
        self.norm_layer2 = nn.LayerNorm(num_dim)
        self.norm_layer3 = nn.LayerNorm(num_dim)
        self.norm_layer4 = nn.LayerNorm(num_dim)

        self.feed_forward = FeedForwardNetwork(num_dim)

    def forward(self, res_emb, enc_out):
        res_emb = res_emb.permute(1, 0, 2)
        attn1_in = self.norm_layer1(res_emb)
        att1_out, _ = self.multihead_attention1(attn1_in, attn1_in, attn1_in, attn_mask=get_mask(CONFIG.seq_len))

        attn2_in = attn1_in + att1_out
        # attn2_in = self.norm_layer2(attn2_in)

        enc_out = self.norm_layer3(enc_out)
        enc_out = enc_out.permute(1, 0, 2)

        attn2_out, _ = self.multihead_attention2(attn2_in, enc_out, enc_out, attn_mask=get_mask(CONFIG.seq_len))
        ffn_in = enc_out + attn2_out

        ffn_in = ffn_in.permute(1, 0, 2)
        ffn_in = self.norm_layer4(ffn_in)
        ffn_out = self.feed_forward(ffn_in)

        out = ffn_in + ffn_out

        return out


class SaintPlus(pl.LightningModule):
    def __init__(self, total_questions, total_categories, total_response, emb_dim, num_layers,
                 num_heads, drop_val, pretrained_ex_path):
        super(SaintPlus, self).__init__()

        self.loss = nn.BCEWithLogitsLoss()
        self.pos_enc = PositionalEncoding(CONFIG.seq_len, CONFIG.emb_dim)
        self.exercise_embed = EncoderInput(total_questions, total_categories, emb_dim, self.pos_enc,
                                           pretrained_ex_path)
        self.response_embed = DecoderInput(total_response, emb_dim, self.pos_enc)

        # self.encoder_layers = get_clones(Encoder(emb_dim, num_heads, drop_val), num_layers)
        #
        # self.decoder_layers = get_clones(Decoder(emb_dim, num_heads, drop_val), num_layers)

        self.transformer = nn.Transformer(d_model=CONFIG.emb_dim, dim_feedforward=1024, activation='relu',
                                          num_encoder_layers=CONFIG.num_layers, num_decoder_layers=CONFIG.num_layers,
                                          nhead=CONFIG.heads)

        self.fc = nn.Linear(emb_dim, 1)  # logits

    def forward(self, x, response):  # elapsed_time, lag_time
        questions, category = x["input_ids"].long().to(CONFIG.device), x['input_cat'].long().to(CONFIG.device)
        # elapsed_time = x["input_rtime"].unsqueeze(-1).float()
        # ela_time = self.elapsed_time(elapsed_time)
        enc_in = self.exercise_embed(questions, category)
        # print("encoder old shape:", enc_in.shape)
        enc_in = enc_in.permute(1, 0, 2)
        # print("encoder new shape:", enc_in.shape)
        # for encoder_num in range(len(self.encoder_layers)):
        #     if encoder_num < 1:
        #         enc_out = self.encoder_layers[encoder_num](enc_in)
        #         enc_in = enc_out
        #     else:
        #         enc_out = self.encoder_layers[encoder_num](enc_in)
        #         enc_in = enc_out

        elapsed_time = x["input_rtime"].unsqueeze(-1).float()
        dec_in = self.response_embed(response.long().to(CONFIG.device), elapsed_time)
        # print("decoder old shpae:", dec_in.shape)

        dec_in = dec_in.permute(1, 0, 2)
        # print("decoder new shpae:",dec_in.shape)

        mask = get_mask(CONFIG.seq_len)
        # print(mask.shape)
        # for decoder_num in range(len(self.decoder_layers)):
        #     if decoder_num < 1:
        #         dec_out = self.decoder_layers[decoder_num](dec_in, enc_out)
        #         dec_in = dec_out
        #     else:
        #         dec_out = self.decoder_layers[decoder_num](dec_in, enc_out)
        #         dec_in = dec_out
        #
        # out = self.fc(dec_out)
        out = self.transformer(enc_in, dec_in, src_mask=mask, tgt_mask=mask, memory_mask=mask)
        out = out.permute(1, 0, 2)
        out = self.fc(out)
        out = out.squeeze()
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        # scheduler = NoamLR(optimizer, CONFIG.emb_dim, CONFIG.warmpup_steps)
        return optimizer

    def training_step(self, batch, batch_ids):
        input, ans, labels = batch
        target_mask = (input["input_ids"] != 0)
        out = self(input, ans)
        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        out = torch.masked_select(out, target_mask)
        out = torch.sigmoid(out)
        labels = torch.masked_select(labels, target_mask)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "outs": out, "labels": labels}

    def validation_step(self, batch, batch_ids):
        input, ans, labels = batch
        target_mask = (input["input_ids"] != 0)
        out = self(input, ans)
        loss = self.loss(out.view(-1).float(), labels.view(-1).float())
        out = torch.masked_select(out, target_mask)
        out = torch.sigmoid(out)
        labels = torch.masked_select(labels, target_mask)
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        output = {"outs": out, "labels": labels}
        return output

    def validation_epoch_end(self, validation_ouput):
        out = torch.cat([i["outs"] for i in validation_ouput]).view(-1)
        labels = torch.cat([i["labels"] for i in validation_ouput]).view(-1)
        auc = roc_auc_score(labels.cpu().detach().numpy(), out.cpu().detach().numpy())
        self.print("val auc", auc)
