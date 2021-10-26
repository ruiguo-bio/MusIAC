import math
from einops import rearrange

import torch
from torch import nn
import transformer

class Classify_transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward,
                 max_seq_length, pos_dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        # self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.transformer = transformer.Classifier(d_model, nhead, num_encoder_layers,
                 dim_feedforward)


        self.classify_0 = nn.Linear(d_model, 128)
        self.classify_1 = nn.Linear(128, 2)

        # self.classify_0 = nn.Linear(d_model, 2)

    def forward(self, src):
        # Reverse the shape of the batches from (num_sentences, num_tokens_in_each_sentence)
        src = src.permute(1, 0)


        # Embed the batches, scale by sqrt(d_model), and add the positional encoding
        src = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))

        # Send the batches to the model
        output = self.transformer(src)

        # Rearrange to batch-first
        # output = output.permute(1,0,2)
        output = rearrange(output, 't n e -> n t e')
        # print("in model, output size", output.size())
        output = output.mean(dim=1)
        # print("in model, output size", output.size())
        output = self.classify_0(output)
        output = self.classify_1(output)
        # print("in model, output size", output.size())

        # print("in model, attention weight size", attention_weight.size())

        # Run the output through an fc layer to return values for each token in the vocab
        return output


class ScoreTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        """
        Initializes the model
                Parameters:
                        vocab_size (int): The amount of tokens in both vocabularies (including start, end, etc tokens)
                        d_model (int): Expected number of features in the encoder/decoder inputs, also used in embeddings
                        nhead (int): Number of heads in the transformer
                        num_encoder_layers (int): Number of sub-encoder layers in the transformer
                        num_decoder_layers (int): Number of sub-decoder layers in the transformer
                        dim_feedforward (int): Dimension of the feedforward network in the transformer
                        max_seq_length (int): Maximum length of each tokenized sentence
                        pos_dropout (float): Dropout value in the positional encoding
                        trans_dropout (float): Dropout value in the transformer
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        # self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.transformer = transformer.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.fc = nn.Linear(d_model, vocab_size)


    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask):
        # Reverse the shape of the batches from (num_sentences, num_tokens_in_each_sentence)
        src = src.permute(1,0)
        tgt = tgt.permute(1,0)

        # Embed the batches, scale by sqrt(d_model), and add the positional encoding
        src = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))

        # Send the batches to the model
        output, attention_weight = self.transformer(src, tgt, tgt_mask=tgt_mask[0], src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        # Rearrange to batch-first
        # output = output.permute(1,0,2)
        output = rearrange(output, 't n e -> n t e')
        attention_weight = attention_weight.permute(1,0,2,3)
        # print("in model, output size", output.size())
        # print("in model, attention weight size", attention_weight.size())

        # Run the output through an fc layer to return values for each token in the vocab
        return self.fc(output), attention_weight


# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
