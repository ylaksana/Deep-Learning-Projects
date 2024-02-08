import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.nn as nn

class Imitator(torch.nn.Module):
    def __init__(self, input_size=56):
        """
           Your code here.
           Set up your detection network
        """
        super().__init__()

        self.layer_1 = nn.Linear(input_size,input_size*2)
        self.layer_2 = nn.Linear(input_size*2,input_size*4)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(input_size*4,6)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20):
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)


    def forward(self, x, batched: bool = False):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        index_list = [i for i in range(0, input_size)]
        indices_to_embed = torch.tensor(index_list, dtype=torch.long).to('mps')
        if batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


class FeedFoward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)
class Head(nn.Module):
    def __init__(self, seq_length, d_model, num_heads, d_internal):
        super().__init__()
        self.K = nn.Linear(d_model, d_internal)
        self.Q = nn.Linear(d_model, d_internal)
        self.V = nn.Linear(d_model, d_internal)
        self.w0 = nn.Linear(d_internal, d_model // num_heads)
        self.register_buffer('tril', torch.tril(torch.ones(seq_length, seq_length)))

    def forward(self, input_vecs):
        keys = self.K(input_vecs) # B, L, d_internal
        d_k = keys.shape[-1]
        queries = self.Q(input_vecs) # B, L, d_internal
        value = self.V(input_vecs) # B, L, d_internal
        weights = torch.matmul(queries, keys.transpose(-2, -1)) * d_k**-0.5# L, L
        weights = weights.masked_fill(self.tril == 0, float('-inf'))
        attention = torch.softmax(weights, dim=-1)

        logit = torch.matmul(attention , value) # B, L, d_internal
        logit = self.w0(logit)
        return logit

class MultiHeadAttention(nn.Module):

    def __init__(self, seq_length, d_model, num_heads, d_internal):
        super().__init__()
        self.heads = nn.ModuleList([Head(seq_length, d_model, num_heads, d_internal) for _ in range(num_heads)])
        self.linear1 = nn.Linear(d_model, d_model)

    def forward(self, input_vecs):
        out = torch.cat([head(input_vecs) for head in self.heads], dim=-1)
        out = self.linear1(out)
        return out

class MHATransformerLayer(nn.Module):
    def __init__(self, seq_length, d_model, num_heads, d_internal):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention( seq_length, d_model, num_heads, d_internal)
        self.ffwd = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, input_vecs):
        x = self.multi_head_attention(self.ln1(input_vecs))
        x += input_vecs
        x = x + self.ffwd(self.ln2(x))

        return x
class SequenceImitator(torch.nn.Module):
    def __init__(self, input_size=58, output_size=6, seq_len=30, n_embed=80, n_hidden=40, num_layers=1, num_heads=4):
        """
           Your code here.
           Set up your detection network
        """
        super().__init__()
        self.seq_len = seq_len
        self.L = []
        for ly in range(num_layers):
            self.L.append(MHATransformerLayer(seq_len, n_embed, num_heads, n_hidden))
        self.first_layer = nn.Linear(input_size, n_embed)
        self.relu = nn.ReLU()
        self.transformer_layers = nn.Sequential(*self.L)
        self.classifier = nn.Linear(n_embed, output_size)
        self.pos_embedding = PositionalEncoding(n_embed, num_positions=seq_len)

    def forward(self, input, batched: bool = False):
        logit = self.relu(self.first_layer(input))
        logit = self.pos_embedding(logit, batched=batched)
        logit = self.transformer_layers(logit)
        logit = self.classifier(logit)
        if batched:
            return logit
        else:
            return logit.squeeze(0)


def save_model(model, id):
    model.eval()
    scripted_model = torch.jit.script(model)
    return scripted_model.save(f'state_agent/{id}.jit')

def load_model(id):
    from torch import load
    from os import path
    model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'{id}.jit'))
    return model
