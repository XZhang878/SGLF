import torch
import torch.nn as nn
import numpy as np


class dot_attention(nn.Module):
    """ 点积注意力机制"""

    def __init__(self, attention_dropout=0.0):
        super(dot_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale        # 是否设置缩放
        if attn_mask:
            attention = attention.masked_fill(attn_mask, -np.inf)     # 给需要mask的地方设置一个负无穷。

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):
    """ 多头自注意力"""
    def __init__(self, model_dim=256, num_heads=4, dropout=0.0, version='v2'):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim//num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = dot_attention(dropout)

        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.version  = version

    def forward(self, key, value, query, attn_mask=None):

        if self.version == 'v2':

            B =1
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            query = query.unsqueeze(1)
            residual = query


            dim_per_head = self.dim_per_head
            num_heads = self.num_heads


            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(key.size(0), B * num_heads, dim_per_head).transpose(0,1)
            value = value.view(value.size(0), B * num_heads, dim_per_head).transpose(0,1)
            query = query.view(query.size(0), B * num_heads, dim_per_head).transpose(0,1)

            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
            # (query, key, value, scale, attn_mask)
            context = context.transpose(0, 1).contiguous().view(query.size(1), B, dim_per_head * num_heads)
            output = self.linear_final(context)
            # dropout
            output = self.dropout(output)

            output = self.layer_norm(residual + output)
            # output = residual + output

        elif self.version == 'v1':

            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            query = query.unsqueeze(0)
            residual = query
            B, L, C = key.size()
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            batch_size = key.size(0)

            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            key = key.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)


            if attn_mask:
                attn_mask = attn_mask.repeat(num_heads, 1, 1)

            # 缩放点击注意力机制
            scale = (key.size(-1) // num_heads) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

            context = context.view(batch_size, -1, dim_per_head * num_heads)

            output = self.linear_final(context)


            output = self.dropout(output)
            output = self.layer_norm(residual + output)


        return output.squeeze(), attention.squeeze()

