import torch
import torch.nn as nn
import numpy as np

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Attention(nn.Module):
    def __init__(self, n_attn_heads, dropout, embedding_dim, hidden_dim1, hidden_dim2):
        super(Attention, self).__init__()

        # Self-Attention
        self.query = nn.Linear(embedding_dim, hidden_dim1)
        self.key = nn.Linear(embedding_dim, hidden_dim1)
        self.value = nn.Linear(embedding_dim, hidden_dim1)
        self.drop_attn = nn.Dropout(dropout)
        self.n_heads = n_attn_heads
        self.size_attn_heads = int(hidden_dim1 / n_attn_heads)

        # Attention Out
        self.attn_out1 = nn.Linear(hidden_dim1, hidden_dim1)
        self.bn_attn_out1 = BertLayerNorm(hidden_dim1)
        self.drop_attn_out1 = nn.Dropout(dropout)
        self.attn_out2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.attn_act2 = nn.ReLU()
        self.attn_out3 = nn.Linear(hidden_dim2, hidden_dim1)
        self.bn_attn_out3 = BertLayerNorm(hidden_dim1)
        self.drop_attn_out3 = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.size_attn_heads)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):

        # Self-Attention
        q, k, v = self.query(x), self.key(x), self.value(x)
        if self.n_heads > 1:  # Multi-Head attention
            q, k, v = self.transpose_for_scores(q), self.transpose_for_scores(k), self.transpose_for_scores(v)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.size_attn_heads)
        attn_probs = self.drop_attn(nn.Softmax(dim=-1)(attn_scores))
        context_v = torch.matmul(attn_probs, v)
        if self.n_heads > 1:  # Multi-Head attention
            context_v = context_v.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_v.size()[:-2] + (-1,)
            context_v = context_v.view(*new_context_layer_shape)

        # Attention Out
        attn_out = self.bn_attn_out1(self.drop_attn_out1(self.attn_out1(context_v)) + x)
        return self.bn_attn_out3(self.drop_attn_out3(self.attn_out3(self.attn_act2(self.attn_out2(attn_out)))) + attn_out)

class RowColumnAttention(nn.Module):
    def __init__(self, color, use_cls, embedding_dim, hidden_dim1, hidden_dim2, n_layers, n_attn_heads, n_classes, dropout,
                 position_embedding_version, segment_embedding_version, attention_version, attention_indv_channels_merge_mode):
        super(RowColumnAttention, self).__init__()
        self.color = color
        self.use_cls = use_cls
        self.position_embedding_version = position_embedding_version
        self.segment_embedding_version = segment_embedding_version
        self.attention_version = attention_version
        self.attention_indv_channels_merge_mode = attention_indv_channels_merge_mode

        # Embeddings
        if color and attention_version == "default":
            embedding_dim_mult = 3
        else:
            embedding_dim_mult = 1
        extra_id = 0
        if use_cls:
            self.cls_embedding = nn.Embedding(1, embedding_dim)
            extra_id = 1
        if self.position_embedding_version != "none":
            self.pos_embedding = nn.Embedding(embedding_dim*2+extra_id, embedding_dim*embedding_dim_mult)  # TODO: 28 + 1 ?
        if self.segment_embedding_version != "none":
            self.seg_embedding = nn.Embedding(2 + extra_id, embedding_dim*embedding_dim_mult)

        self.bn_embedding = BertLayerNorm(embedding_dim*embedding_dim_mult)
        self.drop_embedding = nn.Dropout(dropout)

        # Attention
        self.attention_layers = nn.ModuleList([
            Attention(
                n_attn_heads=n_attn_heads, dropout=dropout, embedding_dim=embedding_dim*embedding_dim_mult,
                hidden_dim1=hidden_dim1*embedding_dim_mult, hidden_dim2=hidden_dim2
            ) for _ in range(n_layers)
            ])

        # Pooler
        if attention_indv_channels_merge_mode == "concat":
            embedding_dim_mult = 3
        self.pooler = nn.Linear(hidden_dim1*embedding_dim_mult, hidden_dim1*embedding_dim_mult)
        self.pooler_act = nn.Tanh()
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # TODO
        if not use_cls:
            self.flat = nn.Linear(embedding_dim*2*hidden_dim1*embedding_dim_mult, embedding_dim*embedding_dim_mult)
            self.act_flat = nn.ReLU()
            self.bn_flat = BertLayerNorm(embedding_dim*embedding_dim_mult)

        # Classification
        self.cls_drop = nn.Dropout(dropout)
        self.cls = nn.Linear(embedding_dim*embedding_dim_mult, n_classes)

    def forward(self, x, position_ids=None, segment_ids=None):

        # Embeddings: We treat each row as a word and their column values as its embedding... and vice versa
        if self.use_cls:
            cls_embedding = self.cls_embedding(torch.zeros((len(x), 1), device=x.device).long())
            cls_embedding = cls_embedding[..., None]
            cls_embedding = torch.cat([cls_embedding, cls_embedding, cls_embedding], dim=-1)
            x = torch.cat([cls_embedding, x], dim=1)
        if self.color:
            if self.attention_version == "default":
                if self.use_cls:
                    x_col = x[:, 1:].permute(0, 2, 1, 3)
                else:
                    x_col = x.permute(0, 2, 1, 3)
                row_channel_cat = torch.cat([x[..., 0], x[..., 1], x[..., 2]], dim=2)
                col_channel_cat = torch.cat([x_col[..., 0], x_col[..., 1], x_col[..., 2]], dim=2)
                x = torch.cat([row_channel_cat, col_channel_cat], dim=1)
                n_examples, seq_len, x_device = len(x), x.shape[1], x.device
            elif self.attention_version == "per_channel":
                if self.use_cls:
                    x = [torch.cat([x[..., 0], x[:, 1:, :, 0].permute(0, 2, 1)], dim=1),
                         torch.cat([x[..., 1], x[:, 1:, :, 1].permute(0, 2, 1)], dim=1),
                         torch.cat([x[..., 2], x[:, 1:, :, 2].permute(0, 2, 1)], dim=1)]
                else:
                    x = [torch.cat([x[..., 0], x[..., 0].permute(0, 2, 1)], dim=1),
                         torch.cat([x[..., 1], x[..., 1].permute(0, 2, 1)], dim=1),
                         torch.cat([x[..., 2], x[..., 2].permute(0, 2, 1)], dim=1)]
                n_examples, seq_len, x_device = len(x[0]), x[0].shape[1], x[0].device
        else:
            if self.use_cls:
                x = torch.cat([x, x[:, 1:].permute(0, 2, 1)], dim=1)
            else:
                x = torch.cat([x, x.permute(0, 2, 1)], dim=1)
            n_examples, seq_len, x_device = len(x), x.shape[1], x.device

        x_pos, x_seg = 0, 0

        if self.position_embedding_version != "none":
            if position_ids is None:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=x_device)
                position_ids = position_ids.unsqueeze(0).expand(size=(n_examples, seq_len))
            x_pos = self.pos_embedding(position_ids)
        if self.segment_embedding_version != "none":
            if segment_ids is None:
                if self.use_cls:
                    segment_ids = torch.cat([
                        torch.zeros((n_examples, 1)), torch.ones((len(x), (seq_len-1)/2)),
                        2*torch.ones((n_examples, (seq_len - 1)/2))], dim=1).long().to(x_device)
                else:
                    segment_ids = torch.cat([
                        torch.zeros((n_examples, seq_len/2)), torch.ones((n_examples, seq_len/2))],
                        dim=1).long().to(x_device)
            x_seg = self.seg_embedding(segment_ids)

        if self.attention_version == "default":
            x = x + x_pos + x_seg  # Now the rows (and columns) are already embedded
        elif self.attention_version == "per_channel":
            for i in range(len(x)):
                x[i] = x[i] + x_pos + x_seg  # We use the same position and segment embeddings for each of the channels

        # Attention Layers
        for attn_layer in self.attention_layers:
            if self.attention_version == "default":
                x = attn_layer(x)
            elif self.attention_version == "per_channel":
                for i in range(len(x)):
                    x[i] = attn_layer(x[i])
        if self.attention_version == "per_channel":
            if self.attention_indv_channels_merge_mode == "sum":
                x = x[0] + x[1] + x[2]
            elif self.attention_indv_channels_merge_mode == "average":
                x = (x[0] + x[1] + x[2])/3
            elif self.attention_indv_channels_merge_mode == "concat":
                x = torch.cat([x[0], x[1], x[2]], dim=2)

        # Pooler
        pooled_output = self.pooler_act(self.pooler(x))
        if self.use_cls:
            pooled_output = pooled_output[:, 0].view(len(pooled_output), -1)
        else:
            pooled_output = self.bn_flat(self.act_flat(self.flat(pooled_output.view(len(pooled_output), -1))))
        # Classification
        return self.cls(self.cls_drop(pooled_output))
