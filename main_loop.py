from get_results import get_results
import os
import random
from time import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sacred import Experiment

# Creates a experiment, or loads an existing one
ex = Experiment('fashionmnist_attention')

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# Creates a hyper-parameter config that will be passed on the main function and that can be changed when running the
# experiment (see example_run_experiment.py)
@ex.config
def my_config():
    random_seed = 42
    lr = 1e-3
    hidden_dim1 = 28  # This can change, but embedding_dim is 28 (images of 28x28) and BERT has hidden_dim1=embedding_dim
    hidden_dim2 = 12
    n_heads = 1  # This needs to be a number that hidden_dim1 is divisible by (usually hidden_dim1 is set by the data)
    n_layers = 1
    n_epochs = 50
    batch_size = 1024
    dropout = 0.2
    position_embedding_version = "none"
    segment_embedding_version = "none"

# %% -------------------------------------- Model Class ----------------------------------------------------------------
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
    def __init__(self, n_attn_heads, dropout, hidden_dim1, hidden_dim2):
        super(Attention, self).__init__()

        # Self-Attention
        self.query = nn.Linear(hidden_dim1, hidden_dim1)
        self.key = nn.Linear(hidden_dim1, hidden_dim1)
        self.value = nn.Linear(hidden_dim1, hidden_dim1)
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
    def __init__(self, color, hidden_dim1, hidden_dim2, n_layers, n_attn_heads, n_classes, dropout,
                 position_embedding_version, segment_embedding_version):
        super(RowColumnAttention, self).__init__()
        self.color = color
        self.position_embedding_version = position_embedding_version
        self.segment_embedding_version = segment_embedding_version

        # Embeddings
        # For now we will be using only images of 28x28 pixels
        hidden_dim_mult = 3 if color else 1
        if self.position_embedding_version != "none":
            self.pos_embedding = nn.Embedding(hidden_dim1*2, hidden_dim1*hidden_dim_mult)  # TODO: 28 + 1 ?
        if self.segment_embedding_version != "none":
            self.seg_embedding = nn.Embedding(2, hidden_dim1*hidden_dim_mult)

        self.bn_embedding = BertLayerNorm(hidden_dim1*hidden_dim_mult)
        self.drop_embedding = nn.Dropout(dropout)

        # Attention
        self.attention_layers = nn.ModuleList([
            Attention(n_attn_heads=n_attn_heads, dropout=dropout, hidden_dim1=hidden_dim1*hidden_dim_mult,
                      hidden_dim2=hidden_dim2) for _ in range(n_layers)
            ])

        # Pooler
        self.pooler = nn.Linear(hidden_dim1*hidden_dim_mult, hidden_dim1*hidden_dim_mult)
        self.pooler_act = nn.Tanh()
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # TODO
        self.flat = nn.Linear(hidden_dim1*2*hidden_dim1*hidden_dim_mult, hidden_dim1*hidden_dim_mult)
        self.act_flat = nn.ReLU()
        self.bn_flat = BertLayerNorm(hidden_dim1*hidden_dim_mult)

        # Classification
        self.cls_drop = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_dim1*hidden_dim_mult, n_classes)

    def forward(self, x, position_ids=None, segment_ids=None):

        # Embeddings: We treat each row as a word and their column values as its embedding... and vice versa
        if self.color:
            x_col = x.permute(0, 2, 1, 3)
            row_channel_cat = torch.cat([x[..., 0], x[..., 1], x[..., 2]], dim=2)
            col_channel_cat = torch.cat([x_col[..., 0], x_col[..., 1], x_col[..., 2]], dim=2)
            x = torch.cat([row_channel_cat, col_channel_cat], dim=1)
        else:
            x = torch.cat([x, x.permute(0, 2, 1)], dim=1)
        x_pos, x_seg = 0, 0
        if self.position_embedding_version != "none":
            if position_ids is None:
                position_ids = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
                position_ids = position_ids.unsqueeze(0).expand(size=(len(x), x.shape[1]))
            x_pos = self.pos_embedding(position_ids)
        if self.segment_embedding_version != "none":
            if segment_ids is None:
                segment_ids = torch.cat([
                    torch.zeros((len(x), x.shape[1]/2)), torch.ones((len(x), x.shape[1]/2))],
                    dim=1, device=x.device).long()
            x_seg = self.seg_embedding(segment_ids)
        x = x + x_pos + x_seg  # Now the rows (and columns) are already embedded

        # Attention Layers
        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        # Pooler
        pooled_output = self.pooler_act(self.pooler(x))
        pooled_output = self.bn_flat(self.act_flat(self.flat(pooled_output.view(len(pooled_output), -1))))
        # Classification
        return self.cls(self.cls_drop(pooled_output))

@ex.automain
def my_main(random_seed, lr, hidden_dim1, hidden_dim2, n_heads, n_layers, n_epochs, batch_size, dropout,
            position_embedding_version, segment_embedding_version):

    # %% --------------------------------------- Set-Up ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    N_CLASSES = 10
    ONLY_BATCH_IN_GPU = False
    COLOR = False

    # %% -------------------------------------- Data Prep --------------------------------------------------------------
    data_train = datasets.FashionMNIST(root='.', train=True, download=True)
    x_train, y_train = data_train.data.float() / 255, data_train.targets
    p_train = torch.arange(x_train.shape[1] * 2, dtype=torch.long).unsqueeze(0).expand(
        size=(len(x_train), x_train.shape[1] * 2))
    s_train = torch.cat([torch.zeros((len(x_train), x_train.shape[1])), torch.ones((len(x_train), x_train.shape[1]))],
                        dim=1).long()
    x_train.requires_grad = True
    if not ONLY_BATCH_IN_GPU:
        x_train, p_train, s_train, y_train = x_train.to(device), p_train.to(device), s_train.to(device), y_train.to(
            device)
    data_test = datasets.FashionMNIST(root='.', train=False, download=True)
    x_test, y_test = data_test.data.float() / 255, data_test.targets
    p_test = torch.arange(x_test.shape[1] * 2, dtype=torch.long).unsqueeze(0).expand(size=(len(x_test), x_test.shape[1] * 2))
    s_test = torch.cat([torch.zeros((len(x_test), x_test.shape[1])), torch.ones((len(x_test), x_test.shape[1]))], dim=1).long()
    if not ONLY_BATCH_IN_GPU:
        x_test, p_test, s_test, y_test = x_test.to(device), p_test.to(device), s_test.to(device), y_test.to(device)

    # %% -------------------------------------- Training Prep ----------------------------------------------------------
    model = RowColumnAttention(COLOR, hidden_dim1, hidden_dim2, n_layers, n_heads, N_CLASSES, dropout,
                               position_embedding_version, segment_embedding_version).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # %% -------------------------------------- Training Loop ----------------------------------------------------------
    print("\n ------------ Doing run number {} with configuration ---------------".format(ex.current_run._id))
    print(ex.current_run.config)
    try:  # Gets the best result so far, so that we only save the model if the result is better (test loss in this case)
        get_results()
        results_so_far = pd.read_excel(os.getcwd() + "/experiments.xlsx")
        acc_test_best = min(results_so_far["test acc"].values)
    except:
        acc_test_best = 0
        print("No results so far, will save the best model out of this run")
    best_epoch, loss_best, acc_best = 0, 1000, 0

    inds_list = list(range(len(x_train)))
    print("Starting training loop...")
    start = time()
    for epoch in range(n_epochs):

        random.shuffle(inds_list)
        loss_train, train_steps = 0, 0
        model.train()
        total = len(x_train) // batch_size + 1
        pred_labels, real_labels = [], []  # Need to get the real labels because we are shuffling after each epoch
        with tqdm(total=total, desc=f"Epoch {epoch}") as pbar:
            for inds in [inds_list[batch * batch_size:(batch + 1) * batch_size] for batch in
                         range(len(inds_list) // batch_size + 1)]:
                if not inds:
                    break
                optimizer.zero_grad()
                if ONLY_BATCH_IN_GPU:
                    logits = model(x_train[inds].to(device), p_train[inds].to(device), s_train[inds].to(device))
                    loss = criterion(logits, y_train[inds].to(device))
                else:
                    logits = model(x_train[inds], p_train[inds], s_train[inds])
                    loss = criterion(logits, y_train[inds])
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                train_steps += 1
                pbar.update(1)
                pbar.set_postfix_str(f"Training Loss: {loss_train / train_steps:.5f}")
                pred_labels += list(np.argmax(logits.detach().cpu().numpy(), axis=1).reshape(-1))
                real_labels += list(y_train[inds].cpu().numpy().reshape(-1))
        acc_train = 100 * accuracy_score(np.array(real_labels), np.array(pred_labels))

        with torch.no_grad():
            loss_test, test_steps = 0, 0
            model.eval()
            total = len(x_test) // batch_size + 1
            pred_labels = []
            with tqdm(total=total, desc=f"Epoch {epoch}") as pbar:
                for batch in range(len(x_test) // batch_size + 1):
                    inds = slice(batch * batch_size, (batch + 1) * batch_size)
                    if len(x_test[inds]) == 0:
                        break
                    if ONLY_BATCH_IN_GPU:
                        logits = model(x_test[inds].to(device), p_test[inds].to(device), s_test[inds].to(device))
                        loss = criterion(logits, y_test[inds].to(device))
                    else:
                        logits = model(x_test[inds], p_test[inds], s_test[inds])
                        loss = criterion(logits, y_test[inds])
                    loss_test += loss.item()
                    test_steps += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"Testing Loss: {loss_test / test_steps:.5f}")
                    pred_labels += list(np.argmax(logits.cpu().numpy(), axis=1).reshape(-1))
        acc_test = 100 * accuracy_score(y_test.cpu().numpy(), np.array(pred_labels))

        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, loss_train / train_steps, acc_train, loss_test / test_steps, acc_test))

        # Only saves the model if it's better than the models from all of the other experiments
        if acc_test > acc_test_best:
            torch.save(model.state_dict(), "PixelAttentionv2_fashionmnist.pt")
            print("A new model has been saved!")
            acc_test_best = acc_test
        if acc_test > acc_best:
            best_epoch, loss_best, acc_best = epoch, loss_test/test_steps, acc_test

        # To keep track of the metrics during the training process on metrics.json
        ex.log_scalar("training loss", loss_train/train_steps, epoch)
        ex.log_scalar("training acc", acc_train, epoch)
        ex.log_scalar("testing loss", loss_test/test_steps, epoch)
        ex.log_scalar("testing acc", acc_test, epoch)
        # To save the best results of this run to info.json. This is used by get_results() to generate the spreadsheet
        ex.info["epoch"], ex.info["test loss"], ex.info["test acc"] = best_epoch, loss_best, acc_best
        ex.info["train loss"], ex.info["train acc"] = loss_train/train_steps, acc_train
        ex.info["time (min)"], ex.info["actual epochs"] = (time() - start)/60, epoch + 1
