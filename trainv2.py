# %% --------------------------------------- Imports -------------------------------------------------------------------
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
N_CLASSES = 10
TRAIN, LOAD_PREVIOUS = True, True
SAVE_MODEL = True
ONLY_BATCH_IN_GPU = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# LR = 1e-5
LR = 1e-3
# HIDDEN_DIM1, HIDDEN_DIM2, N_HEADS = 768, 3072, 1
HIDDEN_DIM1, HIDDEN_DIM2, N_HEADS, N_LAYERS = 28, 12, 1, 3
N_EPOCHS = 200
# BATCH_SIZE = 64
BATCH_SIZE = 512
DROPOUT = 0.2

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x_pixel, x_pos, x_seg, y, return_labels=False):
    pred_labls = []
    with torch.no_grad():
        for bth in range(len(x_pixel) // BATCH_SIZE + 1):
            idxs = slice(bth * BATCH_SIZE, (bth + 1) * BATCH_SIZE)
            lgts = model(x_pixel[idxs], x_pos[idxs], x_seg[idxs])
            pred_labls += list(np.argmax(lgts.cpu().numpy(), axis=1).reshape(-1))
    if return_labels:
        return np.array(pred_labls)
    else:
        return 100*accuracy_score(y.cpu().numpy(), np.array(pred_labls))

def show_mistakes(x_pixel, x_pos, y, idx, label_names):
    y_pred = acc(x_pixel, x_pos, y, return_labels=True)
    idx_mistakes = np.argwhere((y.cpu().numpy() == y_pred) == 0).flatten()
    plt.title("MLP prediction: {} - True label: {}".format(label_names[y_pred[idx_mistakes[idx]]],
                                                           label_names[y[idx_mistakes[idx]]]))
    plt.imshow(x_pixel[idx_mistakes[idx]][1:].reshape(28, 28).cpu().numpy())
    plt.show()

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
    def __init__(self, n_attn_heads=N_HEADS, dropout=DROPOUT, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2):
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
    def __init__(self, color, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2, n_layers=N_LAYERS, n_attn_heads=N_HEADS,
                 n_classes=N_CLASSES, dropout=DROPOUT):
        super(RowColumnAttention, self).__init__()
        self.color = color

        # Embeddings
        # For now we will be using only images of 28x28 pixels
        hidden_dim_mult = 3 if color else 1
        self.pos_embedding = nn.Embedding(hidden_dim1*2, hidden_dim1*hidden_dim_mult)  # TODO: 28 + 1 ?
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
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flat = nn.Linear(hidden_dim1*2*hidden_dim1*hidden_dim_mult, hidden_dim1*hidden_dim_mult)
        self.act_flat = nn.Tanh()
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
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(size=(len(x), x.shape[1]))
        if segment_ids is None:
            segment_ids = torch.cat([
                torch.zeros((len(x), x.shape[1]/2)), torch.ones((len(x), x.shape[1]/2))],
                dim=1, device=x.device).long()
        x = x + self.pos_embedding(position_ids) + self.seg_embedding(segment_ids)  # Now the rows are already embedded
        x = self.drop_embedding(self.bn_embedding(x))

        # Attention Layers
        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        # Pooler
        pooled_output = self.pooler_act(self.pooler(x))
        pooled_output = self.bn_flat(self.act_flat(self.flat(pooled_output.view(len(pooled_output), -1))))
        # Classification
        return self.cls(self.cls_drop(pooled_output))

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
if TRAIN:
    data_train = datasets.FashionMNIST(root='.', train=True, download=True)
    x_train, y_train = data_train.data.float()/255, data_train.targets
    p_train = torch.arange(HIDDEN_DIM1*2, dtype=torch.long).unsqueeze(0).expand(size=(len(x_train), HIDDEN_DIM1*2))
    s_train = torch.cat([torch.zeros((len(x_train), HIDDEN_DIM1)), torch.ones((len(x_train), HIDDEN_DIM1))], dim=1).long()
    x_train.requires_grad = True
    if not ONLY_BATCH_IN_GPU:
        x_train, p_train, s_train, y_train = x_train.to(device), p_train.to(device), s_train.to(device), y_train.to(device)
data_test = datasets.FashionMNIST(root='.', train=False, download=True)
x_test, y_test = data_test.data.float()/255, data_test.targets
p_test = torch.arange(HIDDEN_DIM1*2, dtype=torch.long).unsqueeze(0).expand(size=(len(x_test), HIDDEN_DIM1*2))
s_test = torch.cat([torch.zeros((len(x_test), HIDDEN_DIM1)), torch.ones((len(x_test), HIDDEN_DIM1))], dim=1).long()
if not ONLY_BATCH_IN_GPU:
    x_test, p_test, s_test, y_test = x_test.to(device), p_test.to(device), s_test.to(device), y_test.to(device)

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = RowColumnAttention(color=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
if TRAIN:
    if LOAD_PREVIOUS:
        try:
            model.load_state_dict(torch.load("PixelAttentionv2_fashionmnist.pt"))
            print("A previous model has been loaded!")
        except:
            print("The previous model failed to load, probably due to architecture changes.")
    inds_list = list(range(len(x_train)))
    loss_test_best = 1000
    print("Starting training loop...")
    for epoch in range(N_EPOCHS):

        random.shuffle(inds_list)
        loss_train, train_steps = 0, 0
        model.train()
        total = len(x_train) // BATCH_SIZE + 1
        pred_labels, real_labels = [], []  # Need to get the real labels because we are shuffling after each epoch
        with tqdm(total=total, desc=f"Epoch {epoch}") as pbar:
            for inds in [inds_list[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE] for batch in range(len(inds_list) // BATCH_SIZE + 1)]:
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
        acc_train = 100*accuracy_score(np.array(real_labels), np.array(pred_labels))

        with torch.no_grad():
            loss_test, test_steps = 0, 0
            model.eval()
            total = len(x_test) // BATCH_SIZE + 1
            pred_labels = []
            with tqdm(total=total, desc=f"Epoch {epoch}") as pbar:
                for batch in range(len(x_test) // BATCH_SIZE + 1):
                    inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
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
        acc_test = 100*accuracy_score(y_test.cpu().numpy(), np.array(pred_labels))

        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, loss_train/train_steps, acc_train, loss_test/test_steps, acc_test))

        if loss_test < loss_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), "PixelAttentionv2_fashionmnist.pt")
            print("The model has been saved!")
            loss_test_best = loss_test

# %% ------------------------------------------ Final test -------------------------------------------------------------
model.load_state_dict(torch.load("PixelAttentionv2_fashionmnist.pt"))
model.eval()
y_test_pred = acc(x_test, p_test, y_test, return_labels=True)
print("The accuracy on the test set is", 100*accuracy_score(y_test.cpu().numpy(), y_test_pred), "%")
print(data_test.classes)
print("The confusion matrix is")
print(confusion_matrix(y_test.cpu().numpy(), y_test_pred))

show_mistakes(x_test, p_test, y_test, 0, data_test.classes)
