from modeling import RowColumnAttention
from data_utils import load_data, prep_data
from get_results import get_results
import os
import random
from time import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sacred import Experiment
ex = Experiment('cifar100_attention')

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
@ex.config
def my_config():
    random_seed = 42

    lr = 1e-3
    n_epochs = 200
    batch_size = 64
    dropout = 0.2

    hidden_dim1 = 32  # This can change, but embedding_dim is 32 (images of 32x32) and BERT has hidden_dim1=embedding_dim
    # If using color, the default behaviour is being multiplied by 3 (as embedding dim is)
    # Actually, changing this to be different than embedding_dim will give problems... (for now)
    hidden_dim2 = 48  # In BERT, this = hidden_dim1*4. It's the two linear layers after the self-attention
    n_heads = 1  # This needs to be a number that hidden_dim1 is divisible by
    n_layers = 3  # Number of Attention layers

    use_cls = True  # This includes one more row that is learnt by an embedding and used for classification on the pooler
    # If set to True, the segment embeddings have 3 ids, to distinguish between this special row, actual rows and actual columns
    # And the position embeddings ids also increment by 1
    position_embedding_version = "rows_and_columns"  # "rows_and_columns" means we give id 0 to first row, id 1 to second
    # row, ... id n_rows to first column, etc. and we use nn.Embedding on that and add it up with the other embeddings
    segment_embedding_version = "rows_vs_columns"  # "rows_vs_columns" means we give id 0 to all rows and id 1 to all columns
    # and we use nn.Embedding on that and add it up with the other embeddings

    attention_version = "default"  # "default" = "-" on the spreadsheet. This means that the color channels are concatenated
    # and we do attention on the whole concatenated rows and columns.
    # Options include: "per_channel": Do not concatenate the color channels and do attention on each separately. Then combine
    # them together at the end using one of the options from "attention_indv_channels_merge_mode"
    attention_indv_channels_merge_mode = "concat"  # When attention_version="default", this is not activated. If attention_version="per_channel"
    # then we use: "sum": add each channel, "average": avg each channel, "concat": concat each channel

# %% ------------------------------------------ Experiment -------------------------------------------------------------
@ex.automain
def my_main(random_seed, lr, hidden_dim1, hidden_dim2, n_heads, n_layers, n_epochs, batch_size, dropout,
            position_embedding_version, segment_embedding_version, attention_version,
            attention_indv_channels_merge_mode, use_cls):

    # %% --------------------------------------- Set-Up ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    N_CLASSES = 100
    ONLY_BATCH_IN_GPU = True
    COLOR = True
    DATA_DIR = os.getcwd() + "/cifar-100-python/"

    # %% -------------------------------------- Data Prep --------------------------------------------------------------
    x_train, y_train = load_data(DATA_DIR + "train")
    x_train, p_train, s_train, y_train = prep_data(x_train, y_train, only_batch_in_gpu=ONLY_BATCH_IN_GPU, use_cls=use_cls)
    x_test, y_test = load_data(DATA_DIR + "test")
    x_test, p_test, s_test, y_test = prep_data(x_test, y_test, only_batch_in_gpu=ONLY_BATCH_IN_GPU, use_cls=use_cls)

    # %% -------------------------------------- Training Prep ----------------------------------------------------------
    model = RowColumnAttention(COLOR, use_cls, x_train.shape[1], hidden_dim1, hidden_dim2, n_layers, n_heads, N_CLASSES,
                               dropout, position_embedding_version, segment_embedding_version, attention_version,
                               attention_indv_channels_merge_mode).to(device)
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
            torch.save(model.state_dict(), "PixelAttentionv2_cifar100.pt")
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
