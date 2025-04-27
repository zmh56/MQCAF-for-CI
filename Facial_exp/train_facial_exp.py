# ref in https://github.com/phamquiluan/ResidualMaskingNetwork
# @Time    : 27/4/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university


import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn import metrics
import argparse
import pandas as pd
import tqdm
import scipy.io as sio
import time
import random
from sklearn.model_selection import train_test_split
import multi_head

# Function to adjust tensor size by padding or truncating
def adjust_tensor(tensor):
    first_dim = tensor.shape[0]
    if first_dim < 100:
        padding = torch.zeros((100 - first_dim, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        adjusted_tensor = torch.cat((tensor, padding), dim=0)
    else:
        adjusted_tensor = tensor[:100, :]
    return adjusted_tensor

# Function to pad matrix with zeros until target rows are met
def pad_matrix(matrix, target_rows=40):
    current_rows, cols = matrix.shape
    if current_rows <= target_rows:
        padding = np.zeros((target_rows - current_rows + 5, cols))
        matrix = np.vstack((matrix, padding))
    return matrix

# Dataset class to handle loading and processing of feature data
class CustomFeatureDataset(Dataset):
    def __init__(self, list_mat, args_dataset, audio_class):
        self.audio_list = list_mat
        self.audio_class = audio_class
        self.csv_path = args_dataset.mat_path

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_name = self.audio_list[idx]
        label_indices = torch.zeros(self.audio_class)
        dataFile = os.path.join(self.csv_path, audio_name)

        if os.path.exists(dataFile):
            data = sio.loadmat(dataFile)['data']
            part1 = data[:, 680:696]
            part2 = data[:, 716:723]
            part3 = data[:, 725:726]
            data = np.hstack((part1, part2, part3))
            data = pad_matrix(matrix=data, target_rows=args.win)
            snt_beg = np.random.randint(data.shape[0]-args.win)
            snt_end = snt_beg + args.win
            feature_data = data[snt_beg:snt_end, :]
            feature_data = torch.from_numpy(feature_data)
        else:
            print(f"The file '{dataFile}' does not exist.")
            feature_data = torch.zeros(args.win, feature_num)

        if int(audio_name.split('_')[-3]) < mci_score:
            label_indices[1] = 1.0
        else:
            label_indices[0] = 1.0

        return feature_data.float(), label_indices.float()

# Transformer model for feature fusion with fully connected layers
class TransformerModel_fc_fusion(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.fusion = 'attention'
        self.mulhead_num_hiddens = 32
        self.att_hidden_dims_fc = 512
        self.dropout_fc = 0.3

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc2 = nn.Linear(120, 32)
        self.fc1 = nn.Linear(24, self.mulhead_num_hiddens)

        self.fc_att = nn.Linear(32*self.mulhead_num_hiddens, self.att_hidden_dims_fc//4)
        self.fc = nn.Linear(self.att_hidden_dims_fc//4, output_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)  # Transformer encoding
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)

        if self.fusion == "attention":
            tsne_data = x
        x = torch.flatten(x, 1)
        x = self.fc_att(x)

        if self.fusion == "fc":
            tsne_data = x

        x = self.fc(x)
        return x, tsne_data

# Transformer model with attention mechanism
class TransformerModel_att(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.mul_attention_t = multi_head.MultiHeadAttention(64, 64, 64, 16, 4, 0.5)
        self.mul_attention_c = multi_head.MultiHeadAttention(116, 116, 116, 32, 4, 0.5)

        self.fc = nn.Linear(32*16, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.mul_attention_t(x, x, x, valid_lens=None)
        x = x.permute(0, 2, 1)
        x = self.mul_attention_c(x, x, x, valid_lens=None)
        x = x.permute(0, 2, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Training function
def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.0
    train_bar = tqdm.tqdm(dataloader)
    matrix_label, matrix_pred = np.array([]), np.array([])

    for batch, (X, y) in enumerate(train_bar):
        X, y = X.to(device), y.to(device)
        pred, _ = model(X)
        loss = loss_fn(pred, y)
        y_arg = y.argmax(1)
        pred_arg = pred.argmax(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        matrix_label = np.append(matrix_label, y_arg.cpu().numpy())
        matrix_pred = np.append(matrix_pred, pred_arg.cpu().numpy())

        train_bar.set_postfix(loss=loss.item())

    average_loss = total_loss / len(dataloader)
    accuracy = metrics.accuracy_score(matrix_label, matrix_pred)
    if epoch % args.epochs_val == 0:
        print(f"Epoch Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

# Set the random seed for reproducibility
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Function to read training and validation file lists
def get_txt_from(path_xlsx, seed_datasets):
    data = pd.read_excel(path_xlsx)
    if 'Name' in data.columns:
        data = data.drop(columns=['Name'])

    features = [col for col in data.columns if col in ['ID','CA_Result','MOCA', 'MMSE']]
    X = data[features]
    X = X.fillna(0)
    target = 'CA_Result'
    y = data[target]
    y = y.replace({1: 0, 2: 0, 0: 1, 3: 1, 4: 1})

    feature_names = X.columns
    print(feature_names)

    indices = X.index
    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(X, y, indices, test_size=0.3,
                                                                             random_state=seed_datasets, stratify=y)

    target_id = ['ID', 'CA_Result', 'MOCA', 'MMSE']
    train_file, val_file = [], []

    train_values = data.loc[train_idx, target_id]
    for index, value in train_values.iterrows():
        id_temp = value['ID']
        label_temp = value['CA_Result']
        moca_temp = value['MOCA']
        mmse_temp = value['MMSE']
        str_my_file = f'{id_temp}_game_{moca_temp}_{mmse_temp}_{label_temp}.mat'
        train_file.append(str_my_file)

    val_values = data.loc[val_idx, target_id]
    for index, value in val_values.iterrows():
        id_temp = value['ID']
        label_temp = value['CA_Result']
        moca_temp = value['MOCA']
        mmse_temp = value['MMSE']
        str_my_file = f'{id_temp}_game_{moca_temp}_{mmse_temp}_{label_temp}.mat'
        val_file.append(str_my_file)

    return train_file, val_file

# Function to read file list
def ReadList(list_file):
    with open(list_file, "r") as f:
        lines = f.readlines()
    return [x.rstrip() for x in lines]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DNN model')
    parser.add_argument('--model', type=str, choices=['lstm', 'gru', 'transformer','transformer_fc','transformer_fc_fusion'], default='transformer_fc_fusion', help='Model type')
    parser.add_argument('--mat_path', type=str, default='path_data')
    parser.add_argument('--train_file', type=str, default='path_train_list')
    parser.add_argument('--val_file', type=str, default='path_val_list')
    parser.add_argument('--random_split', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--mci_score', type=int, default=26, help='MCI score threshold')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--feature_num', type=int, default=24, help='Number of features')
    parser.add_argument('--output_folder', type=str, default='exp/video1')

    args = parser.parse_args()

    args.win = args.win_time * args.fps
    from datetime import datetime
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    print("Current time:", time_string)
    args.output_folder = f'{args.output_folder}_{args.model}_{time_string}'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    setup_seed(42)

    batch_size = args.batch_size
    mci_score = args.mci_score
    epochs = args.epochs
    fold = args.fold
    feature_num = args.feature_num


    acc_list = []
    f1_list = []
    auc_list = []

    for fold_i in range(args.fold):
        best_auc = 0
        train_file = f"{args.train_file}{fold_i}.txt"
        val_file = f"{args.val_file}{fold_i}.txt"

        model = TransformerModel_fc_fusion(feature_num, 2, 64, 2, 2).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_data = CustomFeatureDataset(list_mat=train_file, audio_class=2, args_dataset=args)
        val_data = CustomFeatureDataset(list_mat=val_file, audio_class=2, args_dataset=args)

        train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

        for t in range(args.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer, t)
            time.sleep(0.2)
            scheduler.step()
        print("Done!")
