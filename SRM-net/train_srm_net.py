# ref in https://github.com/mravanelli/SincNet
# @Time    : 25/4/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university


import os
import torch
from torch import nn
import numpy as np
import random
import tqdm
import scipy.io
import re
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import dnn_models
from data_io_attention import ReadList, read_conf, str_to_bool

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_matrix = 'auc'

# Load configuration file (replace with actual path)
options = read_conf('')

# Timestamped output folder
time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print("Current time:", time_string)

# Extract config parameters
modal_my = options.modal
name = options.name
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder + '/'
output_folder = options.output_folder + '_' + time_string
data_format = options.data_format

fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)
channel_num = int(options.channel_num)

arch = options.arch
cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))

mulhead_num_hiddens = int(options.mulhead_num_hiddens)
mulhead_num_heads = int(options.mulhead_num_heads)
mulhead_num_query = int(options.mulhead_num_query)
dropout_fc = float(options.dropout_fc)
att_hidden_dims_fc = int(options.att_hidden_dims_fc)
hidden_dims_fc = int(options.hidden_dims_fc)
num_classes = int(options.num_classes)

lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)
fold = int(options.fold)
patience = int(options.patience)

mci_score = 26

# Logging configuration to file
def write_options_to_file(res_file, options):
    option_attributes = [attr for attr in dir(options) if not callable(getattr(options, attr)) and not attr.startswith("__")]
    for attr in option_attributes:
        value = getattr(options, attr)
        if isinstance(value, str) and os.path.isfile(value):
            value = os.path.basename(value)
        res_file.write(f"{attr}: {value}\n")

# Seed setup for reproducibility
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

setup_seed(seed)

# Random batch creation function
def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, fact_amp):
    sig_batch = np.zeros([batch_size, wlen])
    lab_batch = np.zeros(batch_size)
    snt_id_arr = np.random.randint(N_snt, size=batch_size)
    for i in range(batch_size):
        file_path = data_folder + wav_lst[snt_id_arr[i]]
        if data_format == 'mat':
            data = scipy.io.loadmat(file_path)
            if 'ppg' in modal_my:
                signal = data['data'][:, 2]
            elif name == 'ecg2_video1_pre_mj':
                signal = data['data'][:, 0].squeeze() * 1e3
            else:
                signal = data['data']
        else:
            signal = np.loadtxt(file_path)
        signal = np.squeeze(signal)
        if signal.ndim != 1:
            continue
        if len(signal) <= wlen:
            signal = np.pad(signal, (0, wlen + 5 - len(signal)), 'constant')
        snt_beg = np.random.randint(signal.shape[0] - wlen - 1)
        sig_batch[i, :] = signal[snt_beg:snt_beg + wlen]
        temp_wav_lst = wav_lst[snt_id_arr[i]]
        if name == 'eeg1rest_pre':
            label = int(temp_wav_lst.split('_')[-2].split('.')[0])
        elif 'summerv1234' in name:
            label = int(temp_wav_lst.split('_')[-1].split('.')[0])
        else:
            label = int(temp_wav_lst.split('_')[-3].split('.')[0])
        lab_batch[i] = 1 if label < mci_score else 0
    inp = torch.from_numpy(sig_batch).float().to(device).contiguous()
    lab = torch.from_numpy(lab_batch).float().to(device).contiguous()
    return inp, lab

# Metric storage setup
acc_list = [1] * fold
f1_list = [1] * fold
auc_list = [1] * fold
pre_list = [1] * fold
rec_list = [1] * fold

epoch_get_acc = [[] for _ in range(fold)]
epoch_get_f1 = [[] for _ in range(fold)]
epoch_get_auc = [[] for _ in range(fold)]
epoch_get_pre = [[] for _ in range(fold)]
epoch_get_rec = [[] for _ in range(fold)]

# Create output folder
os.makedirs(output_folder, exist_ok=True)
with open(f"{output_folder}/res.txt", "a") as res_file:
    write_options_to_file(res_file, options)

# Extract fold index from filename
def extract_numbers_from_string(input_string):
    return re.findall(r'\d+', input_string)

def model_sort(elem):
    return int(extract_numbers_from_string(elem)[0])

# ======= TRAINING LOOP STARTS =======
for fold_i in range(fold):
    writer = SummaryWriter(f'{output_folder}/tensorboard_fold{fold_i}')
    tr_lst_fold = tr_lst + f'{fold_i}.txt'
    te_lst_fold = te_lst + f'{fold_i}.txt'
    wav_lst_tr = [f for f in ReadList(tr_lst_fold) if f in os.listdir(data_folder)]
    wav_lst_te = [f for f in ReadList(te_lst_fold) if f in os.listdir(data_folder)]
    snt_tr = len(wav_lst_tr)
    snt_te = len(wav_lst_te)
    print(f'Train size: {snt_tr}, Test size: {snt_te}')

    CNN_arch = {
        'wlen': int(fs * cw_len / 1000.0),
        'fs': fs,
        'cnn_N_filt': cnn_N_filt,
        'cnn_len_filt': cnn_len_filt,
        'cnn_max_pool_len': cnn_max_pool_len,
        'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
        'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
        'cnn_use_laynorm': cnn_use_laynorm,
        'cnn_use_batchnorm': cnn_use_batchnorm,
        'cnn_act': cnn_act,
        'cnn_drop': cnn_drop,
        'mulhead_num_hiddens': mulhead_num_hiddens,
        'mulhead_num_heads': mulhead_num_heads,
        'mulhead_num_query': mulhead_num_query,
        'dropout_fc': dropout_fc,
        'hidden_dims_fc': hidden_dims_fc,
        'att_hidden_dims_fc': att_hidden_dims_fc,
        'num_classes': num_classes,
        'channel_num': channel_num,
    }

    CNN_net = dnn_models.ResNet1D_sinc_att(CNN_arch, num_classes=num_classes).to(device)

    if pt_file != 'none':
        model_path_ecg = sorted(os.listdir(pt_file), key=model_sort)
        pt_file_load = f'{pt_file}/{model_path_ecg[fold_i]}'
        checkpoint_load = torch.load(pt_file_load)
        CNN_net.load_state_dict(checkpoint_load['CNN_model_par'], strict=False)

    optimizer_CNN = torch.optim.RMSprop(CNN_net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_CNN, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_value = 0
    counter = 0
    for epoch in range(N_epochs):
        CNN_net.train()
        total_loss = 0
        total_err = 0
        train_label = np.array([])
        train_pred = np.array([])

        for _ in tqdm.tqdm(range(N_batches)):
            inp, lab = create_batches_rnd(batch_size, data_folder, wav_lst_tr, snt_tr, CNN_arch['wlen'], 0.2)
            pout = CNN_net(inp)
            loss = criterion(pout, lab.long())
            pred = torch.max(pout, dim=1)[1]
            err = torch.mean((pred != lab.long()).float())
            optimizer_CNN.zero_grad()
            loss.backward()
            optimizer_CNN.step()

            total_loss += loss.item()
            total_err += err.item()
            train_label = np.append(train_label, lab.cpu().numpy())
            train_pred = np.append(train_pred, pred.cpu().numpy())

        avg_loss = total_loss / N_batches
        avg_err = total_err / N_batches

        print("Train Confusion Matrix:")
        print(confusion_matrix(train_label, train_pred, labels=[1, 0]))
        scheduler.step()
