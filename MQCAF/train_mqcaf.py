# ref in https://github.com/mdswyz/DMD
# @Time    : 27/4/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university

import logging
import os
from pathlib import Path
import numpy as np
import torch
from config import get_json5_config_regression
from utils import assign_gpu, setup_seed
import sys
import torch.nn as nn
import dnn_models
import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import scipy.io


def _set_logger(log_dir, model_name, dataset_name, verbose_level,seed):
    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}-{seed}.log"
    logger = logging.getLogger(f'GAT_{seed}')
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def ReadList(list_file):
    f=open(list_file,"r")
    lines=f.readlines()
    list_sig=[]
    for x in lines:
        list_sig.append(x.rstrip())
    f.close()
    return list_sig

def wav_sort(elem):
    wav_name = elem.split("_")[-1]
    return int(wav_name)


def adjust_array(array, target_length):
    current_length = len(array)

    if current_length < target_length:
        padding = np.zeros(target_length - current_length)
        return np.concatenate((array, padding))
    else:
        return array


def adjust_arrays(arrays, sample_rates, target_time):
    durations = [len(array) / sample_rate for array, sample_rate in zip(arrays, sample_rates)]

    max_duration = max(durations, default=target_time)

    final_target_lengths = [int(max_duration * sample_rate) for sample_rate in sample_rates]

    adjusted_arrays = []
    for array, final_target_length in zip(arrays, final_target_lengths):
        current_length = len(array)
        if current_length < final_target_length:
            padding = np.zeros(final_target_length - current_length)
            adjusted_array = np.concatenate((array, padding))
        else:
            adjusted_array = array
        adjusted_arrays.append(adjusted_array)

    return adjusted_arrays



def pad_matrix(matrix, target_rows=40):
    current_rows, cols = matrix.shape
    if current_rows <= target_rows:
        padding = np.zeros((target_rows - current_rows + 5, cols))
        matrix = np.vstack((matrix, padding))
    return matrix


def create_batches_align_ecg_eeg_video(batch_size,wav_lst,N_snt,args_now):

    sig_batch_ecg = np.zeros([batch_size,args_now.ecg.wlen])
    sig_batch_eeg = np.zeros([batch_size,args_now.eeg.wlen])
    sig_batch_video = np.zeros([batch_size,args_now.video.wlen,args_now.video.feature_num])

    lab_batch = np.zeros(batch_size)

    snt_id_arr = np.random.randint(N_snt, size = batch_size)


    for i in range(batch_size):
        now_name = wav_lst[snt_id_arr[i]]

        if args_now.task_now == 'rest':
            pass
        elif args_now.task_now == 'video':
            ecg_name = now_name.replace('rest', 'rest')
            eeg_name = now_name.replace('rest', 'video1')
            video_name = now_name.replace('ecg', 'videomci2301')

            video_path = args_now.video.data_folder+video_name
            ecg_path = args_now.ecg.data_folder+ecg_name
            eeg_path = args_now.eeg.data_folder+eeg_name


        if os.path.exists(video_path):
            data = scipy.io.loadmat(video_path)['data']
            # data = data[:, 680:696]
            part1 = data[:, 680:696]
            part2 = data[:, 716:723]
            part3 = data[:, 725:726]
            data = np.hstack((part1, part2, part3))
            signal_video = pad_matrix(matrix=data, target_rows=args_now.video.wlen)
        else:
            signal_video = np.zeros((args_now.video.wlen*16,args_now.video.feature_num))


        if now_name.endswith('.mat'):


            data = scipy.io.loadmat(eeg_path)
            signal_eeg = data['data']
            signal_eeg = np.squeeze(signal_eeg)

            if not os.path.exists(ecg_path):
                new_length = len(signal_eeg) * 16

                signal_ecg = np.zeros(new_length)
            else:
                data = scipy.io.loadmat(ecg_path)
                signal_ecg = data['data']

        else:
            signal_ecg = np.loadtxt(ecg_path)
            signal_eeg = np.loadtxt(eeg_path)


        signal_ecg = np.squeeze(signal_ecg)
        signal_eeg = np.squeeze(signal_eeg)
        [signal_ecg, signal_eeg] = adjust_arrays([signal_ecg, signal_eeg], [args_now.ecg.fs, args_now.eeg.fs], args_now.eeg.cw_len/1000)



        snt_time = signal_eeg.shape[0]/args_now.eeg.fs
        snt_beg = np.random.randint(int(snt_time-args_now.eeg.cw_len/1000+1)) #randint(0, snt_len-2*wlen-1)
        snt_end = int(snt_beg + args_now.eeg.cw_len/1000)


        sig_batch_ecg[i,:] = signal_ecg[snt_beg*args_now.ecg.fs : snt_end*args_now.ecg.fs ]
        sig_batch_eeg[i,:] = signal_eeg[snt_beg*args_now.eeg.fs : snt_end*args_now.eeg.fs ]


        snt_time = signal_video.shape[0]/args_now.video.fs
        snt_beg = np.random.randint(int(snt_time-args_now.video.cw_len/1000+1)) #randint(0, snt_len-2*wlen-1)
        snt_end = int(snt_beg + args_now.video.cw_len/1000)
        snt_beg_video = snt_beg * args_now.video.fs
        snt_end_video = snt_end * args_now.video.fs
        sig_batch_video[i,:,:] = signal_video[snt_beg_video:snt_end_video,:]


        if int(now_name.split('_')[-3]) < args_now.mci_score:
            lab_batch[i] = 1.0
        else:
            lab_batch[i] = 0.0

    inp_ecg = torch.from_numpy(sig_batch_ecg).float().cuda().contiguous()
    inp_eeg = torch.from_numpy(sig_batch_eeg).float().cuda().contiguous()

    inp_video = torch.from_numpy(sig_batch_video).float().cuda().contiguous()

    lab = torch.from_numpy(lab_batch).float().cuda().contiguous()

    return inp_ecg, inp_eeg, inp_video, lab

def deleteDuplicatedElementFromList2(list):
    resultList = []
    for item in list:
        if not item in resultList:
            resultList.append(item)
    return resultList

def my_0_1(x):
    tem = 0
    if float(x)>=0.5:
        tem = 1
    return tem

def _run_all(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold = args.fold
    acc_list = []
    f1_list = []
    auc_list = []
    pre_list = []
    rec_list = []

    epoch_get_acc = []
    epoch_get_f1 = []
    epoch_get_auc = []
    epoch_get_pre = []
    epoch_get_rec = []

    for i in range(fold):
        acc_list.append(1)
        f1_list.append(1)
        auc_list.append(1)
        pre_list.append(1)
        rec_list.append(1)

        epoch_get_acc.append([])
        epoch_get_f1.append([])
        epoch_get_auc.append([])
        epoch_get_pre.append([])
        epoch_get_rec.append([])

    import re

    def extract_numbers_from_string(input_string):
        numbers = re.findall(r'\d+', input_string)
        return numbers

    def model_sort(elem):
        num_now = extract_numbers_from_string(elem)[0]
        return int(num_now)

    if args.ecg.pt_file != 'none':
        model_path_ecg = os.listdir(args.ecg.pt_file)
        model_path_ecg.sort(key=model_sort)

    if args.eeg.pt_file != 'none':
        model_path_eeg = os.listdir(args.eeg.pt_file)
        model_path_eeg.sort(key=model_sort)


    if args.modality == "ecg-eeg":
        pass
    elif args.modality == "ecg-eeg-video":
        if args.video.pt_file != 'none':
            model_path_video = os.listdir(args.video.pt_file)
            model_path_video.sort(key=model_sort)

    writer = SummaryWriter(f'{args.output_folder}/tensorboard_my')

    for fold_i in range(fold):
        tr_lst_fold = args.tr_lst + f'{fold_i}.txt'
        te_lst_fold = args.te_lst + f'{fold_i}.txt'

        # training list
        wav_lst_tr = ReadList(tr_lst_fold)
        snt_tr = len(wav_lst_tr)

        # test list
        wav_lst_te = ReadList(te_lst_fold)
        snt_te = len(wav_lst_te)

        person_name = []
        for i, audio in enumerate(wav_lst_te):
            person = audio.split('_')[0]
            person_name.append(person)

        person_name = deleteDuplicatedElementFromList2(person_name)

        logger.info(f'test_len:{snt_te}')
        logger.info(wav_lst_te)

        # # Define Early Stop variables
        # patience = args.patience  # Number of epochs to wait before stopping
        # best_auc = 0  # The best validation loss so far
        # counter = 0  # Number of epochs since the best validation loss improved

        cost = nn.CrossEntropyLoss()
        # l2_loss = nn.MSELoss()
        # cos_loss = nn.CosineEmbeddingLoss(reduction='mean')

        args.ecg.wlen = int(args.ecg.fs * args.ecg.cw_len / 1000.00)
        args.ecg.wshift = int(args.ecg.fs * args.ecg.cw_shift / 1000.00)
        if args.modal_base == 'ResNet1D_sinc_att':
            ecg_net = dnn_models.ResNet1D_sinc_att(option_my=args.ecg, num_classes=2)
        ecg_net.cuda()

        # EEG model
        args.eeg.wlen = int(args.eeg.fs * args.eeg.cw_len / 1000.00)
        args.eeg.wshift = int(args.eeg.fs * args.eeg.cw_shift / 1000.00)
        if args.modal_base == 'ResNet1D_sinc_att':
            eeg_net = dnn_models.ResNet1D_sinc_att(option_my=args.eeg, num_classes=2)
            eeg_net.cuda()


        video_net = dnn_models.TransformerModel_fc(args.video, args.video.feature_num, 2, 64, 2, 2).to(device)
        video_net.cuda()

        GNN_model = dnn_models.att_my_multiquery_video(my_model_option=args.gat)

        GNN_model.cuda()

        if args.ecg.pt_file != 'none':
            pt_file_load = f'{args.ecg.pt_file}/{model_path_ecg[fold_i]}'
            checkpoint_load = torch.load(pt_file_load)
            ecg_net.load_state_dict(checkpoint_load['CNN_model_par'], strict=False)
        if args.eeg.pt_file != 'none':
            pt_file_load = f'{args.eeg.pt_file}/{model_path_eeg[fold_i]}'
            checkpoint_load = torch.load(pt_file_load)
            eeg_net.load_state_dict(checkpoint_load['CNN_model_par'], strict=False)
        if args.video.pt_file != 'none':
            pt_file_load = f'{args.video.pt_file}/{model_path_video[fold_i]}'
            checkpoint_load = torch.load(pt_file_load)
            video_net.load_state_dict(checkpoint_load, strict=False)

        # params = list(ecg_net.parameters()) + \
        #          list(eeg_net.parameters()) + \
        #          list(video_net.parameters()) + \
        #          list(GNN_model.parameters())

        params = GNN_model.parameters()

        optimizer_all = torch.optim.RMSprop(params, lr=args.lr, alpha=0.95, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_all, step_size=10, gamma=0.1)

        for epoch in range(args.N_epochs):

            ecg_net.train()
            eeg_net.train()
            video_net.train()
            GNN_model.train()


            loss_sum = 0
            err_sum = 0
            train_label = np.array([])
            train_pred = np.array([])

            #
            for i in tqdm.tqdm(range(args.N_batches)):
                inp_ecg, inp_eeg, inp_video, lab = create_batches_align_ecg_eeg_video(args.batch_size, wav_lst_tr, snt_tr, args)
                pout_ecg, feature_ecg = ecg_net(inp_ecg)
                pout_eeg, feature_eeg = eeg_net(inp_eeg)
                pout_video, feature_video = video_net(inp_video)

                # cross modal attention
                pout = GNN_model(feature_ecg, feature_eeg,feature_video)
                # loss = cost(pout, lab.long())
                loss = 1 * cost(pout, lab.long()) + 0.8 * cost(pout_eeg, lab.long()) + 0.6 * cost(pout_ecg, lab.long())+ cost(pout_video, lab.long())


                pred = torch.max(pout, dim=1)[1]
                err = torch.mean((pred != lab.long()).float())
                optimizer_all.zero_grad()
                loss.backward()
                optimizer_all.step()
                loss_sum = loss_sum + loss.detach()
                err_sum = err_sum + err.detach()
                train_label = np.append(train_label, lab.cpu().detach().numpy())
                train_pred = np.append(train_pred, pred.cpu().detach().numpy())

            loss_tot = loss_sum / args.N_batches
            err_tot = err_sum / args.N_batches
            conf_matrix = confusion_matrix(train_label, train_pred, labels=[1, 0])
            scheduler.step()

    # res_file.write(f"epoch_get_auc={sum_err}\n")
def GAT_run(
        model_name, dataset_name, config=None, config_file="", seeds=[], log_dir="",
        verbose_level=1
):
    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if config_file != "":
        config_file = Path(config_file)
    else:  # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")

    args = get_json5_config_regression(model_name, dataset_name, config_file)

    if config:
        args.update(config)

    args.ecg.fusion_method = args.fusion_method
    args.eeg.fusion_method = args.fusion_method
    args.video.fusion_method = args.fusion_method
    args.video.align = args.align
    args.eeg.align = args.align
    args.ecg.align = args.align

    from datetime import datetime
    current_time = datetime.now()
    time_string = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    print("time:", time_string)

    args.output_folder_dir = args.output_folder_dir + f'_{args.fusion_method}_' + time_string
    # model_results = []
    try:
        os.stat(args.output_folder_dir)
    except:
        os.mkdir(args.output_folder_dir)

    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args.seed = seed
        args.output_folder = os.path.join(args.output_folder_dir, f"seed_{args.seed}")
        # Folder creation
        try:
            os.stat(args.output_folder)
        except:
            os.mkdir(args.output_folder)

        logger = _set_logger(args.output_folder, model_name, dataset_name, verbose_level,args.seed)
        logger.info(args)
        _run_all(args, logger)

if __name__ == '__main__':
    GAT_run(model_name='MQCAF', dataset_name='mci_nn_pre', config_file='config/config_multimodal.json5',seeds=[1234])