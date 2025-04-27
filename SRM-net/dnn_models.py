# ref in https://github.com/mravanelli/SincNet
# @Time    : 27/4/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
import multi_head

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,-1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
def sinc(band,t_right):
    y_right =  torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left =  flip(y_right,0)

    y = torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y
class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate = 16000, in_channels = 1,
                 stride = 1, padding = 0, dilation = 1, bias = False, groups = 1, min_low_hz = 0.2, min_band_hz = 2):

        super(SincConv_fast,self).__init__()

        if in_channels !=  1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2 == 0:
            self.kernel_size = self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = min_low_hz
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)


        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps = int((self.kernel_size/2))) # computing only half of the window
        self.window_ = 0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes




    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band = (high-low)[:,0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2*band.view(-1,1)
        band_pass_right =  torch.flip(band_pass_left,dims = [1])


        band_pass = torch.cat([band_pass_left,band_pass_center,band_pass_right],dim = 1)


        band_pass = band_pass / (2*band[:,None])


        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        # tem_filters = self.filters.cpu().detach().numpy().reshape(80, -1)

        return F.conv1d(waveforms, self.filters, stride = self.stride,
                        padding = self.padding, dilation = self.dilation,
                        bias = None, groups = 1)
class SincConv_fast_my(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate = 16000, in_channels = 1,
                 stride = 1, padding = 0, dilation = 1, bias = False, groups = 1, min_low_hz = 0.2, min_band_hz = 8):

        super().__init__()

        if in_channels !=  1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2 == 0:
            self.kernel_size = self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz + 5

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = min_low_hz
        # high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        high_hz = self.min_low_hz + 5

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)


        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps = int((self.kernel_size/2))) # computing only half of the window
        self.window_ = 0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes




    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band = (high-low)[:,0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2*band.view(-1,1)
        band_pass_right =  torch.flip(band_pass_left,dims = [1])


        band_pass = torch.cat([band_pass_left,band_pass_center,band_pass_right],dim = 1)


        band_pass = band_pass / (2*band[:,None])


        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        # tem_filters = self.filters.cpu().detach().numpy().reshape(80, -1)

        return F.conv1d(waveforms, self.filters, stride = self.stride,
                        padding = self.padding, dilation = self.dilation,
                        bias = None, groups = 1)
class sinc_conv(nn.Module):

    def __init__(self, N_filt,Filt_dim,fs):
        super(sinc_conv,self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        b1 = np.roll(f_cos,1)
        b2 = np.roll(f_cos,-1)
        b1[0] = 30
        b2[-1] = (fs/2)-100

        self.freq_scale = fs*1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))


        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs


    def forward(self, x):

        filters = Variable(torch.zeros((self.N_filt,self.Filt_dim))).cuda()
        N = self.Filt_dim
        t_right = Variable(torch.linspace(1, (N-1)/2, steps = int((N-1)/2))/self.fs).cuda()


        min_freq = 50.0;
        min_band = 50.0;

        filt_beg_freq = torch.abs(self.filt_b1)+min_freq/self.freq_scale
        filt_end_freq = filt_beg_freq+(torch.abs(self.filt_band)+min_band/self.freq_scale)

        n = torch.linspace(0, N, steps = N)

        # Filter window (hamming)
        window = 0.54-0.46*torch.cos(2*math.pi*n/N);
        window = Variable(window.float().cuda())


        for i in range(self.N_filt):

            low_pass1 = 2*filt_beg_freq[i].float()*sinc(filt_beg_freq[i].float()*self.freq_scale,t_right)
            low_pass2 = 2*filt_end_freq[i].float()*sinc(filt_end_freq[i].float()*self.freq_scale,t_right)
            band_pass = (low_pass2-low_pass1)

            band_pass = band_pass/torch.max(band_pass)

            filters[i,:] = band_pass.cuda()*window

        out = F.conv1d(x, filters.view(self.N_filt,1,self.Filt_dim))

        return out
def act_fun(act_type):

    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim = 1)

    if act_type == "linear":
        return nn.LeakyReLU(1) # initializzed like this, but not used in forward!

class LayerNorm(nn.Module):

    def __init__(self, features, eps = 1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()

        self.input_dim = int(options['input_dim'])
        self.fc_lay = options['fc_lay']
        self.fc_drop = options['fc_drop']
        self.fc_use_batchnorm = options['fc_use_batchnorm']
        self.fc_use_laynorm = options['fc_use_laynorm']
        self.fc_use_laynorm_inp = options['fc_use_laynorm_inp']
        self.fc_use_batchnorm_inp = options['fc_use_batchnorm_inp']
        self.fc_act = options['fc_act']


        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])



        # input layer normalization
        if self.fc_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.fc_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim],momentum = 0.05)


        self.N_fc_lay = len(self.fc_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_fc_lay):

            # dropout
            self.drop.append(nn.Dropout(p = self.fc_drop[i]))

            # activation
            self.act.append(act_fun(self.fc_act[i]))


            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.fc_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.fc_lay[i],momentum = 0.05))

            if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
                add_bias = False


            # Linear operations
            self.wx.append(nn.Linear(current_input, self.fc_lay[i],bias = add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.fc_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.fc_lay[i])),np.sqrt(0.01/(current_input+self.fc_lay[i]))))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))

            current_input = self.fc_lay[i]


    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.fc_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.fc_use_batchnorm_inp):
            x = self.bn0((x))

        for i in range(self.N_fc_lay):

            if self.fc_act[i] != 'linear':

                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

                if self.fc_use_batchnorm[i] == False and self.fc_use_laynorm[i] == False:
                    x = self.drop[i](self.act[i](self.wx[i](x)))

            else:
                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.ln[i](self.wx[i](x)))

                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.bn[i](self.wx[i](x)))

                if self.fc_use_batchnorm[i] == False and self.fc_use_laynorm[i] == False:
                    x = self.drop[i](self.wx[i](x))

        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D_att(nn.Module):
    def __init__(self, option_my, block, layers, num_classes=1000):
        super().__init__()

        self.signal_channels = option_my['channel_num']
        self.in_channels = 64

        self.cnn_N_filt = option_my['cnn_N_filt']
        self.cnn_len_filt = option_my['cnn_len_filt']
        self.fs = option_my['fs']
        self.input_dim = int(option_my['wlen'])
        self.mulhead_num_hiddens = option_my['mulhead_num_hiddens']
        self.mulhead_num_heads = option_my['mulhead_num_heads']
        self.mulhead_num_query = option_my['mulhead_num_query']
        self.att_hidden_dims_fc = option_my['att_hidden_dims_fc']

        self.dropout_fc = option_my['dropout_fc']


        self.layer_norm1 = nn.LayerNorm(normalized_shape=(1, self.input_dim))
        self.sinc_layer1 = SincConv_fast_my(self.cnn_N_filt[0],self.cnn_len_filt[0],self.fs,padding='same')

        self.conv1 = nn.Conv1d(self.cnn_N_filt[0]*self.signal_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(32)

        self.mul_attention_f = multi_head.MultiHeadAttention(512, 512,
                                                           512,
                                                           self.mulhead_num_hiddens, self.mulhead_num_heads, 0.5)
        self.fc_att = nn.Sequential(
            nn.Linear(32*self.mulhead_num_hiddens, self.att_hidden_dims_fc),
            nn.BatchNorm1d(self.att_hidden_dims_fc, momentum=0.05),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_fc),
            nn.Linear(self.att_hidden_dims_fc, self.att_hidden_dims_fc//2),
            nn.BatchNorm1d(self.att_hidden_dims_fc//2, momentum=0.05),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_fc),
            nn.Linear(self.att_hidden_dims_fc//2, self.att_hidden_dims_fc//4),
        )
        self.fc = nn.Linear(self.att_hidden_dims_fc//4, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_now = x.shape[0]

        x = x.reshape(batch_now * self.signal_channels, 1 , -1)
        x = self.layer_norm1((x))
        x = self.sinc_layer1((x))
        x = x.reshape(batch_now, self.signal_channels * self.cnn_N_filt[0] , -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)


        x = x.permute(0, 2, 1)
        x = self.mul_attention_f(x, x, x, valid_lens=None)
        x = x.permute(0, 2, 1)
        x = torch.flatten(x, 1)
        x = self.fc_att(x)
        x = self.fc(x)

        return x
def ResNet1D_sinc_att(option_my, num_classes=1000):
    return ResNet1D_att(option_my, BasicBlock, [2, 2, 2, 2], num_classes)
