import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_5.utilis import org_rnn_input
import math


class CrossProduct(nn.Module):
    def __init__(self, x_dim, v_dim):
        """
            Cross production operation based on factorization machine
        :param x_dim:
        :param v_dim:
        """
        super(CrossProduct, self).__init__()
        self.x_dim = x_dim
        self.k = v_dim
        self.vparam = nn.Parameter(torch.FloatTensor(self.x_dim, self.k))
        self.reset_parameter()

    def forward(self, x):
        """
        :param x:
            Input tensor with shape (batchsize, x_dim)
        :return:
            Output tensor with shape (batch_size, 1)
        """
        # *? need to use sparse operation
        x = torch.diag_embed(x)
        x = torch.matmul(x, self.vparam)
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        x = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return 0.5 * x

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.vparam.size(1))
        self.vparam.data.uniform_(-stdv, stdv)


class CPLayer(nn.Module):
    def __init__(self, x_dim, v_dim, out_feature):
        super(CPLayer, self).__init__()
        self.in_feature = x_dim
        self.k = v_dim
        self.out_feature = out_feature
        self.cps = self.generate_cp()

    def generate_cp(self):
        cp_list = nn.ModuleList()
        for i in range(self.out_feature):
            cp_list.append(CrossProduct(x_dim=self.in_feature, v_dim=self.k))
        return cp_list

    def forward(self, x):
        """
        :param x:
            Input tensor with shape (batch_size, in_feature)
        :return:
            Output tensor with shape (batch_szie, out_feature)
        """
        x = torch.cat([per_cp(x) for per_cp in self.cps], dim=-1)
        return x


class EFMB(nn.Module):
    def __init__(self, x_dim, v_dim, out_feature_1, out_feature_2):
        super(EFMB, self).__init__()
        self.in_deature = x_dim
        self.k = v_dim
        self.out_feature_1 = out_feature_1
        self.out_feature_2 = out_feature_2
        self.cps = CPLayer(self.in_deature, self.k, self.out_feature_1)
        self.aff_transform = nn.Linear(self.out_feature_1, self.out_feature_2)

    def forward(self, x):
        """
        :param x:
            Input tensor with shape (batch_size, in_feature)
        :return:
            Output tensor with shape (batch_szie, out_feature)
        """
        x = self.cps(x)
        x = self.aff_transform(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_feature, hidden_unit, out_feature):
        super(MLP, self).__init__()
        self.in_feature = in_feature
        self.hidden_unit = hidden_unit
        self.out_feature = out_feature
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(in_feature, hidden_unit)
        self.fc2 = nn.Linear(hidden_unit, hidden_unit)
        self.fc3 = nn.Linear(hidden_unit, out_feature)

    def forward(self, x):
        """ To learn the dense global features including time, weather , ...
            using simple full connected layer
            the normalization layer should to be considered
        :param x:
            Input tensor with shape [batch_size, feature]
            represent the several global features
        :return:
        """
        x = self.relu(self.fc1(x))
        residual = x
        # * before the dropout or not
        x = self.relu(self.fc2(x) + residual)
        residual = x
        x = self.relu(self.fc3(x) + residual)
        return x


class SeqNet(nn.Module):
    def __init__(self, in_feature, hidden_unit, out_deep_feature, out_wide_feature):
        super(SeqNet, self).__init__()
        self.in_feature = in_feature
        self.hidden_unit = hidden_unit
        self.out_deep_feature = out_deep_feature
        self.out_wide_feature = out_wide_feature
        self.GRU = nn.GRU(input_size=self.in_feature + self.out_deep_feature + self.out_wide_feature, hidden_size=self.hidden_unit,
                          num_layers=2, batch_first=False, dropout=0.3)

    def forward(self, x, deep_feature, wide_feature, length):
        """ To learn the trip features with the road segment embeddings and the outputs of deepnet
        :param x:
            Input tensor with shape []
            represent the feature of trip (road segment, current status)
        :param deep_feature:
            Input tensor with shape [batch_size, deep_feature]
            represent the learned features of global feature by DeepNet
        :param wide_feature:
            Input tensor with shape [batch_size, wide_feature]
            represent the learned sparse global feature by WideNet
        :param length:
            Input
            represent the seq length
        :return:
        """
        x = org_rnn_input(x, deep_feature, wide_feature, length)
        # batch_first = False
        packed_out, h_n = self.GRU(x)
        unpacked = pad_packed_sequence(packed_out)[0]
        # To split the data along the dimension of seq
        unpacked_list = torch.chunk(unpacked, chunks=unpacked.shape[0], dim=0)
        return unpacked_list


class MCLB(nn.Module):
    def __init__(self, in_dim, at_dim):
        """
        :param in_dim:
            Represent the dimension of global features
        :param at_dim:
            Represent the dimension of RNN out
        """
        super(MCLB, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=at_dim)
        self.sofmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, mask):
        """ To calculate the attention between the trip features and global features
        :param query:
            with shape [batch_size, feature]
        :param context:
            with shape (seq, batch_size, at_feature)
        :return:
        """
        context = context.permute(1, 0, 2).contiguous()
        query = self.tanh(self.linear(query)).unsqueeze(-1)
        # query with shape [batch_size, at_feature, 1]
        attention_scores = torch.bmm(context, query)
        # attention_scores with shape [batch_size, seq, 1] the third dimension is scores
        attention_scores = attention_scores.squeeze(-1)
        # * The padded is 0 but the negative number is exist, that need to masked?
        attention_scores = self.sofmax(attention_scores.masked_fill(mask, float('-inf'))).masked_fill(mask, 0.0).unsqueeze(1)
        # attention_scores with shape [batch_size, 1, seq]
        values = torch.bmm(attention_scores, context)
        values = values.squeeze(1)
        return values


class OutNet(nn.Module):
    def __init__(self, in_feature_1, gru_hidden):
        super(OutNet, self).__init__()
        self.mclb = MCLB(in_feature_1, gru_hidden)
        self.linear = nn.Linear(in_feature_1 + gru_hidden, 1)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, global_feature, seq_out_list, mask):
        """
        :param global_feature:
            the global feature with shape [batch_size, in_feature_1]
        :param seq_out_list:
            the output of each step of seqnet with shape list([batch_size, in_feature_2]), len=seq
        :return:
        """
        context = torch.cat(seq_out_list, dim=0)
        values = self.mclb(global_feature, context, mask)
        # To cat the feature of seq_model and deep_model
        x = torch.cat([values, global_feature], dim=1)
        x = self.linear(x)
        return x


class HSETA(nn.Module):
    def __init__(self, in_w, in_d, in_r, v_w, hidden_w, hidden_d, out_w, out_d, out_r):
        super(HSETA, self).__init__()
        self.in_w = in_w
        self.in_d = in_d
        self.in_r = in_r
        self.k = v_w
        self.hidden_w = hidden_w
        self.hidden_d = hidden_d
        self.out_w = out_w
        self.out_d = out_d
        self.out_r = out_r
        self.efmb = EFMB(x_dim=self.in_w, v_dim=self.k, out_feature_1=self.hidden_w, out_feature_2=self.out_w)
        self.mlp = MLP(in_feature=self.in_d, hidden_unit=self.hidden_d, out_feature=self.out_d)
        self.rnn = SeqNet(in_feature=self.in_r, hidden_unit=self.out_r, out_deep_feature=self.out_d, out_wide_feature=self.out_w)
        self.regression = OutNet(in_feature_1=self.out_w+self.out_d, gru_hidden=self.out_r)

    def generate_boolean_mask(self, length: list):
        """
            To generate the tensor with shape (batch_size, seq) for attention
        :param length:
            List which contact the length of each seq in one batch
        :return:
            Tensor with shape (batch_size, seq)
        """
        mask = torch.zeros((len(length), max(length)))
        for i, per_length in enumerate(length):
            mask[i, per_length:] = 1
        mask = mask > 0
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, x_w, x_d, x_r, length):
        mask = self.generate_boolean_mask(length)
        x_w = self.efmb(x_w)
        x_d = self.mlp(x_d)
        x_r = self.rnn(x_r, x_d, x_w, length)
        x_global = torch.cat([x_d, x_w], dim=-1)
        x = self.regression(x_global, x_r, mask)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.GRU):
                nn.init.orthogonal_(m.weight_ih_l0)
                nn.init.orthogonal_(m.weight_hh_l0)
                nn.init.orthogonal_(m.weight_ih_l1)
                nn.init.orthogonal_(m.weight_hh_l1)
                nn.init.zeros_(m.bias_ih_l0)
                nn.init.zeros_(m.bias_hh_l0)
                nn.init.zeros_(m.bias_ih_l1)
                nn.init.zeros_(m.bias_hh_l1)


class mape(nn.Module):
    def __init__(self):
        super(mape, self).__init__()
        return

    def forward(self, outs, labels):
        loss = torch.mean(torch.abs((outs - labels) / labels))
        return loss


class NoamOpt:
    """
    Optim wrapper that control the learning rate (including warmup and decrease)
    """

    def __init__(self, d_model, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.__step = 0
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self.__lr = 0

    def step(self):
        """
        Replace optim.step()
        :return:
        """
        self.__step = self.__step + 1
        lr = self.learning_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.__lr = lr
        self.optimizer.step()

    def zero_grad(self):
        """
        Replace optim.zero_grad()
        :return:
        """
        self.optimizer.zero_grad()

    def learning_rate(self, step=None):
        """
        Refresh the learning rate
        :param step:
            Auto generation
        :return:
        """
        if step is None:
            step = self.__step
        lr = self.factor * (self.d_model ** (-0.5)) * min((step ** (-0.5)), (step * (self.warmup ** (-1.5))))
        return lr

    def qurey(self):
        return self.__step, self.__lr