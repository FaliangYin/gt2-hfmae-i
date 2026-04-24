import torch
import numpy as np
from torch import nn
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
from torch import optim


def membership_fun(x, m):
    # simplified Gaussian membership function (fixed width)
    return (-(x - m) ** 2).exp()


class FMAE_explainer(nn.Module):
    def __init__(self, ant_dim, con_dim, out_dim, num_fuzzy_set, tr_mode=2, threshold_num=0):
        super().__init__()
        self.ant_dim = ant_dim  # number of antecedent input (semantic feature)
        self.con_dim = con_dim  # number of consequent input (superpixel feature)
        self.out_dim = out_dim  # number of output (classes)
        self.num_fuzzy_set = num_fuzzy_set  # number of fuzzy set for each feature
        self.threshold_num = threshold_num  # number of threshold epsilon
        self.tr_mode = tr_mode  # type reduction mode: 1--maximum; 2--weighted

        # generate the rule base
        rule_base = torch.zeros([num_fuzzy_set ** ant_dim, ant_dim]) # fuzzy rule base
        for i, ii in enumerate(reversed(range(ant_dim))):  # i--forward order；ii--reverse order
            rule_base[:, ii] = torch.tensor(range(num_fuzzy_set)).repeat_interleave(num_fuzzy_set ** i).repeat(
                num_fuzzy_set ** ii)
        self.rule_base = rule_base.long()
        self.num_rule = self.rule_base.shape[0]  # number of rules

        # initialize the parameters
        self.center_mode = '0-1'
        self.center = nn.Parameter(
            torch.stack([torch.zeros(self.ant_dim), torch.ones(self.ant_dim)]))  # center of fuzzy sets
        self.con_param = nn.Parameter(torch.zeros([self.out_dim, self.num_rule, self.con_dim + 1],
                                                  dtype=torch.float64))  # consequent parameters
        if tr_mode == 2:
            self.type_reduce = nn.Parameter(torch.full([self.threshold_num], 1 / self.threshold_num))  # weight of ET2FSs

        self.intercept_ = 0.  # intercept in feature salience explanation
        self.coef_ = np.zeros(self.con_dim)  # feature salience values

    def reinit_center(self, input):
        # initialize the centers of fuzzy sets
        input = input.double()
        if self.tr_mode == 2:
            self.center.data = torch.stack([torch.zeros(self.ant_dim), torch.ones(self.ant_dim)])
            return
        if self.center_mode == 'k-means':
            kmeans = KMeans(n_clusters=self.num_fuzzy_set, n_init="auto")
            kmeans.fit(input.numpy())
            self.center = torch.from_numpy(kmeans.cluster_centers_)
        elif self.center_mode == '0-1':
            self.center = torch.stack([torch.zeros(self.ant_dim),torch.ones(self.ant_dim)])
        else:
            # Take the center equidistant from the minimum to maximum value of the samples
            min_fea = torch.min(input, dim=0).values
            max_fea = torch.max(input, dim=0).values
            for i in range(self.in_dim):
                self.center.data[:, i] = torch.linspace(min_fea[i], max_fea[i], self.num_fuzzy_set)

    def antecedent(self, input):
        # antecedent to calculate the firing strengths
        if self.tr_mode == 2:
            ant_dim, fs_ind = self.ant_dim, self.rule_base
            membership_value_, fir_str_bar_ = [], []
            # self.type_reduce.data = torch.clamp(self.type_reduce.data, 0, 1)
            self.type_reduce.data = nn.functional.relu(self.type_reduce)
            for i, input in enumerate(input):
                membership_value = membership_fun(input.unsqueeze(1), self.center) * self.type_reduce[i]
                membership_value_.append(membership_value)
            membership_value = torch.stack(membership_value_).sum(dim=0) / self.type_reduce.sum()
            fir_str = membership_value[:, fs_ind, range(ant_dim)].prod(dim=2)
            fir_str_bar = fir_str / torch.sum(fir_str, 1).unsqueeze(1)
            return fir_str_bar, membership_value
        else:
            input = input[0]
            num_sam = input.size(0)
            membership_value = torch.cat([membership_fun(input.unsqueeze(1), self.center),
                                          torch.ones([num_sam, 1, self.ant_dim]).to(input.device)], dim=1)
            ant_dim, fs_ind = self.ant_dim, self.rule_base
            fir_str = membership_value[:, fs_ind, range(ant_dim)].prod(dim=2)
            fir_str_bar = fir_str / torch.sum(fir_str, 1).unsqueeze(1)
            return fir_str_bar, membership_value

    def consequent(self, input):
        # consequent containing a set of linear system
        rule_output = (self.con_param[:, :, 1:] @ input.T).T + self.con_param[:, :, 0].T  # [num_sam, num_rule, out_dim]
        return rule_output

    def forward(self, original_input_ant, original_input_con):
        # forward propagation of the FLS
        original_input_ant, original_input_con = original_input_ant.double(), original_input_con.double()
        fir_str_bar, membership_value = self.antecedent(original_input_ant)
        rule_output = self.consequent(original_input_con)
        model_output = torch.einsum('NRC,NR->NC', rule_output, fir_str_bar)
        return model_output, fir_str_bar, membership_value

    def predict(self, original_input_ant, original_input_con):
        # use the FLS to predict
        model_output, _, _ = self.forward(original_input_ant, original_input_con)
        return model_output

    def score(self, original_input_ant, original_input_con, labels, sample_weight=None, mode='rmse'):
        # evaluate approximation ability of the FLS
        prediction = self.predict(original_input_ant, original_input_con)
        if mode == 'acc':
            return accuracy_score(labels.argmax(axis=1), prediction.detach().numpy().argmax(axis=1), sample_weight=sample_weight)
        elif mode == 'r2':
            return r2_score(labels, prediction.detach().numpy(), sample_weight=sample_weight)
        else:
            return mean_squared_error(labels, prediction.detach().numpy(), sample_weight=sample_weight)

    def feature_attribution(self, original_input_ant):
        # generate feature salience explanations
        original_input_ant = original_input_ant.double()
        fir_str_bar, _ = self.antecedent(original_input_ant)
        fs = torch.einsum('CRD,NR->NDC', self.con_param[:, :, 1:], fir_str_bar)   #######
        dim = original_input_ant[0].size(0) if self.tr_mode == 2 else original_input_ant.size(0)
        if dim == 1:
            fs = fs.squeeze(dim=0)
        else:
            fs = fs.mean(dim=0, keepdim=True).squeeze(dim=0)
        if fs.shape[1] == 1:
            self.coef_ = fs.squeeze(dim=1)
            self.intercept_ = (fir_str_bar @ self.con_param[:, :, 1].T).mean(dim=1)
        else:
            self.coef_ = fs
            self.intercept_ = (fir_str_bar @ self.con_param[:, :, 1].T).mean(dim=1)
        return fs


def train_full_batch(model_input_ant, model_input_con, model, target_output,
                     learning_rate=0.01, max_iter=1000, gpu=True):
    """
    train the FLS surrogate model
    :param model_input_ant: input of antecedent
    :param model_input_con: input of consequent
    :param model: FLS
    :param target_output: real output
    """
    if gpu:  # check if cuda is available
        device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device_gpu)
        model_input_ant, model_input_con, target_output = \
            model_input_ant.to(device_gpu), model_input_con.to(device_gpu), target_output.to(device_gpu)

    model_input_ant = model_input_ant.double()
    model_input_con = model_input_con.double()
    target_output = target_output.double()

    # loss function and optimizer
    criterion = mse_loss_fun
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) #
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # iterations
    loss_his = torch.zeros(max_iter)  # record loss
    for i in range(max_iter):
        model_output, _, _ = model.forward(model_input_ant, model_input_con)

        loss = criterion(model_output, target_output)
        loss_his[i] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    if gpu:
        device_cpu = torch.device("cpu")
        model.to(device_cpu)

    return loss_his.data


def mse_loss_fun(y=None, z=None):
    return ((y - z) ** 2).sum() / (2 * y.size(0))
