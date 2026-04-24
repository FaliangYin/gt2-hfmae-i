from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import numpy as np
from sklearn.cluster import KMeans

"""
Numpy implementation of FMAE for compatibility and fast test
"""

def membership_fun(x, m):
    return np.exp((-(x - m) ** 2)/2)


class FMAE_explainer_np:
    def __init__(self, ant_dim, con_dim, out_dim, num_fuzzy_set, tr_mode=1, reduce_weight_num=0):
        super().__init__()
        self.tr_mode = tr_mode
        # self.in_dim = 0
        self.ant_dim = ant_dim
        self.con_dim = con_dim
        self.out_dim = out_dim
        self.num_fuzzy_set = num_fuzzy_set

        # generate the rule base
        fs_ind = np.zeros([num_fuzzy_set ** ant_dim, ant_dim])
        for i, ii in enumerate(reversed(range(ant_dim))):  # i--Ascending subscript, ii--descending subscript
            fs_ind[:, ii] = np.tile(np.repeat(range(num_fuzzy_set), num_fuzzy_set ** i), num_fuzzy_set ** ii)
        self.FRB = fs_ind.astype(np.int32)

        self.num_rule = self.FRB.shape[0]

        # initialize the parameters
        self.center = np.array([np.zeros(self.ant_dim), np.ones(self.ant_dim)]) # np.random.rand(self.num_fuzzy_set, ant_dim)
        self.center_mode = '0-1'
        self.con_param = np.zeros([self.out_dim, self.num_rule, con_dim + 1])  # [out_dim, num_rule, in_dim+1]
        if tr_mode == 2:
            self.type_reduce = np.full([reduce_weight_num], 1 / reduce_weight_num)


        self.intercept_ = 0.
        self.coef_ = np.zeros(con_dim)
        self.sample_weight = None

        # self.scaler = StandardScaler()

    def reinit_center(self, original_input):
        scaled_input = original_input
        if self.tr_mode == 2:
            self.center = np.array([np.zeros(self.ant_dim), np.ones(self.ant_dim)])
            # self.center = np.tile(center, (scaled_input[0].shape[0],1))
            return
        if self.center_mode == 'k-means':
            kmeans = KMeans(n_clusters=self.num_fuzzy_set, n_init="auto")
            kmeans.fit(scaled_input)
            self.center = kmeans.cluster_centers_
        elif self.center_mode == '0-1':
            self.center = np.array([np.zeros(self.ant_dim),np.ones(self.ant_dim)])
        else:
            # Take the center equidistant from the minimum to maximum value of the samples
            min_fea = np.min(scaled_input, axis=0)
            max_fea = np.max(scaled_input, axis=0)
            for i in range(self.ant_dim):
                self.center[:, i] = np.linspace(min_fea[i], max_fea[i], self.num_fuzzy_set)

    def antecedent(self, original_input):
        # scaled_input = self.scaler.transform(original_input)
        if self.tr_mode == 2:
            ant_dim, fs_ind = self.ant_dim, self.FRB
            membership_value_, fir_str_bar_ = [], []
            for scaled_input in original_input:
                num_sam = scaled_input.shape[0]
                membership_value = np.concatenate([membership_fun(np.expand_dims(scaled_input, axis=1), self.center)
                                                   , np.ones([num_sam, 1, self.ant_dim])], axis=1)
                membership_value_.append(membership_value)
            membership_value = np.array(membership_value_).mean(axis=0)
            fir_str = np.prod(membership_value[:, fs_ind, range(ant_dim)], axis=2)  # [num_sam, num_rule]
            fir_str_bar = fir_str / np.expand_dims(np.sum(fir_str, 1), axis=1)  # [num_sam, num_rule]
            # fir_str_bar_.append(fir_str_bar)
            # fir_str_bar = np.array(fir_str_bar_).mean(axis=0)
            return fir_str_bar, membership_value
        else:
            scaled_input = original_input
            num_sam = scaled_input.shape[0]
            membership_value = membership_fun(np.expand_dims(scaled_input, axis=1),
                                              self.center)  # [num_sam, num_fuzzy_set, in_dim]  可能有一列（一个特征）始终为0， 导致后续出现0/0
            membership_value = np.concatenate([membership_value, np.ones([num_sam, 1, self.ant_dim])], axis=1)

            ant_dim, fs_ind = self.ant_dim, self.FRB
            fir_str = np.prod(membership_value[:, fs_ind, range(ant_dim)], axis=2)  # [num_sam, num_rule]
            fir_str_bar = fir_str / np.expand_dims(np.sum(fir_str, 1), axis=1)  # [num_sam, num_rule]
            return fir_str_bar, membership_value

    def consequent(self, original_input):
        # scaled_input = self.scaler.transform(original_input)
        scaled_input = original_input
        rule_output = (self.con_param[:, :, 1:] @ scaled_input.T
                       ).T + self.con_param[:, :, 0].T  # [num_sam, num_rule, out_dim]
        return rule_output

    def forward(self, original_input_ant=None, original_input_con=None):
        fir_str_bar, membership_value = self.antecedent(original_input_ant)
        rule_output = self.consequent(original_input_con)
        model_output = np.einsum('NRC,NR->NC', rule_output, fir_str_bar)  # [num_sam, out_dim]
        return model_output, fir_str_bar, membership_value

    def predict(self, original_input_ant, original_input_con):
        model_output, _, _ = self.forward(original_input_ant, original_input_con)
        return model_output

    def score(self, original_input_ant, original_input_con, labels, sample_weight=None, mode='rmse'):
        # evaluate approximation ability of the FLS
        prediction = self.predict(original_input_ant, original_input_con)
        if mode == 'acc':
            return accuracy_score(labels.argmax(axis=1), prediction.detach().numpy().argmax(axis=1),
                                  sample_weight=sample_weight)
        elif mode == 'r2':
            return r2_score(labels, prediction.detach().numpy(), sample_weight=sample_weight)
        else:
            return mean_squared_error(labels, prediction.detach().numpy(), sample_weight=sample_weight)

    def est_con_param(self, original_input_ant, original_input_con, target_output, sample_weight=None, lbd=0.01):
        # scaled_input = self.scaler.transform(original_input)
        scaled_input = original_input_con
        model_input_plus = np.concatenate([np.ones([scaled_input.shape[0], 1]), scaled_input],
                                          axis=1)  # [num_sam, in_dim+1]
        fir_str_bar, _ = self.antecedent(original_input_ant) ######## Attention
        fir_str_bar_input = np.repeat(fir_str_bar, repeats=model_input_plus.shape[1],
                                      axis=1) * np.tile(model_input_plus, [1, fir_str_bar.shape[1]])  # [num_sam,num_rule*(in_dim+1)]

        if sample_weight is None:
            con_param_temp0 = np.linalg.inv(fir_str_bar_input.T @ fir_str_bar_input +
                                           lbd * np.eye(fir_str_bar_input.shape[1])) @ fir_str_bar_input.T
            for i in range(self.out_dim):
                self.con_param[i, :, :] = (con_param_temp0 @ target_output[:, i]).reshape(self.num_rule,
                                                                                          self.con_dim + 1)
            # self.con_param = np.expand_dims(con_param_temp, axis=0).reshape(self.con_param.shape)  # [out_dim,num_rule,in_dim+1]
        else:
            self.sample_weight = sample_weight
            sample_weight = np.diag(sample_weight)
            con_param_temp0 = np.linalg.inv(fir_str_bar_input.T @ sample_weight @ fir_str_bar_input +
                                           lbd * np.eye(fir_str_bar_input.shape[1])) @ fir_str_bar_input.T \
                             @ sample_weight
            for i in range(self.out_dim):
                self.con_param[i, :, :] = (con_param_temp0 @ target_output[:, i]).reshape(self.num_rule, self.con_dim+1)
            # self.con_param = np.expand_dims(con_param_temp0 @ target_output , axis=0).reshape(self.con_param.shape)

    def fit(self, original_input_ant, original_input_con, target_output, sample_weight=None):
        self.reinit_center(original_input_ant)
        self.est_con_param(original_input_ant, original_input_con, target_output, sample_weight)
        self.feature_attribution(original_input_ant)

    def feature_attribution(self, original_input_ant):
        fir_str_bar, _ = self.antecedent(original_input_ant)
        fs = np.einsum('CRD,NR->NDC', self.con_param[:, :, 1:], fir_str_bar)   #######
        dim = original_input_ant[0].shape[0] if self.tr_mode == 2 else original_input_ant.shape[0]
        if dim == 1:
            fs = np.squeeze(fs, axis=0)
        else:
            fs = np.squeeze(np.mean(fs, axis=0, keepdims=True), axis=0)
        if fs.shape[1] == 1:
            self.coef_ = np.squeeze(fs, axis=1)
            self.intercept_ = np.mean(np.squeeze((fir_str_bar @ self.con_param[:, :, 1].T), axis=1))
        else:
            self.coef_ = fs
            self.intercept_ = np.mean((fir_str_bar @ self.con_param[:, :, 1].T), axis=0)
        return fs