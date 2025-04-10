"""Based on original code by Yongjie Wang https://github.com/wangyongjie-ntu/CFAI/"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class REVISE:

    def __init__(self, data_interface, model_interface, model_vae):
        self.data_interface = data_interface
        self.model_interface = model_interface
        self.model_vae = model_vae
        self.model_vae.eval()

    def generate_counterfactuals(self, query_instance, features_to_vary=None, target=0.7, feature_weights=None,
                                 _lambda=0.001, optimizer="adam", lr=3, max_iter=300):

        start_time = time.time()
        if isinstance(query_instance, dict) or isinstance(query_instance, list):
            query_instance = self.data_interface.prepare_query(query_instance, normalized=True)
        query_instance = torch.FloatTensor(query_instance)
        mask = self.data_interface.get_mask_of_features_to_vary(features_to_vary)
        mask = torch.LongTensor(mask)

        self._lambda = _lambda

        if feature_weights is None:
            feature_weights = torch.ones(query_instance.shape[1])
        else:
            feature_weights = torch.ones(query_instance.shape[0])
            feature_weights = torch.FloatTensor(feature_weights)

        if isinstance(self.data_interface.scaler, MinMaxScaler):
            cf_initialize = torch.rand(query_instance.shape)
        elif isinstance(self.data_interface.scaler, StandardScaler):
            cf_initialize = torch.randn(query_instance.shape)
        else:
            cf_initialize = torch.rand(query_instance.shape)

        cf_initialize = torch.FloatTensor(cf_initialize)
        cf_initialize = mask * cf_initialize + (1 - mask) * query_instance

        with torch.no_grad():
            mu, log_var = self.model_vae.encode(cf_initialize)
            z = self.model_vae.reparameterize(mu, log_var)
            cf = self.model_vae.decode(z)

        if optimizer == "adam":
            optim = torch.optim.Adam([cf], lr)
        else:
            optim = torch.optim.RMSprop([cf], lr)

        for i in range(max_iter):
            cf.requires_grad = True
            optim.zero_grad()
            # cf = self.model_vae.decode(z)
            loss = self.compute_loss(cf, query_instance, target)
            loss.backward()
            optim.step()
            cf.detach_()

        end_time = time.time()
        running_time = time.time()
        final_cf = self.model_vae.decode(z)

        return final_cf.numpy()

    def compute_loss(self, cf_initialize, query_instance, target):

        loss1 = F.relu(target - self.model_interface.predict_tensor(cf_initialize)[1])
        loss2 = torch.sum((cf_initialize - query_instance) ** 2)
        print(loss1, "\t", loss2)
        return loss1 + self._lambda * loss2


class VAE(nn.Module):

    def __init__(self, data_size,
                 encoded_size,
                 data_interface,
                 hidden_dims=[20, 16, 12]
                 ):

        super(VAE, self).__init__()

        self.data_size = data_size
        self.encoded_size = encoded_size
        self.data_interface = data_interface
        self.hidden_dims = hidden_dims

        modules = []

        in_channels = data_size
        # create encoder module
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(12, self.encoded_size)
        self.fc_var = nn.Linear(12, self.encoded_size)

        # create decoder module
        modules = []
        in_channels = encoded_size

        for h_dim in reversed(self.hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        modules.append(nn.Linear(in_channels, self.data_size))
        self.sig = nn.Sigmoid()
        self.decoder = nn.Sequential(*modules)

    def encode(self, input_x):

        output = self.encoder(input_x)
        mu = self.fc_mu(output)
        log_var = self.fc_var(output)

        return [mu, log_var]

    def decode(self, z):

        x = self.decoder(z)
        for v in self.data_interface.encoded_categorical_feature_indices:
            start_index = v[0]
            end_index = v[-1] + 1
            x[:, start_index:end_index] = self.sig(x[:, start_index:end_index])
        return x

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input_x):

        mu, log_var = self.encode(input_x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input_x, mu, log_var]

    def compute_loss(self, output, input_x, mu, log_var):

        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        con_criterion = nn.MSELoss()
        cat_criterion = nn.BCELoss()

        cat_loss = 0
        con_loss = 0

        index_list = []
        v = self.data_interface.encoded_categorical_feature_indices
        for i in range(len(input_x[0])):
            if i in v:
                index_list.append(1)
            else:
                index_list.append(0)
        index_list = torch.tensor(index_list)

        cat_loss += cat_criterion(torch.where(index_list > 0, output, 0), torch.where(index_list > 0, input_x, 0))

        con_loss = con_criterion(torch.where(index_list < 1, output, 0), torch.where(index_list < 1, input_x, 0))
        recon_loss = torch.mean(cat_loss + con_loss)
        total_loss = kl_loss + recon_loss

        return total_loss, recon_loss, kl_loss


class ReviseModel:
    def __init__(self, model, *args, **kwargs):
        self.model = model

    def predict_tensor(self, x):
        return self.model(x)


class ReviseData:
    def __init__(self, dataset: torch.tensor, scaler, cat_feature_idx: torch.tensor):
        self.dataset = dataset
        self.scaler = scaler
        self.encoded_categorical_feature_indices = cat_feature_idx

    def prepare_query(self, query_instance, normalized):
        pass

    def get_mask_of_features_to_vary(self, features_to_vary):
        return torch.zeros_like(self.dataset[0])


def revise_distance(x, x_cf, config):
    # print(f"x: {x},\n x_cf: {x_cf}\n")
    return torch.nn.MSELoss()(x, x_cf)
