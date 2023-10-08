import json
from math import log2, ceil
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score


def normalize(x):
    x = (x - torch.mean(x).unsqueeze(-1)) / torch.std(x).unsqueeze(-1)
    return x

"""
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        y = self.fc(x)
        return y
"""

torch.cuda.is_available = lambda : False

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        y = self.network(x)
        # y[:, -1] = torch.sigmoid(y[:, -1])
        return y


class Enco_Conv_Net(nn.Module):
    def __init__(self, n_channels, output_dim):
        super(Enco_Conv_Net, self).__init__()
        self.features_2x2 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)            
            )
        self.pool1d = nn.MaxPool1d(3, 2)
        self.features_4x4 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=4),
            nn.ReLU(),
            # nn.MaxPool1d(3, 2)        
            )
        self.classifier = nn.Linear(n_channels * 9, output_dim)

    def forward(self, x):
        x = self.transform(x)
        x1 = self.features_2x2(x)
        x2 = self.features_4x4(x)
        x2 = self.pool1d(x2.squeeze(2))
        x1 = x1.flatten(1)        
        x2 = x2.flatten(1)
        x_ = torch.cat((x1, x2), 1)
        y = self.classifier(x_)
        y[:,-1] = torch.sigmoid(y[:,-1])
        return y
    
    def transform(self, x):
        len = x[0].shape[0]
        xbar = torch.cat((x[:, 6:], x[:, :6]), 1)
        x = x.unsqueeze(1).unsqueeze(1)
        xbar = xbar.unsqueeze(1).unsqueeze(1)
        x = torch.cat((x, xbar, x, xbar), 2)
        return x

class Classifier:
    def __init__(self, samples, input_dim, node_id):
        assert type(samples) == type({})
        assert input_dim     >= 1

        self.samples          = samples
        self.input_dim        = input_dim
        self.training_counter = 0
        self.node_layer       = ceil(log2(node_id + 2) - 1)
        self.hidden_dims      = [16, 20, 24, 28, 32]
        self.model            = Encoder(input_dim, self.hidden_dims[self.node_layer], 2)
        if torch.cuda.is_available():
            self.model.cuda()
        self.loss_fn          = nn.MSELoss()
        self.l_rate           = 0.001
        # self.optimizer        = optim.Adam(self.model.parameters(), lr=self.l_rate, betas=(0.9, 0.999), eps=1e-08)
        self.epochs           = []
        self.training_accuracy = [0]
        self.boundary         = -1
        self.nets             = None
        self.maeinv           = None
        self.labels           = None

    def get_label(self, energy):
        label = torch.zeros_like(energy)
        for i in range(energy.shape[0]):
            label[i] = energy[i] > energy.mean()
        return label

    def update_samples(self, latest_samples):
        assert type(latest_samples) == type(self.samples)
        sampled_nets = []
        nets_maeinv  = []
        for k, v in latest_samples.items():
            net = json.loads(k)
            sampled_nets.append(net)
            nets_maeinv.append(v)
        self.nets = torch.from_numpy(np.asarray(sampled_nets, dtype=np.float32).reshape(-1, self.input_dim))
        self.nets = normalize(self.nets)
        self.maeinv = torch.from_numpy(np.asarray(nets_maeinv, dtype=np.float32).reshape(-1, 1))
        self.labels = self.get_label(self.maeinv)
        self.samples = latest_samples
        if torch.cuda.is_available():
            self.nets = self.nets.cuda()
            self.maeinv = self.maeinv.cuda()
            self.labels = self.labels.cuda()


    def train(self):
        if self.training_counter == 0:
            self.epochs = 3000
        else:
            self.epochs = 1000
        self.training_counter += 1
        # in a rare case, one branch has no networks
        if len(self.nets) == 0:
            return
        nets = self.nets
        labels = 2 * self.labels - 1
        maeinv = self.maeinv
        train_data = TensorDataset(nets, maeinv, labels)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.l_rate, betas=(0.9, 0.999), eps=1e-08)
        for epoch in range(self.epochs):
            for x, y, z in train_loader:
                # clear grads
                optimizer.zero_grad()
                # forward to get predicted values
                outputs = self.model(x)
                # loss_s = self.loss_fn(outputs[:, :6], nets[:, 6:])
                loss_mae = self.loss_fn(outputs[:, 0], y.reshape(-1))
                loss_t = self.loss_fn(outputs[:, -1], z.reshape(-1))
                loss = loss_t  # loss_mae + loss_t
                loss.backward()  # back props
                # nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()  # update the parameters

        # training accuracy
        pred = self.model(nets).cpu()
        # split by maeinv
        # pred_label = (pred[:, -1] > self.sample_mean()).float()
        # true_label = (maeinv.reshape(-1) > self.sample_mean()).float()
        # split by label
        pred_label = (pred[:, -1] > 0).float()        
        true_label = self.labels.reshape(-1).cpu()
        acc = accuracy_score(true_label.numpy(), pred_label.numpy())
        self.training_accuracy.append(acc)


    # def predict(self, remaining):
    #     assert type(remaining) == type({})
    #     remaining_archs = []
    #     for k, v in remaining.items():
    #         net = json.loads(k)
    #         remaining_archs.append(net)
    #     remaining_archs = torch.from_numpy(np.asarray(remaining_archs, dtype=np.float32).reshape(-1, self.input_dim))
    #     if torch.cuda.is_available():
    #         remaining_archs = remaining_archs.cuda()
    #     outputs = self.model(remaining_archs)[:, -1].reshape(-1, 1)
    #     if torch.cuda.is_available():
    #         remaining_archs = remaining_archs.cpu()
    #         outputs         = outputs.cpu()
    #     result = {}
    #     for k in range(0, len(remaining_archs)):
    #         arch = remaining_archs[k].detach().numpy().astype(np.int32)
    #         arch_str = json.dumps(arch.tolist())
    #         result[arch_str] = outputs[k].detach().numpy().tolist()[0]
    #     assert len(result) == len(remaining)
    #     return result
    
    def predict(self, remaining):
        assert type(remaining) == type({})
        remaining_archs = []
        for k, v in remaining.items():
            net = json.loads(k)
            remaining_archs.append(net)
        remaining_archs = torch.from_numpy(np.asarray(remaining_archs, dtype=np.float32).reshape(-1, self.input_dim))
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cuda()

        outputs = self.model(normalize(remaining_archs))
        labels = outputs[:, -1].reshape(-1, 1)  #output labels
        xbar = outputs[:, 0].mean().detach().tolist()

        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cpu()
            labels         = labels.cpu()
        result = {}
        for k in range(0, len(remaining_archs)):
            arch = remaining_archs[k].detach().numpy().astype(np.int32)
            arch_str = json.dumps(arch.tolist())
            result[arch_str] = labels[k].detach().numpy().tolist()[0]
        assert len(result) == len(remaining)
        return result, xbar


    def split_predictions(self, remaining, method = None):
        assert type(remaining) == type({})
        samples_badness = {}
        samples_goodies = {}
        xbar = 0
        if len(remaining) == 0:
            return samples_goodies, samples_badness, 0
        if method == None:
            predictions, xbar = self.predict(remaining)  # arch_str -> pred_test_mae
            for k, v in predictions.items():
                # if v < self.sample_mean():
                # split by label
                if v < 0:
                    samples_badness[k] = v
                else:
                    samples_goodies[k] = v
        else:
            predictions = np.mean(list(remaining.values()))
            for k, v in remaining.items():
                if v > predictions:
                    samples_badness[k] = v
                else:
                    samples_goodies[k] = v

        assert len(samples_badness) + len(samples_goodies) == len(remaining)
        return samples_goodies, samples_badness, xbar

    """
    def predict_mean(self):
        if len(self.nets) == 0:
            return 0
        # can we use the actual maeinv?
        outputs = self.model(self.nets)
        pred_np = None
        if torch.cuda.is_available():
            pred_np = outputs.detach().cpu().numpy()
        else:
            pred_np = outputs.detach().numpy()
        return np.mean(pred_np)
    """

    def sample_mean(self):
        if len(self.nets) == 0:
            return 0
        outputs = self.maeinv
        true_np = None
        if torch.cuda.is_available():
            true_np = outputs.cpu().numpy()
        else:
            true_np = outputs.numpy()
        return np.mean(true_np)


    def split_data(self, f1 = None):
        samples_badness = {}
        samples_goodies = {}
        if len(self.nets) == 0:
            return samples_goodies, samples_badness
        self.train()
        outputs = self.model(self.nets)[:, -1].reshape(-1, 1)
        if torch.cuda.is_available():
            self.nets = self.nets.cpu()
            outputs   = outputs.cpu()
        predictions = {}
        for k in range(0, len(self.nets)):
            # arch = self.nets[k].detach().numpy().astype(np.int32)
            # arch_str = json.dumps(arch.tolist())
            arch_str = list(self.samples)[k]
            predictions[arch_str] = outputs[k].detach().numpy().tolist()[0]  # arch_str -> pred_test_mae
        assert len(predictions) == len(self.nets)
        # avg_maeinv = self.sample_mean()
        # self.boundary = avg_maeinv
        for k, v in predictions.items():
            # if v < self.sample_mean():
            if v < 0:
                samples_badness[k] = self.samples[k]  # (val_loss, test_mae)
            else:
                samples_goodies[k] = self.samples[k]  # (val_loss, test_mae)
        assert len(samples_badness) + len(samples_goodies) == len(self.samples)
        return samples_goodies, samples_badness
