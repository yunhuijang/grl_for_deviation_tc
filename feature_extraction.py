import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class AE(torch.nn.Module):
    def __init__(self, data_loader):
        super().__init__()
        for data in data_loader:
            input_size = data[0].shape[0]
            break
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 16 ==> 4
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
        )
          
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 4 ==> 16
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, input_size),
            torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoder(self, x):
        encoded = self.encoder(x)
        return encoded


def feature_extraction_ae(case_dict):
    '''
    input: case_dict (key: caseid / value: event log transformation feature)
    output: case_dict_ae (key: caseid / value: autoencoder feature value)
    '''
    
    # TODO: make autoencoder input size flexible
    
    loader = DataLoader(Tensor(np.array([*case_dict.values()])), batch_size=32)
    # Model Initialization
    model = AE(loader)

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-4)
    
    epochs = 20
    outputs = []
    losses = []
    last_epoch_output = []
    for epoch in range(epochs):
        print(f'{epoch} epoch')
        for data in loader:
            # Output of Autoencoder
            reconstructed = model(data)

            # Calculating the loss function
            loss = loss_function(reconstructed, data)

            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss)
            outputs.append((epoch, data, reconstructed))
            if epoch == epochs-1:
                last_epoch_output.extend(model.get_encoder(data))
    
    plt.plot([loss.detach().numpy() for loss in losses])
    
    case_dict_ae = dict.fromkeys(case_dict.keys())
    for caseid, batch_output in zip(case_dict.keys(), last_epoch_output):
        case_dict_ae[caseid] = batch_output.detach().numpy()

    return case_dict_ae