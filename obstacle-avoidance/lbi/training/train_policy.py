import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import IPython as ipy
from lbi.training.policy_networks import MLPSoftmax
import argparse 


def main(raw_args=None):


    ###################################################################
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=1, help="print more (default: 1")

    args = parser.parse_args(raw_args)
    verbose = args.verbose

    # Load data
    data = np.load('lbi/training/training_data.npz')
    lidar_all_training = data['lidar_all_training']
    labels_all_training = data['labels_all_training']

    num_rays = np.shape(lidar_all_training)[1]
    num_primitives = np.shape(labels_all_training)[1]


    # Device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # Create dataset object
    xx_tensor = torch.Tensor(lidar_all_training)    # transform to torch tensor
    yy_tensor = torch.Tensor(labels_all_training)

    dataset = TensorDataset(xx_tensor, yy_tensor)    # dataset
    params = {'batch_size': 1000, # 1000
              'shuffle': True}
              # 'num_workers': 12}
    dataloader = DataLoader(dataset, **params)    # dataloader

    # Initialize MLP
    num_in = num_rays
    num_out = num_primitives
    model = MLPSoftmax(num_in, num_out)
    # model.to(device)

    # Define the loss function
    loss_function = nn.BCELoss()
    # loss_function = nn.MSELoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # , weight_decay=1e-5)

    # Run the training loop
    num_epochs = 5000 # 5000
    for epoch in range(0, num_epochs):

        current_loss = 0.0
        num_batches = 0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(dataloader, 0):

            # Get inputs
            inputs, targets = data
            # inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Update current loss
            current_loss += loss.item()
            num_batches += 1

        # Print 
        if verbose and (epoch % 1000 == 0):
            print("epoch: ", epoch, "; loss: ", current_loss/num_batches)

    # Process is complete.
    if verbose:
        print('Training complete.')

    # Save model
    torch.save(model.state_dict(), "lbi/training/models/trained_model")
    if verbose:
        print('Saved trained model.')


#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main() 

