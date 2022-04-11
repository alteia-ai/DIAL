import numpy as np
import torch

def fgsm_attack(data, epsilon=1/255):
    data_grad = data.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed data by adjusting each pixel of the input data
    perturbed_data = data + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    # Return the perturbed data
    return perturbed_data

def entropy(data, n_classes):
    return (- torch.sum(data * torch.log(data), dim=0) / np.log(n_classes)).cpu().detach()