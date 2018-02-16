import torch.nn.functional as F


def margin_loss(class_probs, labels):
    correct = F.relu(0.9 - class_probs) ** 2
    wrong = F.relu(class_probs - 0.1) ** 2
    loss = (labels * correct) + 0.5 * ((1. - labels) * wrong)
    loss = loss.sum(dim=-1)
    return loss.mean()


def reconstruction_loss(recons, images):
    loss = (recons - images.view(-1, 784)) ** 2
    loss = loss.sum(dim=-1)
    return loss.mean()