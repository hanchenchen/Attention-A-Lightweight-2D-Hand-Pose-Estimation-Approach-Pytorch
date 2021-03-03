import torch.nn as nn


def sum_mse_loss(pred, target):
    """
    :param pred:    Tensor  B,num_stage(3),num_channel(21),46,46
    :param target:
    :return:
    """
    criterion = nn.MSELoss(reduction='sum')
    loss = criterion(pred, target)
    return loss




