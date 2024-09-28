"""
Focal Loss Wrapper.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from focal_loss.py!")


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """

    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    
    # First, calculate the effective number, the relative contribution of each sample to the training
    E_n = [(1-beta**cls_num)/(1-beta) for cls_num in cls_num_list]
    
    # Get weights from each effective number by inverse rule
    per_cls_weights = [1.0 / E_i for E_i in E_n]
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight #torch.tensor(weight, dtype=torch.float32)
        
    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################

        # Get softmax probabilities
        probs = F.softmax(input,dim=1)
        target_probs = probs.gather(1, target.view(-1, 1)).squeeze(1)
        focal_loss = -((1-target_probs)**self.gamma)*torch.log(target_probs)
        focal_loss = focal_loss * self.weight[target.long()]
        # self.weight = self.weight.tolist()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return focal_loss.mean()
