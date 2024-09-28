"""
Vanilla CNN model.  (c) 2021 Georgia Tech

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


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from cnn.py!")


class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and no padding                           #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################

        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            
            # Conv layer, 2D becuase input is image matrix
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=7,stride=1,padding=0), # create a 32 feature map with a 7x7 kernel
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(32*13*13,out_features=10)
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        out = self.cnn(x)
        out = self.flatten(out)
        outs = self.fully_connected(out)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
