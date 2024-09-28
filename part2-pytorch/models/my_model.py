"""
MyModel model.  (c) 2021 Georgia Tech

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
    print("Roger that from my_model.py!")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        
        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            
            # Conv layer, 2D becuase input is image matrix
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1,padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
                        
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(1024),
            
            nn.AdaptiveAvgPool2d((1,1)),
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(1024,out_features=10)
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
        out = out.view(out.size(0), -1)
        outs = self.fully_connected(out)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
