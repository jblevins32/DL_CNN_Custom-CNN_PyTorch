"""
2d Convolution Module.  (c) 2021 Georgia Tech

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

import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        # Number of strides for the kernel in each direction to loop over
        N,C,H,W = x.shape
        
        # Pad the data
        if self.padding != 0:
            x_padded = np.zeros((N,C,H+(self.padding*2),W+(self.padding*2)))
            for N_iter in range(N): # Each input in batch
                for C_iter in range(C): # Each channel
                    x_pad = np.concatenate((np.zeros((H, self.padding)), x[N_iter, C_iter], np.zeros((H, self.padding))), axis=1)
                    W_pad = x_pad.shape[1]
                    x_pad = np.concatenate((np.zeros((self.padding,W_pad)), x_pad, np.zeros((self.padding,W_pad))), axis=0)
                    x_padded[N_iter, C_iter] = x_pad
            H = x_padded.shape[2]
            W = x_padded.shape[3]
        else:
            x_padded = x # No padding
            
        # Determine height and width of the output
        H_out = int(((H-self.kernel_size)/self.stride)+1)
        W_out = int(((W-self.kernel_size)/self.stride)+1)
        
        # Initialize the output matrix - 4D
        out = np.zeros((N,self.out_channels,H_out,W_out))

        # loop over all items in batch, channels, rows, and columns
        for N_iter in range(N): # Each input in batch
            for C_iter in range(self.out_channels): # Each channel
                for W_iter in range(W_out): # Slide across width
                    for H_iter in range(H_out): # Slide across height
                        
                        # Form starting and ending portions of stride
                        h_start = H_iter * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = W_iter * self.stride
                        w_end = w_start + self.kernel_size
            
                        # Determine the max in each kernel. Receptive field is the area of the input that the filter is working on
                        receptive_field = x_padded[N_iter, :, h_start:h_end, w_start:w_end]
                        kernel = self.weight[C_iter]
                        out[N_iter,C_iter,H_iter,W_iter] = np.sum(receptive_field*kernel) + self.bias[C_iter]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x_padded,H_out,W_out,N,C,H,W)
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x_padded,H_out,W_out,N,C,H,W = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        self.dw = np.zeros((self.out_channels, self.in_channels,self.kernel_size,self.kernel_size))
        self.dx = np.zeros((N, self.in_channels,H,W))
        self.db = np.zeros(self.out_channels) # one bias per kernel
        
        # Compute gradients with respect to weights, input (dx), and bias (db)
        for N_iter in range(N): # Each input in batch
            for CO_iter in range(self.out_channels):  # For each output channel (filter)
                for CI_iter in range(self.in_channels):  # For each input channel
                    for W_iter in range(W_out):  # Slide across width
                        for H_iter in range(H_out):  # Slide across height

                            # Determine receptive field in the padded input
                            h_start = H_iter * self.stride
                            h_end = h_start + self.kernel_size
                            w_start = W_iter * self.stride
                            w_end = w_start + self.kernel_size

                            # Compute weight gradients
                            receptive_field = x_padded[N_iter, CI_iter, h_start:h_end, w_start:w_end] # patch of the input to look at
                            self.dw[CO_iter, CI_iter] += receptive_field * dout[N_iter, CO_iter, H_iter, W_iter] # 

                            # Compute input gradients
                            self.dx[N_iter, CI_iter, h_start:h_end, w_start:w_end] += dout[N_iter, CO_iter, H_iter, W_iter] * self.weight[CO_iter, CI_iter]
                
                # Compute bias gradients            
                self.db[CO_iter] += np.sum(dout[N_iter,CO_iter,:,:])

        # Remove the padding
        if self.padding != 0:
            self.dx = self.dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
