"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################

        # Number of strides for the kernel in each direction to loop over
        N = x.shape[0]
        C = x.shape[1]       
        H = x.shape[2]
        W = x.shape[3]
        
        # Determine height and width of the output
        H_out = int(((H-self.kernel_size)/self.stride)+1)
        W_out = int(((W-self.kernel_size)/self.stride)+1)
        
        # Initialize the output matrix - 4D
        out = np.zeros((N,C,H_out,W_out))
        idx_out = np.zeros((N,C,H_out,W_out))

        # loop over all items in batch, channels, rows, and columns to take max
        for N_iter in range(N): # Each input in batch
            for C_iter in range(C): # Each channel
                for W_iter in range(W_out): # Slide across width
                    for H_iter in range(H_out): # Slide across height
                        
                        # Determine the max in each kernel
                        h_start = H_iter * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = W_iter * self.stride
                        w_end = w_start + self.kernel_size
                        
                        kernel_out = np.max(x[N_iter,C_iter,h_start:h_end, w_start:w_end])
                        out[N_iter,C_iter,H_iter,W_iter] = kernel_out
                        
                        # Determine the indices of that max within each kernel
                        kernel_idx = np.argmax(x[N_iter,C_iter,h_start:h_end, w_start:w_end])
                        idx_out[N_iter,C_iter,H_iter,W_iter] = kernel_idx

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out, idx_out,N,C)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out, idx_out,N,C = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################

        # Initialize weight changes
        self.dx = np.zeros((x.shape))
        
        # loop over all items in batch, channels, rows, and columns to take max
        for N_iter in range(N): # Each input in batch
            for C_iter in range(C): # Each channel
                for W_iter in range(W_out): # Slide across width
                    for H_iter in range(H_out): # Slide across height
                        
                        # Determine the max in each kernel
                        h_start = H_iter * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = W_iter * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Setup the gradient to make with a gradient of kernel size with the appropriate max index chosen
                        change = np.zeros((self.kernel_size,self.kernel_size))
                        idx = int(idx_out[N_iter,C_iter,H_iter,W_iter])
                        r,c = np.unravel_index(idx, (self.kernel_size, self.kernel_size))
                        
                        # Assign the gradient to the total dx matrix
                        change[r,c] = dout[N_iter, C_iter, H_iter, W_iter]
                        self.dx[N_iter,C_iter,h_start:h_end, w_start:w_end] = change
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
