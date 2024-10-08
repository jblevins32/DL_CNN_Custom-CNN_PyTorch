# Run this code to train a model according to the selected parameters

import yaml
from solver import Solver
import torch
from cs7643.cifar10 import CIFAR10
import sys

sys.path.append('.')

cifar10_ds = CIFAR10('./data/cifar10', download=True, train=True)

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device = " + device)
if device == 'cpu':
    print("WARNING: Using CPU will cause slower train times")

# Change this line to choose model to train!!!
config_file = "config_mymodel" # @param ["config_mymodel", "config_twolayer", "config_vanilla_cnn","config_resnet32"]

config_file = "./configs/" + config_file + ".yaml"

print("Training a model using configuration file " + config_file)

with open(config_file, "r") as read_file:
  config = yaml.safe_load(read_file)

kwargs = {}
for key in config:
  for k, v in config[key].items():
    if k != 'description':
      kwargs[k] = v

kwargs['device'] = device
kwargs['prefix_path'] = '.'

print(kwargs)

solver = Solver(**kwargs)
solver.train()