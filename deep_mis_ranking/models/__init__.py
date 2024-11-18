from __future__ import absolute_import
import torch
import torch.nn as nn
from collections import OrderedDict

from .DenseNet import *
from .MuDeep import *
from .AlignedReID import *
from .PCB import *
from .HACNN import *
from .IDE import *
from .LSRO import *

__factory = {
  # 1. 
  'hacnn': HACNN,
  'densenet121': DenseNet121,
  'ide': IDE,
  # 2.
  'aligned': ResNet50,
  'pcb': PCB,
  'mudeep': MuDeep,
  # 3.
  'cam': IDE,
  'hhl': IDE, 
  'lsro': DenseNet121,
  'spgan': IDE,
}

def get_names():
  return __factory.keys()

def init_model(name, pre_dir, *args, **kwargs):
  if name not in __factory.keys(): 
    raise KeyError("Unknown model: {}".format(name))

  print("Initializing model: {}".format(name))
  net = __factory[name](*args, **kwargs)
  # # load pretrained model
  # checkpoint = torch.load(pre_dir) # for Python 2
  # # checkpoint = torch.load(pre_dir, encoding="latin1") # for Python 3
  # state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
  # change = False
  # for k, v in state_dict.items():
  #   if k[:6] == 'module':
  #     change = True
  #     break
  # if not change: 
  #   new_state_dict = state_dict
  # else:
  #   from collections import OrderedDict
  #   new_state_dict = OrderedDict()
  #   for k, v in state_dict.items():
  #     name = k[7:] # remove 'module.' of dataparallel
  #     new_state_dict[name]=v
  # net.load_state_dict(new_state_dict)
  # # freeze

  # Load pretrained model
  checkpoint = torch.load(pre_dir)
  if 'state_dict' in checkpoint:
      pretrained_dict = checkpoint['state_dict']
  else:
      pretrained_dict = checkpoint
  # Handling 'module.' prefix in state_dict keys if model was trained using DataParallel
  new_state_dict = OrderedDict()
  for k, v in pretrained_dict.items():
      name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
      new_state_dict[name] = v
  # Filter out unnecessary keys
  model_dict = net.state_dict()
  pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
  model_dict.update(pretrained_dict)  # Update current model's dict with pretrained dict
  # Load the updated dictionary
  net.load_state_dict(model_dict)

  net.eval()
  net.volatile = True
  return net