import torch
import torch.nn.functional as F
import torch.nn as nn
from numpy import dtype
import numpy as np
import json


if __name__=="__main__":
    file_path = './data/task1.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    if isinstance(data, list):
        data = data[:150]
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    else:
        print("None")
    print("end")