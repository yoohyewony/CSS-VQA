import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

with open('ans_cpv2_UpDn.pkl', 'rb') as f:
    data = pickle.load(f)

#print(data['236390000'].size())

new = {}
for key, val in data.items():
    new[key] = F.softmax(torch.unsqueeze(val,0), dim=1)

#print(new['236390000'].size())
#print(new['236390000'])

with open('ans_cpv2_UpDn1.pkl', 'wb') as f:
    pickle.dump(new, f)
