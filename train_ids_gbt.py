import numpy as np
import torch
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import time
import math
import pickle
from sklearn.ensemble import GradientBoostingClassifier

from utils import *
from models import *

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

# Hyperparameters of the code
b_m_ratio=2

gbt_ids_model_dir = './trained_models/gbt_ids_model.pkl'

X_tr_pool, Y_tr_pool, X_te, Y_te = create_datasets(train_ratio=0.8, b_m_ratio=b_m_ratio)

# train gradient boosted tree ids model
gbt_ids_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=4, random_state=0).fit(X_tr_pool, Y_tr_pool)
test_acc_gbt = gbt_ids_model.score(X_te, Y_te)
print("Test accuracy of gradient boosted tree IDS: {:.2f}%".format(test_acc_gbt))
# with open(gbt_ids_model_dir, 'wb') as f:
# 	pickle.dump(gbt_ids_model, f)