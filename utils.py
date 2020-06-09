import torch
import numpy as np
import math
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import Dataset, DataLoader

def create_datasets_v0(input_dir='./NBx.npy', label_dir='./NBy.npy', train_ratio=0.8, b_m_ratio=10, train_data_usage=1.0):
	"""
	create training dataset and test dataset
	"""
	# Load the dataset and labels
	X_total = np.load(input_dir)
	Y_total = np.load(label_dir)

	mal_idx = np.argwhere(Y_total == 0)
	benign_idx_total = np.argwhere(Y_total == 1)

	# Get indices of benign samples
	benign_size = mal_idx.shape[0] * b_m_ratio
	np.random.seed(1)
	np.random.shuffle(benign_idx_total)
	benign_idx = benign_idx_total[0:benign_size]

	# Set splitting point between training and test data
	SP_B = math.ceil(benign_idx.shape[0] * train_ratio)
	SP_M = math.ceil(mal_idx.shape[0] * train_ratio)

	# Get indices of training and test data
	benign_idx_tr = benign_idx[0:SP_B]
	benign_idx_te = benign_idx[SP_B:]
	mal_idx_tr = mal_idx[0:SP_M]
	mal_idx_te = mal_idx[SP_M:]

	combined_idx_tr = np.concatenate((benign_idx_tr, mal_idx_tr), axis=0)
	np.random.shuffle(combined_idx_tr)
	combined_idx_te = np.concatenate((benign_idx_te, mal_idx_te), axis=0)
	np.random.shuffle(combined_idx_te)

	# Create training and test datasets
	X_tr = np.take(X_total, combined_idx_tr[:,0], axis=0)
	Y_tr = np.take(Y_total, combined_idx_tr[:,0], axis=0)
	X_te = np.take(X_total, combined_idx_te[:,0], axis=0)
	Y_te = np.take(Y_total, combined_idx_te[:,0], axis=0)

	train_size = math.ceil(X_tr.shape[0] * train_data_usage)
	X_tr = X_tr[:train_size]
	Y_tr = Y_tr[:train_size]

	return X_tr, Y_tr, X_te, Y_te

def create_datasets(input_dir='./NBx.npy', label_dir='./NBy.npy', train_ratio=0.8, b_m_ratio=10, train_data_usage=1.0):
	"""
	create training dataset and test dataset
	"""
	# Load the dataset and labels
	X_total = np.load(input_dir)
	# Y_total = np.load(label_dir)

	load_gbt_model_path = './trained_models/gbt_ids_model.pkl'
	with open(load_gbt_model_path, 'rb') as f:
		gbt_ids_model = pickle.load(f)
	Y_total = gbt_ids_model.predict(X_total)

	mal_idx = np.argwhere(Y_total == 0)
	benign_idx_total = np.argwhere(Y_total == 1)

	# Get indices of benign samples
	benign_size = mal_idx.shape[0] * b_m_ratio
	np_seed_value = 0
	np.random.seed(np_seed_value)
	print('Numpy seed value: ', np_seed_value)
	np.random.shuffle(benign_idx_total)
	benign_idx = benign_idx_total[0:benign_size]

	# Set splitting point between training and test data
	SP_B = math.ceil(benign_idx.shape[0] * train_ratio)
	SP_M = math.ceil(mal_idx.shape[0] * train_ratio)

	# Get indices of training and test data
	benign_idx_tr = benign_idx[0:SP_B]
	benign_idx_te = benign_idx[SP_B:]
	mal_idx_tr = mal_idx[0:SP_M]
	mal_idx_te = mal_idx[SP_M:]

	combined_idx_tr = np.concatenate((benign_idx_tr, mal_idx_tr), axis=0)
	np.random.shuffle(combined_idx_tr)
	combined_idx_te = np.concatenate((benign_idx_te, mal_idx_te), axis=0)
	np.random.shuffle(combined_idx_te)

	# Create training and test datasets
	X_tr = np.take(X_total, combined_idx_tr[:,0], axis=0)
	Y_tr = np.take(Y_total, combined_idx_tr[:,0], axis=0)
	X_te = np.take(X_total, combined_idx_te[:,0], axis=0)
	Y_te = np.take(Y_total, combined_idx_te[:,0], axis=0)

	train_size = math.ceil(X_tr.shape[0] * train_data_usage)
	X_tr = X_tr[:train_size]
	Y_tr = Y_tr[:train_size]

	return X_tr, Y_tr, X_te, Y_te


class IDS_Dataset(Dataset):
	def __init__(self, x, y):
		self.len = x.shape[0]
		self.x_data = torch.from_numpy(x).float()
		self.y_data = torch.from_numpy(y[:,np.newaxis]).float()

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len