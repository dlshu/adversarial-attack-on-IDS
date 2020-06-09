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

num_workers = 8 if cuda else 0 
b_m_ratio=2

batch_size = 32
learning_rate = 0.005
num_epoch = 20

NUM_QUERY_Init = 15
NUM_QUERY = 5
NUM_ROUND = 4

load_gbt_model_path = './trained_models/gbt_ids_model.pkl'

torch_seed_value = 1
torch.manual_seed(torch_seed_value)
print('PyTorch see value: ', torch_seed_value)

X_tr_pool, Y_tr_pool, X_te, Y_te = create_datasets(b_m_ratio=b_m_ratio)

# iniialize the data indices for query and the data indices remaining unlabeled
idx_queried = np.arange(NUM_QUERY_Init) # indices of queried (labeled) data points in the training data pool
idx_unlabled = np.arange(NUM_QUERY_Init, X_tr_pool.shape[0]) # indices of unlabeled data points in the training data pool
X_tilde_queried = np.zeros((NUM_QUERY, X_tr_pool.shape[1])) # perturbed values of the queried data
Y_tilde = np.zeros(NUM_QUERY) # S-IDS labels of the perturbed values of the queried data
X_tr = np.zeros([1]) # training data
Y_tr = np.zeros([1]) # labels of training data



def update_datasets(idx_queried, idx_unlabled, X_unlabeled_pool, Y_unlabeled_pool, X_tilde_queried, Y_tilde):
	global train_loader, test_loader, X_unlabeled, Y_unlabeled, X_tr, Y_tr

	if X_tr.shape[0] == 1:
		X_tr = X_unlabeled_pool[idx_queried]
	else:
		X_tr = np.vstack((X_tr, X_unlabeled_pool[idx_queried], X_tilde_queried))
		# X_tr = np.vstack((X_tr, X_tilde_queried))
	if Y_tr.shape[0] == 1:
		Y_tr = Y_unlabeled_pool[idx_queried]
	else:
		Y_tr = np.hstack((Y_tr, Y_unlabeled_pool[idx_queried], Y_tilde))
		# Y_tr = np.hstack((Y_tr, Y_tilde))

	X_unlabeled = X_unlabeled_pool[idx_unlabled] # update X_unlabeled with the indices: dis.argsort()[NUM_QUERY:]
	Y_unlabeled = Y_unlabeled_pool[idx_unlabled] # update Y_unlabeled with the indices: dis.argsort()[NUM_QUERY:]

	train_dataset = IDS_Dataset(X_tr, Y_tr)
	test_dataset = IDS_Dataset(X_te, Y_te)
	train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
	test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)


# update training data
update_datasets(idx_queried, idx_unlabled, X_tr_pool, Y_tr_pool, X_tilde_queried, Y_tilde)

#load pretrained gradient boosted tree ids model
with open(load_gbt_model_path, 'rb') as f:
	gbt_ids_model = pickle.load(f)

test_acc_gbt = gbt_ids_model.score(X_te, Y_te)

ids_model = IDS_model()
ids_model.to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(ids_model.parameters(), lr=learning_rate)

def train_nn_ids(data_loader=train_loader, criterion=criterion, optimizer=optimizer):
	device = torch.device('cuda' if cuda else 'cpu')

	train_loss = 0
	num_data = 0
	num_correct = 0
	acc = 0

	for batch_idx, (data, target) in enumerate(data_loader):
		data, target = data.to(device), target.to(device)
		pred = ids_model(data)

		loss = criterion(pred, target)
		train_loss += loss.item()
		train_loss_avg = train_loss / (batch_idx + 1)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Count number of correct predictions
		pred_int = torch.round(pred.data).long()
		ground_truth = target.long()
		num_correct += pred_int.eq(ground_truth).sum().item()
		num_data += data.size(0)

	train_loss = train_loss / (batch_idx + 1)
	acc = num_correct / num_data * 100.
	
	return train_loss, acc

def test_nn_ids(data_loader=test_loader, criterion=criterion):
	device = torch.device('cuda' if cuda else 'cpu')

	test_loss = 0
	num_data = 0
	num_correct = 0
	acc = 0

	ids_model.eval()
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(data_loader):
			data, target = data.to(device), target.to(device)
			pred = ids_model(data)
			pred_gbt = gbt_ids_model.predict(data.data.cpu().numpy())

			loss = criterion(pred, target)
			test_loss += loss.item()
			
			# Count number of correct predictions
			pred_int = torch.round(pred.data).long()
			ground_truth = target.long()
			# pred_gbt_int = np.argmax(pred_gbt, axis=1)
			pred_gbt_tensor = torch.from_numpy(pred_gbt).long().to(device)
			pred_gbt_tensor = torch.unsqueeze(pred_gbt_tensor, 1)

			# num_correct += pred_int.eq(ground_truth).sum().item()
			num_correct += pred_int.eq(pred_gbt_tensor).sum().item()
			num_data += data.size(0)

	test_loss = test_loss / (batch_idx + 1)
	acc = num_correct / num_data * 100.

	return test_loss, acc


def cal_dis(x, model):
	nx = torch.unsqueeze(x, 0)
	nx.requires_grad_()
	eta = torch.zeros(nx.shape)
	out = model(nx+eta)

	py = torch.round(out.data).long()
	ny = torch.round(out.data).long()

	max_iter = 100 # maximal iteration of DeepFool

	i_iter = 0
	while py == ny and i_iter < max_iter:
		out.backward(retain_graph=True)
		grad_i = nx.grad.data.clone()

		fi = out[0] - 0.5
		grad_i_np = grad_i.numpy().flatten()
		# ri = -fi.data / np.dot(grad_i_np, grad_i_np) * grad_i
		ri = -fi.data / np.sqrt(np.dot(grad_i_np, grad_i_np)) * grad_i

		eta += ri.clone()
		nx.grad.data.zero_()
		out = model(nx+eta)
		py = torch.round(out.data).long()
		
		i_iter += 1

	x_dis = (eta*eta).sum()
	x_tilde = nx+eta
	x_tilde = x_tilde[0].data

	return  x_dis, x_tilde


list_dis_n_rounds = []
list_attack_success_rate = []
list_avg_dis = []
for round_i in range(NUM_ROUND):
	print('=========')
	print("round: ", round_i)
	print("training data size: {}/{}".format(X_tr.shape[0], X_tr_pool.shape[0]))
	print("training data used: {:.2f}%".format(X_tr.shape[0] / X_tr_pool.shape[0] * 100))

	# Train IDS model
	print('\nTraining S-IDS model.')
	for i in range(num_epoch):
		train_loss, train_acc = train_nn_ids()
		test_loss, test_acc = test_nn_ids()
	print('Training Loss: {:.4f}   Training Accuracy: {:.4f}'.format(train_loss, train_acc))
	print('Test Loss: {:.4f}   Test Accuracy: {:.4f}'.format(test_loss, test_acc))

	# Query
	print('\nQuery IDS model about labels')
	ids_model.cpu()
	ids_model.eval()
	dis = np.zeros(X_unlabeled.shape[0])
	X_tilde_unlabeled = torch.zeros(X_unlabeled.shape)

	for i in range(X_unlabeled.shape[0]):
		x = torch.from_numpy(X_unlabeled[i, :]).float()
		dis[i], X_tilde_unlabeled[i,:] = cal_dis(x=x, model=ids_model)
		if i % 500 == 0 or i == X_unlabeled.shape[0] - 1:
			print("i = {:2d}, x_dis = {:.4f}".format(i, dis[i].item()))

	idx_queried = dis.argsort()[:NUM_QUERY]
	idx_unlabled = dis.argsort()[NUM_QUERY:]

	X_tilde_queried = X_tilde_unlabeled[idx_queried].data.cpu().numpy()
	Y_tilde = gbt_ids_model.predict(X_tilde_queried)

	# test the success rate of adversarial attack on malicious data
	print('\nTest DeepFool on malicious test data.')
	mal_idx_te = np.argwhere(Y_te == 0)
	X_te_df = np.take(X_te, mal_idx_te[:,0], axis=0) # X_te_df: malicious data points in X_te
	dis_te = np.zeros(X_te_df.shape[0]) # perturbation values of datapoints in X_te_df
	X_tilde_te = torch.zeros(X_te_df.shape) # perturbed data points given X_te_df

	# compute perturbation values and the perturbed data points
	for i in range(X_te_df.shape[0]):
		x = torch.from_numpy(X_te_df[i, :]).float()
		dis_te[i], X_tilde_te[i,:] = cal_dis(x=x, model=ids_model)
		if i % 200 == 0 or i == X_te_df.shape[0] - 1:
			print("i = {:2d}, x_dis = {:.4f}".format(i, dis_te[i].item()))

	dis_nan_test = np.isnan(dis_te) # check nan values in dis_te
	num_nan = dis_nan_test.astype(int).sum() # count number of nan values in dis_te
	valid_rate_df = (1 - num_nan / dis_te.shape[0]) # calculate the portion of non-nan values in dis_te

	valid_idx = [] # indices of non-nan values in dis_te
	for j in range(dis_te.shape[0]):
		flag_inf = False
		X_j = X_tilde_te[j,:]
		for k in range(X_j.shape[0]):
			if np.isinf(X_j[k]):
				flag_inf = True
		if np.isnan(dis_te[j]) or np.isinf(dis_te[j]) or flag_inf:
			continue
		else:
			valid_idx.append(j)

	print('Valid adversarial examples: {} / {}'.format(len(valid_idx), mal_idx_te.shape[0]))
	dis_valid = np.take(dis_te, valid_idx) # non-nan valued perturbations
	X_tilde_te_valid = np.take(X_tilde_te, valid_idx, axis=0) # non-nan valued perturbed datapoints
	Y_te_valid = np.zeros(X_tilde_te_valid.shape[0]) # gbt labels of X_tilde_te_valid
	
	test_acc_gbt_df = gbt_ids_model.score(X_tilde_te_valid, Y_te_valid) * valid_rate_df # compute the test accuracy of DFAL using gbt model
	attack_success_rate_te = (1 - test_acc_gbt_df) * 100
	print("\nAdversarial attack success rate on test data: {:.2f}%".format(attack_success_rate_te))
	list_attack_success_rate.append(attack_success_rate_te)

	dis_valid_avg = np.mean(np.sqrt(dis_valid)) # compute the average perturbation values on test data
	print("Average perturbation on test data: {:.4f}".format(dis_valid_avg))
	list_avg_dis.append(dis_valid_avg)
			
	ids_model.to(device)

	# update datasets with queried data
	update_datasets(idx_queried, idx_unlabled, X_unlabeled, Y_unlabeled, X_tilde_queried, Y_tilde)
	print('\nDatasets updated\n')

print('\nPyTorch seed value: ', torch_seed_value)
print('\nAttack success rate:')
for i in range(len(list_attack_success_rate)):
	print('{:.2f}  '.format(list_attack_success_rate[i]), end = ' ')
print('\nPerturbation magnitude:')
for i in range(len(list_avg_dis)):
	print('{:.3f}  '.format(list_avg_dis[i]), end = ' ')
print('\n')