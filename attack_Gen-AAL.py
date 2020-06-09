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
num_workers = 8 if cuda else 0
b_m_ratio=2

batch_size_s_ids = 32
batch_size_vae = 128
learning_rate_s_ids = 0.005
learning_vae = 2e-4

NUM_QUERY_Init = 15
NUM_QUERY = 5
NUM_EPOCH_S_IDS = 20
NUM_EPOCH_VAE = 50
vae_loss_weight = 1.5

NUM_ROUND = 4 # number of model retraining for active learning


gbt_ids_model_dir = './trained_models/gbt_ids_model.pkl'
vae_gen_model_dir = './trained_models/pretrained_vae_KLD_epoch500.pth'

torch_seed_value = 6
torch.manual_seed(torch_seed_value)
print('PyTorch seed value: ', torch_seed_value)

X_tr_pool, Y_tr_pool, X_te, Y_te = create_datasets(train_ratio=0.8, b_m_ratio=b_m_ratio)

# train_dataset_vae = IDS_Dataset(X_tr_pool, Y_tr_pool)
train_dataset_vae = IDS_Dataset(X_tr_pool, Y_tr_pool)
train_loader_vae = DataLoader(dataset=train_dataset_vae, shuffle=True, batch_size=batch_size_vae, num_workers=num_workers)

# iniialize the data indices for query and the data indices remaining unlabeled
idx_queried = np.arange(NUM_QUERY_Init) # indices of queried (labeled) data points in the training data pool
idx_unlabled = np.arange(NUM_QUERY_Init, X_tr_pool.shape[0]) # indices of unlabeled data points in the training data pool
X_tilde_queried = np.zeros((NUM_QUERY, X_tr_pool.shape[1])) # perturbed values of the queried data
Y_tilde = np.zeros(NUM_QUERY) # S-IDS labels of the perturbed values of the queried data
X_tr = np.zeros([1]) # training data
Y_tr = np.zeros([1]) # labels of the training data


def update_datasets(idx_queried, idx_unlabled, X_unlabeled_pool, Y_unlabeled_pool, X_tilde_queried, Y_tilde):
	global train_loader_s_ids, test_loader, train_loader_vae, X_unlabeled, Y_unlabeled, X_tr, Y_tr

	if X_tr.shape[0] == 1:
		X_tr = X_unlabeled_pool[idx_queried]
	else:
		X_tr = np.vstack((X_tr, X_unlabeled_pool[idx_queried], X_tilde_queried))
	if Y_tr.shape[0] == 1:
		Y_tr = Y_unlabeled_pool[idx_queried]
	else:
		Y_tr = np.hstack((Y_tr, Y_unlabeled_pool[idx_queried], Y_tilde))

	X_unlabeled = X_unlabeled_pool[idx_unlabled] # update X_unlabeled with the indices: dis.argsort()[NUM_QUERY:]
	Y_unlabeled = Y_unlabeled_pool[idx_unlabled] # update Y_unlabeled with the indices: dis.argsort()[NUM_QUERY:]

	train_dataset_s_ids = IDS_Dataset(X_tr, Y_tr)
	test_dataset = IDS_Dataset(X_te, Y_te)
	train_loader_s_ids = DataLoader(dataset=train_dataset_s_ids, shuffle=True, batch_size=batch_size_s_ids, num_workers=num_workers)
	train_loader_vae = DataLoader(dataset=train_dataset_s_ids, shuffle=True, batch_size=batch_size_vae, num_workers=num_workers)
	test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size_s_ids, num_workers=num_workers)
	

# update training data
update_datasets(idx_queried, idx_unlabled, X_tr_pool, Y_tr_pool, X_tilde_queried, Y_tilde)

#load pretrained gradient boosted tree ids model
with open(gbt_ids_model_dir, 'rb') as f:
	gbt_ids_model = pickle.load(f)

test_acc_gbt = gbt_ids_model.score(X_te, Y_te)
# print("test_acc_gbt: ", test_acc_gbt)

# load pretrained VAE generator model
vae_model = VAE().to(device)
vae_model.load_state_dict(torch.load(vae_gen_model_dir))

s_ids_model = IDS_model()
s_ids_model.to(device)

s_ids_criterion = nn.BCELoss()

optimizer_s_ids = optim.Adam(s_ids_model.parameters(), lr=learning_rate_s_ids)
optimizer_vae = optim.Adam(vae_model.parameters(), lr=learning_vae)

def vae_loss_function(recon_x, x, mu, logvar):
	coeff1 = 1.5
	coeff2 = 1
	coeff3 = 0.00015

	num_features = x.size(1)
	MSE = F.mse_loss(recon_x, x.view(-1, num_features))
	BCE = F.binary_cross_entropy(recon_x, x.view(-1, num_features))

	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	loss_vae = coeff1 * MSE + coeff2 * BCE + coeff3 * KLD

	return loss_vae, MSE


def train_s_ids(criterion=s_ids_criterion, optimizer=optimizer_s_ids):
	device = torch.device('cuda' if cuda else 'cpu')

	train_loss = 0
	num_data = 0
	num_correct = 0
	acc = 0

	s_ids_model.train()
	for batch_idx, (data, target) in enumerate(train_loader_s_ids):
		data, target = data.to(device), target.to(device)
		pred = s_ids_model(data)

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
	# print('Training Loss: {:.4f}   Training Accuracy: {:.4f}'.format(train_loss, acc))
	# print('num_data: ', num_data)
	
	return train_loss, acc


def test_s_ids(criterion=s_ids_criterion):
	device = torch.device('cuda' if cuda else 'cpu')

	test_loss = 0
	num_data = 0
	num_correct = 0
	acc = 0

	s_ids_model.eval()
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			pred = s_ids_model(data)
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
	# print('Test Loss: {:.4f}   Test Accuracy: {:.4f}'.format(test_loss, acc))
	# print('num_data: ', num_data)

	return test_loss, acc


def train_vae(epoch, coeff_vae, optimizer=optimizer_vae):
	vae_model.train()
	s_ids_model.eval()

	train_loss = 0
	train_mse_loss = 0
	train_ids_loss = 0
	num_data = 0
	num_correct = 0
	acc = 0

	for batch_idx, (data, _) in enumerate(train_loader_vae):
		data = data.to(device)
		optimizer.zero_grad()

		ny = s_ids_model(data) #ny: ids prediction of the original data
		recon_data, mu, logvar = vae_model(data)
		py = s_ids_model(recon_data) #py: ids prediction of the perturbed data

		bs = ny.size(0)
		target = 1 - torch.round(ny.data)

		loss_IDS = s_ids_criterion(py, target)
		loss_vae, mse_loss = vae_loss_function(recon_data, data, mu, logvar)
		loss = loss_IDS + coeff_vae * loss_vae
		# loss = loss_IDS

		loss.backward()
		train_loss += loss.item()
		train_mse_loss += mse_loss.item()
		train_ids_loss += loss_IDS.item()
		optimizer.step()

		# Count number of correct predictions
		py_int = torch.round(py.data.cpu()).long()
		ny_int = torch.round(ny.data.cpu()).long()
		num_correct += py_int.eq(ny_int).sum().item()
		num_data += data.size(0)

	train_loss = train_loss / (batch_idx + 1)
	train_mse_loss = train_mse_loss / (batch_idx + 1)
	train_ids_loss = train_ids_loss / (batch_idx + 1)
	acc = num_correct / num_data * 100.

	if epoch == 1 or (epoch + 1) % 10 == 0 or epoch + 1 == NUM_EPOCH_VAE:
		print('Epoch: {:3}   Training Loss: {:.4f}   Training MSE Loss: {:.4f}   Training IDS loss: {:.4f}   S-IDS Accuracy: {:.2f}%'.format(
			epoch, train_loss, train_mse_loss, train_ids_loss, acc))


	return train_loss, train_mse_loss, train_ids_loss, acc

def print_vae_data_samples():
	vae_model.eval()
	with torch.no_grad():
		for i, (data, _) in enumerate(train_loader_vae):
			data = data.to(device)
			input_batch = data
			recon_batch, _, _ = vae_model(data)

			for j in range(5):
				input_data = input_batch[j]
				recon_data = recon_batch[j]

				input_array = input_data.cpu().detach().numpy()
				recon_array = recon_data.cpu().detach().numpy()

				print("####")
				# print("input: {}\noutput:{}".format(input_data, recon_data))
				np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
				print("input:\n", input_array)
				print("output:\n", recon_array)

			break


def test_vae():
	# test success rate of adversarial attack using the malicious test data and the gbt IDS model
	vae_model.cpu()
	vae_model.eval()

	mal_idx_te = np.argwhere(Y_te == 0)
	X_te_vae = np.take(X_te, mal_idx_te[:,0], axis=0)
	Y_te_vae = np.zeros_like(mal_idx_te[:,0])
	x = torch.from_numpy(X_te_vae).float()

	with torch.no_grad():
		recon_x, _, _ = vae_model(x)
	X_tilde_vae = recon_x.data.numpy()
	test_acc_gbt_vae = gbt_ids_model.score(X_tilde_vae, Y_te_vae)
	test_attack_success_rate = (1 - test_acc_gbt_vae) * 100
	# print("\nIDS model test accuracy on VAE-generated data: {:.2f}%".format(test_acc_gbt_vae * 100))
	print("\nAdversarial attack success rate on test data: {:.2f}%".format(test_attack_success_rate))

	dis = F.mse_loss(recon_x, x.view(-1, X_te_vae.shape[1]))
	print("MSE between the original and the perturbed data: {:.4f}".format(dis))

	eta = recon_x - x
	eta = eta.data.numpy()
	
	return eta, test_attack_success_rate, X_te_vae, X_tilde_vae
	

def cal_dis(x, vae_model, s_ids_model):
	# For x that can be successfully perturbed for adversarial attack, save the distance as "x_dis = mse(x and x_tilde)",
	# otherwise, save the distance as "x_dis = np.inf".
	x=x.view(1, -1)
	with torch.no_grad():
		# print("x.size(): ", x.size())
		ny = s_ids_model(x) #ny: ids prediction of the original data
		recon_x, _, _ = vae_model(x)
		py = s_ids_model(recon_x) #py: ids prediction of the perturbed data
		py_int = torch.round(py.data).long()
		ny_int = torch.round(ny.data).long()

	if py_int == ny_int:
		x_dis = np.inf
		x_tilde = x[0].data.numpy()
	else:
		x_array = x[0].data.numpy()
		recon_x_array = recon_x[0].data.numpy()
		x_dis = np.sum(x_array * recon_x_array)
		x_tilde = recon_x_array

	return x_dis, x_tilde

# Active learning with VAE
list_attack_success_rate = []
list_avg_dis = []
for round_i in range(NUM_ROUND):
	print('=========')
	print('round: ', round_i)
	print("training data size: {}/{}".format(X_tr.shape[0], X_tr_pool.shape[0]))
	print("training data used: {:.2f}%".format(X_tr.shape[0] / X_tr_pool.shape[0] * 100))
	
	# Train IDS model
	print('\nTraining S-IDS model.')
	for i in range(NUM_EPOCH_S_IDS):
		train_loss, train_acc = train_s_ids()
		test_loss, test_acc = test_s_ids()
		if (i+1)%4 == 0 or i+1 == NUM_EPOCH_S_IDS:
			print('epoch: ', i)
			print('Training Loss: {:.4f}   Training Accuracy: {:.4f}'.format(train_loss, train_acc))
			print('Test Loss: {:.4f}   Test Accuracy: {:.4f}'.format(test_loss, test_acc))
	
	# Train VAE model
	print('\nTraining VAE model.')
	for i in range(NUM_EPOCH_VAE):
		train_vae(epoch=i, coeff_vae=vae_loss_weight)

	Eta_te, attack_success_rate_te, X_te_vae, X_tilde_vae = test_vae()
	# if round_i == 1:
	# 	print('\nSave X_te_vae and X_tilde_vae.')
	# 	save_data_path = './data_save/X_and_X_tilde_samples.pkl'
	# 	X_and_X_tilde = [X_te_vae, X_tilde_vae]
	# 	with open(save_data_path, 'wb') as f:
	# 		pickle.dump(X_and_X_tilde, f)
	# 	print('\nSave Eta_te.')
	# 	save_data_path = './data_save/eta_r2.pkl'
	# 	with open(save_data_path, 'wb') as f:
	# 		pickle.dump(Eta_te, f)

	list_attack_success_rate.append(attack_success_rate_te)

	Eta_te_array = np.zeros(Eta_te.shape[0])
	for k in range(Eta_te.shape[0]):
		Eta_te_array[k] = np.sqrt(np.dot(Eta_te[k,:], Eta_te[k,:]))
	Eta_te_avg = np.mean(Eta_te_array)
	list_avg_dis.append(Eta_te_avg)

	# Query
	print('\nQuery IDS model about labels')
	s_ids_model.cpu()
	vae_model.cpu()
	s_ids_model.eval()
	vae_model.eval()

	dis = np.zeros(X_unlabeled.shape[0])
	X_tilde_unlabeled = np.zeros(X_unlabeled.shape)

	for i in range(X_unlabeled.shape[0]):
		x = torch.from_numpy(X_unlabeled[i, :]).float()
		dis[i], X_tilde_unlabeled[i,:] = cal_dis(x=x, vae_model=vae_model, s_ids_model=s_ids_model)

	s_ids_model.to(device)
	vae_model.to(device)

	idx_queried = dis.argsort()[:NUM_QUERY]
	idx_unlabled = dis.argsort()[NUM_QUERY:]

	X_tilde_queried = X_tilde_unlabeled[idx_queried]
	Y_tilde = gbt_ids_model.predict(X_tilde_queried)

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