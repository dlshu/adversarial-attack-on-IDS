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

batch_size_vae = 128
learning_vae = 1e-4
NUM_EPOCH_VAE = 10

save_model_path = './models/'

torch.manual_seed(0)

X_tr_pool, Y_tr_pool, X_te, Y_te = create_datasets(b_m_ratio=b_m_ratio)

train_dataset_vae = IDS_Dataset(X_tr_pool, Y_tr_pool)
train_loader_vae = DataLoader(dataset=train_dataset_vae, shuffle=True, batch_size=batch_size_vae, num_workers=num_workers)
test_dataset = IDS_Dataset(X_te, Y_te)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size_vae, num_workers=num_workers)


#load pretrained VAE generator model
vae_model = VAE().to(device)
# vae_model.load_state_dict(torch.load(vae_gen_model_dir))

optimizer_vae = optim.Adam(vae_model.parameters(), lr=learning_vae)

def vae_loss_function(recon_x, x, mu, logvar):
	coeff1 = 1.5
	coeff2 = 1
	coeff3 = 0.00015

	num_features = x.size(1)
	MSE = F.mse_loss(recon_x, x.view(-1, num_features))
	BCE = F.binary_cross_entropy(recon_x, x.view(-1, num_features))
	# l1_loss = nn.L1Loss()
	# L1 = l1_loss(recon_x, x.view(-1, num_features))

	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	loss_vae = coeff1 * MSE + coeff2 * BCE + coeff3 * KLD

	return loss_vae, MSE


def train_vae(epoch, train_loader=train_loader_vae, optimizer=optimizer_vae):
	vae_model.train()
	
	train_loss = 0
	train_mse_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):
		data = data.to(device)
		optimizer.zero_grad()
		recon_data, mu, logvar = vae_model(data)
		loss_vae, mse_loss = vae_loss_function(recon_data, data, mu, logvar)
		
		loss_vae.backward()
		optimizer.step()

		train_loss += loss_vae.item()
		train_mse_loss += mse_loss.item()

	train_loss = train_loss / (batch_idx + 1)
	train_mse_loss = train_mse_loss / (batch_idx + 1)

	# if epoch == 1 or (epoch + 1) % 10 == 0 or epoch + 1 == NUM_EPOCH_VAE:
	# 	print('Epoch: {:3}   Training Loss: {:.4f}   Training MSE Loss: {:.4f}'.format(epoch, train_loss, train_mse_loss))

	return train_loss, train_mse_loss

def print_vae_data_samples(data_loader=test_loader):
	vae_model.eval()
	with torch.no_grad():
		for i, (data, _) in enumerate(data_loader):
			data = data.to(device)
			input_batch = data
			recon_batch, _, _ = vae_model(data)

			for j in range(5):
				input_data = input_batch[j]
				recon_data = recon_batch[j]

				input_array = input_data.cpu().detach().numpy()
				recon_array = recon_data.cpu().detach().numpy()

				print("\nSample #{}:".format(j))
				# print("input: {}\noutput:{}".format(input_data, recon_data))
				np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
				print("input:\n", input_array)
				print("output:\n", recon_array)

			break


def test_vae(epoch, test_loader=test_loader):
	# test success rate of adversarial attack using the test data and the gbt IDS model
	vae_model.eval()
	test_loss = 0
	test_mse_loss = 0
	with torch.no_grad():
		for i, data in enumerate(test_loader):
			data = data[0]
			data = data.to(device)
			recon_data, mu, logvar = vae_model(data)
			loss_vae, mse_loss = vae_loss_function(recon_data, data, mu, logvar)
			test_loss += loss_vae.item()
			test_mse_loss += mse_loss.item()

	# if epoch == 1 or (epoch + 1) % 10 == 0 or epoch + 1 == NUM_EPOCH_VAE:
	# 	print('Epoch: {:3}   Testg Loss: {:.4f}   Test MSE Loss: {:.4f}'.format(epoch, test_loss, test_mse_loss))

	return test_loss, test_mse_loss


print('\nTraining VAE model.')
for i in range(NUM_EPOCH_VAE):
	train_loss, train_mse_loss = train_vae(i)
	test_loss, test_mse_loss = test_vae(i)
	print('Epoch: {:3}   Training Loss: {:.4f}   Training MSE Loss: {:.4f}   Testg Loss: {:.4f}   Test MSE Loss: {:.4f}'.format(
		i, train_loss, train_mse_loss, test_loss, test_mse_loss))
	
# 	# Save models
# 	if i >0 and (i+1) % 100 == 0:
# 		torch.save(vae_model.state_dict(), save_model_path + 'pretrained_vae_KLD_epoch{}.pth'.format(i))
# 		print("model saved at epoch {}.\n".format(i))
# # Save models
# torch.save(vae_model.state_dict(), save_model_path + 'pretrained_vae_KLD_epoch{}.pth'.format(i))
# print("model saved at epoch {}.\n".format(i))

print('\nSamples of input / output feature points')
print_vae_data_samples()