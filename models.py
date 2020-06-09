import torch
import torch.nn as nn
from torch.nn import functional as F

class IDS_model(nn.Module):
	def __init__(self):
		super(IDS_model, self).__init__()
		self.sequence = nn.Sequential(
			nn.Linear(78,128),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(128,64),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(64,32),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(32,1),
			nn.Sigmoid()
			)

	def forward(self, x):
		output = self.sequence(x)

		return output

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        H1 = 40
        H2 = 40
        H3 = 32
        H4 = 16
        H5 = 8

        # Four Hidden Layers
        self.fc1 = nn.Linear(78, H1)
        self.fc1_1 = nn.Linear(H1, H2)
        self.fc1_2 = nn.Linear(H2, H3)
        self.fc1_3 = nn.Linear(H3, H4)
        self.fc21 = nn.Linear(H4, H5)
        self.fc22 = nn.Linear(H4, H5)
        self.fc3 = nn.Linear(H5, H4)
        self.fc3_1 = nn.Linear(H4, H3)
        self.fc3_2 = nn.Linear(H3, H2)
        self.fc3_3 = nn.Linear(H2, H1)
        self.fc4 = nn.Linear(H1, 78)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc1_1(h1))
        h1 = F.relu(self.fc1_2(h1))
        h1 = F.relu(self.fc1_3(h1))
        
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc3_1(h3))
        h3 = F.relu(self.fc3_2(h3))
        h3 = F.relu(self.fc3_3(h3))

        return torch.sigmoid(self.fc4(h3))
        # return self.fc4(h3)
        # return torch.tanh(self.fc4(h3))
        # return F.relu(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 78))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar