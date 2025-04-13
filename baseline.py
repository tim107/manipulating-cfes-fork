from train_models import *
from datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.tanh3 = nn.Tanh()
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, return_logit=False):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
        out = self.tanh3(out)
        out1 = self.fc4(out)
        out = self.sigmoid(out1)

        if return_logit:
            return out1
        else:
            return out

#baseline model cc
## Setup data  ###########################
data, labels, protected, data_t, labels_t, protected_t, cat_features = get_presplit_data("cc")
data, data_v, labels, labels_v, protected, protected_v = train_test_split(data, labels, protected, test_size=0.11)
ss = StandardScaler()
data = ss.fit_transform(data)
data_t = ss.transform(data_t)
data_v = ss.transform(data_v)

data = torch.from_numpy(data).float().to(device)
labels = torch.from_numpy(labels).float().to(device)
protected = torch.from_numpy(protected).float().to(device)
data_t = torch.from_numpy(data_t).float().to(device)
labels_t = torch.from_numpy(labels_t).float().to(device)
protected_t = torch.from_numpy(protected_t).float().to(device)
data_v = torch.from_numpy(data_v).float().to(device)
labels_v = torch.from_numpy(labels_v).float().to(device)
protected_v = torch.from_numpy(protected_v).float().to(device)
data.requires_grad = True
protected.requires_grad = True

model = NeuralNet(data.shape[1], HIDDEN, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in in range(50):
    model.train()
    loss_func = nn.BCELoss()
    optimizer.zero_grad()
    output = model(data)
    loss = loss_func(output, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{50}, Loss: {loss.item():.4f}")
