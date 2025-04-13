import torch
from torch import nn
from datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

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


def baseline_model(dataset):
    data, labels, protected, data_t, labels_t, protected_t, cat_features = get_presplit_data(dataset)
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
    protected.requires_grad = True

    model = NeuralNet(data.shape[1], 200, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_ds = TensorDataset(data, labels.unsqueeze(1))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ds = TensorDataset(data_v, labels_v.unsqueeze(1))
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_ds = TensorDataset(data_t, labels_t.unsqueeze(1))
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    loss_fn = nn.BCELoss()
    lowest_val_loss = float('inf')
    val_patience = 5

    for epoch in range(50):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_dl:
            optimizer.zero_grad()
            batch_loss = loss_fn(model(x_batch), y_batch)
            batch_loss.backward()
            optimizer.step()
            total_train_loss += batch_loss.item()
        avg_train_loss = total_train_loss / len(train_dl)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_dl:
                val_loss += loss_fn(model(x_val), y_val).item()
        avg_val_loss = val_loss / len(val_dl)

        print(f"Epoch {epoch+1}/50, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < lowest_val_loss:
            lowest_val_loss = avg_val_loss
            val_patience = 5
            torch.save(model.state_dict(), f"models/baseline_model_{dataset}.pt")
        else:
            val_patience -= 1

        print(f"Validation patience left: {val_patience}")
        if val_patience == 0:
            print("Training aborted after no validation improvement")
            break
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_test, y_test in test_dl:
            preds = model(x_test)
            predicted = (preds >= 0.5).float()
            correct += (predicted == y_test).sum().item()
            total += y_test.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")


# baseline_model("cc")

# baseline_model("german")

baseline_model("dc3")