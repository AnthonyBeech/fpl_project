import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs




def load_data(file_path, squeeze=False):
    data = pd.read_csv(file_path)
    if squeeze:
        data = data.squeeze()
    return data

def prepare_data_loader(X, y, batch_size=64, shuffle=True):
    tensor_X = torch.tensor(X.values).float().to(device)
    tensor_y = torch.tensor(y.values).float().to(device)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, criterion, optimizer, train_loader, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 4 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            actual.extend(targets.tolist())
            predicted.extend(outputs.tolist())
    return actual, predicted

def plot_results(x, y, xlabel, ylabel, title, plot_file):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x, x, color='red', linestyle='--')
    plt.savefig(plot_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
data_loc = "data/"
num_epochs = 10
batch_size = 64

# Data loading
X_train = load_data(data_loc + "X_train.csv")
y_train = load_data(data_loc + "y_train.csv", squeeze=True)
X_test = load_data(data_loc + "X_test.csv")
y_test = load_data(data_loc + "y_test.csv", squeeze=True)

# Prepare DataLoaders
train_loader = prepare_data_loader(X_train, y_train, batch_size)
test_loader = prepare_data_loader(X_test, y_test, batch_size, shuffle=False)

# Model, Loss, Optimizer
model = PlayerScorePredictor(input_size=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization

# Train
train_model(model, criterion, optimizer, train_loader, num_epochs)

# Evaluate
actual, predicted = evaluate_model(model, test_loader)
print("Mean Absolute Error:", mean_absolute_error(actual, predicted))
print("Root Mean Squared Error:", mean_squared_error(actual, predicted, squared=False))
print("R² Score:", r2_score(actual, predicted))

# Plotting
plot_results(actual, predicted, 'Actual Scores', 'Predicted Scores', 
             'Actual vs Predicted Player Scores', 'plot/actual_vs_predicted.png')

# Save Model
torch.save(model.state_dict(), 'data/player_score_predictor_model.pkl')
