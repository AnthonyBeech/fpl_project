import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PlayerScorePredictor(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, lstm_layers=2, p_dropout=0.5):
        super(PlayerScorePredictor, self).__init__()

        # LSTM layers for capturing temporal dynamics
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=lstm_layers, batch_first=True, dropout=p_dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Output layer
        self.fc3 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        # LSTM layers expect input of shape (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        x = lstm_out  # We use the output of the last time step

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)
        return x.squeeze()

def load_data(file_path, squeeze=False):
    data = pd.read_csv(file_path)
    if squeeze:
        data = data.squeeze()
    return data

def prepare_data_loader(X, y, batch_size=64, shuffle=True):
    tensor_X = torch.tensor(X.values).float()  # Convert DataFrame to NumPy array
    tensor_y = torch.tensor(y.values).float() if isinstance(y, pd.DataFrame) else torch.tensor(y).float()
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, criterion, optimizer, train_loader, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
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

# Configuration
data_loc = "data/"
num_epochs = 100
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
model = PlayerScorePredictor(input_size=X_train.shape[1])
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
