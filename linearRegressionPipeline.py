import numpy as np
import pandas as pd
torch.manual_seed(42)

# Step 1: Data Generation with Numpy
np.random.seed(42)
X = np.random.rand(1000, 5)  # 1000 samples, 5 features
weights_true = np.array([2.5, -1.8, 3.3, 0.8, -2.2])
y = X @ weights_true + np.random.randn(1000) * 0.5  # Linear relation with noise

# Step 2: Load data into Pandas DataFrame
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
data['target'] = y

# Step 3: Data Preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['target'], test_size=0.2, random_state=42)

# Step 4: Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Step 5: Define a PyTorch Linear Regression Model
import torch.nn as nn
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(X_train.shape[1])

# Step 6: Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 7: Training Loop
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')

# Step 8: Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    mse = criterion(predictions, y_test_tensor)
print(f'Test MSE: {mse.item()}')

# Step 9: Pandas Data Analysis
weights_learned = model.linear.weight.detach().numpy().flatten()
pd.DataFrame({
    'Feature': [f'feature_{i}' for i in range(5)],
    'True Weight': weights_true,
    'Learned Weight': weights_learned
})
