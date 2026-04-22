import torch
import torch.nn as nn
import torch.optim as optim
from data_pipeline import build_dataset
from model import Enhancer

# Load data
X, Y = build_dataset(2000)

# Model
model = Enhancer()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    total_loss = 0
    for i in range(len(X)):
        inp = X[i].unsqueeze(0)
        target = Y[i].unsqueeze(0)

        optimizer.zero_grad()
        output = model(inp)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(X)}")

# Save model
torch.save(model.state_dict(), "enhancer.pth")