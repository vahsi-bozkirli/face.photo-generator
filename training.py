#training.py
#!/usr/bin/env/python3

"""
model traning parameters = model, data_loader, num_epochs
"""

import torch.nn as nn
import torch.optim as optim

def train_model(model, data_loader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for data in data_loader:
            images, _ = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    
    return model
