#!pyvenv_3.10.12/bin/python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import os

# Define dataset class for loading point cloud data
class PointCloudDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        data = pd.read_csv(csv_file)
        self.is_train = is_train
        self.points = data[['x', 'y', 'z']].values
        if is_train:
            self.instance_ids = data['instance_id'].values

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = self.points[idx]
        if self.is_train:
            return torch.tensor(point, dtype=torch.float32), torch.tensor(self.instance_ids[idx], dtype=torch.long)
        return torch.tensor(point, dtype=torch.float32)

# Define the neural network that predicts hop vectors (x, y, z)
class HopNet(nn.Module):
    def __init__(self):
        super(HopNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)  # 3 outputs for hop vector (x, y, z)

    def forward(self, point):
        x = torch.relu(self.fc1(point))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        hop_vector = self.fc4(x)
        return hop_vector  # 3D vector for hopping

# Training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for points, instance_ids in dataloader:
        points = points.to(device)
        instance_ids = instance_ids.to(device)

        optimizer.zero_grad()

        # Predict hop vectors for all points
        hop_vectors = model(points)

        # Find nearest points based on hop vectors
        points_np = points.cpu().numpy()
        hop_vectors_np = hop_vectors.cpu().detach().numpy()
        new_points = points_np + hop_vectors_np

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points_np)
        distances, indices = nbrs.kneighbors(new_points)

        # Aggregate point pairs
        predicted_instance_ids = np.arange(len(points_np))  # Initially, each point is its own instance
        for i, j in enumerate(indices[:, 0]):
            if distances[i, 0] < 0.1:  # Threshold for assigning to the same instance
                predicted_instance_ids[i] = predicted_instance_ids[j]

        # Compute loss by comparing predicted instances with ground truth
        predicted_instance_ids = torch.tensor(predicted_instance_ids, dtype=torch.long, device=device)
        loss = criterion(predicted_instance_ids, instance_ids)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Test function
def test_model(model, dataloader, device, save_csv=False, output_file=None):
    model.eval()
    all_points = []
    all_hop_vectors = []
    
    with torch.no_grad():
        for points in dataloader:
            points = points.to(device)
            hop_vectors = model(points)
            
            all_points.append(points.cpu().numpy())
            all_hop_vectors.append(hop_vectors.cpu().numpy())
    
    all_points = np.vstack(all_points)
    all_hop_vectors = np.vstack(all_hop_vectors)
    
    # Generate new points by adding hop vectors
    new_points = all_points + all_hop_vectors
    
    # Nearest neighbors search to assign instance IDs
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(all_points)
    distances, indices = nbrs.kneighbors(new_points)

    predicted_instance_ids = np.arange(len(all_points))
    for i, j in enumerate(indices[:, 0]):
        if distances[i, 0] < 0.1:  # Threshold for instance assignment
            predicted_instance_ids[i] = predicted_instance_ids[j]
    
    # Save the output to CSV if required
    if save_csv and output_file:
        output_data = np.hstack((all_points, predicted_instance_ids.reshape(-1, 1)))
        output_df = pd.DataFrame(output_data, columns=['x', 'y', 'z', 'predicted_instance_id'])
        output_df.to_csv(output_file, index=False)
    
    return predicted_instance_ids

# Main training loop
def main(train_csv, test_csv, output_csv, device):
    # Hyperparameters
    batch_size = 64
    lr = 0.001
    num_epochs = 50
    
    # Dataset and DataLoader
    train_dataset = PointCloudDataset(train_csv, is_train=True)
    test_dataset = PointCloudDataset(test_csv, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and loss
    model = HopNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')
    
    # Testing and saving output
    test_model(model, test_loader, device, save_csv=True, output_file=output_csv)

# Entry point
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_csv = 'train_point_cloud.csv'  # Replace with actual path
    test_csv = 'test_point_cloud.csv'    # Replace with actual path
    output_csv = 'segmented_output.csv'  # Replace with desired output file path
    main(train_csv, test_csv, output_csv, device)
