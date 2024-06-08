#preprocess.py
#!/usr/bin/env/python3

"""
this file preprocess the face images what get from kaggle dataset
and provides preprocessed images shape = (224,224)
"""

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

dataset_path = "data/train"
batch_size = 5
num_training_samples = 10

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
])

def read_preprocess(batch_size=batch_size, num_training_samples=num_training_samples):
    dataset_folder = datasets.ImageFolder(root=dataset_path, transform=transform)
    subset_dataset, _ = random_split(dataset_folder, [num_training_samples, len(dataset_folder) - num_training_samples])
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def read(dataset_path):
    dataset_folder = datasets.ImageFolder(root=dataset_path, transform=transform)
    data_loader = DataLoader(dataset_folder, batch_size = 1)
    return data_loader

