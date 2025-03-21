import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split


def get_california_housing_data(validation_split=0.1, test_split=0.1):
    # Load the California housing dataset
    dataset = fetch_california_housing()
    data, target = dataset.data, dataset.target
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(data, target, test_size=(validation_split + test_split), random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split / (validation_split + test_split), random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_normalized_data(train_images, val_images, test_images):
    mean = np.mean(train_images, axis=0)
    std = np.std(train_images, axis=0) + 1e-8  # Avoid division by zero
    
    train_images = (train_images - mean) / std
    val_images = (val_images - mean) / std
    test_images = (test_images - mean) / std
    
    return train_images, val_images, test_images


def get_MNIST_data(validation_split=0.1):
    # Define transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the MNIST training dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training data into training and validation sets
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Convert datasets to NumPy arrays
    def dataset_to_numpy(dataset):
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        images, labels = next(iter(data_loader))
        return images.view(images.shape[0], -1).numpy(), labels.numpy()  # Flatten images to 784D
    
    train_images, train_labels = dataset_to_numpy(train_dataset)
    val_images, val_labels = dataset_to_numpy(val_dataset)
    test_images, test_labels = dataset_to_numpy(test_dataset)
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels
