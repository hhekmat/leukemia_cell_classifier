import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# processing images into flattened grayscale representations of pixel intensity
def flatten_images():
    images = []
    cancerous_images = []
    healthy_images = []
    labels = []

    images_folder = os.path.join(os.path.dirname(__file__), "images/")
    cancerous_images_folder = os.path.join(images_folder, 'all')
    healthy_images_folder = os.path.join(images_folder, 'hem')
    
    for filename in os.listdir(cancerous_images_folder):
        if len(images) < 2500:
            if filename.endswith('.bmp'):
                img = cv2.imread(os.path.join(cancerous_images_folder, filename), cv2.IMREAD_GRAYSCALE)
                img = img.flatten()
                images.append(img)
                cancerous_images.append(img)
                labels.append(1)
        else:
            break

    for filename in os.listdir(healthy_images_folder):
        if len(images) < 5000:
            if filename.endswith('.bmp'):
                img = cv2.imread(os.path.join(healthy_images_folder, filename), cv2.IMREAD_GRAYSCALE)
                img = img.flatten()
                images.append(img)
                healthy_images.append(img)
                labels.append(0)
        else:
            break

    all_X = np.array(images)
    cancerous_X = np.array(cancerous_images)
    healthy_X = np.array(healthy_images)
    all_y = np.array(labels)

    return all_X, all_y, cancerous_X, healthy_X

# use sklearn package to compute top n principle components
# visualize effect on variance
# reconstruct components to visualize individually
def doPCA(X, n, visComp):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8)
    plt.title('Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='orange')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Ratio')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, visComp, figsize=(12, 4))

    for i in range(visComp):
        pc_image = pca.components_[i].reshape(450, 450)  # Assuming img_shape is the original image shape
        axes[i].imshow(pc_image, cmap='gray')
        axes[i].set_title(f'PC {i + 1}')

    plt.show()

# split dataset into train and test
def splitDataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# use sklearn package to train and test random forest classifier
def doRandomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# split test set into validation and test (now have 3 subsets)
# convert data into dataloaders from tensors (for compatibility w PyTorch)
def prepDataNN(X_train, X_test, y_train, y_test, batch_size):
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).float()

    X_val_tensor = torch.tensor(X_val).float()
    y_val_tensor = torch.tensor(y_val).float()

    X_test_tensor = torch.tensor(X_test).float()
    y_test_tensor = torch.tensor(y_test).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

# initializer for binary classifier class from PyTorch
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=1):
        super(BinaryClassifier, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    # forward pass - pass data through layers
    # note: backward pass built into PyTorch
    def forward(self, x):
        return self.layers(x)
    
# train custom neural network using training and validation sets
def trainNN(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Training
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels.unsqueeze(1))
                total_val_loss += val_loss.item()

        print(f'Epoch {epoch+1}, Validation Loss: {total_val_loss / len(val_loader)}')

# evaluate trained neural network on test set
def evalNN(model, test_loader, criterion):
    model.eval()
    total_test_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss = criterion(outputs, labels.unsqueeze(1))
            total_test_loss += test_loss.item()

        predictions = (outputs > 0.5).float()
        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.reshape(-1).tolist())

    average_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {average_test_loss}')

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print('Classification Report:\n', report)

# call on all above functions properly
def main():
    X, y, cancerous_X, healthy_X = flatten_images()
    #doPCA(cancerous_X, 50)
    doPCA(healthy_X, 50, 15)

    X_train, X_test, y_train, y_test = splitDataset(X, y)
    doRandomForest(X_train, X_test, y_train, y_test)
    
    lr = 0.001
    inputSize = X_train.shape[1]
    hiddenLayerConfig = [64, 128, 256, 256]

    train_loader, val_loader, test_loader = prepDataNN(X_train, X_test, y_train, y_test, batch_size=32)
    customNN = BinaryClassifier(inputSize, hiddenLayerConfig)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(customNN.parameters(), lr)
    trainNN(customNN, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    evalNN(customNN, test_loader, criterion)

if __name__ == '__main__':
    main()