import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import logging

# Disable logging to suppress "Files already downloaded" message
logging.getLogger("torchvision").setLevel(logging.ERROR)

# Model Definition: CNN + LSTM
class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        
        # CNN part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # LSTM part
        self.lstm_input_size = 256 * 4 * 4  # Input size to LSTM after CNN layers
        self.lstm_hidden_size = 128  # LSTM hidden layer size
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_hidden_size, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.lstm_hidden_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN part
        x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool2(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool3(nn.ReLU()(self.bn3(self.conv3(x))))
        
        # Flattening the output of CNN layers
        x = x.view(x.size(0), -1)  # Flatten
        x = x.unsqueeze(1)  # Add an extra dimension to simulate sequence input for LSTM
        
        # LSTM part
        lstm_out, _ = self.lstm(x)  # _ is hidden state, not needed
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
        
        # Fully connected layers
        x = nn.ReLU()(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Data Loading and Augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Prepare CIFAR10 dataset
trainset = CIFAR10(root='./cifar10', train=True, download=False, transform=transform_train)
testset = CIFAR10(root='./cifar10', train=False, download=False, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model, loss, optimizer, and scheduler
model = CNN_LSTM_Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training and Testing functions
train_losses = []  # Track training losses
test_losses = []  # Track testing losses

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_acc = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_acc = correct / total
        # if (batch_idx + 1) % 100 == 0:
        #     print(f'[INFO] Epoch-{epoch + 1}-Batch-{batch_idx + 1}: Train Loss-{loss.item():.4f}, Accuracy-{train_acc:.4f}')
    
    avg_train_loss = train_loss / len(trainloader)
    train_losses.append(avg_train_loss)  # Store average training loss
    return train_acc

def test(epoch):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_test_loss = test_loss / len(testloader)
    test_losses.append(avg_test_loss)  # Store average test loss
    test_acc = correct / total
    print(f'[INFO] Epoch-{epoch + 1}-Train Accuracy: {train_acc:.4f} -Test Accuracy: {test_acc:.4f}')
    return test_acc

def predict_images():
    model.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = outputs.max(1)

    # 显示5张图像和其预测类别
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax = axes[i]
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f'Pred: {classes[predicted[i]]} ({classes[labels[i]]})')
        ax.axis('off')
    plt.savefig('output/XD CNN-CIFAR10-Prediction-Result.jpg')
    plt.show()

# Training loop
# Main function with if __name__ == '__main__'
if __name__ == '__main__':
    best_acc = 0
    total_train_acc = []
    total_test_acc = []
    epoch = 50
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    for epoch in range(epoch):
        train_acc = train(epoch)
        test_acc = test(epoch)
        total_train_acc.append(train_acc)
        total_test_acc.append(test_acc)
        scheduler.step()

        if test_acc > best_acc:
            print('Saving model...')
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot training and test losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')

    plt.legend()

    # Plot training and test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(total_train_acc, label='Train Accuracy')
    plt.plot(total_test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.savefig(f'.//output//XD CNN-CIFAR10-Loss-Accurancy.jpg')

    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f'Best Test Accuracy: {best_acc * 100:.2f}%')

    # Visualization of Predictions
    # Get one batch of test images and display predictions
    model.eval()
    inputs, targets = next(iter(testloader))  # Get one batch of data
    inputs, targets = inputs.to(device), targets.to(device)

    # Make predictions
    outputs = model(inputs)
    _, predicted = outputs.max(1)

    # Display the first few images with their predicted labels
    predict_images()


