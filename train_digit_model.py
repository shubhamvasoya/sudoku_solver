import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from models.digit_recognizer import DigitCNN


def train_model():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Device configuration with detailed GPU info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("CUDA not available. Using CPU.")
        print("To use GPU, install PyTorch with CUDA support:")

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset with num_workers for faster data loading
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    # Use more workers for GPU training and pin memory for faster transfer
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # Model, loss, and optimizer
    model = DigitCNN().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Optional: Use learning rate scheduler for better training
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print("Starting training...")

    # Training loop
    model.train()
    for epoch in range(10):  # Increased epochs for better training
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to GPU
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/10, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')

        # Step the scheduler
        scheduler.step()

        # Print epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1} completed - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    # Test the model
    print("Testing model...")
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_dataset)

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Save the model (move to CPU before saving to ensure compatibility)
    model.cpu()
    torch.save(model.state_dict(), 'data/mnist_model.pth')
    print("Model saved to data/mnist_model.pth")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")


if __name__ == "__main__":
    train_model()