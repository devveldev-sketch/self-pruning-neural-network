import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import PrunableMLP
from utils import (
    compute_sparsity_loss,
    compute_sparsity,
    count_active_params,
    plot_gate_distribution,
)
import config


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset (FIXED normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Model
    model = PrunableMLP().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining Started...\n")

    # Training Loop
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        total_cls = 0
        total_sparse = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            classification_loss = criterion(outputs, labels)
            sparsity_loss = compute_sparsity_loss(model)

            loss = classification_loss + config.LAMBDA * sparsity_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += classification_loss.item()
            total_sparse += sparsity_loss.item()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Total Loss: {total_loss:.2f} | "
            f"Cls Loss: {total_cls:.2f} | "
            f"Sparse Loss: {total_sparse:.2f}"
        )

    # Evaluation
    acc = evaluate(model, testloader, device)
    sparsity = compute_sparsity(model)
    active, total = count_active_params(model)

    print("\n===== FINAL RESULTS =====")
    print(f"Lambda: {config.LAMBDA}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Active Params: {active}/{total}")

    plot_gate_distribution(model)


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


if __name__ == "__main__":
    train()