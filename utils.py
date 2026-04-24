import torch
import matplotlib.pyplot as plt
from model import PrunableLinear


def compute_sparsity_loss(model):
    loss = 0.0

    for m in model.modules():
        if hasattr(m, "gate_scores"):
            gates = m.get_gates()
            loss += (gates * (1 - gates)).mean()   # 🔥 KEY CHANGE

    return loss


def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = m.get_gates()
            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total


def count_active_params(model, threshold=1e-2):
    total = 0
    active = 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = m.get_gates()
            total += gates.numel()
            active += (gates > threshold).sum().item()

    return active, total


def plot_gate_distribution(model):
    all_gates = []

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = m.get_gates().detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()