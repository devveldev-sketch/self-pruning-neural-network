# Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns to prune its own weights during training using learnable gating mechanisms and sparsity regularization.

---

## 🚀 Overview

In real-world deployment, neural networks must be efficient in terms of memory and computation.

Instead of pruning after training, this model learns which weights are unnecessary **during training itself**.

---

## 🧠 Method

Each weight is associated with a learnable gate:

Pruned Weight = Weight × Sigmoid(Gate Score)

The model is trained using:

Total Loss = Classification Loss + λ × Sparsity Loss

- Classification loss ensures prediction accuracy  
- Sparsity loss encourages the network to deactivate unnecessary connections  

---

## ⚙️ Architecture

- Custom `PrunableLinear` layer  
- Fully connected neural network (MLP)  
- Applied on CIFAR-10 dataset  

---

## 📊 Results

- The network successfully learns to prune redundant connections  
- Gate values show a **bimodal distribution** (near 0 and 1)  
- Demonstrates trade-off between:
  - Model accuracy  
  - Network sparsity  

---

## 📈 Observations

- Increasing λ increases sparsity  
- Higher sparsity can reduce accuracy  
- The model effectively identifies important and unimportant weights  

---

## 🛠 Tech Stack

- Python  
- PyTorch  
- Torchvision  
- Matplotlib  

---

## ▶️ How to Run

python3 train.py

---

## 📁 Project Structure

self_pruning_nn/
│── model.py
│── train.py
│── utils.py
│── config.py
│── report.md
│── requirements.txt


---

## 👩‍💻 Author

**Devadharshini S**
Integrated M.Tech CSE (Business Analytics), VIT Chennai

* Skills: Python, R, SQL, Tableau
* Interests: Deep Learning, Predictive Analytics, AI Applications

GitHub: [https://github.com/devveldev-sketch](https://github.com/devveldev-sketch)

---

## 💡 Future Improvements

* Use Hard Concrete gates for true L0 regularization
* Extend to CNN-based architectures
* Apply structured pruning (channel-level pruning)
