# hybrid_qcnn_mnist_4x4.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import PauliSumOp, StateFn, ExpectationFactory, AerPauliExpectation
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector

# ----------------------
# 1) Utilities: convert 4x4 image -> 4 features (quadrant means)
# ----------------------
def img4x4_to_quadrant_features(img4x4):
    """
    img4x4: numpy array shape (4,4) or torch tensor (1,4,4) etc.
    returns: 1d numpy array length 4 with values scaled to [0, pi]
    """
    arr = np.array(img4x4).reshape(4, 4)
    # quadrants: top-left, top-right, bottom-left, bottom-right (each 2x2)
    q0 = arr[0:2, 0:2].mean()
    q1 = arr[0:2, 2:4].mean()
    q2 = arr[2:4, 0:2].mean()
    q3 = arr[2:4, 2:4].mean()
    feats = np.array([q0, q1, q2, q3], dtype=np.float64)
    # assume pixel values in [0,1]; map to rotation angles in [0, pi]
    feats = feats.clip(0.0, 1.0) * np.pi
    return feats

# ----------------------
# 2) Build a parametrized QCNN-like circuit
#    - 4 qubits
#    - input_params: 4 angles (encoding)
#    - weight_params: parameters for convolution/pooling layers
# ----------------------
def build_qcnn_circuit(num_qubits=4, conv_layers=2):
    # input parameters (angle encodings)
    input_params = ParameterVector('x', length=num_qubits)

    # trainable parameters (we'll use a small set)
    # for each conv layer we have parameterized rotations on each qubit and two-qubit entanglers
    w_len = conv_layers * (num_qubits * 1 + (num_qubits - 1) * 2)  # rough
    weight_params = ParameterVector('w', length=w_len)

    qc = QuantumCircuit(num_qubits)

    # Encoding
    for i in range(num_qubits):
        qc.ry(input_params[i], i)

    # Parameter indexing helper
    idx = 0
    for layer in range(conv_layers):
        # local single-qubit rotations (like convolution bias)
        for q in range(num_qubits):
            qc.ry(weight_params[idx], q); idx += 1

        # local entangling blocks (two-qubit unitaries between neighbors)
        # use CNOT + rotations + CNOT to make a controlled-Ry-like entangler
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
            qc.rz(weight_params[idx], q + 1); idx += 1
            qc.ry(weight_params[idx], q + 1); idx += 1
            qc.cx(q, q + 1)

        # small "pooling-like" rotation (single-qubit)
        for q in range(0, num_qubits, 2):  # on qubits 0 and 2 (reduce locality)
            qc.rx(0.2, q)  # small fixed rotation (non-learned) to mix info
    # final local rotations (ensure idx consumed safely; if any leftover params, attach them)
    while idx < len(weight_params):
        qc.ry(weight_params[idx], idx % num_qubits)
        idx += 1

    return qc, input_params, weight_params

# ----------------------
# 3) Wrap circuit into a CircuitQNN (outputs expectation of Z on qubit 0)
# ----------------------
def make_qnn(qc, input_params, weight_params):
    # Observable: measure Z on qubit 0 (you can measure a sum of Paulis to get multi-output)
    pauli_z0 = PauliSumOp.from_list([('Z' + 'I' * (qc.num_qubits - 1), 1.0)])  # Z on qubit 0

    # Choose backend + quantum instance
    backend = AerSimulator(method='statevector')
    qi = QuantumInstance(backend, shots=None)

    # CircuitQNN gives expectation values; set input_gradients to True for parameter-shift capability
    qnn = CircuitQNN(
        circuit=qc,
        input_params=list(input_params),
        weight_params=list(weight_params),
        forward_output=None,
        sampling=False,
        observable=pauli_z0,
        quantum_instance=qi
    )
    return qnn

# ----------------------
# 4) PyTorch model that uses TorchConnector + small FC head
# ----------------------
class HybridQCNN(nn.Module):
    def __init__(self, qnn, n_qubit_outputs=1):
        super().__init__()
        # Torch layer that wraps the Qiskit QNN
        self.qtnn = TorchConnector(qnn)  # this appears like an nn.Module
        # a small classical head that maps qnn output to 10 classes
        # qnn returns a single expectation scalar in our design; use small linear layers
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        # x: batch x 1 x 4 x 4 (torch tensor)
        batch_size = x.size(0)
        # convert to quadrant features
        feats = []
        x_np = x.detach().cpu().numpy()
        for i in range(batch_size):
            f = img4x4_to_quadrant_features(x_np[i, 0])
            feats.append(f)
        feats = np.stack(feats, axis=0).astype(np.float64)  # shape (B,4)

        # The TorchConnector QNN expects torch tensors; it will map input params first
        feats_t = torch.from_numpy(feats)  # dtype torch.float64 may be required by qiskit connector
        if feats_t.dtype != torch.float64:
            feats_t = feats_t.double()

        # QNN forward: returns tensor shape (batch, 1) (the expectation)
        q_out = self.qtnn(feats_t)

        # Ensure dtype is float32 for torch linear layers (convert)
        q_out = q_out.float()
        logits = self.fc(q_out)
        return logits

# ----------------------
# 5) Small training / demonstration code (toy)
# ----------------------
def demo_train():
    # toy dataset: use random data to demo. Replace this with scaled MNIST 4x4 tensors
    # Assume you have `train_images` as torch tensor shape (N,1,4,4) normalized [0,1],
    # and `train_labels` as integers 0..9
    # For demo, create a tiny synthetic dataset:
    N = 200
    X = np.random.rand(N, 1, 4, 4).astype(np.float32)
    y = np.random.randint(0, 10, size=(N,)).astype(np.int64)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    # Build circuit and qnn
    qc, input_params, weight_params = build_qcnn_circuit(num_qubits=4, conv_layers=2)
    qnn = make_qnn(qc, input_params, weight_params)

    # Build hybrid model
    model = HybridQCNN(qnn)
    model.train()
    model = model  # already on CPU; if you want GPU for classical parts you can move later

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for data, target in loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * data.size(0)
            _, preds = outputs.max(1)
            correct += (preds == target).sum().item()
            total += data.size(0)

        print(f"Epoch {epoch+1}: Loss={epoch_loss/total:.4f}, Acc={100.0*correct/total:.2f}%")

    print("Demo training finished. Replace synthetic data with your 4x4 MNIST tensors to train properly.")

if __name__ == "__main__":
    demo_train()