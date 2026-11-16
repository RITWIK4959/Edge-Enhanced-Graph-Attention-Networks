# Edge-Enhanced Graph Attention Networks for Node Classification

A reproducible research project comparing **GAT, GATv2, GATv2 with Static Edge Features, and GATv2 with Learned Edge Features** for semi-supervised node classification on the **Cora, CiteSeer, and PubMed** citation-network datasets.

The project investigates whether incorporating edge information into graph attention mechanisms provides consistent improvements over standard Graph Attention Networks.

---

## 1. Research Objective

The main research question is:

> **Does incorporating edge information into GATv2 improve node classification compared with standard GAT and GATv2?**

Four models are evaluated under the same experimental protocol:

1. **GAT**
2. **GATv2**
3. **GATv2 + Static Edge**
4. **GATv2 + Learned Edge**

The experiments use identical dataset splits, model depth, hidden dimensionality, optimization settings, checkpoint selection criteria, and evaluation procedures.

---

## 2. Models

### 2.1 GAT

The standard Graph Attention Network uses attention coefficients to determine the importance of neighboring nodes.

\[
\alpha_{ij} =
\text{softmax}_j
\left(
\text{LeakyReLU}
\left(
\mathbf{a}^{T}
[\mathbf{W}h_i || \mathbf{W}h_j]
\right)
\right)
\]

Neighbor representations are then aggregated using the learned attention weights.

---

### 2.2 GATv2

GATv2 uses dynamic attention in which the ordering of the query and key representations is handled more flexibly than in the original GAT formulation.

The model is implemented using PyTorch Geometric's GATv2 convolution.

---

### 2.3 GATv2 + Static Edge

This model extends GATv2 by incorporating deterministic edge representations.

For an edge connecting nodes \(i\) and \(j\), the static representation is constructed from the node representations:

\[
e_{ij} =
\frac{h_i+h_j}{2}
\]

The resulting representation is projected/truncated to the configured edge dimension.

No additional trainable edge encoder is used.

This provides a controlled test of whether simple edge information can improve GATv2.

---

### 2.4 GATv2 + Learned Edge

The learned-edge model uses a small trainable edge encoder.

For an edge \((i,j)\), the encoder receives:

\[
[h_i || h_j || |h_i-h_j|]
\]

and produces a learned edge representation:

\[
e_{ij}=f_{\theta}(h_i,h_j)
\]

The edge encoder is intentionally small because the benchmark datasets contain very limited labeled training nodes.

---

## 3. Experimental Architecture

All models use the same general architecture wherever applicable.

### Node representation

```text
Input Node Features
        |
Linear Projection
        |
Layer Normalization
        |
ELU
        |
Feature Dropout
        |
Graph Attention Layer 1
        |
Layer Normalization
        |
ELU
        |
Residual Connection
        |
Graph Attention Layer 2
        |
Layer Normalization
        |
ELU
        |
Residual Connection
        |
Dropout
        |
Linear Classifier
        |
Class Prediction
```

### Configuration

| Parameter | Value |
|---|---:|
| Hidden dimension | 64 |
| Number of GNN layers | 2 |
| Attention heads | 4 |
| Edge dimension | 16 |
| Learned edge hidden dimension | 16 |
| Feature dropout | 0.45 |
| Attention dropout | 0.35 |
| Edge dropout | 0.10 |
| Classifier dropout | 0.45 |
| Learning rate | 0.002 |
| Weight decay | 0.005 |
| Label smoothing | 0.02 |
| Maximum epochs | 400 |
| Early stopping patience | 50 |
| Gradient clipping | 1.0 |

---

# 4. Datasets

The project evaluates the models on three standard citation-network datasets.

| Dataset | Nodes | Edges | Features | Classes | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 | 500 | 1,000 |
| CiteSeer | 3,327 | 9,104 | 3,703 | 6 | 120 | 500 | 1,000 |
| PubMed | 19,717 | 88,648 | 500 | 3 | 60 | 500 | 1,000 |

The same predefined train, validation, and test splits are used for all four models.

---

# 5. Reproducibility

The final experiment uses five random seeds:

```python
SEEDS = (42, 43, 44, 45, 46)
```

The final reported results are calculated over all five seeds.

### Seed execution

```text
Seed 42
Seed 43
Seed 44
Seed 45
Seed 46
```

The final paper reports:

\[
\text{Mean} \pm \text{Standard Deviation}
\]

over the five runs.

---

# 6. Training Procedure

Each model is trained independently for each dataset and seed.

### Optimizer

AdamW:

```text
Learning Rate = 0.002
Weight Decay  = 0.005
```

### Loss

Cross-entropy with label smoothing:

```text
Label Smoothing = 0.02
```

### Learning-rate scheduler

`ReduceLROnPlateau` is applied using validation Macro-F1.

```text
Scheduler Patience = 10
Factor             = 0.5
Minimum LR         = 1e-5
```

### Early stopping

Training stops when validation Macro-F1 does not improve for 50 epochs.

```text
Patience = 50
```

### Checkpoint selection

The best checkpoint is selected using:

```text
Validation Macro-F1
```

Validation accuracy is used as a secondary tie-breaking criterion.

The test set is evaluated only after restoring the best validation checkpoint.

---

# 7. Evaluation Metrics

The primary evaluation metric is:

### Macro-F1

Macro-F1 is calculated by taking the unweighted mean of the F1 score across all classes.

This is particularly useful for these datasets because class distributions are not necessarily balanced.

Additional reported metrics:

- Test Accuracy
- Test Macro-F1
- Validation Macro-F1
- Parameter Count
- Training Time

---

# 8. Final Experimental Results

The following results are based on five random seeds: **42, 43, 44, 45, and 46**.

## Cora

| Model | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| GAT | **77.70 ± 1.28%** | **76.61 ± 1.15%** |
| GATv2 | 77.50 ± 1.19% | 76.45 ± 1.14% |
| GATv2 + Static Edge | 77.10 ± 1.07% | 76.03 ± 0.69% |
| GATv2 + Learned Edge | 77.48 ± 0.92% | 76.42 ± 0.99% |

**Best model:** GAT

---

## CiteSeer

| Model | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| GAT | 67.14 ± 0.28% | 63.82 ± 0.71% |
| GATv2 | 67.14 ± 0.97% | 63.82 ± 0.84% |
| GATv2 + Static Edge | **67.60 ± 1.33%** | **64.05 ± 1.10%** |
| GATv2 + Learned Edge | 66.50 ± 0.96% | 62.73 ± 0.71% |

**Best model:** GATv2 + Static Edge

---

## PubMed

| Model | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| GAT | 75.78 ± 0.64% | 75.79 ± 0.53% |
| GATv2 | 75.58 ± 0.95% | 75.51 ± 0.82% |
| GATv2 + Static Edge | **75.84 ± 1.21%** | **75.82 ± 0.80%** |
| GATv2 + Learned Edge | 74.16 ± 2.39% | 74.41 ± 1.80% |

**Best model:** GATv2 + Static Edge

---

# 9. Overall Results

| Dataset | Best Model | Test Macro-F1 |
|---|---|---:|
| Cora | GAT | **76.61 ± 1.15%** |
| CiteSeer | GATv2 + Static Edge | **64.05 ± 1.10%** |
| PubMed | GATv2 + Static Edge | **75.82 ± 0.80%** |

The results demonstrate that edge information provides **dataset-dependent benefits** rather than a universal improvement.

---

# 10. Comparison with GATv2

The primary comparison is between vanilla GATv2 and its edge-enhanced variants.

## Static Edge vs GATv2

| Dataset | GATv2 | Static Edge | Difference |
|---|---:|---:|---:|
| Cora | 76.45% | 76.03% | -0.42 pp |
| CiteSeer | 63.82% | 64.05% | **+0.23 pp** |
| PubMed | 75.51% | 75.82% | **+0.31 pp** |

Static edge representations improve GATv2 on CiteSeer and PubMed, while producing slightly lower performance on Cora.

---

## Learned Edge vs GATv2

| Dataset | GATv2 | Learned Edge | Difference |
|---|---:|---:|---:|
| Cora | 76.45% | 76.42% | -0.03 pp |
| CiteSeer | 63.82% | 62.73% | -1.09 pp |
| PubMed | 75.51% | 74.41% | -1.10 pp |

The learned-edge model does not consistently improve over GATv2.

---

# 11. Main Findings

### Finding 1 — Edge information is dataset-dependent

Static edge information improves GATv2 on CiteSeer and PubMed but does not improve performance on Cora.

Therefore, edge information should not be considered universally beneficial.

---

### Finding 2 — Static edge representations are more reliable than learned edges

The static-edge formulation performs better than the learned-edge formulation on CiteSeer and PubMed.

The learned edge model also shows higher variability on PubMed.

This suggests that additional trainable edge parameters may not be beneficial when the number of labeled nodes is extremely small.

---

### Finding 3 — GAT remains competitive

On Cora, standard GAT achieves the highest five-seed Macro-F1:

```text
76.61 ± 1.15%
```

This demonstrates that the additional complexity of GATv2 and edge-enhanced variants does not automatically produce better results.

---

### Finding 4 — Small training sets make overfitting important

The datasets provide only:

```text
Cora     : 140 training nodes
CiteSeer : 120 training nodes
PubMed   : 60 training nodes
```

During training, several models reach near-perfect training accuracy while validation performance remains substantially lower.

This makes regularization, validation-based checkpointing, and multi-seed evaluation important.

---

# 12. Research Conclusion

The experiments indicate that incorporating edge information into GATv2 provides **dataset-dependent rather than universally consistent improvements**.

Static edge representations achieved the strongest performance among the GATv2 variants on CiteSeer and PubMed, while standard GAT remained the strongest model on Cora.

The learned edge formulation did not consistently outperform GATv2 and showed comparatively larger variability, particularly on PubMed.

Overall, the results suggest that **simple fixed edge representations can provide more reliable benefits than additional trainable edge parameters under limited-label settings**.

---

# 13. Project Structure

```text
DL_Final_Project/
│
├── data/
│   ├── Cora/
│   ├── CiteSeer/
│   └── PubMed/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── edge_encoder.py
│   ├── evaluate.py
│   ├── main.py
│   ├── models.py
│   ├── train.py
│   └── tune.py
│
├── results/
│   ├── all_results.csv
│   ├── all_results.json
│   └── GAT_Research_Results.xlsx
│
├── venv/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 14. Installation

## Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd DL_Final_Project
```

## Create a virtual environment

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# 15. Running the Experiment

For the complete five-seed experiment:

```bash
rm -rf results
python -m src.main
```

The program executes:

```text
Seed 42
    Cora
        GAT
        GATv2
        GATv2 + Static Edge
        GATv2 + Learned Edge

    CiteSeer
        GAT
        GATv2
        GATv2 + Static Edge
        GATv2 + Learned Edge

    PubMed
        GAT
        GATv2
        GATv2 + Static Edge
        GATv2 + Learned Edge

Seed 43
...
Seed 44
...
Seed 45
...
Seed 46
...
```

---

# 16. Output Files

After execution, the following files are generated:

```text
results/all_results.csv
results/all_results.json
results/GAT_Research_Results.xlsx
```

These contain the results for all datasets, models, and seeds.

---

# 17. Reproducibility Checklist

Before producing final paper results, verify:

- [x] Four models only
- [x] Same datasets
- [x] Same train/validation/test splits
- [x] Same hidden dimension
- [x] Same number of layers
- [x] Same optimization settings
- [x] Same checkpoint criterion
- [x] Five random seeds
- [x] Test evaluation after best validation checkpoint
- [x] Mean and standard deviation reported
- [x] No artificial result inflation
- [x] Static and learned edge variants evaluated under the same protocol

---

# 18. Important Experimental Note

The results in this README should be interpreted as the outcome of the current controlled five-seed configuration.

Earlier development runs used different architectures and hyperparameters. Those results should **not** be directly combined with the final five-seed experiment.

For the final paper, use the five-seed results from the controlled configuration as the primary results.

---

# 19. Technologies

- Python
- PyTorch
- PyTorch Geometric
- NumPy
- scikit-learn
- Pandas
- openpyxl

---

# 20. Research Scope

### Included

- Graph Neural Networks
- Graph Attention Networks
- GATv2
- Static edge representations
- Learned edge representations
- Node classification
- Multi-seed evaluation
- Robustness through statistical reporting