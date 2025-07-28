# Lottery Ticket Hypothesis in Variational Quantum Classifiers

This repository contains the implementation and experimental code for investigating the Lottery Ticket Hypothesis (LTH) in the context of Variational Quantum Classifiers (VQCs). The study explores whether sparse, trainable subnetworks (winning tickets) exist within quantum neural networks, similar to their classical counterparts.

## Project Structure

```
lottery-ticket-vqc/
├── data/                           # Dataset files
│   ├── iris.txt                   # Iris dataset
│   └── wine.txt                   # Wine dataset
├── dataframes/                    # Experimental results (CSV format)
│   ├── slth/                      # Strong LTH results
│   └── wlth/                      # Weak LTH results
├── plots/                         # Generated visualizations
│   ├── slth/                      # SLTH plots
│   └── wlth/                      # WLTH plots
├── scripts/                       # Experimental scripts
│   ├── optuna/                    # Hyperparameter optimization
│   ├── slth/                      # Strong LTH experiments
│   └── wlth/                      # Weak LTH experiments
└── src/                           # Source code
    ├── config.py                  # Configuration management
    ├── dataset.py                 # Dataset handling
    ├── models.py                  # Model implementations
    ├── main_slth.py              # Strong LTH experiments
    ├── main_wlth.py              # Weak LTH experiments
    └── plot_*.py                  # Visualization scripts
```

## Models

### Quantum Models

- **VQC (Variational Quantum Classifier)**: Standard variational quantum circuit with parameterized gates
- **BVQC (Barren-plateau-free VQC)**: Modified VQC architecture designed to mitigate barren plateau effects

### Classical Models

- **NN (Neural Network)**: Standard fully-connected neural network
- **SNN (Simple Neural Network)**: Simplified neural network architecture for comparison

## Datasets

The experiments are conducted on two classical machine learning datasets:

- **Iris Dataset**: 3-class classification (both 2D and 3D variants)
- **Wine Dataset**: 3-class classification (both 2D and 3D variants)

All datasets are preprocessed using standard scaling.

## Experimental Setup

### Strong Lottery Ticket Hypothesis (SLTH)

- Tests whether sparse subnetworks can be trained from scratch to achieve comparable performance
- Uses evolutionary algorithms for subnet discovery
- Parameters:
  - 75 generations
  - 25 individuals per generation
  - Selection, recombination, and mutation operations

### Weak Lottery Ticket Hypothesis (WLTH)

- Tests whether sparse subnetworks exist after standard training and pruning
- Implements both iterative and one-shot pruning strategies
- Iterative pruning: 20% pruning rate per iteration
- One-shot pruning: Various pruning rates from 0% to 100%

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **PennyLane**: Quantum machine learning framework
- **PyTorch**: Neural network implementation
- **Optuna**: Hyperparameter optimization
- **scikit-learn**: Dataset preprocessing and metrics
- **matplotlib/seaborn**: Visualization
- **pandas/numpy**: Data manipulation

## Usage

### Running SLTH Experiments

```bash
cd src/
python main_slth.py
```

This will:

1. Run evolutionary algorithm-based subnet discovery
2. Test performance of discovered sparse networks
3. Save results to `dataframes/slth/`

### Running WLTH Experiments

```bash
cd src/
python main_wlth.py
```

This will:

1. Train models to convergence
2. Apply iterative and one-shot pruning
3. Evaluate pruned network performance
4. Save results to `dataframes/wlth/`

### Generating Plots

```bash
cd src/
python plot_slth.py      # Generate SLTH visualizations
python plot_wlth.py      # Generate WLTH visualizations
```

### Hyperparameter Optimization

```bash
cd src/
python optuna_script.py
```

## Configuration

Model and experiment configurations are managed in `src/config.py`. Key parameters include:

- **Model type**: VQC, BVQC, NN, SNN
- **Dataset**: 2iris, 3iris, 2wine, 3wine
- **Training parameters**: learning rate, batch size, epochs
- **Quantum-specific**: number of layers, data re-uploading, uniform range

## Results

Results are automatically saved as:

- **CSV files**: Detailed experimental data in `dataframes/`
- **PDF plots**: Visualizations in `plots/`
- **Performance metrics**: Accuracy, remaining weights, training time

### Key Findings

The experiments investigate:

1. **Existence of winning tickets** in quantum neural networks
2. **Pruning behavior** differences between quantum and classical models
3. **Performance degradation** under various pruning strategies
4. **Scalability** of lottery ticket hypothesis to quantum computing

## Reproducibility

- All experiments use fixed random seeds for reproducibility
- Multiple runs (typically 10) are performed for statistical significance
- Detailed hyperparameter configurations are logged
- Results include confidence intervals and statistical analysis

## File Naming Convention

- Results files follow the pattern: `{MODEL}_{DATASET}_{METHOD}.csv`
- Plot files include smoothed and unsmoothed variants
- Model types: VQC, BVQC, NN, SNN
- Datasets: 2iris, 3iris, 2wine, 3wine
- Methods: 75x25 (SLTH), ITERATIVE, ONE_SHOT (WLTH)

---

**Note**: This repository contains the complete experimental setup for reproducing the results presented in our submission on the Lottery Ticket Hypothesis in Variational Quantum Classifiers.
