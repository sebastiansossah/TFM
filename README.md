# TFM

## Overview

This repository hosts a Master's Thesis project focused on anomaly detection in time-series data from an industrial control system. The project evaluates and compares the performance of classical machine learning algorithms against various deep learning autoencoder architectures for identifying anomalous behavior in a water treatment plant dataset.

The primary objective is to build models capable of distinguishing between normal and anomalous operational states, which is crucial for predictive maintenance and ensuring system reliability.

## Dataset

The project utilizes a time-series dataset from a Secure Water Treatment (SWaT) testbed. The data, located in `data/df.zip`, represents sensor readings and actuator states over time and includes labeled periods of normal and anomalous operation. The notebooks are configured to load this dataset directly from the repository.

## Methodology

The core methodology is based on unsupervised learning, where models are trained exclusively on data from normal operational periods to learn a representation of the system's standard behavior.

1.  **Data Preprocessing**: The continuous time-series data is transformed into a set of fixed-size sliding windows (60 seconds). This approach captures temporal patterns and structures the data for model input.
2.  **Feature Engineering**:
    *   For classical models, Principal Component Analysis (PCA) is applied to the flattened windows to reduce dimensionality.
    *   For all models, sensor (continuous) and actuator (binary) data are handled distinctly. Continuous sensor data is standardized using `StandardScaler`.
3.  **Anomaly Detection**: Models are trained to reconstruct the input windows. The reconstruction error (e.g., Mean Absolute Error) is calculated for each window. Windows with an error exceeding a statistically determined threshold (e.g., the 85th percentile of the training errors) are flagged as anomalous.

## Models Explored

The project is divided into two main experimental parts: classical machine learning and deep learning.

### Classical Models

The notebooks in the `modelos_clasicos/` directory contain experiments and hyperparameter tuning for the following algorithms:

*   **Isolation Forest (`IF.ipynb`)**: An ensemble method that isolates anomalies by building a forest of random decision trees.
*   **K-Nearest Neighbors (`KNN.ipynb`)**: A proximity-based algorithm where the anomaly score is determined by a data point's distance to its k-th nearest neighbor.
*   **One-Class SVM (`OCSVM.ipynb`)**: An algorithm that learns a decision boundary around normal data points to identify outliers that fall outside this boundary.

### Deep Learning Models

The `modelos_deep_learning/pipeline-colab.ipynb` notebook implements a complete pipeline to train and evaluate several autoencoder architectures designed for time-series data:

*   **Dense Autoencoder**: A standard autoencoder with fully connected layers that flattens the time-series window.
*   **LSTM Autoencoder**: Utilizes Long Short-Term Memory (LSTM) layers to model temporal sequences and dependencies within each data window.
*   **CNN Autoencoder**: Employs 1D Convolutional Neural Networks (CNN) to capture local patterns and motifs in the time-series signals.
*   **Transformer Autoencoder**: A more advanced architecture using self-attention mechanisms to weigh the importance of different time steps within a window.

## Repository Structure

```
.
├── data/
│   └── df.zip                  # Zipped time-series dataset
├── modelos_clasicos/
│   ├── IF.ipynb                # Notebook for Isolation Forest experiments
│   ├── KNN.ipynb               # Notebook for K-Nearest Neighbors experiments
│   └── OCSVM.ipynb             # Notebook for One-Class SVM experiments
└── modelos_deep_learning/
    └── pipeline-colab.ipynb    # Comprehensive pipeline for deep learning models
```

## Usage

### Prerequisites
*   Python 3.x
*   Jupyter Notebook or JupyterLab
*   Required packages can be installed via pip. Key dependencies are listed in the notebooks (e.g., `tensorflow`, `scikit-learn`, `pyod`).

### Running the Experiments

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sebastiansossah/TFM.git
    cd TFM
    ```

2.  **Run the notebooks:**
    *   The notebooks in `modelos_clasicos/` can be run individually to execute the experiments for each classical model.
    *   The `modelos_deep_learning/pipeline-colab.ipynb` is a complete experimental pipeline designed for a GPU-enabled environment like Google Colab. It trains all specified deep learning models, evaluates their performance, and saves results (metrics, training history plots, reconstruction visualizations) to `results/` and `history/` directories.

## Results

Each notebook executes a hyperparameter search and evaluates models based on F1-score, precision, and recall for detecting anomalous windows. Results are presented in pandas DataFrames. The deep learning pipeline provides a more extensive analysis, including visualizations of training loss, error distributions, and signal reconstructions to assess model performance qualitatively and quantitatively.
