# Audio Denoising using Spiking Neural Networks

This repository contains the implementation associated with my **Dual Degree Project (DDP) at the Indian Institute of Technology Bombay**, titled:

> **Audio Denoising using Spiking Neural Networks**

The project investigates the use of **Spiking Neural Networks (SNNs)** for real-time speech enhancement by replacing the recurrent components of an **NSNet2-based speech enhancement architecture** with biologically inspired spiking neuron models.

**Author:** Nirmal S.
**Institution:** Indian Institute of Technology Bombay
**Department:** Electrical Engineering
**Project:** Dual Degree Project (DDP)
**Supervisor:** Prof. Udayan Ganguly

---

## Project Overview

The objective of this project was to investigate whether spiking neural networks could replace conventional recurrent neural networks in a lightweight speech-enhancement system while maintaining competitive denoising performance and reducing model complexity.

The project uses an **NSNet2-based architecture** as the starting point and replaces its recurrent processing components with spiking neural network layers.

The overall processing pipeline is:

```text
Noisy Audio
     │
     ▼
   STFT
     │
     ▼
Log Power Spectrum
     │
     ▼
NSNet2-based Hybrid Network
     │
     ├── Linear Projection
     │
     ├── Spiking Neural Network Layer
     │
     ├── Spiking Neural Network Layer
     │
     └── Dense Layers
     │
     ▼
Spectral Mask
     │
     ▼
Noisy STFT × Mask
     │
     ▼
Enhanced STFT
     │
     ▼
Inverse STFT
     │
     ▼
Enhanced Audio
```

The model operates in the time-frequency domain and predicts a spectral mask that is applied to the noisy complex STFT to reconstruct enhanced speech.

---

## Spiking Neural Network Architecture

A major component of the project was the investigation of different spiking neuron dynamics as alternatives to the recurrent GRU layers in the baseline architecture.

Four neuron models were investigated:

| Neuron Model | Description                                 |
| ------------ | ------------------------------------------- |
| **LIF**      | Leaky Integrate-and-Fire                    |
| **RLIF**     | Recurrent Leaky Integrate-and-Fire          |
| **adLIF**    | Adaptive Leaky Integrate-and-Fire           |
| **RadLIF**   | Recurrent Adaptive Leaky Integrate-and-Fire |

The project investigated these models in the context of speech enhancement and compared their denoising performance and model complexity.

The SNN-based architecture reduced the number of trainable parameters by approximately **60%** relative to the GRU-based baseline. Among the evaluated neuron models, **adLIF** provided the strongest overall performance and trade-off between model efficiency and speech-enhancement quality.

---

## Experimental Studies

The project consisted of three main experimental studies.

### Experiment 1 — Replacement of GRU Blocks with Spiking Neurons

The first experiment investigated replacing the GRU blocks of the NSNet2-based architecture with different SNN neuron models:

* LIF
* RLIF
* adLIF
* RadLIF

The models were compared in terms of speech-enhancement performance and parameter count.

The SNN-based architectures achieved a substantial reduction in trainable parameters, with the best-performing variant achieving approximately **60% fewer parameters** than the original GRU-based model.

The **adLIF** model demonstrated the strongest performance among the evaluated SNN variants and provided a competitive trade-off between model efficiency and denoising performance.

---

### Experiment 2 — Effect of Surrogate Gradient Functions

Spiking neurons contain a non-differentiable spike-generation function. During backpropagation, this non-differentiability is addressed using **surrogate gradients**.

The project investigated the effect of different surrogate-gradient functions:

* **Boxcar**
* **Exponential**
* **Gaussian**
* **Multi-Gaussian**

The experiments compared the resulting speech-enhancement performance using **SI-SDR**.

The results showed relatively small differences between the tested surrogate-gradient functions, suggesting that the hybrid SNN architecture was relatively robust to the choice of surrogate gradient over the tested settings.

---

### Experiment 3 — Effect of Spiking Threshold

The third experiment investigated the sensitivity of the SNN-based architecture to the neuronal firing threshold.

The default threshold was:

```text
1
```

Additional threshold values were tested:

```text
0.05
5
10
```

All other experimental settings were kept constant while varying the threshold.

The results showed relatively small changes in SI-SDR across the tested threshold values, indicating that the model was relatively robust to threshold variations over the investigated range.

---

## Audio Representation

The speech signals are represented in the time-frequency domain using the **Short-Time Fourier Transform (STFT)**.

For each noisy-clean audio pair:

1. The noisy waveform is transformed using STFT.
2. The clean waveform is independently transformed using STFT.
3. The noisy power spectrum is calculated.
4. A logarithmic power-spectrum representation is provided to the model.
5. The SNN-based network predicts a spectral mask.
6. The predicted mask is applied to the noisy complex STFT.
7. The enhanced STFT is reconstructed into the time domain using inverse STFT.

### STFT Configuration

| Parameter           | Value |
| ------------------- | ----: |
| FFT size            |   508 |
| Hop length          |   160 |
| Window length       |   320 |
| Training batch size |     1 |

---

## Spectral Mask Estimation

The hybrid network first projects the input spectral features into a higher-dimensional representation and processes them using SNN layers.

The resulting representation is passed through fully connected layers and a sigmoid activation to estimate the spectral mask:

```text
Input Log-Power Spectrum
          │
          ▼
  Linear Projection
          │
          ▼
       SNN Layer
          │
          ▼
       SNN Layer
          │
          ▼
   Fully Connected Layers
          │
          ▼
       Sigmoid
          │
          ▼
     Spectral Mask
```

The predicted mask is applied element-wise to the noisy complex STFT:

```python
enhanced_stft = noisy_stft * predicted_mask
```

The enhanced waveform is then reconstructed using inverse STFT.

---

## Loss Function

The training implementation uses a custom **Compressed Complex Loss**.

The loss combines:

* a compressed complex spectral loss
* a compressed magnitude loss

The spectral representation is normalized and magnitude compression is applied using:

```text
c = 0.3
alpha = 0.3
```

The resulting loss is used to optimize the predicted spectral mask against the clean speech representation.

---

## Training

The main training implementation uses:

| Parameter          |             Value |
| ------------------ | ----------------: |
| Optimizer          |             AdamW |
| Learning rate      |            `1e-4` |
| Scheduler          | ReduceLROnPlateau |
| Scheduler factor   |             `0.9` |
| Scheduler patience |                 5 |
| Batch size         |                 1 |

During training, the model predicts a spectral mask, applies the mask to the noisy STFT, computes the compressed complex loss against the clean STFT, and updates the network parameters through backpropagation using surrogate gradients for the spiking layers.

---

## Evaluation

The enhanced waveforms are reconstructed using inverse STFT and evaluated using standard speech-enhancement metrics.

### SNR

Signal-to-noise ratio is computed between the clean and enhanced waveforms:

```text
SNR = 10 log10(signal power / residual noise power)
```

where the residual noise is calculated from the difference between the clean and enhanced signals.

### SI-SDR

**Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)** is used to measure the similarity between the clean reference speech and the enhanced waveform while accounting for scale differences.

The validation pipeline reports:

```text
Validation Loss
SNR
SI-SDR
```

---

## Key Findings

The main findings of the project were:

* Replacing GRU layers with SNN layers resulted in an approximately **60% reduction in trainable parameters**.
* **LIF, RLIF, adLIF, and RadLIF** were investigated as alternative spiking neuron models.
* **adLIF** provided the strongest overall performance among the evaluated SNN variants.
* Different surrogate-gradient functions, including **Gaussian, Exponential, and Multi-Gaussian**, produced relatively similar speech-enhancement performance.
* The model showed relatively low sensitivity to the tested spiking threshold values.
* The results suggest that SNNs are a promising direction for **efficient and low-latency speech enhancement**, particularly for resource-constrained and real-time applications.

---

# Repository Contents and Code Organization

The repository separates the **main speech-enhancement implementation**, the **SNN implementation**, and the **surrogate-gradient definitions**.

```text
audio-denoising-snn/
│
├── src/
│   ├── audio_denoising_snn_adLIF_exponential_threshold5_.ipynb
│   ├── snns_with_surrogates.py
│   └── surrogate_gradients.py
│
└── README.md
```

### 1. `audio_denoising_snn_adLIF_exponential_threshold5_.ipynb`

**Purpose:** Main end-to-end implementation of the audio-denoising system.

This notebook is the primary entry point for the project. It brings together:

* audio data loading
* STFT preprocessing
* logarithmic power-spectrum features
* NSNet2-based hybrid architecture
* SNN layers
* spectral-mask prediction
* compressed complex loss
* model training
* validation
* waveform reconstruction using inverse STFT
* SNR evaluation
* SI-SDR evaluation

The current notebook is configured as a **specific reproducible experiment using adLIF neurons, an Exponential surrogate gradient, and a firing threshold of 5.0**.

The experimental configuration is:

```python
NEURON_TYPE = "adLIF"
SURROGATE = "exponential"
THRESHOLD = 5.0
```

**Intention:** provide a complete, inspectable end-to-end implementation of the speech-enhancement system and a concrete example of one experimental configuration.

---

### 2. `snns_with_surrogates.py`

**Purpose:** SNN implementation with configurable neuron type, surrogate gradient, and firing threshold.

This file contains the SNN layers used by the speech-enhancement model and exposes the experimental choices as explicit configuration parameters.

The supported neuron models are:

```text
LIF
RLIF
adLIF
RadLIF
```

The SNN can also select a surrogate gradient and firing threshold for a particular experiment.

For example:

```python
neuron_type = "adLIF"
surrogate = "exponential"
threshold = 5.0
```

These parameters are passed into the SNN layers so that the same architecture can be evaluated under different experimental settings.

**Intention:** keep the SNN implementation modular so that the neuron dynamics and training configuration can be changed without rewriting the NSNet2-based speech-enhancement architecture.

---

### 3. `surrogate_gradients.py`

**Purpose:** Defines the surrogate-gradient functions used for training the spiking neurons.

The forward spike-generation function is non-differentiable. During backpropagation, surrogate gradients provide an approximate derivative so that gradient-based optimization can be used.

This file implements:

* **Boxcar**
* **Exponential**
* **Gaussian**
* **Multi-Gaussian**

All four use the same hard spike in the forward pass and differ in the gradient substituted during the backward pass.

For example, the Exponential surrogate uses:

```text
g(x) = α exp(-β |x|)
```

The current experiment selects:

```python
surrogate = "exponential"
```

**Intention:** isolate the surrogate-gradient component of SNN training so that different gradient approximations can be investigated independently of the rest of the speech-enhancement architecture.

---

## How the Files Work Together

```text
                  Main Notebook
                       │
                       │ builds and trains
                       ▼
                NSNet2-based model
                       │
                       │ uses
                       ▼
             snns_with_surrogates.py
                       │
                       │ selects
                       ▼
          ┌──────────────────────────┐
          │ Neuron:     adLIF        │
          │ Surrogate:  Exponential  │
          │ Threshold:  5.0          │
          └──────────────────────────┘
                       │
                       │ uses surrogate gradient
                       ▼
              surrogate_gradients.py
```

In other words:

* The **notebook** defines and trains the complete speech-enhancement system.
* `snns_with_surrogates.py` provides the configurable spiking-neuron layers used inside that system.
* `surrogate_gradients.py` provides the gradient approximations used to train the non-differentiable spiking neurons.

This separation makes the experimental setup easier to inspect, reproduce, and modify.

---

## Connection to the Thesis Experiments

The thesis investigated three aspects of the SNN-based speech-enhancement architecture:

### Experiment 1

**Neuron architecture**

```text
LIF
RLIF
adLIF
RadLIF
```

### Experiment 2

**Surrogate-gradient function**

```text
Boxcar
Exponential
Gaussian
Multi-Gaussian
```

### Experiment 3

**Spiking threshold**

```text
0.05
1
5
10
```

The repository exposes these components separately so that a specific configuration can be selected without rewriting the complete speech-enhancement model.

The current notebook provides a **specific reproducible configuration**:

```text
Neuron      : adLIF
Surrogate   : Exponential
Threshold   : 5.0
```

rather than automatically running every possible combination.

---

## Dataset

The experiments use speech and noise data derived from the **VCTK** and **DEMAND** corpora.

The datasets themselves are **not included in this repository**.

The notebook uses local/Google Colab paths for accessing the dataset, so these paths must be modified when running the project in another environment.

---

## Code Attribution

This project builds upon existing open-source implementations and research.

### NSNet2

The baseline speech-enhancement architecture is based on the **NSNet2** implementation by Noah Zhy.

NSNet2 served as the starting point for the speech-enhancement architecture used in this project. The contribution of this project was to investigate replacing its recurrent processing components with spiking neural network layers and evaluate different spiking architectures.

### SpArch

The SNN components and neuron models are based on the **SpArch (Spiking Architectures for Speech Technology)** project from the Idiap Research Institute.

The SpArch framework provides implementations of the neuron models investigated in this work, including:

* LIF
* RLIF
* adLIF
* RadLIF

The underlying NSNet2 and SpArch implementations are **not claimed as original work** in this repository. The research contribution of this project is the investigation and integration of these SNN components into an NSNet2-based speech-enhancement pipeline, together with the experimental analysis of neuron models, surrogate gradients, and spiking thresholds.

Please refer to the respective upstream repositories for their original implementations, licenses, and attribution requirements.

---

## Reproducibility Notes

The original implementation was developed primarily in **Google Colab**.

The notebook contains paths referring to local/Google Drive storage. These paths need to be changed before running the project in another environment.

The datasets and trained model checkpoints are not included in this repository.

---

## Author

**Nirmal S.**

Dual Degree, Electrical Engineering
**Indian Institute of Technology Bombay**

**Dual Degree Project:** Audio Denoising using Spiking Neural Networks
