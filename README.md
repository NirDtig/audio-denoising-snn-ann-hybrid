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
Kindly look at the Repository Contents and Code Organization section to know the code structure of this repo. The repo also contains my thesis named as Dual_Degree_Btech_plus_Mtech.pdf

## Project Overview

The objective of this project was to investigate whether spiking neural networks could replace conventional recurrent neural networks in a lightweight speech-enhancement system while maintaining competitive denoising performance and reducing model complexity.

The project uses an **NSNet2-based architecture** as the starting point and replaces its recurrent processing components with spiking neural network layers.

The overall speech-enhancement pipeline is:

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

The repository contains two reference notebooks at the root level and the modified experimental implementation under the `src/` directory.

```text
audio-denoising-snn/
│
├── audio_denoising_snn.ipynb
├── audio_denoising_thesis_faithful_clean.ipynb
│
├── src/
│   ├── audio_denoising_snn_adLIF_exponential_threshold5_.ipynb
│   ├── snns_with_surrogates.py
│   └── surrogate_gradients.py
│
└── README.md
```

### `audio_denoising_snn.ipynb`

**Purpose:** Original/reference implementation of the audio-denoising system.

This notebook contains the NSNet2-based speech-enhancement pipeline and uses the **SpArch (`sparch`) library** for the underlying SNN implementation.

It contains the main workflow for:

* loading noisy and clean speech
* STFT preprocessing
* logarithmic power-spectrum representation
* NSNet2-based hybrid architecture
* SNN-based processing
* spectral-mask estimation
* compressed complex loss
* model training
* validation
* waveform reconstruction
* SNR evaluation
* SI-SDR evaluation

The SNN implementation in this notebook comes from the SpArch library and is **not a custom SNN implementation developed in this project**.

**Intention:** provide the original/reference implementation from which the project was developed.

---

### `audio_denoising_thesis_faithful_clean.ipynb`

**Purpose:** Clean/faithful version of the reference implementation.

This notebook is a similar, cleaned-up version of `audio_denoising_snn.ipynb` that represents the implementation and workflow described in the thesis.

It contains the same major speech-enhancement components:

* STFT-based audio representation
* NSNet2-based architecture
* SNN processing
* spectral-mask estimation
* compressed complex loss
* training and validation
* SNR and SI-SDR evaluation

The underlying SNN implementation is still based on the SpArch framework.

**Intention:** provide a cleaner and easier-to-inspect version of the implementation corresponding to the thesis.

---

### `src/audio_denoising_snn_adLIF_exponential_threshold5_.ipynb`

**Purpose:** Reproducible example of a specific experimental configuration.

This notebook uses the modified SNN implementation in `src/snns_with_surrogates.py` and explicitly configures:

```python
NEURON_TYPE = "adLIF"
SURROGATE = "exponential"
THRESHOLD = 5.0
```

Thus, this notebook represents:

```text
Neuron      : adLIF
Surrogate   : Exponential
Threshold   : 5.0
```

The notebook retains the NSNet2-based speech-enhancement pipeline while using the configurable SNN implementation provided in `snns_with_surrogates.py`.

**Intention:** provide a concrete, reproducible example of how the experimental parameters investigated in the thesis can be configured in the model.

---

### `src/snns_with_surrogates.py`

**Purpose:** Modified SNN implementation that exposes the experimental parameters used in the thesis.

The underlying SNN implementation is based on the SpArch framework. This file modifies the SNN interface so that the following can be specified explicitly:

* neuron type
* surrogate-gradient function
* firing threshold

The supported neuron models include:

```text
LIF
RLIF
adLIF
RadLIF
```

For example:

```python
neuron_type = "adLIF"
surrogate = "exponential"
threshold = 5.0
```

These parameters are passed into the SNN layers, allowing the same speech-enhancement architecture to be evaluated under different experimental configurations.

**Intention:** provide the configurable SNN implementation required for the experimental studies in the thesis, while building upon the existing SpArch implementation rather than claiming the underlying SNN architecture as original work.

---

### `src/surrogate_gradients.py`

**Purpose:** Implements the surrogate-gradient functions required for the surrogate-gradient experiments.

The hard spike-generation function used by spiking neurons is non-differentiable. During backpropagation, a surrogate gradient is substituted for its true derivative.

This file implements:

* **Boxcar**
* **Exponential**
* **Gaussian**
* **Multi-Gaussian**

All four use the same hard spike in the forward pass and differ in the gradient used during the backward pass.

For example:

```python
surrogate = "exponential"
```

selects the Exponential surrogate gradient.

**Intention:** provide the additional surrogate-gradient implementations required to investigate the effect of different gradient approximations on SNN-based speech enhancement.

---

## How the Files Work Together

The relationship between the repository components is:

```text
              Reference / Clean Notebooks
                       │
                       ▼
              NSNet2-based Architecture
                       │
                       ▼
                SpArch SNN model
                       │
                       │
              Thesis experimental studies
                       │
                       ▼
             snns_with_surrogates.py
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        Neuron Models     Surrogate Gradients
        LIF / RLIF /      Boxcar / Exponential /
        adLIF / RadLIF    Gaussian / Multi-Gaussian
              │                 │
              └────────┬────────┘
                       ▼
       Specific experimental configuration
                       │
                       ▼
audio_denoising_snn_adLIF_exponential_threshold5_.ipynb
                       │
                       ▼
          adLIF + Exponential + Threshold 5
```

In summary:

* **`audio_denoising_snn.ipynb`** is the original/reference implementation using the SpArch SNN library.
* **`audio_denoising_thesis_faithful_clean.ipynb`** is a similar, cleaner version of that implementation.
* **`snns_with_surrogates.py`** provides the modified/configurable SNN implementation needed for the experimental studies.
* **`surrogate_gradients.py`** provides the surrogate-gradient implementations.
* **`audio_denoising_snn_adLIF_exponential_threshold5_.ipynb`** combines the modified SNN implementation with the speech-enhancement pipeline for one specific reproducible configuration.

The repository therefore separates the **reference implementation**, the **experimental modifications**, and the **specific experimental configuration** rather than presenting the entire SNN implementation as original work.

---

## Connection to the Thesis Experiments

The thesis investigated three main experimental dimensions.

### Experiment 1 — SNN Neuron Architecture

Different spiking neuron models were investigated:

```text
LIF
RLIF
adLIF
RadLIF
```

The objective was to investigate different spiking-neuron dynamics as replacements for the recurrent components of the NSNet2-based architecture.

### Experiment 2 — Surrogate Gradient

Different surrogate-gradient functions were investigated:

```text
Boxcar
Exponential
Gaussian
Multi-Gaussian
```

The objective was to investigate how the choice of surrogate gradient used during backpropagation affects the performance of the SNN-based speech-enhancement model.

### Experiment 3 — Spiking Threshold

Different firing thresholds were investigated:

```text
0.05
1
5
10
```

The objective was to investigate the sensitivity of the SNN-based architecture to the firing threshold.

The repository provides the modified SNN and surrogate-gradient code required to configure these experimental dimensions.

The notebook:

```text
src/audio_denoising_snn_adLIF_exponential_threshold5_.ipynb
```

provides one concrete example:

```text
Neuron      : adLIF
Surrogate   : Exponential
Threshold   : 5.0
```

This is a **specific reproducible configuration**, while the broader set of configurations investigated in the thesis is documented in the thesis itself.

---

## Dataset

The experiments use speech and noise data derived from the **VCTK** and **DEMAND** corpora.

The datasets themselves are **not included in this repository**.

The notebooks use local/Google Colab paths for accessing the dataset, so these paths need to be modified when running the project in another environment.

---

## Code Attribution

This project builds upon existing open-source implementations and research.

### NSNet2

The baseline speech-enhancement architecture is based on the **NSNet2** implementation by Noah Zhy.

https://github.com/noahzhy/NSNet2

NSNet2 served as the starting point for the speech-enhancement architecture used in this project. The contribution of this project was to investigate replacing its recurrent processing components with spiking neural network layers and evaluate different spiking architectures.

### SpArch

The underlying SNN implementation and neuron models are based on **SpArch (Spiking Architectures for Speech Technology)** from the Idiap Research Institute.

The original SpArch implementation is **not claimed as original work** in this repository.

https://github.com/idiap/sparch/tree/main

The contribution of this project was the **application and experimental investigation of SNNs within an NSNet2-based speech-enhancement system**, including:

* comparison of different SNN neuron models
* investigation of surrogate-gradient functions
* investigation of firing-threshold sensitivity
* evaluation using speech-enhancement metrics
* analysis of model complexity and parameter efficiency

The modified files under `src/` support these experiments while retaining attribution to the underlying SpArch implementation.

Please refer to the respective upstream repositories for their original implementations, licenses, and attribution requirements.

---

## Reproducibility Notes

The original implementation was developed primarily in **Google Colab**.

The notebooks contain paths referring to local/Google Drive storage. These paths need to be changed before running the project in another environment.

The datasets and trained model checkpoints are not included in this repository.

---

## Author

**Nirmal S.**

Dual Degree, Electrical Engineering
**Indian Institute of Technology Bombay**

**Dual Degree Project:** Audio Denoising using Spiking Neural Networks
