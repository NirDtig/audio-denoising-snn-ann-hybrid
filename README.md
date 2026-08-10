# Audio Denoising using Spiking Neural Networks

This repository contains the implementation of my **Dual Degree Project (DDP) at IIT Bombay**, titled:

> **Audio Denoising using Spiking Neural Networks**

The project investigates the use of **Spiking Neural Networks (SNNs)** for speech enhancement by replacing the recurrent components of an **NSNet2-style speech enhancement architecture** with biologically inspired spiking neuron models.

Multiple spiking neuron models were investigated as part of the project:

* **Leaky Integrate-and-Fire (LIF)**
* **Recurrent Leaky Integrate-and-Fire (RLIF)**
* **Adaptive Leaky Integrate-and-Fire (adLIF)**
* **Recurrent Adaptive Leaky Integrate-and-Fire (RadLIF)**

The final implementation in this repository uses **RadLIF** SNN layers within the NSNet2-style hybrid architecture.

---

## Project Overview

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
NSNet2-style Hybrid Network
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

The model operates in the time-frequency domain and predicts a spectral mask that is applied to the noisy complex STFT to obtain an enhanced speech representation.

---

## Architecture

The model follows an **NSNet2-style speech enhancement architecture**.

The recurrent processing component is replaced with SNN layers. The final implementation contains two SNN blocks followed by fully connected layers for spectral-mask estimation.

### Spiking Neuron Models

A major part of the project was the investigation of different spiking neuron dynamics for audio denoising.

The evaluated neuron models were:

| Neuron model | Description                                 |
| ------------ | ------------------------------------------- |
| **LIF**      | Leaky Integrate-and-Fire                    |
| **RLIF**     | Recurrent Leaky Integrate-and-Fire          |
| **adLIF**    | Adaptive Leaky Integrate-and-Fire           |
| **RadLIF**   | Recurrent Adaptive Leaky Integrate-and-Fire |

The final executable implementation in this repository uses **RadLIF** neurons.

The RadLIF-based layers use:

* Bidirectional processing
* Batch normalization
* Dropout
* Hidden layer configuration `[128, 128, 200]`

---

## Audio Representation

The input audio is transformed into the time-frequency domain using the **Short-Time Fourier Transform (STFT)**.

For each noisy-clean audio pair:

1. The noisy waveform is transformed using STFT.
2. The clean waveform is independently transformed using STFT.
3. The noisy power spectrum is calculated.
4. A logarithmic power-spectrum representation is used as the model input.
5. The SNN-based model predicts a spectral mask.
6. The predicted mask is applied to the noisy complex STFT.
7. The enhanced STFT is converted back into the time domain using inverse STFT.

### STFT Configuration

| Parameter           | Value |
| ------------------- | ----: |
| FFT size            |   508 |
| Hop length          |   160 |
| Window length       |   320 |
| Training batch size |     1 |

---

## Spectral Mask Estimation

The hybrid network first projects the input spectral features to a higher-dimensional representation and processes them using two SNN layer blocks.

The resulting representation is passed through fully connected layers and a sigmoid activation to produce the spectral mask:

```text
Input log-power spectrum
        │
        ▼
Linear projection
        │
        ▼
RadLIF SNN
        │
        ▼
RadLIF SNN
        │
        ▼
Fully connected layers
        │
        ▼
Sigmoid
        │
        ▼
Spectral mask
```

The predicted mask is applied element-wise to the noisy complex STFT:

```python
enhanced_stft = noisy_stft * predicted_mask
```

The enhanced waveform is then reconstructed using inverse STFT.

---

## Loss Function

The final training implementation uses a custom **Compressed Complex Loss**.

The loss combines:

* A compressed complex spectral loss
* A compressed magnitude loss

The predicted and target complex spectra are normalized, magnitude compression is applied with exponent `c = 0.3`, and the complex and magnitude terms are combined using `alpha = 0.3`.

The training configuration is:

```text
CompressedComplexLoss(
    c = 0.3,
    alpha = 0.3
)
```

---

## Training

The final training configuration in the notebook uses:

| Parameter          |             Value |
| ------------------ | ----------------: |
| Optimizer          |             AdamW |
| Learning rate      |            `1e-4` |
| Scheduler          | ReduceLROnPlateau |
| Scheduler factor   |             `0.9` |
| Scheduler patience |                 5 |
| Epochs             |                10 |
| Batch size         |                 1 |

During each training iteration, the model predicts a spectral mask, applies it to the noisy STFT, computes the compressed complex loss against the clean STFT, and updates the model parameters through backpropagation.

---

## Evaluation

The validation pipeline reconstructs enhanced waveforms using inverse STFT and evaluates speech-enhancement performance using:

### SNR

Signal-to-noise ratio is calculated between the clean and enhanced waveforms:

```text
SNR = 10 log10(signal power / noise power)
```

The residual noise is calculated as the difference between the clean and enhanced signals.

### SI-SDR

**Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)** is also computed between the clean reference and enhanced waveform.

The validation procedure reports:

```text
Validation Loss
SNR
SI-SDR
```

for each training epoch.

---

## Repository Contents

```text
audio-denoising-snn/
│
├── audio_denoising_thesis_faithful_clean.ipynb
└── README.md
```

The notebook contains the implementation of:

* Audio data loading
* STFT preprocessing
* SNN model construction
* NSNet2-style hybrid architecture
* Spectral-mask estimation
* Compressed Complex Loss
* Model training
* Validation
* SNR evaluation
* SI-SDR evaluation

---

## Code Attribution

The SNN components used in this project are based on the **SpArch** framework developed by the **Idiap Research Institute**.

**SpArch — Spiking Architectures for Speech Technology**

[SpArch GitHub Repository](https://github.com/idiap/sparch?utm_source=chatgpt.com)

The framework provides the `SNN` implementation and the spiking neuron models used in this project.

The SNN framework and underlying neuron implementations are **not claimed as original work in this repository**. The project contribution is the investigation and integration of different SNN neuron models into an NSNet2-style audio-denoising pipeline.

Please refer to the original SpArch repository for its license and attribution requirements.

---

## Notes

The original implementation was developed in Google Colab.

The repository version contains the thesis implementation with development-time commented-out material removed, while preserving the executable model, training configuration, and evaluation pipeline.

The dataset paths in the notebook refer to local/Colab storage and therefore need to be modified when running the notebook in another environment.

---

## Author

**Nirmal S.**

Dual Degree, Electrical Engineering
**Indian Institute of Technology Bombay**

**Dual Degree Project:** Audio Denoising using Spiking Neural Networks
