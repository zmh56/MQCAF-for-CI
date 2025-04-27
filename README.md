# Multi-Query Cross-Modal Attention Fusion (MQCAF) for Cognitive Impairment Recognition

This study proposes a multi-query cross-modal attention fusion (MQCAF) model based on synchronized multimodal signals for cognitive impairment recognition. We constructed a standardized cognitive impairment assessment dataset (EEV-CI) containing synchronized EEG, ECG, and video signals. The MQCAF model integrates EEG, ECG, and video features to improve cognitive impairment prediction.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Feature Engineering](#feature-engineering)
- [SRM-net](#srm-net)
- [Facial_exp](#facial_exp)
- [MQCAF](#mqcaf)
- [Data](#data)
- [Citation](#citation)
- [License](#license)

## Installation

To install the necessary dependencies for this repository, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Feature Engineering

This section outlines the traditional feature extraction techniques used in the machine learning experiments. These features are essential for comparing the performance of traditional machine learning methods with the proposed deep learning-based methods.

### SRM-net (Physiological Signal Encoder Training)

This section describes the training procedure for the **SRM-net**, the encoder responsible for processing the physiological signals (EEG, ECG). The model is designed to effectively extract meaningful features from these signals, which are then used for cognitive impairment prediction.

To train SRM-net, use the following command:

```bash
python train_srm_net.py
```

### Facial_exp (Facial Expression Encoder Training)

This section describes the training procedure for the **Facial Expression Encoder**, which is responsible for processing video data (such as facial expressions and aus sequence). The encoder extracts key facial features related to emotion recognition.

To train the Facial Expression Encoder, run:

```bash
python train_facial_exp.py
```

### MQCAF (Multi-query Cross-modal Attention Fusion Training)

The **MQCAF model** fuses three modalities—EEG, ECG, and video signals—using a multi-query attention mechanism. The fusion of these modalities helps improve the performance of cognitive impairment recognition.

To train the MQCAF model, use the following command:

```bash
python train_mqcaf.py
```

## Data

The dataset used in this study is the **EEV-CI dataset**, which includes synchronized EEG, ECG, and video signals for cognitive impairment recognition. Due to privacy and permission-related concerns, the dataset is not yet publicly available. We are currently in the process of securing the necessary permissions, and any updates regarding its availability will be promptly shared on this GitHub repository.


## Citation

If you use this repository in your research, please cite our work:

```bibtex
@article{xx,
  title={Multi-query Cross-modal Attention Fusion (MQCAF) for Cognitive Impairment Recognition},
  author={},
  journal={},
  year={2025},
  volume={XX},
  number={YY},
  pages={ZZZZ-ZZZZ},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
