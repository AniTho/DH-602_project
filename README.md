# Introduction

This project aims to tackle the problem of data augmentation in medical imaging by presenting a method that integrates ControlNet, a diffusion-based generative model, with medical domain expertise. The proposed approach is crafted to enhance and generate samples for medical imaging datasets, with a specific emphasis on improving multi-class classification accuracy.

# Getting Started

Environment
- Tested OS: Linux
- Python >= 3.9
- PyTorch == 2.0


Installation
- Clone the repository to your local machine.
- Navigate to the project directory: cd DH-602_project
- Create the environment and install the requirements using source `pip install requirements.txt`
- Download the dataset using `source scripts/down_process_kneeOA.sh`

# Training

- Update the configuration file from `configs/cfg.yaml` by setting the path of dataset directory and boolean variable to train the corresponding architecture component in the following order (VAE -> DIFF -> CONTROL -> CLS_MODEL)

- To train the component, run the following command:

```python
python main.py
```
