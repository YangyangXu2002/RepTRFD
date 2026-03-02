# RepTRFD
Official implementation of "Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery", CVPR 2026.

## 📂 Dataset

The datasets used in our experiments are publicly available. Please download the required data from our Google Drive and place them into the `data/` directory.

🔗 **[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1rphapDHEcFwBZXH-nHEKFGMynZabOKPT?usp=sharing)**

## 🗂️ Directory Structure

After downloading the dataset, your project structure should look like this:

```text
RepTRFD/
├── data/                           # Datasets directory
│   ├── airplane.tiff
│   ├── Toy.mat
│   ├── Washington_DC.mat
│   ├── news.mat
│   ├── 0809.png
│   └── mario011.ply
│
├── model.py                        # Core RepTRFD network architectures
├── utils.py                        # Utility functions
│
├── Demo_inpainting.py              # Script for Image/Video Inpainting
├── Demo_denoising.py               # Script for MSI/HSI Denoising
├── Demo_super_resolution.py        # Script for Image Super-Resolution
└── Demo_point_cloud.py             # Script for Point Cloud Recovery
