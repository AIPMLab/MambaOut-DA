# Official code base for MambaOut-DA: Reforming Unsupervised Domain Adaptation with MambaOut in an Efficient Manner.
## Author: Yihang Wu, Ahmad Chaddad.


<p align="center">
<img src="image.png" alt="intro" width="90%"/>
</p>

**Figure 1**. Pipeline of MambaOut-DA based on standard UDA. The process begins with extracting features from both labeled source and unlabeled target domains by a pre-trained MambaOut model (frozen). Furthermore, these features pass through a bottleneck module consisting of two fully connected layers. The output from the bottleneck is used to compute the adaptation loss ($\mathcal{L}_{CMMD},\mathcal{L}_{IM},\mathcal{L}_{SSL}$) and the classification loss ($\mathcal{L}_{CE}$) on the source domain to fine-tune the bottleneck module.

### Table 1: Comparison with SOTA methods on **Office-31**. The best performance is marked in **bold**. The methods above the horizontal line are CNN-based methods, while the methods below the horizontal line are ViT-based methods.

| Method | A→W | D→W | W→D | A→D | D→A | W→A | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN-based Methods** | | | | | | | |
| ResNet-50 | 68.4 | 96.7 | 99.3 | 68.9 | 62.5 | 60.7 | 76.1 |
| DANN | 82.0 | 96.9 | 99.1 | 79.7 | 68.2 | 67.4 | 82.2 |
| rRGrad+CAT | 94.4 | 98.0 | **100** | 90.8 | 72.2 | 70.2 | 87.6 |
| SAFN+ENT | 90.1 | 98.6 | 99.8 | 90.7 | 73.0 | 70.2 | 87.1 |
| CDAN+TN | 95.7 | 98.7 | **100** | 94.0 | 73.4 | 74.2 | 89.3 |
| TAT | 92.5 | 99.3 | **100** | 93.2 | 73.1 | 72.1 | 88.4 |
| SHOT | 90.1 | 98.4 | 99.9 | 94.0 | 74.7 | 74.3 | 88.6 |
| MDD+SCDA | 95.3 | 99.0 | **100** | 95.4 | 77.2 | 75.9 | 90.5 |
| **ViT-based Methods** | | | | | | | |
| ViT | 91.2 | 99.2 | **100** | 93.6 | 80.7 | 80.7 | 91.1 |
| TVT | 96.4 | 99.4 | **100** | 96.4 | 84.9 | 86.1 | 93.9 |
| CDTrans | 96.7 | 99.0 | **100** | 97.0 | 81.1 | 81.9 | 92.6 |
| SSRT | 97.7 | 99.2 | **100** | 98.6 | 83.5 | 82.2 | 93.5 |
| PMTrans | **99.5** | 99.4 | **100** | **99.8** | **86.7** | **86.5** | **95.3** |
| EUDA | 95.3 | **100** | **100** | 93.4 | 80.5 | 82.9 | 92.0 |
| **MambaOut-based Method** | | | | | | | |
| **Ours** | 97.2 | **100** | **100** | 96.8 | 84.1 | 85.8 | 94.0 |
