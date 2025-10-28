# Official code base for MambaOut-DA: Reforming Unsupervised Domain Adaptation with MambaOut in an Efficient Manner.
## Author: Yihang Wu, Ahmad Chaddad.


<p align="center">
<img src="image.png" alt="intro" width="90%"/>
</p>
"Pipeline of MambaOut-DA based on standard UDA. The process begins with extracting features from both labeled source and unlabeled target domains by a pre-trained MambaOut model (frozen). Furthermore, these features pass through a bottleneck module consisting of two fully connected layers. The output from the bottleneck is used to compute the adaptation loss ($\mathcal{L}_{CMMD},\mathcal{L}_{IM},\mathcal{L}_{SSL}$) and the classification loss ($\mathcal{L}_{CE}$) on the source domain to fine-tune the bottleneck module."
