# KrabbyPatty_Pytorch

## Introduction
ICLR 2021 paper <a href="https://openreview.net/forum?id=1FvkSpWosOl">Is Attention Better Than Matrix Decomposition?</a> Pytorch implementation. I haved tested this on IWSLT for the correctness and the efficacy. The hamburger-pytorch is not correct.


## Usage

```python
import torch
from krabbypatty_pytorch import KrabbyPatty

x = torch.randn(42,64,512)  # [Sequence Length, Batch Size, Hidden Dimension]
krabbypatty = KrabbyPatty(input_dim=512)
output = krabbypatty(x) + x
```

## Citations
```bibtex
@inproceedings{
    title={Is Attention Better Than Matrix Decomposition?},
    author={Geng, Zhengyang and Guo, Meng-Hao and Chen, Hongxu and Li, Xia and Wei, Ke and Lin Zhouchen}
    year={2021},
    url={https://openreview.net/forum?id=1FvkSpWosOl},
    note={ICLR 2021}
}
```
