"""Vendored CRAFT model (region+affinity character-region detector).

Source: clovaai/CRAFT-pytorch (Baek et al., CVPR 2019), MIT License, (c) 2019 NAVER Corp.
Only the model definition is vendored (`craft.py`, `vgg16_bn.py`), patched for modern torchvision
(`pretrained=`/`model_urls` were removed). Training/inference live in `craft_train.py` /
`craft_segmenter.py`. Weights are NOT committed — see ocr/experiments/README.md.
"""

from .craft import CRAFT

__all__ = ["CRAFT"]
