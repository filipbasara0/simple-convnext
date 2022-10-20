# simple-convnext

Simple implementation of the [ConvNext](https://arxiv.org/abs/2201.03545) architecture in PyTorch.

![image](https://user-images.githubusercontent.com/29043871/196708595-074ad9dc-781b-4911-a5b9-b614e7ff9f19.png)

Supports RandAug and Cutmix / Mixup augmentation and Label Smoothing, Random Erasing and Stochastic Depth regularization.

For now, tested on CIFAR10, and obtained **0.92** F1.

Architecture is as follows:

- Patch size: `1`
  - Patch sizes of **2** and **4** resulted in roughly **6** and **12** lower F1 score
- Layer depths: `[2, 2, 2, 2]`
- Block dims: `[64, 128, 256, 512]`
- This model is almost 5 times smaller (`6.4M params`) compared to `ConvNeXt-T` - (`29M params`)

Augmentation / Regularization params:

- Mixup: `0.8`, Cutmix: `1.0`, Prob: `0.4`, Label smoothing: `0.1`
- Stochastic Depth Rate: `0.0`
- RandAug: `ops=2, magnitude=9`

AdamW was used as an optimizer, with a learning rate of `1e-3` and weight decay `1e-1`.
Training was done for `100` epochs with a batch size of `256` on GTX 1070Ti.

Next step is to get a decent performance on the tiny-imagenet dataset.

Hoping to get some better hardware to train on ImageNet-1k and fine tune the model for object detection. 🙏😅
