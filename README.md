# simple-convnext

Simple implementation of the [ConvNext](https://arxiv.org/abs/2201.03545) architecture in PyTorch.

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/29043871/197199637-9cd8b61b-632a-4cad-9361-b2db0af8c574.png"> 
</p>

Supports RandAug and Cutmix / Mixup augmentation and Label Smoothing, Random Erasing and Stochastic Depth regularization.

### CIFAR-10

After several iterations, I managed to get **0.92** F1 on CIFAR-10. The model should perform better with additional parameter optimization and longer training.

All models were trained from scratch, without fine-tuning.

Architecture is as follows:

- Patch size: `1`
  - Patch sizes of **2** and **4** resulted in roughly **6** and **12** lower F1 score, emphasizing the importance of a small patch size for low resolution images
- Layer depths: `[2, 2, 2, 2]`
- Block dims: `[64, 128, 256, 512]`
- This model is almost 5 times smaller (`6.4M params`) compared to `ConvNeXt-T` - (`29M params`)
- Image size: `64`

Augmentation / Regularization params:

- Mixup: `0.8`, Cutmix: `1.0`, Prob: `0.4`, Label smoothing: `0.1`
- Stochastic Depth Rate: `0.0`
- RandAug: `ops: 2, magnitude: 9`

AdamW was used as an optimizer, with a learning rate of `1e-3` and weight decay `1e-1`.
Training was done for `100` epochs with a batch size of `256` on GTX 1070Ti.

```
              precision    recall  f1-score   support

           0       0.92      0.94      0.93      1000
           1       0.94      0.98      0.96      1000
           2       0.90      0.89      0.90      1000
           3       0.81      0.80      0.80      1000
           4       0.93      0.91      0.92      1000
           5       0.86      0.84      0.85      1000
           6       0.93      0.95      0.94      1000
           7       0.95      0.95      0.95      1000
           8       0.96      0.96      0.96      1000
           9       0.97      0.93      0.95      1000

    accuracy                           0.92     10000
   macro avg       0.92      0.92      0.92     10000
weighted avg       0.92      0.92      0.92     10000
```

### Tiny ImageNet

The first run resulted with a F1 score of **0.64**.

There is deinitely more room to improve, since I didn't invest much time into tuning training and aug parameters.
Also, there is no doubt the model would continue improving, if trained for more than 100 epochs.

All models were trained from scratch, without fine-tuning or additional data.

Architecture for Tiny ImageNet is as follows:

- Patch size: `2`
- Layer depths: `[3, 3, 9, 3]`
- Block dims: `[96, 192, 384, 768]`
- This model is still smaller (`22M params`) compared to `ConvNeXt-T` - (`29M params`)
- Image size: `64`

Augmentation / Regularization params:

- Mixup: `0.8`, Cutmix: `1.0`, Prob: `0.6`, Label smoothing: `0.1`
- Stochastic Depth Rate: `0.1`
- RandAug: `ops: 2, magnitude: 9`

AdamW was used as an optimizer, with a learning rate of `2e-3` and weight decay `5e-2`.
Training was done for `100` epochs with a batch size of `128` on GTX 1070Ti and took ~21 hours.

```
            precision    recall  f1-score   support

           0       0.81      0.86      0.83        50
           1       0.83      0.80      0.82        50
           2       0.71      0.60      0.65        50
           3       0.62      0.58      0.60        50
           4       0.71      0.68      0.69        50
           ...
           ...
         195       0.78      0.78      0.78        50
         196       0.43      0.30      0.35        50
         197       0.48      0.42      0.45        50
         198       0.46      0.46      0.46        50
         199       0.55      0.58      0.56        50

    accuracy                           0.64     10000
   macro avg       0.65      0.64      0.64     10000
weighted avg       0.65      0.64      0.64     10000
```

### Future steps

Hoping to get some better hardware to train on ImageNet-1k and fine tune the model for object detection. 🙏😅
