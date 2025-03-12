# Advent of Adversaries (Aperion 2025)

![banner.png](banner.png)

## What's in here?
### `imagenet-subset-224x224/`
- Contains a 100-image subset of ImageNet
- All images are PNGs.
- To match the input dimensions of the ResNet models provided (3 color channels x 224 x 224), all images have already been scaled and cropped to 224 x 224.
- The name of an image-file represents its respective class index in the output layer of the models.
- For example, the file `imagenet-subset-224x224/014.png` has the class index 14 and corresponds to the logit `model(input)[0, 14]`.
- From the above, it should be obvious that each ImageNet class has atmost one image here.

### `index-to-label.json`
- A JSON file representing an array that maps a class index to its respective label
- For example, the file `imagenet-subset-224x224/014.png` has the class label "indigo bunting", since that is the string at index 14 of the JSON array.
- The labels of the images are not _strictly_ important to the problem statement, but they can be handy for debugging nonetheless.

### `models/`
- Contains three ResNet models, saved as PyTorch state dicts
- Example usage:
```python
import torch
from torchvision import models

model = models.resnet18()
model.load_state_dict(torch.load('resnet18.pt'))
model.eval()

...
```
