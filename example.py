import os
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms import functional as TF
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

for model_name in ['resnet18', 'resnet34', 'resnet50']:
    # Set up the model
    model = getattr(models, model_name)()
    model.load_state_dict(torch.load(f'models/{model_name}.pt'))
    model.eval().to(device=device)

    print(model_name)
    for file_name in tqdm(os.listdir('imagenet-subset-224x224')):
        # Convert PIL Image to PyTorch Tensor
        with Image.open(f'imagenet-subset-224x224/{file_name}') as img:
            img_tensor = TF.to_tensor(img).to(device=device)

        # Normalize the image tensor
        norm_img_tensor = TF.normalize(img_tensor, imagenet_mean, imagenet_std)
        
        # BE CAREFUL THAT YOU USE THE NORMALIZED (NOT RAW) TENSOR WHEN PASSING IT TO THE MODEL!
        #pred = model(norm_img_tensor.unsqueeze(0))

        # BE CAREFUL THAT YOU CHECK FOR PERTURBATION LIMITS ON THE RAW (NOT NORMALIZED) TENSOR!
        # Perturbation logic goes here.
        # Since this is only an example, we simply add random noise within the perturbation limit
        img_tensor += 1e-3 * torch.rand_like(img_tensor)

        # BE CAREFUL THAT YOU SAVE THE RAW (NOT NORMALIZED) TENSOR!
        dir_path = f'example-submission/{model_name}'
        os.makedirs(dir_path, exist_ok=True)
        torch.save(img_tensor, f'{dir_path}/{file_name.split(".")[0] + ".pt"}')