import numpy as np
import os
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms import functional as TF
from tqdm import tqdm

# Change this to your submission directory
submission_dir = 'example-submission'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

mcs = [] # No. of misclassifications
mean_rmses = [] # Mean RMSEs

models_ls = os.listdir('models')
models_ls.sort()
for model_name in models_ls:
    mcs.append(0)
    mean_rmses.append(0)

    model = getattr(models, model_name.split('.')[0])()
    model.load_state_dict(torch.load(f'models/{model_name}'))
    model.eval().to(device=device)

    print(model_name)
    for file_name in tqdm(os.listdir('imagenet-subset-224x224')):
        gt_idx = int(file_name.split('.')[0])

        with Image.open(f'imagenet-subset-224x224/{file_name}') as orig:
            orig_tensor = TF.to_tensor(orig).to(device=device)

        adv_tensor = torch.load(f'{submission_dir}/{model_name.split(".")[0]}/{file_name.split(".")[0] + ".pt"}', map_location=device)
            
        diff = adv_tensor - orig_tensor
        diff = diff.clamp(min=-1e-3, max=1e-3)
        
        adv_tensor = orig_tensor + diff

        norm_adv_tensor = TF.normalize(adv_tensor, imagenet_mean, imagenet_std)
        
        pred = model(norm_adv_tensor.unsqueeze(0))
        mcs[-1] += int(pred.argmax().item() != gt_idx)
        mean_rmses[-1] += diff.square().mean().sqrt().item() / 100

scores_per_model = np.array(mcs) - 4e7 * (np.array(mean_rmses) ** 2)

print(f'misclassifications: {dict(zip(models_ls, mcs))}')
print(f'mean_rmses:         {dict(zip(models_ls, mean_rmses))}')
print(f'scores_per_model:   {dict(zip(models_ls, scores_per_model.tolist()))}')
print(f'final_score:        {scores_per_model.sum().item()}')