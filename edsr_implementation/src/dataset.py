import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CustomImageDataset(Dataset):
    """
    Датасет для SR: загружает пары LR и HR изображений.
    Основываясь на датасете DIV2K ожидает, что имена пары файлов соответствуют:
        HR: image0001.png
        LR: image0001x{scale_factor}.png
    """
    def __init__(self, hr_dir, lr_dir, scale_factor=2, patch_size=64):
        """
        Аргументы:
            hr_dir: путь к HR изображениям
            lr_dir: путь к LR изображениям
            scale_factor: масштаб
            patch_size: размер патча в LR, тогда размер HR будет patch_size * scale_factor
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale_factor = scale_factor
        self.patch_size_lr = patch_size
        self.patch_size_hr = patch_size * scale_factor

        # Список HR файлов
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg'))])
        self.to_tensor = T.ToTensor() 

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_file, lr_file = self.valid_pairs[idx]

        base_name, ext = os.path.splitext(hr_file)
        lr_file = f"{base_name}x{self.scale_factor}{ext}"
        
        hr_path = os.path.join(self.hr_dir, hr_file)
        lr_path = os.path.join(self.lr_dir, lr_file)

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")
        
        # Получение случайного патча
        w_lr, h_lr = lr_img.size
        
        left_lr = random.randint(0, w_lr - self.patch_size_lr) 
        top_lr = random.randint(0, h_lr - self.patch_size_lr)

        left_hr = left_lr * self.scale_factor
        top_hr = top_lr * self.scale_factor

        lr_patch = lr_img.crop((left_lr, top_lr, left_lr + self.patch_size_lr, top_lr + self.patch_size_lr))
        hr_patch = hr_img.crop((left_hr, top_hr, left_hr + self.patch_size_hr, top_hr + self.patch_size_hr))

        lr_tensor = self.to_tensor(lr_patch) 
        hr_tensor = self.to_tensor(hr_patch)

        return lr_tensor, hr_tensor