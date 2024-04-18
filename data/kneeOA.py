from torch.utils.data import Dataset
import pathlib
from PIL import Image
import random

class KneeOADataset(Dataset):
    def __init__(self, cfg, subset = 'train', transform=None):
        self.transform = transform
        base_path = pathlib.Path(cfg.DATASET_PATH)
        imgs_dir = base_path / subset
        self.imgs = list(imgs_dir.glob('*/*.png'))
        self.descriptions = {
        "0": "Healthy knee image.",
        "1": "Doubtful joint narrowing with possible osteophytic lipping.",
        "2": "Definite presence of osteophytes and possible joint space narrowing.",
        "3": "Multiple osteophytes, definite joint space narrowing, with mild sclerosis.",
        "4": "Large osteophytes, significant joint narrowing, and severe sclerosis."
        }
        self.classwise_imgs = dict()
        for img in self.imgs:
            class_name = img.parts[-2]
            img_list = self.classwise_imgs.get(class_name, [])
            img_list.append(img)
            self.classwise_imgs[class_name] = img_list

        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        kl_grade = img_path.parts[-2]
        img_class = int(kl_grade)
        progressive_classes = ['0', '1', '2', '3', '4']
        del progressive_classes[img_class]
        progressive_class = random.choice(progressive_classes)
        prog_img_path = random.choice(self.classwise_imgs[progressive_class])
        img = Image.open(img_path)
        prog_img = Image.open(prog_img_path)
        kl_description = self.descriptions[kl_grade]
        prompt_text = f"Generate a high-resolution X-ray image of the knee with Kellgren and Lawrence grade (KL Grade): {progressive_class}, depicting {self.descriptions[progressive_class]} from a conditioned image of Kellgren and Lawrence grade (KL Grade): {kl_grade}, depicting {kl_description}"
        if self.transform:
            img = self.transform(img)
            prog_img = self.transform(prog_img)
        return dict(image = img, prompt = prompt_text, kl_grade = int(kl_grade),
                    hint = prog_img)