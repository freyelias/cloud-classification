import torch
from torch.utils.data import Dataset
import cv2


class CloudDataset(Dataset):
    def __init__(self, img_lab_list, label_category, transform=None):
        self.img_lab_list = img_lab_list
        self.label_category = label_category
        self.transform = transform

    def __len__(self):
        return len(self.img_lab_list)

    def __getitem__(self, index):
        img_path = self.img_lab_list[index][0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.label_category == 'cloud':
            label = torch.tensor(int(self.img_lab_list[index][1]))
        elif self.label_category == 'base':
            label = torch.tensor(int(self.img_lab_list[index][2]))
        else:
            raise ValueError(f"Label category has to be 'cloud' or 'base' but is {self.label_category} instead!")

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label
