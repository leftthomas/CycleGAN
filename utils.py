import glob
import random

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.transform = transforms.Compose(
                [transforms.RandomResizedCrop(256, (1.0, 1.12), interpolation=Image.BICUBIC),
                 transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.files_A = sorted(glob.glob('{}/{}/A/*.jpg'.format(root, mode)))
        self.files_B = sorted(glob.glob('{}/{}/B/*.jpg'.format(root, mode)))

    def __getitem__(self, index):
        a_name = self.files_A[index]
        a = self.transform(Image.open(a_name).convert('RGB'))
        if self.mode == 'train':
            b_name = self.files_B[random.randint(0, len(self.files_B) - 1)]
            b = self.transform(Image.open(b_name).convert('RGB'))
        else:
            b_name = self.files_B[index]
            b = self.transform(Image.open(b_name))
        return a, b, a_name, b_name

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.detach():
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)

