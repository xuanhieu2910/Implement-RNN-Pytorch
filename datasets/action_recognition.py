from typing import Any

from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.datasets.vision as VisionDataset

class ActionRecognitionDataset(VisionDataset):

    def __init__(self):
        super().__init__()


    def __getitem__(self, index):
        pass

    def __len__(self) -> int:
        pass

