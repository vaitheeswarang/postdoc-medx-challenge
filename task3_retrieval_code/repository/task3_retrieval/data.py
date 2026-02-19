from dataclasses import dataclass
import numpy as np
from PIL import Image
from medmnist import PneumoniaMNIST


@dataclass
class DataItem:
    index: int
    label: int
    image: Image.Image


def load_split(split: str = "test", download: bool = True) -> PneumoniaMNIST:
    return PneumoniaMNIST(split=split, download=download)


def get_item(ds: PneumoniaMNIST, idx: int) -> DataItem:
    img, label = ds[idx]
    label = int(np.asarray(label).reshape(-1)[0])
    return DataItem(index=idx, label=label, image=img)
