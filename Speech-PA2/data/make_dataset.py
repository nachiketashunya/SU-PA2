import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from zipfile import ZipFile

# Get voxceleb data 
dataset_dir = "/home/nachiketa/Speech-PA2/data/processed"
meta_url = "https://mm.kaist.ac.kr/datasets/voxceleb/meta/list_test_hard2.txt"

voxceleb_ds = torchaudio.datasets.VoxCeleb1Verification(root=dataset_dir, download=True, meta_url=meta_url)



