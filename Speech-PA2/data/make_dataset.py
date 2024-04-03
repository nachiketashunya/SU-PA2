import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os

# Get voxceleb data 
dataset_dir = "/home/nachiketa/Speech-PA2/data/processed"
meta_url = "https://mm.kaist.ac.kr/datasets/voxceleb/meta/list_test_hard2.txt"

class VoxCelebDataset(Dataset):
  def __init__(self, data_dir, text_file, max_frames=1000):
    self.data_dir = data_dir
    self.text_file = text_file
    self.max_frames = max_frames
    self.data_list = []

    # Read data from text file
    with open(text_file, 'r') as f:
      for line in f:
        label, first_wav, second_wav = line.strip().split()
        self.data_list.append((label, first_wav, second_wav))

  def __getitem__(self, index):
    label, first_wav_path, second_wav_path = self.data_list[index]
    first_wav_tensor, _ = self.process_sample(first_wav_path)
    second_wav_tensor, _ = self.process_sample(second_wav_path)
    return first_wav_tensor, second_wav_tensor, label

  def process_sample(self, path):
    file_path = os.path.join(self.data_dir, path)
    
    wav, sample_rate = torchaudio.load(file_path)
    num_frames = wav.shape[1]

    if num_frames < self.max_frames:
      padding_size = self.max_frames - num_frames
      wav = torch.nn.functional.pad(wav, (0, padding_size), value=0)
    elif num_frames > self.max_frames:
      wav = wav[:, :self.max_frames]

    wav = wav.unsqueeze(0)  # Add channel dimension if required

    return wav, sample_rate

def get_test_loader():
    max_frames = 30000
    wav_dir = os.path.join(dataset_dir, "wav")
    text_file = os.path.join(dataset_dir, "list_test_hard2.txt")

    test_dataset = VoxCelebDataset(wav_dir, text_file, max_frames)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2, shuffle=True)

    return test_loader