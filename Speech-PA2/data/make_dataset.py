import torch
import torchaudio
import os

class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, text_file, max_frames=32000):
    self.data_dir = data_dir
    self.text_file = text_file
    self.max_frames = max_frames
    self.data_list = []

    # Read data from text file
    with open(text_file, 'r') as f:
      for line in f:
        label, first_wav, second_wav = line.strip().split()

        first_w_path = os.path.join(self.data_dir, first_wav)
        second_w_path = os.path.join(self.data_dir, second_wav)

        if os.path.exists(first_w_path) and os.path.exists(second_w_path):
          self.data_list.append((label, first_wav, second_wav))

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, index):
    label, first_wav_path, second_wav_path = self.data_list[index]
    first_wav_tensor, first_sample_rate = self.process_sample(first_wav_path)
    second_wav_tensor, second_sample_rate = self.process_sample(second_wav_path)

    label = torch.tensor(np.array(int(label)))

    return first_wav_tensor.squeeze(0), second_wav_tensor.squeeze(0), label

  def process_sample(self, path):
    filename, _ = os.path.splitext(path)
    path = filename + ".wav"
    file_path = os.path.join(self.data_dir, path)

    wav, sample_rate = torchaudio.load(file_path)
    num_frames = wav.shape[1]

    if num_frames < self.max_frames:
      padding_size = self.max_frames - num_frames
      wav = torch.nn.functional.pad(wav, (0, padding_size), value=0)
    elif num_frames > self.max_frames:
      wav = wav[:, :self.max_frames]

    return wav, sample_rate