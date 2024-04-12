import os
import soundfile as sf
import torch
import torch.nn.functional as F
from .metrics import compute_eer
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, UniSpeechSatForXVector, HubertForSequenceClassification
from speechbrain.inference.speaker import EncoderClassifier
import torch
import wandb 
import sys

sys.path.append("Speech-PA2")
from data.make_dataset import CustomDataset
from evaluation.utils import evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_dir = "data/processed/"
wav_dir = os.path.join(dataset_dir, "VoxCeleb1_subset/dev")
text_file = os.path.join(dataset_dir, "VoxCeleb1_subset/list_test_hard.txt")

vox_test_dataset = CustomDataset(wav_dir, text_file, 32000)
vox_test_loader = DataLoader(vox_test_dataset, batch_size=64, shuffle=True)

# Initialize wandb
wandb.init(project="Speaker Verification", name="Eval on Voxceleb1-H Dataset")

"""
1. Wavlm Base+

"""
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

model = model.to(device)
evaluation(feature_extractor, vox_test_loader, log_title="Wavlm Base+", model=model)

"""
2. Unispeech SAT Base

"""

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/unispeech-sat-base-sv')
model = UniSpeechSatForXVector.from_pretrained('microsoft/unispeech-sat-base-sv')

model = model.to(device)
evaluation(feature_extractor, vox_test_loader, log_title="Unispeech", model=model)


"""
3. Ecapa TDNN

"""

if torch.cuda.is_available():
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"} )
else:
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

evaluation(classifier, vox_test_loader, log_title="Ecapa TDNN", is_et=True)

