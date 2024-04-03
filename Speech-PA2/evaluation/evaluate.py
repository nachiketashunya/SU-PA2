import os
import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from ...models.pretrained.ecapa_tdnn import ECAPA_TDNN_SMALL
from .metrics import compute_eer
from ..data.make_dataset import get_test_loader 

model_path = "/content/drive/MyDrive/SpeechAss2/models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_LIST = ['hubert_large', 'wav2vec2_xlsr', "wavlm_large"]

def init_model(model_name, checkpoint=None):
    if model_name == 'wavlm_large':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')
    elif model_name == 'hubert_large':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k')
    elif model_name == 'wav2vec2_xlsr':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr')
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict['model'], strict=False)

    return model

def evaluation(model_name, test_loader, checkpoint=None, threshold=0.8):
    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)

    model = init_model(model_name, checkpoint)
    model = model.to(device)

    total_eer = 0

    # Iterate through test loader
    for wav1, wav2, label, _, _ in test_loader:
      print(f"Wav 1 shape: {wav1.shape}")
      print(f"Label: {label}, Label Shape: {label.shape}")

      wav1 = wav1.to(device)
      wav2 = wav2.to(device)

      model.eval()
      with torch.no_grad():
          emb1 = model(wav1)
          emb2 = model(wav2)

          sim = F.cosine_similarity(emb1, emb2)
          print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))

          sim[sim < threshold] = 0
          sim[sim >= threshold] = 1

          total_eer += compute_eer(label.detach().cpu().numpy(), sim.cpu().numpy())
    
    return total_eer

# Evaluate 
wavlm_large = os.path.join(model_path, models[3])

# Get test loader
test_loader = get_test_loader()
eer = evaluation("hubert_large", test_loader, wavlm_large)
print(eer)