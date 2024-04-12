import torch
import torch.nn as nn
from metrics import compute_eer
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluation(fext_or_clf, test_loader, log_title, model=None, is_et=False):
    total_eer = 0
    # the resulting embeddings can be used for cosine similarity-based retrieval
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    for i, (wav1, wav2, labels) in enumerate(test_loader):
      wav1 = wav1.to(device)
      wav2 = wav2.to(device)

      model.eval()
      with torch.no_grad():
          # audio files are decoded on the fly
            if is_et:
                audio1_features = fext_or_clf(wav1.squeeze(0), return_tensors="pt", sampling_rate=16000)
                audio1 = audio1_features.input_values.squeeze(0)
                audio1 = audio1.to(device)

                audio2_features = fext_or_clf(wav2.squeeze(0), return_tensors="pt", sampling_rate=16000)
                audio2 = audio2_features.input_values.squeeze(0)
                audio2 = audio2.to(device)

                embeddings1 = model(input_values=audio1).embeddings
                embeddings2 = model(input_values=audio2).embeddings
            else:
                embeddings1 = fext_or_clf.encode_batch(wav1)
                embeddings2 = fext_or_clf.encode_batch(wav2)
               
            embeddings1 = torch.nn.functional.normalize(embeddings1, dim=-1).cpu()
            embeddings2 = torch.nn.functional.normalize(embeddings2, dim=-1).cpu()

            sim = cosine_sim(embeddings1, embeddings2)

            sim = torch.sigmoid(sim)
            eer = compute_eer(labels, sim)

            if i % 100 == 0:
                print(f"{i+1}/{len(test_loader)} EER: {eer}")

            total_eer += eer
    
    avg_eer = round(total_eer / len(test_loader), 4)

    wandb.log({
        f'{log_title} EER': avg_eer
    })

    print(f"Average EER: {avg_eer}")
    print(f"Average EER(%): {avg_eer * 100}%")