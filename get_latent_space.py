import numpy as np
import pickle
import torch

def get_latents(test_loader, model, device):
    # Set model in training mode, if not already
    model.eval()
    labels = []
    latent_space = []
    with torch.no_grad():
        for i, (x, label) in enumerate(test_loader):

            x = x.to(device)#cuda(gpu, non_blocking=True)

            # compute output and loss
            z = model.forward_single(x)
            z = z.cpu().numpy()
            if len(latent_space) == 0:
                latent_space = z
            else:
                latent_space = np.concatenate((latent_space, z))

            label = list(label.cpu().numpy())
            labels.extend(label)

    return np.array(labels), latent_space


def get_and_save_latents(test_loader, model, device):
    labels, latent_space = get_latents(test_loader, model, device)
    with open("labels_and_latents_export.pkl", "wb") as f:
        pickle.dump((labels, latent_space), f)