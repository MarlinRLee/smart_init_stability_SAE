import os
import torch
import torch.nn as nn
from overcomplete.sae import TopKSAE, RATopKSAE
from utils import cosine_kmeans

def get_sae_model(model_type, d_brain, d_model, k, device, dataloader, config):
    """
    Factory to build or load the correct SAE variant.
    Handles the specific initialization logic for RA-SAE and SI-SAE.
    """
    if model_type == "RA-SAE":
        # RA-SAE requires pre-computed centers
        centers_path = f"../centers/centers_{config['n_candidates']}.pt"
        if os.path.exists(centers_path):
            print(f"Loading cached centers from {centers_path}")
            c_tensor = torch.load(centers_path, map_location=device, weights_only=True)
        else:
            c_tensor = cosine_kmeans(dataloader, config['n_candidates'], d_brain)
            torch.save(c_tensor, centers_path)
            
        sae = RATopKSAE(d_brain, nb_concepts=d_model, top_k=k, points=torch.tensor(c_tensor).float().cuda(),
                delta=config['delta'], device=device)
        return sae

    sae = TopKSAE(input_shape=d_brain, 
                  nb_concepts=d_model, 
                  top_k=k, 
                  device=device)
    
    if model_type == "SAE":
        return sae
    elif model_type == "SI-SAE":
        # SI-SAE or base SAE Logic: Initialize Standard SAE with Noisy K-Means Centroids
        si_path = f"../centers/si_centers_{d_model}.pt"
        if os.path.exists(si_path):
            si_centers = torch.load(si_path, map_location=device, weights_only=True)
        else:
            si_centers = cosine_kmeans(dataloader, d_model, d_brain)
            torch.save(si_centers, si_path)

        # Apply SI initialization logic
        centers = si_centers.cpu()
        noise = torch.randn_like(centers) * centers.std() * config['per_init']
        weights = (1 - config['per_init']) * centers + noise
        
        # Set Dictionary and Encoder
        norm_weights = torch.nn.functional.normalize(weights.to(device), p=2, dim=-1)
        sae.dictionary._weights.data = norm_weights
        
        enc_weights = norm_weights.clone() * (1.0 / (k ** 0.5))
        sae.encoder.final_block[0].weight.data = enc_weights
        
        return sae