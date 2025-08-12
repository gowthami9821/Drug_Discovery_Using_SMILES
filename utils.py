# utils.py
import torch
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def sample(model, n_samples, n_steps, device):
    x = torch.randn(n_samples, 2048).to(device)
    for t in reversed(range(n_steps)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long).to(device)
        with torch.no_grad():
            predicted_noise = model(x, t_batch)
        alpha_t = 1 - (0.0001 + (0.02 - 0.0001) * (t / n_steps))
        x = (x - (1 - alpha_t) * predicted_noise) / (alpha_t ** 0.5)
    return x.cpu()

def nearest_smiles_from_fp(fp_generated, fps_real, smiles_real):
    closest_smiles = []
    for fp in fp_generated:
        bin_str = ''.join(['1' if bit > 0.5 else '0' for bit in fp.numpy()])
        try:
            gen_fp = DataStructs.CreateFromBitString(bin_str)
        except:
            continue
        best_sim, best_smiles = -1, None
        for real_fp, smi in zip(fps_real, smiles_real):
            ref_str = ''.join(['1' if bit else '0' for bit in real_fp])
            ref_fp = DataStructs.CreateFromBitString(ref_str)
            sim = DataStructs.TanimotoSimilarity(gen_fp, ref_fp)
            if sim > best_sim:
                best_sim = sim
                best_smiles = smi
        closest_smiles.append(best_smiles)
    return closest_smiles
