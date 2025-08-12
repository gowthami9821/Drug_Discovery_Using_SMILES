from flask import Flask, request, jsonify, render_template
from diffusion_model import SimpleDiffusionModel
from rdkit.Chem import AllChem
from utils import sample, nearest_smiles_from_fp
import torch
import base64
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO

# Flask app setup
app = Flask(__name__, template_folder='templates')

# Load your model
device = torch.device("cpu")
model = SimpleDiffusionModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Load training data for SMILES recovery
smiles_list = np.load("smiles_list.npy", allow_pickle=True)
fingerprints = np.load("fingerprints.npy", allow_pickle=True)

# Homepage route (optional)
@app.route('/')
def home():
    return render_template('index.html')

# Generate molecules and return image
@app.route('/generate', methods=['POST'])

def generate():
    try:
        data = request.get_json()
        input_smiles = data.get("smiles")

        if not input_smiles:
            return jsonify({'error': 'SMILES string is required'}), 400

        # Convert SMILES to fingerprint
        mol = Chem.MolFromSmiles(input_smiles)
        if not mol:
            return jsonify({'error': 'Invalid SMILES'}), 400

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.array(fp).astype(np.float32)
        input_tensor = torch.tensor(arr).unsqueeze(0).to(device)

        # Apply noise + denoise (simulating one step of reverse diffusion)
        t = torch.tensor([500], dtype=torch.long).to(device)  # arbitrary timestep
        noise = torch.randn_like(input_tensor)
        noisy_input = input_tensor * (1 - 0.5) + noise * 0.5  # simple q_sample

        with torch.no_grad():
            denoised_fp = model(noisy_input, t).squeeze(0)

        # Find the closest real molecule
        result_smiles = nearest_smiles_from_fp([denoised_fp], fingerprints, smiles_list)[0]

        # Draw result molecule
        result_mol = Chem.MolFromSmiles(result_smiles)
        img = Draw.MolToImage(result_mol, size=(400, 300))

        buf = BytesIO()
        img.save(buf, format='PNG')
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({'image_base64': img_base64, 'output_smiles': result_smiles})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Start the server
if __name__ == "__main__":
    app.run(debug=True)
