import os
import torch
import faiss
import json
from tqdm import tqdm
from clip_dual_encoder import CLIPStyleDualEncoder
from vjepa_bridge import VJEPAtoPRISM

# TODO: Replace with actual dataset paths and video-caption pairs
DATASET_DIR = "data/videos"
ANNOTATIONS_FILE = "data/annotations.json"  # Format: [{"video": "x.mp4", "caption": "..."}, ...]
OUTPUT_INDEX = "video_index.faiss"
OUTPUT_METADATA = "video_metadata.json"

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = CLIPStyleDualEncoder().to(device)
    model.load_state_dict(torch.load("dual_encoder.pt"))
    model.eval()

    # Load V-JEPA encoder
    vjepa = VJEPAtoPRISM("configs/pretrain/vitl16.yaml", device=str(device))

    # Load annotations
    with open(ANNOTATIONS_FILE, 'r') as f:
        annotations = json.load(f)

    # Init FAISS index and metadata tracker
    dim = 512  # latent/text shared embedding size
    index = faiss.IndexFlatL2(dim)
    metadata = {}

    for i, sample in enumerate(tqdm(annotations)):
        video_path = os.path.join(DATASET_DIR, sample['video'])

        try:
            latent = vjepa.extract_latents(vjepa.load_video(video_path))
            latent_embed = model.latent_encoder(latent.unsqueeze(0).to(device))
            latent_embed = latent_embed / latent_embed.norm(dim=-1, keepdim=True)

            vec = latent_embed.cpu().detach().numpy().astype('float32')
            index.add(vec)
            metadata[i] = {"video": sample['video'], "caption": sample['caption']}

        except Exception as e:
            print(f"[ERROR] Skipped {video_path}: {e}")

    # Save FAISS index and metadata
    faiss.write_index(index, OUTPUT_INDEX)
    with open(OUTPUT_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[✓] Saved index to {OUTPUT_INDEX} and metadata to {OUTPUT_METADATA}")
