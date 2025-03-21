import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from clip_dual_encoder import CLIPStyleDualEncoder

# TODO: Replace with actual latent extractor and dataset annotations
class DummyLatentTextDataset(Dataset):
    def __init__(self):
        self.examples = [
            {"latent": torch.randn(1024), "text": "a person is speaking at a courtroom podium"},
            {"latent": torch.randn(1024), "text": "a group of people signing a document"},
            {"latent": torch.randn(1024), "text": "a man walks through a courthouse lobby"},
            {"latent": torch.randn(1024), "text": "a legal deposition is taking place"},
        ] * 100  # TODO: Replace with real dataset

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["latent"], ex["text"]


# InfoNCE loss
def contrastive_loss(logits_per_video, logits_per_text):
    labels = torch.arange(logits_per_video.size(0)).to(logits_per_video.device)
    loss_v2t = F.cross_entropy(logits_per_video, labels)
    loss_t2v = F.cross_entropy(logits_per_text, labels)
    return (loss_v2t + loss_t2v) / 2


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPStyleDualEncoder().to(device)
    dataset = DummyLatentTextDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(10):  # TODO: Add validation and early stopping
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            latents, texts = batch
            latents = latents.to(device)

            logits_per_video, logits_per_text = model(latents, texts)
            loss = contrastive_loss(logits_per_video, logits_per_text)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")

    # TODO: Save final model checkpoint
    torch.save(model.state_dict(), "dual_encoder.pt")
