import torch
from torch.utils.data import DataLoader
from torch import optim
import os

from unet import UNetSmall
from dataset_segmentation import FundusSegDataset
from loss_functions import DiceLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 512
BATCH = 4
EPOCHS = 40
LR = 1e-4

def train_one_model(name, root):
    print(f"\n====== TRAINING {name.upper()} MODEL ======\n")

    # Dataset
    trainset = FundusSegDataset(root=root, size=SIZE, augment=True)
    loader = DataLoader(trainset, batch_size=BATCH, shuffle=True)

    # Model
    model = UNetSmall(in_ch=3, out_ch=1, base=32).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = DiceLoss()

    for epoch in range(EPOCHS):
        model.train()
        losses = []

        for img, mask in loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            pred = model(img)
            loss = criterion(pred, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss = {sum(losses)/len(losses):.4f}")

    # Save
    os.makedirs("models", exist_ok=True)
    outfile = f"models/model_{name}.pth"
    torch.save(model.state_dict(), outfile)

    print(f"\n✔ Saved {name} model → {outfile}\n")


if __name__ == "__main__":
    # Train vessel model
    train_one_model("vessels", "dataset_seg/vessels")

    # Train optic disc model
    train_one_model("disc", "dataset_seg/disc")