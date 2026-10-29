"""Minimal standalone training scaffold; use the Kaggle notebook for the full workflow."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Folder containing all/ and hem/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="bloodvision_efficientnet_b3.pth")
    args = parser.parse_args()

    root = Path(args.data_root)
    if not (root / "all").is_dir() or not (root / "hem").is_dir():
        raise FileNotFoundError("Expected data-root/all and data-root/hem directories.")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(root, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} loss={running_loss/total:.4f} acc={correct/total:.4f}")

    torch.save({"model_state_dict": model.state_dict(), "class_to_idx": ds.class_to_idx}, args.output)
    print(f"Saved checkpoint: {args.output}")

if __name__ == "__main__":
    main()
