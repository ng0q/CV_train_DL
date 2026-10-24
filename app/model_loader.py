import torch
from pathlib import Path
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from torchvision import transforms

MODEL_DIR = Path('model')
MODEL_PATH = MODEL_DIR / 'model.pth'

CLASS_NAMES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

model = None
transform = None

def load_artifacts() -> None:
    global model, transform


    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}. "
            "Run `git lfs pull` to fetch model/model.pth."
        )

    net = models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(512, 10)

    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    net.load_state_dict(state_dict)

    net.eval()

    model = net

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print(f"Model loaded from {MODEL_PATH.resolve()}")

def predict(image: Image.Image) -> tuple[str, float]:
    if model is None or transform is None:
        raise RuntimeError("Model is not loaded")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, idx = probs.max(dim=1)

    return CLASS_NAMES[idx.item()], float(confidence.item())