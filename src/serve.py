import io
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import get_model

app = FastAPI(title="CIFAR-10 Model Serving API")

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

infer_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

def load_checkpoint():
    global model
    checkpoint_dir = Path(os.getenv("CHECKPOINT_DIR", "/app/checkpoints"))
    checkpoint_path = checkpoint_dir / os.getenv("MODEL_NAME", "classifier_v1.pt")
    
    if not checkpoint_path.exists():
        checkpoint_path = Path("checkpoints/classifier_v1.pt")
        
    if checkpoint_path.exists():
        m = get_model("resnet18", num_classes=10)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        m.load_state_dict(checkpoint["model_state_dict"])
        m.to(device)
        m.eval()
        model = m
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

@app.on_event("startup")
async def startup_event():
    load_checkpoint()

@app.get("/health")
async def health():
    if model is None:
        load_checkpoint()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = infer_transforms(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0].tolist()
            
        probabilities = {CLASSES[i]: round(probs[i], 4) for i in range(len(CLASSES))}
        predicted_class = CLASSES[max(range(len(probs)), key=lambda i: probs[i])]
        
        return {
            "predicted_class": predicted_class,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
