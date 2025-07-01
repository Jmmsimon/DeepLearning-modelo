from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import io
from pathlib import Path

class BinaryRetinaNet(nn.Module):
    def __init__(self):
        super().__init__()
        w = EfficientNet_B0_Weights.IMAGENET1K_V1
        bb = efficientnet_b0(weights=w)
        self.features = bb.features
        in_f = bb.classifier[1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_f, 1)
        )
    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)

transform = transforms.Compose([
    transforms.Resize((240,240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

MODEL_PATH = Path('models/retina_binary_best.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BinaryRetinaNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Retina Binary Classification Model"}

@app.post("/predict/")
async def predict_retina(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            prediction = int(prob >= 0.5)
        return JSONResponse(content={"prediction": prediction, "probability": prob})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
