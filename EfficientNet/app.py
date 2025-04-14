from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision.models import efficientnet_b2
from torchvision import transforms
import torch.nn as nn

app = FastAPI()  # API instance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = efficientnet_b2(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[-1].in_features, 1)  # change output layer to regression
)
model.load_state_dict(torch.load("efficient_b2_model_retrain.pth", map_location=torch.device('cpu')))  # trained weights
model.to(device)
model.eval()

# data processing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # preprocess test image
        image = Image.open(file.file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # do prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()

        # prediction
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

