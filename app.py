import torchvision.transforms as transforms
import torch.nn.functional as F
from model import ResNet18CIFAR
from PIL import Image
import gradio as gr
import torch
import os

CLASSES = [
    'Airplane (طائرة)', 'Automobile (سيارة)', 'Bird (طائر)', 'Cat (قطة)',
    'Deer (غزال)', 'Dog (كلب)', 'Frog (ضفدع)', 'Horse (حصان)',
    'Ship (سفينة)', 'Truck (شاحنة)'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'resnet18_cifar10.pth')

model = ResNet18CIFAR(num_classes=10)
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print("The form was successfully uploaded for ✅ Gradio App")
else:
    raise FileNotFoundError(f"❌ Weights file not found at : {weights_path}")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def predict(input_image):
    if input_image is None:
        return None

    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)

    input_tensor = transform(input_image.convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    confidences = {CLASSES[i]: float(probabilities[i]) for i in range(10)}
    return confidences

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image for the test"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions"),
    title="🖼️ CIFAR-10 ResNet-18 Image Classifier",
    description="Upload any image, and the model will... ResNet-18 The trainer, by classifying it into one of the categories... CIFAR-10 The tenth in terms of precision and aesthetics.",
    theme="soft"
)

if __name__ == '__main__':
    demo.launch()