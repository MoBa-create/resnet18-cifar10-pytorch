import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import ResNet18CIFAR

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

def load_trained_model(weights_path, device):
    model = ResNet18CIFAR(num_classes=10)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ The weights were successfully loaded from {weights_path}")
        return model
    else:
        raise FileNotFoundError(f"❌ Weights file not found: {weights_path}")

def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top3_prob, top3_catid = torch.topk(probabilities, 3)

    print(f"\n Image prediction results'{image_path}':")
    print("-" * 40)
    for i in range(top3_prob.size(0)):
        idx = top3_catid[i].item()
        prob = top3_prob[i].item() * 100
        print(f"{i+1}. {CLASSES[idx]} - trust : {prob:.2f}%")

    predicted_class = CLASSES[top3_catid[0].item()]
    confidence = top3_prob[0].item() * 100

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = 'resnet18_cifar10.pth'
    
    model = load_trained_model(weights_path, device)
    
    test_image_path = 'test_image.jpg' 
    
    if os.path.exists(test_image_path):
        predict_image(test_image_path, model, device)
    else:
        print(f"⚠️ Add a picture with a name.'{test_image_path}'Run the script again in the folder.")