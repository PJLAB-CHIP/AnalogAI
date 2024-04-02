import torch
import torchvision.models as models


model = models.resnet18(pretrained=True)
model.eval()


from torch.quantization import QuantStub, DeQuantStub, prepare, convert
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


dataset = ImageNet(root="/path/to/imagenet", split="val", transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

model = prepare(model)


def calibrate_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, _ in data_loader:
            model(image)

calibrate_model(model, data_loader)


quantized_model = convert(model)
