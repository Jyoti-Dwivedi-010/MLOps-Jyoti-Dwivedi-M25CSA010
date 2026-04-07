import torchvision.models as models



def resnet18_cifar10(num_classes: int = 10):
    model = models.resnet18(weights=None)
    model.fc = __import__("torch").nn.Linear(model.fc.in_features, num_classes)
    return model



def resnet34_binary(num_classes: int = 2):
    model = models.resnet34(weights=None)
    model.fc = __import__("torch").nn.Linear(model.fc.in_features, num_classes)
    return model
