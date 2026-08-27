import torch
from src.model import get_model

def test_model_forward_shape():
    model = get_model("resnet18", num_classes=10)
    model.eval()
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
