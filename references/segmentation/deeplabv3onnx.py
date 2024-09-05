""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ File for exporting model (Here deeplabv3) to ONNX format                 │
  └──────────────────────────────────────────────────────────────────────────┘
"""

#* Import

import torch
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import torchvision

def export_deeplabv3_to_onnx(model_path):
    """
    Export the DeepLabV3 model to ONNX format.

    Parameters:
    model_path (str): The path to save the ONNX model.
    """
    PATH = "D:\\pytorch_cuda\\segmentaion_data\\dataset6\\output\\checkpoint.pth"
    
    model = torchvision.models.get_model(
        name='deeplabv3_resnet101',
        #model='deeplabv3_resnet101',
        weights=None,
        weights_backbone='ResNet101_Weights.IMAGENET1K_V1',
        num_classes=5,
        aux_loss=True
    )
    #model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT) # up-to-date weights
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model'])

    model.eval()  

    dummy_input = torch.randn(1, 3,360, 640) # Image with 3 channels, 224x224 pixels

    torch.onnx.export(
        model, 
        dummy_input, 
        model_path, 
        opset_version=11,
        input_names=["input"], 
        output_names=["output"]
    )

    print(f"Model exported to {model_path} successfully!")


if __name__ == "__main__":
    onnx_model_path = "./deeplabv3.onnx"
    export_deeplabv3_to_onnx(onnx_model_path)