import torch
import torchvision.models as models
from ultralyticsplus import YOLO, render_result

model = models.resnet50(weights = True)
modelLeaf = torch.load('best.pt')

#modelLeaf.eval()
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)
dummy_input_text = "GingerLeaf.jpg"

input_names = ["actual_input"]
output_names = ["output"]

torch.onnx.export(model, dummy_input,
                   "plant_detection.onnx",
                   verbose=True, 
                   opset_version=9,
                   input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )
