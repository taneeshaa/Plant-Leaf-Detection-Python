import onnx

modelONNX = onnx.load('plant_detection.onnx')

inputs = modelONNX.graph.input

for input in inputs:
  print(input)