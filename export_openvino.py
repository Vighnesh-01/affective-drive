from ultralytics import YOLO

# 1. Load your custom weights from the project folder
model = YOLO("best.pt")

# 2. Export to OpenVINO INT8 format
exported_path = model.export(
    format="openvino",
    quantize="int8",          # Use 'int8' or 8 for INT8 quantization
    imgsz=640,
    data="path/to/data.yaml"  # Path to your dataset's data.yaml for INT8 calibration
)

print(f"\n✅ Model converted successfully! Saved at: {exported_path}")