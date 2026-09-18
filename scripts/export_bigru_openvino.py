# scripts/export_bigru_openvino.py
import torch
from src.model.bigru_model import AffectiveBiGRU

model = AffectiveBiGRU(input_size=41, hidden_size=64, num_classes=3)
model.load_state_dict(torch.load("data/bigru_affective.pth", map_location="cpu"))
model.eval()

dummy_input = torch.zeros(1, 30, 41)  # (batch=1, seq_len=30, features=41)
torch.onnx.export(
    model, dummy_input,
    "data/bigru_affective.onnx",
    input_names=["temporal_features"],
    output_names=["class_logits"],
    dynamic_axes={"temporal_features": {0: "batch_size"}},
    opset_version=17
)

# Convert ONNX → OpenVINO IR
# mo --input_model data/bigru_affective.onnx --output_dir data/bigru_openvino/