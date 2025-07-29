import torch
import os
from anomalib.models import Patchcore


def pt_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224)):
    model = Patchcore()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    return onnx_path
