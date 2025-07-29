import numpy as np
import cv2
import torch
from PIL import Image
from anomalib.data import Folder
from anomalib.engine import Engine

def infer_and_visualize(engine, model, image_path, data_root, output_path, mode="heatmap"):
    """
    用engine.test完成单张图片推理，生成热力图或边缘图，保存到output_path。
    Args:
        engine: 已实例化的Anomalib Engine
        model: 已加载权重的Patchcore模型
        image_path: 目标图片路径
        data_root: 用于构造datamodule的根目录
        output_path: 临时结果图保存路径
        mode: "heatmap" or "edge"
    Returns:
        output_path
    """
    # 构造单张图片的datamodule
    datamodule = Folder(
        name="single_image",
        root=data_root,
        normal_dir=None,
        abnormal_dir=None,
        predict_dir=image_path,
        eval_batch_size=1,
        num_workers=0,
    )
    with torch.no_grad():
        results = engine.test(datamodule=datamodule, model=model)
    # results结构可能为[{...}], 或直接dict
    if isinstance(results, list):
        result = results[0]
    else:
        result = results
    anomaly_map = result.get("anomaly_map")
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    if anomaly_map is None:
        img.save(output_path)
        return output_path
    map_norm = ((anomaly_map - anomaly_map.min()) / max(anomaly_map.ptp(), 1e-6) * 255).astype(np.uint8)
    if mode == "heatmap":
        heatmap = cv2.applyColorMap(map_norm, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        vis_img = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)
    elif mode == "edge":
        _, thresh = cv2.threshold(map_norm, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis_img = img_np.copy()
        cv2.drawContours(vis_img, contours, -1, (0,255,0), 2)
    else:
        vis_img = img_np
    Image.fromarray(vis_img).save(output_path)
    return output_path