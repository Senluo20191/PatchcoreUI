import torch
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
import os
import cv2
import numpy as np
from PIL import Image

def visualize_inference(model_ckpt, image_path):
    """
    单张图片推理并展示结果（热力图叠加），无需保存到文件。
    Args:
        model_ckpt: Patchcore模型权重文件路径
        image_path: 待检测图片文件路径
    """
    # 1. 加载模型
    if not os.path.exists(model_ckpt):
        print(f"模型文件不存在: {model_ckpt}")
        return
    model = Patchcore.load_from_checkpoint(model_ckpt)
    model.eval()

    # 2. 构造单图 datamodule
    img_dir = os.path.dirname(image_path)
    datamodule = Folder(
        name="single_infer",
        root=img_dir,
        normal_dir=None,
        abnormal_dir=None,
        predict_dir=image_path,
        eval_batch_size=1,
        num_workers=0,
    )

    # 3. 推理引擎
    engine = Engine(
        default_root_dir=None,
        callbacks=None,
        devices='1',
        max_epochs=1
    )
    with torch.no_grad():
        results = engine.test(datamodule=datamodule, model=model)
    # 4. 结果解析与可视化
    if isinstance(results, list):
        result = results[0]
    else:
        result = results
    anomaly_map = result.get("anomaly_map")
    if anomaly_map is None:
        print("未获得 anomaly_map，直接展示原图")
        img = Image.open(image_path)
        img.show()
        return

    # 5. 热力图叠加
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    map_norm = ((anomaly_map - anomaly_map.min()) / max(anomaly_map.ptp(), 1e-6) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(map_norm, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    vis_img = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)

    # 6. 显示结果
    vis_pil = Image.fromarray(vis_img)
    vis_pil.show()
    print("推理完成并已展示结果图")

if __name__ == "__main__":
    # 修改为你的模型权重和待检测图片路径
    model_ckpt = "Amb-bottle/model/patchcore_last.pt"
    image_path = "bottle/test/broken_large/000.png"
    visualize_inference(model_ckpt, image_path)