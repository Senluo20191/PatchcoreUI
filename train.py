import os
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
import time

def train_patchcore(
    name="NPL_0703_900",
    root="./",
    train_batch_size=16,
    eval_batch_size=16,
    num_workers=2,
    num_epochs=10,
    save_root=os.getcwd(),
    progress_callback=None
):
    exp_dir = os.path.join(save_root, name)
    model_dir = os.path.join(exp_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_pt_path = os.path.join(model_dir, "patchcore_last.pt")
    print(f"实验目录: {exp_dir}")
    print(f"模型保存路径: {model_pt_path}")

    datamodule = Folder(
        name=name,
        root=root,
        normal_dir="train/good",
        abnormal_dir="test",
        mask_dir="group_truth",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    model = Patchcore(pre_trained=True)
    engine = Engine(
        default_root_dir=exp_dir,
        callbacks=None,
        devices="1" if torch.cuda.is_available() else "cpu",
        max_epochs=num_epochs
    )

    time1 = time.time()
    engine.train(datamodule=datamodule, model=model)
    time2 = time.time()
    print(f"训练总耗时: {time2 - time1:.2f}秒")
    torch.save(model.state_dict(), model_pt_path)
    print("模型已保存。")
    return {
        "model_pt": model_pt_path
    }

# if __name__ == "__main__":
#     result = train_patchcore()
#     print(result)