import os
import threading
import torch
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal,Qt
from patchcore_infer import infer_and_visualize
from anomalib.engine import Engine

class DetectWorker(QObject):
    finished_signal = pyqtSignal(dict)
    log_signal = pyqtSignal(str)

    def __init__(self, engine, model, img_path, data_root, result_dir, mode="heatmap"):
        super().__init__()
        self.engine = engine
        self.model = model
        self.img_path = img_path
        self.data_root = data_root
        self.result_dir = result_dir
        self.mode = mode

    def run(self):
        try:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            result_img_path = os.path.join(
                self.result_dir, f"result_{os.path.basename(self.img_path)}"
            )
            infer_and_visualize(self.engine, self.model, self.img_path, self.data_root, result_img_path, mode=self.mode)
            self.finished_signal.emit({"orig": self.img_path, "result": result_img_path})
        except Exception as e:
            self.log_signal.emit(f"检测异常: {e}\n")
            self.finished_signal.emit({"orig": self.img_path, "result": ""})

    def cleanup(self):
        # 释放模型/engine等资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def bind_validate_events(ui, controller=None):
    state = {
        "model": None,
        "engine": None,
        "model_path": None,
        "image_paths": [],
        "result_images": [],
        "current_index": 0,
        "result_dir": os.path.join(os.getcwd(), "detect_results"),
        "data_root": None,
        "mode": "heatmap",
    }

    def update_enabled():
        model_loaded = state["model"] is not None
        ui.btn_select_image.setEnabled(model_loaded)
        ui.label_image_dir.setProperty("dimmed", not model_loaded)
        ui.label_image_dir.style().unpolish(ui.label_image_dir)
        ui.label_image_dir.style().polish(ui.label_image_dir)
        images_loaded = bool(state["image_paths"])
        ui.btn_prev.setEnabled(images_loaded)
        ui.btn_next.setEnabled(images_loaded)
        ui.checkbox_original.setEnabled(images_loaded)
        ui.checkbox_result.setEnabled(images_loaded)
        ui.label_title.setProperty("dimmed", not images_loaded)
        ui.label_title.style().unpolish(ui.label_title)
        ui.label_title.style().polish(ui.label_title)

    def select_model():
        path, _ = QFileDialog.getOpenFileName(ui, '选择模型文件', '', 'Model Files (*.ckpt)')
        if path:
            state["model_path"] = path
            ui.label_model_path.setText(f"模型路径: {path}")
            try:
                from anomalib.models import Patchcore
                model = Patchcore()
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                state["model"] = model
                state["engine"] = Engine()
                QMessageBox.information(ui, "模型加载", f"模型已成功加载：{os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(ui, "模型加载错误", str(e))
        update_enabled()

    def select_image_or_folder():
        if not state["model"]:
            QMessageBox.warning(ui, "未加载模型", "请先加载模型文件！")
            return
        reply = QMessageBox.question(
            ui, "选择模式", "是否选择文件夹？（否则选择单张图像）",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            folder = QFileDialog.getExistingDirectory(ui, '选择图片文件夹')
            if folder:
                paths = [os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                state["image_paths"] = paths
                ui.label_image_dir.setText(f"图片路径: {folder}")
                state["data_root"] = folder  # 以图片文件夹为数据根
        else:
            path, _ = QFileDialog.getOpenFileName(ui, '选择单张图片', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
            if path:
                state["image_paths"] = [path]
                ui.label_image_dir.setText(f"图片路径: {os.path.dirname(path)}")
                state["data_root"] = os.path.dirname(path)
        state["result_images"] = ["" for _ in state["image_paths"]]
        state["current_index"] = 0
        if state["image_paths"]:
            run_detection(state["current_index"])
        update_enabled()

    def run_detection(idx):
        img_path = state["image_paths"][idx]
        show_result = ui.checkbox_result.isChecked()
        mode = "heatmap" if show_result else "edge"
        state["mode"] = mode
        worker = DetectWorker(
            state["engine"], state["model"], img_path, state["data_root"], state["result_dir"], mode=mode
        )
        worker.finished_signal.connect(lambda result: on_detect_finished(result, idx))
        thread = threading.Thread(target=worker.run)
        thread.start()
        ui.btn_prev.setEnabled(False)
        ui.btn_next.setEnabled(False)
        ui.progress_bar.setValue(0)
        ui.progress_bar.setMaximum(100)
        ui.progress_bar.setFormat("检测进行中...")

    def on_detect_finished(result_dict, idx):
        orig_path = result_dict.get("orig", "")
        result_path = result_dict.get("result", "")
        if result_path:
            state["result_images"][idx] = result_path
        ui.btn_prev.setEnabled(True)
        ui.btn_next.setEnabled(True)
        ui.progress_bar.setValue(100)
        ui.progress_bar.setFormat("检测完成")
        show_image()

    def show_image():
        if not state["image_paths"]:
            ui.label_image.setText('未选择图片')
            return
        show_result = ui.checkbox_result.isChecked()
        idx = state["current_index"]
        img_path = state["result_images"][idx] if show_result and state["result_images"][idx] else state["image_paths"][idx]
        from PyQt5.QtGui import QPixmap
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            ui.label_image.setPixmap(pixmap.scaled(
                ui.label_image.size(), aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation
            ))
        else:
            ui.label_image.setText('图片无法加载')

    def prev_image():
        if state["current_index"] > 0:
            state["current_index"] -= 1
            run_detection(state["current_index"])

    def next_image():
        if state["current_index"] < len(state["image_paths"]) - 1:
            state["current_index"] += 1
            run_detection(state["current_index"])

    ui.btn_select_model.clicked.connect(select_model)
    ui.btn_select_image.clicked.connect(select_image_or_folder)
    ui.btn_prev.clicked.connect(prev_image)
    ui.btn_next.clicked.connect(next_image)
    ui.checkbox_original.clicked.connect(show_image)
    ui.checkbox_result.clicked.connect(show_image)
    update_enabled()