import sys
import os
import threading
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QSpinBox, QMessageBox, QProgressBar,
    QTextEdit, QCheckBox, QButtonGroup, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QFont

import torch
from anomalib.models import Patchcore

##########################################
# 全局样式设定
##########################################
def apply_custom_style(widget):
    """
    设置主窗口和控件的灰底、蓝绿字体、亮灰白禁用字体、分割线等美化样式
    """
    style = """
    QWidget {
        background-color: #2e2e2e;
        color: #1ecbe1;
        font-family: 'Microsoft YaHei', 'Arial', sans-serif;
        font-size: 14px;
    }
    QPushButton {
        background-color: #3a7bd5;
        color: #ffffff;
        border-radius: 6px;
        padding: 6px 12px;
    }
    QPushButton:disabled {
        background-color: #555;
        color: #e0e0e0;
    }
    QLabel[sectionTitle="true"] {
        font-size: 18px;
        font-weight: bold;
        color: #39e179;
        margin-bottom: 8px;
    }
    QLabel {
        color: #39e179;
        background-color: #232323;
        border: 1px solid #555;
        border-radius: 4px;
    }
    QLabel[dimmed="true"] {
        color: #e0e0e0;
    }
    QProgressBar {
        background-color: #333;
        color: #1ecbe1;
        border-radius: 4px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #39e179;
    }
    QLineEdit, QSpinBox {
        background-color: #232323;
        color: #1ecbe1;
        border-radius: 4px;
        border: 1px solid #555;
    }
    QCheckBox {
        color: #39e179;
    }
    QTextEdit {
        background-color: #232323;
        color: #39e179;
        border: 1px solid #555;
        border-radius: 4px;
    }
    QFrame[line="true"] {
        background-color: #555;
        min-width: 2px;
        max-width: 2px;
    }
    """
    widget.setStyleSheet(style)

##########################################
# 检测区后台线程（检测单张图片）
##########################################
class DetectWorker(QObject):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict) # 检测完成，返回{"orig": 原图路径, "result": 结果图路径}

    def __init__(self, model, img_path, result_dir):
        super().__init__()
        self.model = model
        self.img_path = img_path
        self.result_dir = result_dir

    def run(self):
        """
        检测逻辑（可接入你的模型推理代码）
        当前仅复制原图作为结果图，建议替换为Patchcore推理和可视化代码
        """
        import time
        result_img_path = ""
        try:
            import shutil
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            result_img_path = os.path.join(
                self.result_dir,
                f"result_{os.path.basename(self.img_path)}"
            )
            shutil.copy(self.img_path, result_img_path)  # 演示：复制原图到结果目录
            time.sleep(0.2)  # 模拟检测耗时
            self.finished_signal.emit({"orig": self.img_path, "result": result_img_path})
        except Exception as e:
            self.finished_signal.emit({"orig": self.img_path, "result": ""})

##########################################
# 检测区UI
##########################################
class ModelValidationArea(QWidget):
    """
    验证区：用于加载模型和图片，自动检测与结果展示。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.connect_signals()
        self.model = None
        self.model_path = None
        self.image_paths = []
        self.result_images = []
        self.current_index = 0
        self.result_dir = os.path.join(os.getcwd(), "detect_results")
        self.detect_thread = None
        self.worker = None

    def init_ui(self):
        # 区域标题
        self.label_title = QLabel("测试区")
        self.label_title.setProperty("sectionTitle", True)
        # 控件初始化
        self.btn_select_model = QPushButton('选择模型(.pt)')
        self.btn_select_image = QPushButton('选择图像/文件夹')
        self.btn_prev = QPushButton('上一张')
        self.btn_next = QPushButton('下一张')
        self.btn_save_project = QPushButton('保存工程')
        self.btn_load_project = QPushButton('读取工程')
        self.label_image = QLabel('效果图展示区')
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.checkbox_original = QCheckBox('显示原图')
        self.checkbox_result = QCheckBox('显示结果图')
        self.checkbox_group = QButtonGroup()
        self.checkbox_group.setExclusive(True)
        self.checkbox_group.addButton(self.checkbox_original)
        self.checkbox_group.addButton(self.checkbox_result)
        self.checkbox_original.setChecked(True)
        self.label_model_path = QLabel("模型路径: 未选择")
        self.label_image_dir = QLabel("图片路径: 未选择")
        # 控制台输出
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)

        # 布局
        hbox_btns = QHBoxLayout()
        hbox_btns.addWidget(self.btn_select_model)
        hbox_btns.addWidget(self.btn_select_image)

        hbox_nav = QHBoxLayout()
        hbox_nav.addWidget(self.btn_prev)
        hbox_nav.addWidget(self.btn_next)

        hbox_checkbox = QHBoxLayout()
        hbox_checkbox.addWidget(self.checkbox_original)
        hbox_checkbox.addWidget(self.checkbox_result)

        hbox_project = QHBoxLayout()
        hbox_project.addWidget(self.btn_save_project)
        hbox_project.addWidget(self.btn_load_project)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label_title)
        vbox.addLayout(hbox_btns)
        vbox.addWidget(self.label_model_path)
        vbox.addWidget(self.label_image_dir)
        vbox.addWidget(self.label_image, stretch=2)
        vbox.addLayout(hbox_checkbox)
        vbox.addLayout(hbox_nav)
        vbox.addLayout(hbox_project)
        vbox.addWidget(self.progress_bar)
        vbox.addWidget(QLabel("控制台输出："))
        vbox.addWidget(self.log_text, stretch=1)
        vbox.addStretch()
        self.setLayout(vbox)

    def connect_signals(self):
        self.btn_select_model.clicked.connect(self.select_model)
        self.btn_select_image.clicked.connect(self.select_image_or_folder)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.checkbox_original.clicked.connect(self.show_image)
        self.checkbox_result.clicked.connect(self.show_image)
        self.btn_save_project.clicked.connect(self.save_project)
        self.btn_load_project.clicked.connect(self.load_project)

    def set_dimmed(self, dimmed: bool):
        """
        设置禁用/启用样式（亮灰白色/正常色）
        """
        for widget in [
            self.btn_select_model, self.btn_select_image,
            self.btn_prev, self.btn_next,
            self.btn_save_project, self.btn_load_project,
            self.checkbox_original, self.checkbox_result
        ]:
            widget.setEnabled(not dimmed)
        self.label_title.setProperty("dimmed", dimmed)
        self.label_title.style().unpolish(self.label_title)
        self.label_title.style().polish(self.label_title)

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择模型文件', '', 'Model Files (*.pt)')
        if path:
            self.model_path = path
            self.label_model_path.setText(f"模型路径: {path}")
            try:
                self.model = Patchcore()
                self.model.load_state_dict(torch.load(path, map_location='cpu'))
                self.model.eval()
                self.log_text.append(f"模型已成功加载：{os.path.basename(path)}")
                QMessageBox.information(self, "模型加载", f"模型已成功加载：{os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "模型加载错误", str(e))

    def select_image_or_folder(self):
        if not self.model:
            QMessageBox.warning(self, "未加载模型", "请先加载模型文件！")
            return
        reply = QMessageBox.question(
            self, "选择模式", "是否选择文件夹？（否则选择单张图像）",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            folder = QFileDialog.getExistingDirectory(self, '选择图片文件夹')
            if folder:
                self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                self.label_image_dir.setText(f"图片路径: {folder}")
        else:
            path, _ = QFileDialog.getOpenFileName(self, '选择单张图片', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
            if path:
                self.image_paths = [path]
                self.label_image_dir.setText(f"图片路径: {os.path.dirname(path)}")
        self.result_images = ["" for _ in self.image_paths]
        self.current_index = 0
        if self.image_paths:
            self.run_detection(self.current_index)

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.run_detection(self.current_index)

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.run_detection(self.current_index)

    def run_detection(self, idx):
        img_path = self.image_paths[idx]
        self.worker = DetectWorker(self.model, img_path, self.result_dir)
        self.worker.log_signal.connect(self.log_text.append)
        self.worker.finished_signal.connect(self.on_detect_finished)
        self.detect_thread = threading.Thread(target=self.worker.run)
        self.detect_thread.start()
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setFormat("检测进行中...")

    def on_detect_finished(self, result_dict):
        idx = self.current_index
        orig_path = result_dict.get("orig", "")
        result_path = result_dict.get("result", "")
        if result_path:
            self.result_images[idx] = result_path
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("检测完成")
        self.show_image()

    def show_image(self):
        if not self.image_paths:
            self.label_image.setText('未选择图片')
            return
        show_result = self.checkbox_result.isChecked()
        img_path = self.result_images[self.current_index] if show_result and self.result_images[self.current_index] else self.image_paths[self.current_index]
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            self.label_image.setPixmap(pixmap.scaled(
                self.label_image.size(), aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation
            ))
        else:
            self.label_image.setText('图片无法加载')

    def save_project(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "保存工程", "", "工程文件 (*.json)")
        if save_path:
            project_data = {
                "model_path": self.model_path,
                "image_paths": self.image_paths,
                "result_images": self.result_images,
                "result_dir": self.result_dir,
                "current_index": self.current_index,
            }
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "保存工程", "已保存工程文件！")

    def load_project(self):
        load_path, _ = QFileDialog.getOpenFileName(self, "读取工程", "", "工程文件 (*.json)")
        if load_path and os.path.exists(load_path):
            with open(load_path, "r", encoding="utf-8") as f:
                project_data = json.load(f)
            self.model_path = project_data.get("model_path", "")
            self.image_paths = project_data.get("image_paths", [])
            self.result_images = project_data.get("result_images", [])
            self.result_dir = project_data.get("result_dir", os.path.join(os.getcwd(), "detect_results"))
            self.current_index = project_data.get("current_index", 0)
            self.label_model_path.setText(f"模型路径: {self.model_path}")
            if self.image_paths:
                self.label_image_dir.setText(f"图片路径: {os.path.dirname(self.image_paths[0])}")
            else:
                self.label_image_dir.setText("图片路径: 未选择")
            self.show_image()
            QMessageBox.information(self, "读取工程", "已读取工程文件！")

##########################################
# 训练区后台线程
##########################################
class TrainWorker(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(self, train_args):
        super().__init__()
        self.train_args = train_args

    def run(self):
        class EmittingStream:
            def __init__(self, text_write_func):
                self.text_write_func = text_write_func
            def write(self, text):
                self.text_write_func(str(text))
            def flush(self):
                pass
        sys.stdout = EmittingStream(self.log_signal.emit)
        sys.stderr = EmittingStream(self.log_signal.emit)
        try:
            import time
            self.log_signal.emit("训练开始...\n")
            name = self.train_args.get("name", "default_exp")
            root = self.train_args.get("root", os.getcwd())
            train_batch_size = self.train_args.get("train_batch_size", 16)
            eval_batch_size = self.train_args.get("eval_batch_size", 16)
            num_workers = self.train_args.get("num_workers", 2)
            num_epochs = self.train_args.get("num_epochs", 10)
            exp_dir = os.path.join(os.getcwd(), name)
            model_dir = os.path.join(exp_dir, 'model')
            os.makedirs(model_dir, exist_ok=True)
            model_pt_path = os.path.join(model_dir, "patchcore_last.pt")
            self.log_signal.emit(f"实验目录: {exp_dir}\n")
            self.log_signal.emit(f"模型保存路径: {model_pt_path}\n")
            # Patchcore训练流程（建议替换为你自己的训练代码）
            from anomalib.data import Folder
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
            from anomalib.engine import Engine
            engine = Engine(
                default_root_dir=exp_dir,
                callbacks=None,
                devices="1" if torch.cuda.is_available() else "cpu",
                max_epochs=num_epochs
            )
            time1 = time.time()
            for epoch in range(1, num_epochs + 1):
                self.log_signal.emit(f"Epoch {epoch}/{num_epochs}...\n")
                engine.train(datamodule=datamodule, model=model)
                progress = int(epoch / num_epochs * 100)
                self.progress_signal.emit(progress)
                time.sleep(0.5)
            time2 = time.time()
            self.log_signal.emit(f"训练总耗时: {time2 - time1:.2f}秒\n")
            torch.save(model.state_dict(), model_pt_path)
            self.log_signal.emit("模型已保存。\n")
            self.progress_signal.emit(100)
            self.finished_signal.emit({"model_pt": model_pt_path})
        except Exception as e:
            self.log_signal.emit(f"训练发生异常: {e}\n")
            self.finished_signal.emit({})
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

##########################################
# 训练区UI
##########################################
class TrainArea(QWidget):
    """
    训练区：用于输入训练参数、启动训练、显示日志与进度
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.train_thread = None
        self.worker = None

    def initUI(self):
        # 区域标题
        self.label_title = QLabel("训练区")
        self.label_title.setProperty("sectionTitle", True)
        # 控件布局
        layout = QVBoxLayout()
        hbox_name = QHBoxLayout()
        hbox_name.addWidget(QLabel("工程名:"))
        self.edit_name = QLineEdit("")
        hbox_name.addWidget(self.edit_name)
        hbox_root = QHBoxLayout()
        self.btn_root = QPushButton("选择数据根目录")
        self.btn_root.clicked.connect(self.select_root)
        self.label_root = QLabel("未选择")
        hbox_root.addWidget(self.btn_root)
        hbox_root.addWidget(self.label_root)
        hbox_batch = QHBoxLayout()
        hbox_batch.addWidget(QLabel("train_batch_size:"))
        self.spin_train_batch = QSpinBox()
        self.spin_train_batch.setRange(1, 128)
        self.spin_train_batch.setValue(16)
        hbox_batch.addWidget(self.spin_train_batch)
        hbox_batch.addWidget(QLabel("eval_batch_size:"))
        self.spin_eval_batch = QSpinBox()
        self.spin_eval_batch.setRange(1, 128)
        self.spin_eval_batch.setValue(16)
        hbox_batch.addWidget(self.spin_eval_batch)
        hbox_workers = QHBoxLayout()
        hbox_workers.addWidget(QLabel("num_workers:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(2)
        hbox_workers.addWidget(self.spin_workers)
        hbox_epoch = QHBoxLayout()
        hbox_epoch.addWidget(QLabel("num_epochs:"))
        self.spin_epoch = QSpinBox()
        self.spin_epoch.setRange(1, 100)
        self.spin_epoch.setValue(10)
        hbox_epoch.addWidget(self.spin_epoch)
        self.btn_train = QPushButton("开始训练")
        self.btn_train.clicked.connect(self.start_train)
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        # 控制台输出
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 布局顺序调整：进度条在前，日志填满余下空间
        layout.addWidget(self.label_title)
        layout.addLayout(hbox_name)
        layout.addLayout(hbox_root)
        layout.addLayout(hbox_batch)
        layout.addLayout(hbox_workers)
        layout.addLayout(hbox_epoch)
        layout.addWidget(self.btn_train)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("控制台输出："))
        layout.addWidget(self.log_text, stretch=1)
        layout.addStretch()
        self.setLayout(layout)

    def set_dimmed(self, dimmed: bool):
        """
        设置禁用/启用样式（亮灰白色/正常色）
        """
        for widget in [
            self.btn_root, self.btn_train, self.edit_name, self.spin_train_batch,
            self.spin_eval_batch, self.spin_workers, self.spin_epoch
        ]:
            widget.setEnabled(not dimmed)
        self.label_title.setProperty("dimmed", dimmed)
        self.label_title.style().unpolish(self.label_title)
        self.label_title.style().polish(self.label_title)

    def select_root(self):
        root = QFileDialog.getExistingDirectory(self, "选择数据根目录")
        if root:
            self.label_root.setText(root)

    def start_train(self):
        root = self.label_root.text()
        name = self.edit_name.text()
        train_batch_size = self.spin_train_batch.value()
        eval_batch_size = self.spin_eval_batch.value()
        num_workers = self.spin_workers.value()
        num_epochs = self.spin_epoch.value()
        train_args = {
            "name": name,
            "root": root,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "num_workers": num_workers,
            "num_epochs": num_epochs
        }
        self.worker = TrainWorker(train_args)
        self.worker.log_signal.connect(self.log_text.append)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.on_train_finished)
        self.train_thread = threading.Thread(target=self.worker.run)
        self.train_thread.start()
        self.btn_train.setEnabled(False)
        parent = self.parentWidget()
        if parent and hasattr(parent, "validate_area"):
            parent.validate_area.set_dimmed(True)
            self.set_dimmed(False)

    def on_train_finished(self, result):
        self.btn_train.setEnabled(True)
        pt_path = result.get("model_pt", "未知路径")
        QMessageBox.information(self, "训练结束", f"模型已保存至: {pt_path}")
        parent = self.parentWidget()
        if parent and hasattr(parent, "validate_area"):
            parent.validate_area.set_dimmed(False)
            self.set_dimmed(False)

##########################################
# 主窗口：左训练、分割线、右验证
##########################################
class MainUI(QWidget):
    """
    主窗口，左侧训练，右侧验证，一体化界面管理。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore训练与验证")
        self.setMinimumSize(1600, 900)
        self.resize(1600, 900)
        hbox = QHBoxLayout()
        self.train_area = TrainArea(self)
        self.validate_area = ModelValidationArea(self)
        self.train_area.set_dimmed(False)
        self.validate_area.set_dimmed(False)
        # 分割线
        self.line = QFrame()
        self.line.setFrameShape(QFrame.VLine)
        self.line.setProperty("line", True)
        hbox.addWidget(self.train_area, stretch=1)
        hbox.addWidget(self.line)
        hbox.addWidget(self.validate_area, stretch=1)
        self.setLayout(hbox)
        apply_custom_style(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainUI()
    win.show()
    sys.exit(app.exec_())