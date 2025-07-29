import os
import threading
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal

class TrainWorker(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(self, train_args):
        super().__init__()
        self.train_args = train_args

    def run(self):
        import time
        try:
            name = self.train_args.get("name", "default_exp")
            root = self.train_args.get("root", os.getcwd())
            train_batch_size = self.train_args.get("train_batch_size", 16)
            eval_batch_size = self.train_args.get("eval_batch_size", 16)
            num_workers = self.train_args.get("num_workers", 2)
            num_epochs = self.train_args.get("num_epochs", 10)
            base_dir = os.path.abspath(os.getcwd())
            exp_dir = os.path.join(base_dir, name)
            model_dir = os.path.join(exp_dir, 'model')
            os.makedirs(model_dir, exist_ok=True)
            model_pt_path = os.path.join(model_dir, "patchcore_last.ckpt")
            self.log_signal.emit(f"实验目录: {exp_dir}\n")
            self.log_signal.emit(f"模型保存路径: {model_pt_path}\n")
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
            from anomalib.models import Patchcore
            model = Patchcore(pre_trained=True)
            from anomalib.engine import Engine
            engine = Engine(
                default_root_dir=exp_dir,
                callbacks=None,
                devices="1",
                max_epochs=num_epochs
            )
            time1 = time.time()
            engine.train(datamodule=datamodule, model=model)
            time2 = time.time()
            self.log_signal.emit(f"训练总耗时: {time2 - time1:.2f}秒\n")
            import torch
            torch.save(model.state_dict(), model_pt_path)
            self.log_signal.emit("模型已保存。\n")
            self.progress_signal.emit(100)
            self.finished_signal.emit({"model_pt": model_pt_path})
        except Exception as e:
            self.log_signal.emit(f"训练发生异常: {e}\n")
            self.finished_signal.emit({})

def bind_train_events(ui, controller=None):
    def update_enabled():
        name_ok = bool(ui.edit_name.text().strip())
        ui.btn_root.setEnabled(name_ok)
        ui.label_root.setProperty("dimmed", not name_ok)
        ui.label_root.style().unpolish(ui.label_root)
        ui.label_root.style().polish(ui.label_root)
        root_ok = ui.label_root.text() != "未选择"
        ui.btn_train.setEnabled(name_ok and root_ok)
        ui.label_title.setProperty("dimmed", not (name_ok and root_ok))
        ui.label_title.style().unpolish(ui.label_title)
        ui.label_title.style().polish(ui.label_title)

    def select_root():
        root = QFileDialog.getExistingDirectory(ui, "选择数据根目录")
        if root:
            ui.label_root.setText(root)
        update_enabled()

    def start_train():
        name = ui.edit_name.text().strip()
        if not name:
            QMessageBox.warning(ui, "错误", "请先输入工程名")
            return
        root = ui.label_root.text()
        train_args = {
            "name": name,
            "root": root,
            "train_batch_size": ui.spin_train_batch.value(),
            "eval_batch_size": ui.spin_eval_batch.value(),
            "num_workers": ui.spin_workers.value(),
            "num_epochs": ui.spin_epoch.value()
        }
        worker = TrainWorker(train_args)
        worker.log_signal.connect(ui.log_text.append)
        worker.progress_signal.connect(ui.progress_bar.setValue)
        def finished(result):
            pt_path = result.get("model_pt")
            if pt_path:
                QMessageBox.information(ui, "训练结束", f"模型已保存至: {pt_path}")
            else:
                QMessageBox.warning(ui, "训练失败", "训练发生异常")
            update_enabled()
        worker.finished_signal.connect(finished)
        thread = threading.Thread(target=worker.run)
        thread.start()
        ui.btn_train.setEnabled(False)

    ui.edit_name.textChanged.connect(update_enabled)
    ui.btn_root.clicked.connect(select_root)
    ui.btn_train.clicked.connect(start_train)
    update_enabled()