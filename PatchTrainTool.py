import sys
import os
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QSpinBox, QMessageBox, QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

import train

class TrainWorker(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(self, train_args):
        super().__init__()
        self.train_args = train_args

    def run(self):
        # 重定向stdout到信号
        class EmittingStream:
            def __init__(self, text_write_func):
                self.text_write_func = text_write_func
            def write(self, text):
                self.text_write_func(str(text))
            def flush(self):
                pass
        sys.stdout = EmittingStream(self.log_signal.emit)
        sys.stderr = EmittingStream(self.log_signal.emit)

        # 训练过程
        try:
            result = train.train_patchcore(progress_callback=self.progress_signal.emit, **self.train_args)
            self.finished_signal.emit(result if result else {})
        except Exception as e:
            self.log_signal.emit(f"训练发生异常: {e}\n")
            self.finished_signal.emit({})
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore Training UI")
        self.setGeometry(300, 200, 700, 500)

        # 参数
        self.root = ""
        self.name = "NPL_0703_900"
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.num_workers = 2
        self.num_epochs = 10

        self.initUI()
        self.apply_custom_style()
        self.train_thread = None
        self.worker = None

    def initUI(self):
        layout = QVBoxLayout()

        # 工程名
        hbox_name = QHBoxLayout()
        hbox_name.addWidget(QLabel("工程名:"))
        self.edit_name = QLineEdit(self.name)
        hbox_name.addWidget(self.edit_name)

        # 数据根目录(root)
        hbox_root = QHBoxLayout()
        self.btn_root = QPushButton("选择数据根目录")
        self.btn_root.clicked.connect(self.select_root)
        self.label_root = QLabel("未选择")
        hbox_root.addWidget(self.btn_root)
        hbox_root.addWidget(self.label_root)

        # batch size参数
        hbox_batch = QHBoxLayout()
        hbox_batch.addWidget(QLabel("train_batch_size:"))
        self.spin_train_batch = QSpinBox()
        self.spin_train_batch.setRange(1, 128)
        self.spin_train_batch.setValue(self.train_batch_size)
        hbox_batch.addWidget(self.spin_train_batch)
        hbox_batch.addWidget(QLabel("eval_batch_size:"))
        self.spin_eval_batch = QSpinBox()
        self.spin_eval_batch.setRange(1, 128)
        self.spin_eval_batch.setValue(self.eval_batch_size)
        hbox_batch.addWidget(self.spin_eval_batch)

        # num_workers参数
        hbox_workers = QHBoxLayout()
        hbox_workers.addWidget(QLabel("num_workers:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(self.num_workers)
        hbox_workers.addWidget(self.spin_workers)

        # 训练批次
        hbox_epoch = QHBoxLayout()
        hbox_epoch.addWidget(QLabel("训练轮数:"))
        self.spin_epoch = QSpinBox()
        self.spin_epoch.setRange(1, 100)
        self.spin_epoch.setValue(self.num_epochs)
        hbox_epoch.addWidget(self.spin_epoch)

        # 训练按钮
        self.btn_train = QPushButton("开始训练")
        self.btn_train.clicked.connect(self.start_training)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        # 控制台日志窗口
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setFixedHeight(200)

        # 加入主布局
        layout.addLayout(hbox_name)
        layout.addLayout(hbox_root)
        layout.addLayout(hbox_batch)
        layout.addLayout(hbox_workers)
        layout.addLayout(hbox_epoch)
        layout.addWidget(self.btn_train)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.text_log)
        self.setLayout(layout)

    def apply_custom_style(self):
        # （可选：美化样式，可按需简化）
        from PyQt5.QtGui import QColor, QPalette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#23272E"))
        palette.setColor(QPalette.WindowText, QColor("#D3D3D3"))
        palette.setColor(QPalette.Button, QColor("#393E46"))
        palette.setColor(QPalette.ButtonText, QColor("#D3D3D3"))
        palette.setColor(QPalette.Base, QColor("#23272E"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setStyleSheet("""
            QWidget { font-family: 'Segoe UI', 'Microsoft YaHei', Arial; color: #D3D3D3; background: #23272E; }
            QLabel { color: #D3D3D3; font-size: 14px; }
            QLineEdit { background: #1C1F23; color: #D3D3D3; border: 1px solid #5A96B8; border-radius: 5px; padding: 4px; }
            QPushButton { background-color: #393E46; color: #D3D3D3; border: 1px solid #5A96B8; border-radius: 7px; padding: 6px 18px; font-weight: bold; font-size: 15px; }
            QPushButton:hover { color: #27D2C9; background: #323946; }
            QPushButton:pressed { color: #27D2C9; background: #232F3E; }
            QSpinBox { background: #1C1F23; color: #D3D3D3; border: 1px solid #5A96B8; border-radius: 5px; padding: 2px; }
        """)

    def select_root(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据根目录")
        if dir_path:
            self.root = dir_path
            self.label_root.setText(dir_path)

    def start_training(self):
        self.name = self.edit_name.text().strip() or "NPL_0703_900"
        self.train_batch_size = self.spin_train_batch.value()
        self.eval_batch_size = self.spin_eval_batch.value()
        self.num_workers = self.spin_workers.value()
        self.num_epochs = self.spin_epoch.value()

        if not self.root:
            QMessageBox.warning(self, "错误", "请先选择数据根目录")
            return

        # 禁用控件
        self.set_all_controls_enabled(False)
        self.btn_train.setText("训练中...")

        # 清空进度和日志
        self.progress_bar.setValue(0)
        self.text_log.clear()

        args = dict(
            name=self.name,
            root=self.root,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            num_epochs=self.num_epochs,
            save_root=os.getcwd()
        )
        self.worker = TrainWorker(args)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.training_finished)

        self.train_thread = threading.Thread(target=self.worker.run)
        self.train_thread.start()

    def append_log(self, text):
        self.text_log.moveCursor(self.text_log.textCursor().End)
        self.text_log.insertPlainText(text)
        self.text_log.moveCursor(self.text_log.textCursor().End)

    def set_all_controls_enabled(self, enabled):
        for widget in [self.edit_name, self.btn_root, self.spin_train_batch, self.spin_eval_batch, self.spin_workers, self.spin_epoch]:
            widget.setEnabled(enabled)
        self.btn_train.setEnabled(enabled)

    def training_finished(self, result):
        self.set_all_controls_enabled(True)
        self.btn_train.setText("开始训练")
        if result and "model_pt" in result:
            QMessageBox.information(self, "训练完成", f"模型已保存为: {result['model_pt']}")
        else:
            QMessageBox.warning(self, "训练失败", "训练过程出现异常或未生成模型文件。")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())