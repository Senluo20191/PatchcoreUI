from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QSpinBox, QProgressBar, QCheckBox, QButtonGroup, QFrame, QTextEdit, QSizePolicy
)
from PyQt5.QtCore import Qt

def apply_custom_style(widget):
    style = """
    QWidget { background-color: #2e2e2e; color: #1ecbe1; font-size: 14px; }
    QPushButton { background-color: #3a7bd5; color: #ffffff; border-radius: 6px; padding: 6px 12px; }
    QPushButton:disabled { background-color: #555; color: #e0e0e0; }
    QLabel[sectionTitle="true"] { font-size: 18px; font-weight: bold; color: #39e179; }
    QLabel[dimmed="true"] { color: #e0e0e0; }
    QLabel { color: #39e179; background-color: #232323; border: 1px solid #555; border-radius: 4px; }
    QProgressBar { background-color: #333; color: #1ecbe1; border-radius: 4px; }
    QProgressBar::chunk { background-color: #39e179; }
    QLineEdit, QSpinBox { background-color: #232323; color: #1ecbe1; border-radius: 4px; border: 1px solid #555; }
    QCheckBox { color: #39e179; }
    QTextEdit { background-color: #232323; color: #39e179; border: 1px solid #555; border-radius: 4px; }
    QFrame[line="true"] { background-color: #555; min-width: 2px; max-width: 2px; }
    """
    widget.setStyleSheet(style)

class TrainArea(QWidget):
    def __init__(self):
        super().__init__()
        self.label_title = QLabel("训练区")
        self.label_title.setProperty("sectionTitle", True)
        self.edit_name = QLineEdit()
        self.btn_root = QPushButton("选择数据根目录")
        self.label_root = QLabel("未选择")
        self.spin_train_batch = QSpinBox()
        self.spin_train_batch.setRange(1, 128)
        self.spin_train_batch.setValue(16)
        self.spin_eval_batch = QSpinBox()
        self.spin_eval_batch.setRange(1, 128)
        self.spin_eval_batch.setValue(16)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(2)
        self.spin_epoch = QSpinBox()
        self.spin_epoch.setRange(1, 100)
        self.spin_epoch.setValue(10)
        self.btn_train = QPushButton("开始训练")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_layout()

    def _setup_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.label_title)
        hbox_name = QHBoxLayout()
        hbox_name.addWidget(QLabel("工程名:"))
        hbox_name.addWidget(self.edit_name)
        layout.addLayout(hbox_name)
        hbox_root = QHBoxLayout()
        hbox_root.addWidget(self.btn_root)
        hbox_root.addWidget(self.label_root)
        layout.addLayout(hbox_root)
        hbox_batch = QHBoxLayout()
        hbox_batch.addWidget(QLabel("train_batch_size:"))
        hbox_batch.addWidget(self.spin_train_batch)
        hbox_batch.addWidget(QLabel("eval_batch_size:"))
        hbox_batch.addWidget(self.spin_eval_batch)
        layout.addLayout(hbox_batch)
        hbox_workers = QHBoxLayout()
        hbox_workers.addWidget(QLabel("num_workers:"))
        hbox_workers.addWidget(self.spin_workers)
        layout.addLayout(hbox_workers)
        hbox_epoch = QHBoxLayout()
        hbox_epoch.addWidget(QLabel("num_epochs:"))
        hbox_epoch.addWidget(self.spin_epoch)
        layout.addLayout(hbox_epoch)
        layout.addWidget(self.btn_train)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("控制台输出："))
        layout.addWidget(self.log_text, stretch=1)
        layout.addStretch()
        self.setLayout(layout)

class ValidateArea(QWidget):
    def __init__(self):
        super().__init__()
        self.label_title = QLabel("测试区")
        self.label_title.setProperty("sectionTitle", True)
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
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_layout()

    def _setup_layout(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.label_title)
        hbox_btns = QHBoxLayout()
        hbox_btns.addWidget(self.btn_select_model)
        hbox_btns.addWidget(self.btn_select_image)
        vbox.addLayout(hbox_btns)
        vbox.addWidget(self.label_model_path)
        vbox.addWidget(self.label_image_dir)
        vbox.addWidget(self.label_image, stretch=2)
        hbox_checkbox = QHBoxLayout()
        hbox_checkbox.addWidget(self.checkbox_original)
        hbox_checkbox.addWidget(self.checkbox_result)
        vbox.addLayout(hbox_checkbox)
        hbox_nav = QHBoxLayout()
        hbox_nav.addWidget(self.btn_prev)
        hbox_nav.addWidget(self.btn_next)
        vbox.addLayout(hbox_nav)
        hbox_project = QHBoxLayout()
        hbox_project.addWidget(self.btn_save_project)
        hbox_project.addWidget(self.btn_load_project)
        vbox.addLayout(hbox_project)
        vbox.addWidget(self.progress_bar)
        vbox.addWidget(QLabel("控制台输出："))
        vbox.addWidget(self.log_text, stretch=1)
        vbox.addStretch()
        self.setLayout(vbox)

class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore训练与验证")
        self.setMinimumSize(1600, 900)
        self.resize(1600, 900)
        self.train_area = TrainArea()
        self.validate_area = ValidateArea()
        self.line = QFrame()
        self.line.setFrameShape(QFrame.VLine)
        self.line.setProperty("line", True)
        hbox = QHBoxLayout()
        hbox.addWidget(self.train_area, stretch=1)
        hbox.addWidget(self.line)
        hbox.addWidget(self.validate_area, stretch=1)
        self.setLayout(hbox)
        apply_custom_style(self)