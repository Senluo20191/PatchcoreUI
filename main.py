import sys
from PyQt5.QtWidgets import QApplication
from ui_config import MainUI
import train_controller
import validate_controller

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainUI()
    train_controller.bind_train_events(win.train_area)
    validate_controller.bind_validate_events(win.validate_area)
    win.show()
    sys.exit(app.exec_())