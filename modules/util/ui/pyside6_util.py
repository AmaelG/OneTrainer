import signal
import sys
from abc import ABCMeta

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget


class QtABCMeta(type(QWidget), ABCMeta):
    # Combined metaclass that resolves the conflict between Qt's Shiboken metaclass and ABCMeta.
    pass


def _apply_light_theme(app: QApplication) -> None:
    # The PySide6 UI is not dark-theme complete yet. Force a known light
    # baseline instead of inheriting the desktop theme, which can produce dark
    # backgrounds with light-only widget styling and unreadable controls.
    app.setStyle("Fusion")
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#f6f6f6"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#202020"))
    palette.setColor(QPalette.ColorRole.Base, QColor("white"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f0f0f0"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("white"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#202020"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#202020"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#f0f0f0"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#202020"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#c00000"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2a82da"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#777777"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#777777"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#777777"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#e9e9e9"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor("#e9e9e9"))
    app.setPalette(palette)

    app.setStyleSheet("""
        QWidget {
            background-color: #f6f6f6;
            color: #202020;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
            background-color: white;
            color: #202020;
            border: 1px solid #adadad;
            padding: 2px 2px;
            selection-background-color: #2a82da;
            selection-color: white;
        }
        QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
        QTextEdit:disabled, QPlainTextEdit:disabled {
            background-color: #e9e9e9;
            color: #777777;
        }
        QPushButton, QComboBox {
            background-color: #f0f0f0;
            color: #202020;
            border: 1px solid #adadad;
            border-radius: 2px;
            padding: 2px 6px;
        }
        QPushButton:disabled, QComboBox:disabled {
            background-color: #e9e9e9;
            color: #777777;
        }
        QComboBox QAbstractItemView {
            background-color: white;
            color: #202020;
            selection-background-color: #2a82da;
            selection-color: white;
        }
        QTabWidget::pane, QGroupBox, QFrame, QScrollArea {
            background-color: #f6f6f6;
        }
        QTabBar::tab {
            background-color: #e9e9e9;
            color: #202020;
            border: 1px solid #adadad;
            padding: 3px 10px;
        }
        QTabBar::tab:selected {
            background-color: white;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            background-color: white;
            border: 1px solid #adadad;
        }
        QCheckBox::indicator:checked {
            background-color: #2a82da;
        }
        QProgressBar {
            background-color: #e0e0e0;
            color: #202020;
            border: 1px solid #adadad;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #c8c8c8;
        }
    """)


def create_application() -> QApplication:
    # Restore the OS default SIGINT handler so Ctrl+C terminates the process
    # directly at the C level. Qt's event loop blocks inside C++, so Python's
    # own SIGINT handler would never get a chance to run while app.exec() is
    # active and Ctrl+C would be ignored.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Prevent Qt from seeding the app with the system dark palette before the
    # explicit light theme is applied.
    QApplication.setDesktopSettingsAware(False)
    app = QApplication(sys.argv)
    _apply_light_theme(app)

    return app
