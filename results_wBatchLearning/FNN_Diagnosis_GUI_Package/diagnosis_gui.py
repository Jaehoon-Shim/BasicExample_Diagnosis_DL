import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


CHECKPOINT_NAME = "fnn_fault_checkpoint.pt"


def checkpoint_path() -> Path:
    """외부 PT가 있으면 우선 사용하고, 없으면 EXE 내부 PT를 사용한다."""
    if getattr(sys, "frozen", False):
        external_path = Path(sys.executable).resolve().parent / CHECKPOINT_NAME
        if external_path.exists():
            return external_path
        return Path(getattr(sys, "_MEIPASS")) / CHECKPOINT_NAME
    return Path(__file__).resolve().parent / CHECKPOINT_NAME


class CheckpointFNN(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        layers = []
        for layer_index in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[layer_index], architecture[layer_index + 1]))
            if layer_index < len(architecture) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DiagnosisEngine:
    def __init__(self):
        self.path = checkpoint_path()
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)

        required_keys = {
            "model_state_dict",
            "architecture",
            "x_mean",
            "x_scale",
            "class_ids",
        }
        missing = required_keys.difference(checkpoint)
        if missing:
            raise KeyError(f"체크포인트 필수 항목 누락: {sorted(missing)}")

        self.architecture = [int(v) for v in checkpoint["architecture"]]
        self.input_size = self.architecture[0]
        self.class_ids = np.asarray(checkpoint["class_ids"], dtype=np.int64)
        self.x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
        self.x_scale = np.asarray(checkpoint["x_scale"], dtype=np.float32)

        if self.x_mean.size != self.input_size or self.x_scale.size != self.input_size:
            raise ValueError("PT의 표준화 정보 크기가 입력 크기와 일치하지 않습니다.")
        if np.any(self.x_scale == 0):
            raise ValueError("PT의 x_scale에 0이 포함되어 있습니다.")
        if self.architecture[-1] != self.class_ids.size:
            raise ValueError("PT의 출력 노드 수와 class_ids 개수가 일치하지 않습니다.")

        self.model = CheckpointFNN(self.architecture)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != self.input_size:
            raise ValueError(
                f"입력은 {self.input_size}개여야 합니다. 현재 {values.size}개입니다."
            )

        x_scaled = (values - self.x_mean) / self.x_scale
        x_tensor = torch.from_numpy(x_scaled.reshape(1, -1))

        with torch.inference_mode():
            logits = self.model(x_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_index = int(np.argmax(probabilities))
        predicted_class = int(self.class_ids[predicted_index])
        return predicted_class, probabilities


class DiagnosisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = None
        self.setWindowTitle("FNN 3상 전류 고장 진단")
        self.resize(860, 700)
        self._build_ui()
        self._load_engine()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        title = QLabel("FNN 3상 전류 고장 진단")
        title.setFont(QFont("맑은 고딕", 20, QFont.Bold))
        main_layout.addWidget(title)

        self.model_status = QLabel("모델을 불러오는 중입니다...")
        self.model_status.setStyleSheet("color: #5f6b7a;")
        main_layout.addWidget(self.model_status)

        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.StyledPanel)
        input_layout = QVBoxLayout(input_frame)

        input_header = QHBoxLayout()
        input_title = QLabel("입력 데이터")
        input_title.setFont(QFont("맑은 고딕", 12, QFont.Bold))
        input_header.addWidget(input_title)
        input_header.addStretch()
        self.count_label = QLabel("0 / 600")
        input_header.addWidget(self.count_label)
        input_layout.addLayout(input_header)

        guide = QLabel(
            "Ia 200개 → Ib 200개 → Ic 200개 순서로 총 600개 값을 "
            "쉼표, 공백 또는 줄바꿈으로 입력하세요."
        )
        guide.setWordWrap(True)
        guide.setStyleSheet("color: #5f6b7a;")
        input_layout.addWidget(guide)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText(
            "예: 0.12, 0.35, ... (Ia 200개, Ib 200개, Ic 200개)"
        )
        self.input_edit.setMinimumHeight(145)
        self.input_edit.textChanged.connect(self._update_input_count)
        input_layout.addWidget(self.input_edit)

        button_layout = QHBoxLayout()
        load_button = QPushButton("CSV/TXT 불러오기")
        load_button.clicked.connect(self._load_input_file)
        clear_button = QPushButton("입력 초기화")
        clear_button.clicked.connect(self._clear_all)
        self.predict_button = QPushButton("진단 실행")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self._run_diagnosis)
        self.predict_button.setStyleSheet(
            "QPushButton {background:#173a63; color:white; font-weight:bold; "
            "padding:9px 22px; border-radius:4px;}"
            "QPushButton:disabled {background:#aeb7c2;}"
        )
        button_layout.addWidget(load_button)
        button_layout.addWidget(clear_button)
        button_layout.addStretch()
        button_layout.addWidget(self.predict_button)
        input_layout.addLayout(button_layout)
        main_layout.addWidget(input_frame)

        result_layout = QGridLayout()
        result_layout.setHorizontalSpacing(18)

        probability_frame = QFrame()
        probability_frame.setFrameShape(QFrame.StyledPanel)
        probability_layout = QVBoxLayout(probability_frame)
        probability_title = QLabel("클래스별 진단 확률")
        probability_title.setFont(QFont("맑은 고딕", 12, QFont.Bold))
        probability_layout.addWidget(probability_title)

        self.probability_table = QTableWidget(0, 2)
        self.probability_table.setHorizontalHeaderLabels(["Class", "Probability"])
        self.probability_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.probability_table.verticalHeader().setVisible(False)
        self.probability_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.probability_table.setSelectionMode(QAbstractItemView.NoSelection)
        probability_layout.addWidget(self.probability_table)
        result_layout.addWidget(probability_frame, 0, 0)

        final_frame = QFrame()
        final_frame.setFrameShape(QFrame.StyledPanel)
        final_layout = QVBoxLayout(final_frame)
        final_layout.setAlignment(Qt.AlignCenter)
        final_title = QLabel("최종 진단 Class")
        final_title.setAlignment(Qt.AlignCenter)
        final_title.setFont(QFont("맑은 고딕", 12, QFont.Bold))
        self.final_class_label = QLabel("-")
        self.final_class_label.setAlignment(Qt.AlignCenter)
        self.final_class_label.setFont(QFont("맑은 고딕", 38, QFont.Bold))
        self.final_class_label.setStyleSheet("color:#173a63;")
        self.confidence_label = QLabel("진단을 실행해 주세요.")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        final_layout.addStretch()
        final_layout.addWidget(final_title)
        final_layout.addWidget(self.final_class_label)
        final_layout.addWidget(self.confidence_label)
        final_layout.addStretch()
        result_layout.addWidget(final_frame, 0, 1)
        result_layout.setColumnStretch(0, 3)
        result_layout.setColumnStretch(1, 2)
        main_layout.addLayout(result_layout, 1)

        self.setStyleSheet(
            "QMainWindow, QWidget {background:#f7f9fc; color:#17202a; "
            "font-family:'맑은 고딕'; font-size:10pt;}"
            "QFrame {background:white; border:1px solid #dce2ea; border-radius:6px;}"
            "QTextEdit, QTableWidget {background:white; border:1px solid #cfd7e3;}"
            "QPushButton {padding:8px 13px;}"
        )

    def _load_engine(self):
        try:
            self.engine = DiagnosisEngine()
            architecture_text = "-".join(map(str, self.engine.architecture))
            self.model_status.setText(
                f"모델 로드 완료  |  구조: {architecture_text}  |  "
                f"PT: {self.engine.path.name}"
            )
            self.predict_button.setEnabled(True)
            self._prepare_probability_table()
            self._update_input_count()
        except Exception as error:
            self.model_status.setText("모델 로드 실패")
            QMessageBox.critical(
                self,
                "모델 로드 오류",
                f"{CHECKPOINT_NAME}을 불러올 수 없습니다.\n\n{error}",
            )

    def _prepare_probability_table(self):
        self.probability_table.setRowCount(len(self.engine.class_ids))
        for row, class_id in enumerate(self.engine.class_ids):
            class_item = QTableWidgetItem(str(int(class_id)))
            class_item.setTextAlignment(Qt.AlignCenter)
            probability_item = QTableWidgetItem("-")
            probability_item.setTextAlignment(Qt.AlignCenter)
            self.probability_table.setItem(row, 0, class_item)
            self.probability_table.setItem(row, 1, probability_item)

    @staticmethod
    def _parse_text(text):
        tokens = [token for token in re.split(r"[\s,;]+", text.strip()) if token]
        return np.asarray([float(token) for token in tokens], dtype=np.float32)

    def _update_input_count(self):
        try:
            count = self._parse_text(self.input_edit.toPlainText()).size
        except ValueError:
            self.count_label.setText("숫자가 아닌 값 포함")
            self.count_label.setStyleSheet("color:#b42318; font-weight:bold;")
            return

        expected = self.engine.input_size if self.engine else 600
        self.count_label.setText(f"{count} / {expected}")
        color = "#067647" if count == expected else "#b54708"
        self.count_label.setStyleSheet(f"color:{color}; font-weight:bold;")

    def _load_input_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "입력 데이터 선택",
            "",
            "Data files (*.csv *.txt);;All files (*.*)",
        )
        if not path:
            return

        try:
            expected = self.engine.input_size
            values = self._read_numeric_file(path, expected)
            if values.size == expected + 1:
                values = values[:expected]
            if values.size != expected:
                raise ValueError(
                    f"사용 가능한 첫 데이터는 {expected}개여야 합니다. "
                    f"현재 {values.size}개입니다."
                )
            self.input_edit.setPlainText(", ".join(f"{v:.9g}" for v in values))
        except Exception as error:
            QMessageBox.warning(self, "파일 입력 오류", str(error))

    @staticmethod
    def _read_numeric_file(path, expected):
        last_error = None
        for delimiter in (",", None):
            try:
                data = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64)
                if data.ndim == 2:
                    for row in data:
                        finite_row = row[np.isfinite(row)]
                        if finite_row.size in (expected, expected + 1):
                            return finite_row.astype(np.float32)
                finite = np.asarray(data).reshape(-1)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    return finite.astype(np.float32)
            except Exception as error:
                last_error = error
        raise ValueError(f"숫자 데이터를 읽지 못했습니다: {last_error}")

    def _run_diagnosis(self):
        try:
            values = self._parse_text(self.input_edit.toPlainText())
            predicted_class, probabilities = self.engine.predict(values)
        except Exception as error:
            QMessageBox.warning(self, "입력 확인", str(error))
            return

        predicted_index = int(np.argmax(probabilities))
        for row, probability in enumerate(probabilities):
            item = self.probability_table.item(row, 1)
            item.setText(f"{probability * 100:.3f} %")
            highlight = row == predicted_index
            background = QColor("#d8e8fb") if highlight else QColor("white")
            for column in range(2):
                self.probability_table.item(row, column).setBackground(background)

        confidence = float(probabilities[predicted_index]) * 100
        self.final_class_label.setText(str(predicted_class))
        self.confidence_label.setText(f"최대 확률: {confidence:.3f} %")

    def _clear_all(self):
        self.input_edit.clear()
        self.final_class_label.setText("-")
        self.confidence_label.setText("진단을 실행해 주세요.")
        if self.engine:
            self._prepare_probability_table()


def main():
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("FNN Diagnosis GUI")
    window = DiagnosisWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
