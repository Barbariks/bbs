import sys
import numpy as np
import cv2
import win32gui
import win32ui
import win32con
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QPushButton, QFileDialog,
    QMainWindow, QWidget, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

def list_windows():
    """
    Получает список видимых окон и их заголовков.
    """
    windows = []

    def enum_windows_proc(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append((hwnd, title))
        return True

    win32gui.EnumWindows(enum_windows_proc, None)
    return windows


def capture_window(hwnd):
    """
    Захватывает содержимое окна по его HWND.
    """
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

        bmp_info = bitmap.GetInfo()
        bmp_data = bitmap.GetBitmapBits(True)

        # Проверка на пустоту изображения
        if not bmp_data:
            print("Не удалось захватить изображение, кадр пуст.")
            return None

        img = QImage(bmp_data, bmp_info['bmWidth'], bmp_info['bmHeight'], QImage.Format_RGB32)

        # Проверка на пустоту изображения после создания QImage
        if img.isNull():
            print("Изображение пустое после конвертации в QImage.")
            return None

        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        return img
    except Exception as e:
        print(f"Ошибка при захвате окна: {e}")
        return None


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Media Viewer with Neural Network Integration")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

    def initUI(self):
        # Центральный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Основной макет
        self.layout = QVBoxLayout()

        # Поле отображения контента
        self.image_label = QLabel("Здесь будет отображаться контент")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Кнопка для загрузки фото
        self.photo_button = QPushButton("Загрузить фото")
        self.photo_button.clicked.connect(self.load_photo)
        self.layout.addWidget(self.photo_button)

        # Кнопка для загрузки видео
        self.video_button = QPushButton("Загрузить видео")
        self.video_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.video_button)

        # Выпадающий список для выбора окна
        self.window_selector = QComboBox()
        self.refresh_window_list()
        self.layout.addWidget(self.window_selector)

        # Кнопка для трансляции окна
        self.window_button = QPushButton("Показать окно")
        self.window_button.clicked.connect(self.start_window_stream)
        self.layout.addWidget(self.window_button)

        # Выпадающий список для выбора камеры
        self.camera_selector = QComboBox()
        self.refresh_camera_list()
        self.layout.addWidget(self.camera_selector)

        # Кнопка для камеры
        self.camera_button = QPushButton("Открыть выбранную камеру")
        self.camera_button.clicked.connect(self.start_camera)
        self.layout.addWidget(self.camera_button)

        # Установка макета
        self.central_widget.setLayout(self.layout)

    def refresh_window_list(self):
        """
        Обновляет список доступных окон.
        """
        self.window_selector.clear()
        windows = list_windows()
        for hwnd, title in windows:
            self.window_selector.addItem(title, hwnd)

    def refresh_camera_list(self):
        """
        Обновляет список доступных камер.
        """
        self.camera_selector.clear()
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            self.camera_selector.addItem(f"Камера {index}", index)
            cap.release()
            index += 1

    def load_photo(self):
        """
        Загружает фото, обрабатывает его через нейросеть и отображает результат.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", "Images (*.png *.xpm *.jpg *.bmp *.jpeg)"
        )
        if file_path:
            img = cv2.imread(file_path)
            processed_img = self.process_with_neural_network(img)  # Обработка через нейросеть
            pixmap = self.convert_to_pixmap(processed_img)
            self.set_resized_image(pixmap)

    def load_video(self):
        """
        Загружает видео и обрабатывает его кадры через нейросеть.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "", "Videos (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            self.play_video(file_path)

    def start_window_stream(self):
        """
        Начинает отображение содержимого выбранного окна с обработкой через нейросеть.
        """
        hwnd = self.window_selector.currentData()
        if hwnd:
            self.display_window(hwnd)

    def display_window(self, hwnd):
        """
        Показывает содержимое окна в реальном времени с обработкой через нейросеть.
        """
        try:
            while True:
                img = capture_window(hwnd)
                if img is None:
                    print("Не удалось захватить окно или изображение пустое.")
                    break

                # Конвертация QImage в OpenCV формат
                frame = self.qimage_to_cv(img)

                # Проверка, не пустой ли кадр
                if frame is None or frame.size == 0:
                    print("Кадр пустой, пропускаем его.")
                    continue

                processed_frame = self.process_with_neural_network(frame)  # Обработка через нейросеть
                pixmap = self.convert_to_pixmap(processed_frame)
                self.set_resized_image(pixmap)
                QApplication.processEvents()
        except Exception as e:
            print(f"Ошибка при отображении окна: {e}")
            traceback.print_exc()

    def play_video(self, source):
        """
        Воспроизводит видео с обработкой кадров через нейросеть.
        """
        try:
            cap = cv2.VideoCapture(source)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = self.process_with_neural_network(frame)  # Обработка через нейросеть
                pixmap = self.convert_to_pixmap(processed_frame)
                self.set_resized_image(pixmap)
                QApplication.processEvents()
            cap.release()
        except Exception as e:
            print(f"Ошибка при воспроизведении видео: {e}")
            traceback.print_exc()

    def start_camera(self):
        """
        Запускает выбранную камеру и обрабатывает её кадры через нейросеть.
        """
        camera_index = self.camera_selector.currentData()
        if camera_index is None:
            return

        try:
            cap = cv2.VideoCapture(camera_index)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = self.process_with_neural_network(frame)  # Обработка через нейросеть
                pixmap = self.convert_to_pixmap(processed_frame)
                self.set_resized_image(pixmap)
                QApplication.processEvents()
            cap.release()
        except Exception as e:
            print(f"Ошибка при работе с камерой: {e}")
            traceback.print_exc()

    def process_with_neural_network(self, frame):
        """
        Заготовка для обработки кадра через нейросеть.
        :param frame: Кадр в формате OpenCV.
        :return: Обработанный кадр.
        """
        
        model = YOLO("weights/best.pt")

        results = model(frame)
        
        # Здесь подключается модель нейросети
        # Например, можно использовать PyTorch, TensorFlow и т. д.
        # Пока просто возвращаем исходный кадр
        return results[0]

    def convert_to_pixmap(self, frame):
        """
        Конвертирует OpenCV изображение в QPixmap для отображения.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def qimage_to_cv(self, img):
        """
        Конвертация QImage в OpenCV формат.
        """
        try:
            # Получаем байтовые данные изображения
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            data = np.array(ptr).reshape(img.height(), img.width(), 4)  # 4 канала для RGBA

            # Преобразуем данные в формат BGR для OpenCV
            frame = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)

            # Проверка на пустоту данных
            if frame is None or frame.size == 0:
                print("Ошибка: пустое изображение в OpenCV.")
                return None

            return frame
        except Exception as e:
            print(f"Ошибка при конвертации QImage в OpenCV: {e}")
            return None

    def set_resized_image(self, pixmap):
        """
        Масштабирует изображение, чтобы оно вписывалось в окно приложения.
        """
        max_width = self.image_label.width()
        max_height = self.image_label.height()

        scaled_pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = App()
    viewer.show()
    sys.exit(app.exec_())
