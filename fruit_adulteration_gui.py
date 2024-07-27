import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt6.QtGui import QPixmap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class FruitAdulterationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the trained model
        self.model_path = r"path of fruit_adulteration_model_updated.h5"
        self.model = load_model(self.model_path)

        # Initialize UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Fruit Adulteration Prediction")
        self.setGeometry(100, 100, 600, 400)

        # Create QLabel for displaying the selected image
        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 200, 200)

        # Create QLabel for displaying the prediction result
        self.result_label = QLabel(self)
        self.result_label.setGeometry(50, 270, 500, 30)

        # Create QPushButton for inserting an image
        self.insert_button = QPushButton("Insert Image", self)
        self.insert_button.setGeometry(300, 50, 200, 50)
        self.insert_button.clicked.connect(self.insert_image)

    def preprocess_image(self, img_path, target_size=(128, 128)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values
        return img_array

    def predict_adulteration(self, img_path):
        img_array = self.preprocess_image(img_path)
        prediction = self.model.predict(img_array)
        return "Adulterated" if prediction > 0.5 else "Not Adulterated"

    def insert_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Insert Image", "", "Image Files (*.png *.jpg *.jpeg)")

        if file_path:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(200, 200)
            self.image_label.setPixmap(pixmap)

            result = self.predict_adulteration(file_path)
            self.result_label.setText(f"The fruit is predicted to be: {result}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = FruitAdulterationApp()
    mainWin.show()
    sys.exit(app.exec())
