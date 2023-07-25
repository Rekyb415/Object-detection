import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import requests
import io

# Replace this with the URL of your Flask server
SERVER_URL = "http://127.0.0.1:5000/predict"

class ImageUploadApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Upload Client")

        self.image_path = None
        self.result_text = tk.StringVar()
        self.create_widgets()

        # Allow the window to accept dropped files
        root.drop_target_register(DND_FILES)
        root.dnd_bind('<<Drop>>', self.handle_drop)

    def create_widgets(self):
        # Image display area
        self.image_label = tk.Label(self.root, width=300, height=300)
        self.image_label.pack()

        # Result display area
        self.result_label = tk.Label(self.root, textvariable=self.result_text, font=("Helvetica", 16))
        self.result_label.pack(pady=10)

    def handle_drop(self, event):
        file_path = event.data
        if file_path:
            self.image_path = file_path
            self.display_image()

            # Send image to server for prediction
            response = self.send_image_to_server(file_path)
            if response:
                self.update_result(response)
            else:
                self.result_text.set("Error while communicating with the server.")

    def display_image(self):
        image = Image.open(self.image_path)
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def send_image_to_server(self, file_path):
        try:
            with open(file_path, 'rb') as image_file:
                files = {'image': image_file}
                response = requests.post(SERVER_URL, files=files)
                if response.status_code == 200:
                    return response.json()
                else:
                    return None
        except Exception as e:
            print("Error:", str(e))
            return None

    def update_result(self, response):
        predicted_class = response.get("predicted_class")
        predicted_label = response.get("predicted_label")
        self.result_text.set(f"Predicted Class: {predicted_class}, Predicted Label: {predicted_label}")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ImageUploadApp(root)
    root.mainloop()
