import tkinter as tk
from PIL import Image, ImageTk
from commands import *
from ui import setup_ui

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.original_img = None
        self.displayed_img = None
        self.command_stack = []

        setup_ui(self) 

        self.image_label = tk.Label(root)
        self.image_label.pack(expand=True)

    def open_image(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff *.jfif")]
        )
        if not filepath:
            return

        self.original_img = Image.open(filepath).convert("RGB")
        self.displayed_img = self.original_img.copy()
        self.show_image(self.displayed_img)
        self.command_stack.clear()

    def show_image(self, img):
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def apply_command(self, command):
        command.execute()
        self.command_stack.append(command)

    def undo(self):
        if self.command_stack:
            cmd = self.command_stack.pop()
            cmd.undo()
