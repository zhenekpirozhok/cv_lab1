import tkinter as tk
from PIL import Image, ImageTk
from commands import *
from ui import setup_ui

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")
        self.original_img = None
        self.displayed_img = None
        self.command_stack = []

        setup_ui(self)

        # Label для отображения изображения
        self.image_label = tk.Label(root, bg="gray")
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Обработчик изменения размера окна
        self.root.bind("<Configure>", self.on_resize)

        self.step_command = None

    def open_image(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff *.jfif, *.webp")],
        )
        if not filepath:
            return

        self.original_img = Image.open(filepath).convert("RGB")
        self.displayed_img = self.original_img.copy()
        self.show_image(self.displayed_img)
        self.command_stack.clear()

    def show_image(self, img):
        if img is None:
            return

        max_width = self.image_label.winfo_width()
        max_height = self.image_label.winfo_height()

        if max_width <= 1 or max_height <= 1:
            max_width = self.root.winfo_width()
            max_height = self.root.winfo_height()

        img_copy = img.copy()
        img_copy.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_copy)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk


    def apply_command(self, command):
        command.execute()
        self.command_stack.append(command)

    def undo(self):
        if self.command_stack:
            cmd = self.command_stack.pop()
            cmd.undo()

    def on_resize(self, event):
        """Автоматическое масштабирование при изменении размера окна"""
        if self.displayed_img:
            self.show_image(self.displayed_img)
