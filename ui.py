import tkinter as tk
from commands import *

def setup_ui(editor):
    root = editor.root

    editor.open_button = tk.Button(root, text="📂 Открыть изображение", command=editor.open_image, font=("Arial", 12))
    editor.open_button.pack(pady=10)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    editor.btn_frame = btn_frame

    buttons_row1 = [
        ("Ч/Б (усреднение)", GrayscaleAverageCommand),
        ("Ч/Б (HSV V)", GrayscaleHSVCommand),
        ("Бинаризация Otsu", BinarizationCommand),
    ]
    for col, (text, cmd_class) in enumerate(buttons_row1):
        btn = tk.Button(btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=15)
        btn.grid(row=0, column=col, padx=5)
    
    editor.btn_undo = tk.Button(root, text="Отменить", command=editor.undo, width=15, bg="lightcoral")
    editor.btn_undo.pack(pady=5)

    buttons_row2 = [
        ("Нормализация", NormalizeCommand),
        ("Растяжение контраста", ContrastStretchCommand),
        ("Эквализация", EqualizeHistCommand)
    ]
    for col, (text, cmd_class) in enumerate(buttons_row2):
        btn = tk.Button(btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=15)
        btn.grid(row=1, column=col, padx=5, pady=5)

    buttons_row3 = [
        ("Размытие Гаусса", GaussianBlurCommand),
        ("Резкость", SharpenCommand),
        ("Собель", SobelEdgeCommand)
    ]

    for col, (text, cmd_class) in enumerate(buttons_row3):
        btn = tk.Button(editor.btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=15)
        btn.grid(row=2, column=col, padx=5, pady=5)

    buttons_row4 = [
        ("Сдвиг", ShiftCommand),
        ("Поворот", RotateCommand)
    ]

    for col, (text, cmd_class) in enumerate(buttons_row4):
        btn = tk.Button(editor.btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=15)
        btn.grid(row=3, column=col, padx=5, pady=5)
