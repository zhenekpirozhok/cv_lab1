import tkinter as tk
from commands import *
from detect_time import RecognizeTimeCommand

def setup_ui(editor):
    root = editor.root

    editor.open_button = tk.Button(root, text="📂 Открыть изображение", command=editor.open_image, font=("Arial", 12))
    editor.open_button.pack(pady=10)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    editor.btn_frame = btn_frame

    # --- Базовые преобразования ---
    buttons_row1 = [
        ("Ч/Б (усреднение)", GrayscaleAverageCommand),
        ("Ч/Б (HSV V)", GrayscaleHSVCommand),
        ("Бинаризация Otsu", BinarizationCommand),
    ]
    for col, (text, cmd_class) in enumerate(buttons_row1):
        btn = tk.Button(btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=18)
        btn.grid(row=0, column=col, padx=5, pady=5)
    
    # --- Undo ---
    editor.btn_undo = tk.Button(root, text="Отменить", command=editor.undo, width=15, bg="lightcoral")
    editor.btn_undo.pack(pady=5)

    # --- Нормализация и контраст ---
    buttons_row2 = [
        ("Нормализация", NormalizeCommand),
        ("Растяжение контраста", ContrastStretchCommand),
        ("Эквализация", EqualizeHistCommand)
    ]
    for col, (text, cmd_class) in enumerate(buttons_row2):
        btn = tk.Button(btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=18)
        btn.grid(row=1, column=col, padx=5, pady=5)

    # --- Фильтрация ---
    buttons_row3 = [
        ("Размытие Гаусса", GaussianBlurCommand),
        ("Резкость", SharpenCommand),
        ("Собель", SobelEdgeCommand)
    ]
    for col, (text, cmd_class) in enumerate(buttons_row3):
        btn = tk.Button(editor.btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=18)
        btn.grid(row=2, column=col, padx=5, pady=5)

    # --- Геометрические преобразования ---
    buttons_row4 = [
        ("Сдвиг", ShiftCommand),
        ("Поворот", RotateCommand)
    ]
    for col, (text, cmd_class) in enumerate(buttons_row4):
        btn = tk.Button(editor.btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=18)
        btn.grid(row=3, column=col, padx=5, pady=5)

    # --- Hough ---
    buttons_row5 = [
        ("Hough Линии", HoughLinesCommand),
        ("Hough Круги", HoughCirclesCommand),
    ]
    for col, (text, cmd_class) in enumerate(buttons_row5):
        btn = tk.Button(btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=18)
        btn.grid(row=4, column=col, padx=5, pady=5)

    # --- Локальные признаки и сегментация ---
    buttons_row6 = [
        ("Локальные признаки", LocalStatsCommand),
        ("Сегментация (по клику)", RegionGrowCommand),
        ("Поиск времени на изображении", RecognizeTimeCommand),
    ]
    for col, (text, cmd_class) in enumerate(buttons_row6):
        btn = tk.Button(btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=18, bg="lightblue")
        btn.grid(row=5, column=col, padx=5, pady=5)


    # --- Пошаговое распознавание времени ---
    def start_recognize_time():
        editor.step_command = RecognizeTimeCommand(editor)
        editor.step_command.start()

    def next_step_recognize_time():
        if hasattr(editor, 'step_command'):
            editor.step_command.next_step()
        else:
            messagebox.showinfo("Инфо", "Сначала нажмите 'Начать распознавание времени'")

    buttons_row6 = [
        ("Локальные признаки", LocalStatsCommand),
        ("Сегментация (по клику)", RegionGrowCommand),
    ]
    for col, (text, cmd_class) in enumerate(buttons_row6):
        btn = tk.Button(btn_frame, text=text, command=lambda c=cmd_class: editor.apply_command(c(editor)), width=18, bg="lightblue")
        btn.grid(row=5, column=col, padx=5, pady=5)

    # Кнопки для времени
    btn_start_time = tk.Button(btn_frame, text="Начать распознавание времени", command=start_recognize_time, width=18, bg="lightgreen")
    btn_start_time.grid(row=5, column=2, padx=5, pady=5)

    btn_next_step = tk.Button(btn_frame, text="Следующий шаг", command=next_step_recognize_time, width=18, bg="orange")
    btn_next_step.grid(row=5, column=3, padx=5, pady=5)
