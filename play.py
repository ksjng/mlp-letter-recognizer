import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sys

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

try:
    model = np.load("model.npz")
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]
except FileNotFoundError:
    print("Błąd: plik model.npz nie istnieje.")
    print("Najpierw uruchom train.py aby wytrenować model.")
    sys.exit(1)
except Exception:
    print("Błąd: nie udało się odczytać pliku model.npz (plik może być uszkodzony).")
    sys.exit(1)

mapa = [ "C", "O", "T" ]

def predict(img):

    pixels = np.sum(img)

    # reject drawings that are too small (almost empty canvas)
    if pixels < 5:
        return "Nic (za mało)", 1

    # reject drawings that are too large (random scribbles)
    if pixels > 300:
        return "Nic (za dużo)", 1

    x = img.reshape(1, 784)

    z1 = x@W1 + b1
    a1 = relu(z1)

    z2 = a1@W2 + b2

    probs = softmax(z2)

    cls = np.argmax(probs)
    conf = float(probs[0,cls])

    # reject unknown symbols
    if cls == 3:
        return "Nic (inne)", conf

    # reject predictions with low confidence
    if conf < 0.75:
        return "Nic", conf

    return mapa[cls], conf

class App:

    def __init__(self, root):

        self.root = root
        self.root.title("Rozpoznawanie liter C O T")
        self.root.protocol("WM_DELETE_WINDOW", self.exit)
        self.fig, self.ax = plt.subplots(figsize=(3, 3))
        self.ax.set_xlim(0, 28)
        self.ax.set_ylim(28, 0)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()
        self.img = np.zeros((28, 28))
        self.im = self.ax.imshow(self.img, cmap="gray", vmin=0, vmax=1)

        frame = tk.Frame(root)
        frame.pack()

        tk.Button(frame, text="Rozpoznaj", command=self.recognize).grid(row=0, column=0, padx=5)
        tk.Button(frame, text="Wyczyść", command=self.clear).grid(row=0, column=1, padx=5)

        self.label = tk.Label(root, text="Narysuj C, O lub T", font=("Arial", 14))
        self.label.pack(pady=10)

        self.drawing = False
        self.canvas.mpl_connect("motion_notify_event", self.draw)
        self.canvas.get_tk_widget().bind("<Button-1>", self.start)
        self.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.stop)

    def start(self, e):
        self.drawing = True

    def stop(self, e):
        self.drawing = False

    def draw(self, e):

        if not self.drawing or e.xdata is None or e.ydata is None:
            return

        x = int(e.xdata)
        y = int(e.ydata)

        for dx in range(-1, 2):
            for dy in range(-1, 2):

                if 0 <= x + dx < 28 and 0 <= y + dy < 28:
                    self.img[y + dy, x + dx] = 1

        self.im.set_data(self.img)

        self.canvas.draw_idle()


    def clear(self):

        self.img[:] = 0
        self.im.set_data(self.img)
        self.canvas.draw_idle()
        self.label.config(text="Narysuj C, O lub T")

    def recognize(self):

        letter, p = predict(self.img)

        self.label.config(text=f"Sieć rozpoznała: {letter}\nPewność: {p * 100:.1f}%")

    def exit(self):
        self.root.destroy()
        self.root.quit()

root = tk.Tk()

App(root)

root.mainloop()