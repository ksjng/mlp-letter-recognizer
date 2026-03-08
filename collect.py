import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

data = []
labels = []

count = { 0:0, 1:0, 2:0, 3:0 }

class App:

    def __init__(self,root):

        self.root = root
        self.root.title("Zbieranie próbek C, O i T")
        self.root.protocol("WM_DELETE_WINDOW", self.exit)
        self.img = np.zeros((28, 28))
        self.fig, self.ax = plt.subplots(figsize=(3, 3))
        self.ax.set_xlim(0, 28)
        self.ax.set_ylim(28, 0)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()
        self.im = self.ax.imshow(self.img, cmap="gray", vmin=0, vmax=1)
        self.label = tk.Label(root, text="Narysuj i naciśnij C / O / T aby zapisać próbkę litery\nNarysuj i naciśnij X aby zapisać próbkę innego znaku\nNaciśnij ESC aby zapisać i wyjść")
        self.label.pack()
        self.drawing = False
        self.canvas.mpl_connect("motion_notify_event", self.draw)

        root.bind("c", self.save_c)
        root.bind("o", self.save_o)
        root.bind("t", self.save_t)
        root.bind("x", self.save_x)

        root.bind("<Escape>", self.exit)

        self.canvas.get_tk_widget().bind("<Button-1>", self.start)
        self.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.stop)

        self.update_label()

    def update_label(self):
        self.label.config(text=f"C: {count[0]}  O: {count[1]}  T: {count[2]}  Inne: {count[3]}\nNarysuj i naciśnij C / O / T aby zapisać próbkę litery\nNarysuj i naciśnij X aby zapisać próbkę innego znaku\nNaciśnij ESC aby zapisać i wyjść")

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

    def save(self, label):

        data.append(self.img.copy())
        labels.append(label)

        count[label] += 1

        print("Zapisano próbkę")

        self.img[:] = 0
        self.im.set_data(self.img)
        self.canvas.draw_idle()

        self.update_label()


    def save_c(self, e): self.save(0)
    def save_o(self, e): self.save(1)
    def save_t(self, e): self.save(2)
    def save_x(self, e): self.save(3)

    def exit(self, e=None):

        print("Zapisywanie datasetu...")

        np.savez("dataset.npz", X=np.array(data), Y=np.array(labels))

        print("Dataset zapisany:", len(data), "próbek")

        self.root.destroy()
        self.root.quit()

root = tk.Tk()

app = App(root)

root.mainloop()