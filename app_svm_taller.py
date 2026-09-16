"""Aplicación interactiva para el Taller 10: fronteras SVM no lineales."""

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.svm import SVC


BASE_X = np.array([[2, 2], [3, 3], [4, 2], [6, 6], [7, 8], [8, 7]], dtype=float)
BASE_Y = np.array([0, 0, 0, 1, 1, 1])


class TallerSVM(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Taller 10 · SVM: fronteras no lineales")
        self.geometry("1240x760")
        self.minsize(1040, 660)
        self.configure(bg="#f4f7f5")

        self.x = BASE_X.copy()
        self.y = BASE_Y.copy()
        self.modelo = None
        self.kernel = tk.StringVar(value="linear")
        self.pred_x = tk.StringVar(value="5")
        self.pred_y = tk.StringVar(value="4")
        self.estado = tk.StringVar(value="Carga el conjunto inicial y entrena un modelo.")
        self._crear_interfaz()
        self._actualizar_tabla()
        self._dibujar()

    def _crear_interfaz(self):
        encabezado = tk.Frame(self, bg="#167a45", padx=24, pady=16)
        encabezado.pack(fill="x")
        tk.Label(encabezado, text="TALLER DE LABORATORIO: FRONTERAS NO LINEALES",
                 font=("Segoe UI", 18, "bold"), bg="#167a45", fg="white").pack(anchor="w")
        tk.Label(encabezado, text="Explora cómo una SVM lineal y una SVM con kernel RBF clasifican los mismos datos.",
                 font=("Segoe UI", 10), bg="#167a45", fg="#dff7e9").pack(anchor="w", pady=(3, 0))

        cuerpo = tk.Frame(self, bg="#f4f7f5", padx=18, pady=16)
        cuerpo.pack(fill="both", expand=True)
        controles = tk.Frame(cuerpo, bg="white", padx=16, pady=14, highlightbackground="#d4ddd7", highlightthickness=1)
        controles.pack(side="left", fill="y")
        grafica = tk.Frame(cuerpo, bg="white", padx=8, pady=8, highlightbackground="#d4ddd7", highlightthickness=1)
        grafica.pack(side="left", fill="both", expand=True, padx=(16, 0))

        tk.Label(controles, text="1. Conjunto de datos", font=("Segoe UI", 13, "bold"), bg="white", fg="#123526").pack(anchor="w")
        tk.Label(controles, text="Clase 0 = naranja · Clase 1 = azul", font=("Segoe UI", 9), bg="white", fg="#607167").pack(anchor="w", pady=(2, 8))
        self.tabla = tk.Listbox(controles, height=8, width=34, font=("Consolas", 10), borderwidth=0, bg="#f2f6f3", highlightthickness=0)
        self.tabla.pack(fill="x")

        ttk.Separator(controles).pack(fill="x", pady=14)
        tk.Label(controles, text="2. Añadir el punto que desafía la recta", font=("Segoe UI", 11, "bold"), bg="white", fg="#123526").pack(anchor="w")
        self.boton_punto = tk.Button(controles, text="Agregar [5, 5] como clase 0", command=self.agregar_punto,
                                     bg="#e99838", fg="white", activebackground="#cf7f22", relief="flat", padx=10, pady=7, cursor="hand2")
        self.boton_punto.pack(fill="x", pady=(7, 4))
        tk.Button(controles, text="Restablecer conjunto inicial", command=self.restablecer,
                  bg="#edf2ee", fg="#254535", relief="flat", padx=10, pady=7, cursor="hand2").pack(fill="x")

        ttk.Separator(controles).pack(fill="x", pady=14)
        tk.Label(controles, text="3. Elegir y entrenar", font=("Segoe UI", 11, "bold"), bg="white", fg="#123526").pack(anchor="w")
        for valor, texto in [("linear", "Kernel lineal (una recta)"), ("rbf", "Kernel RBF (frontera curva)")]:
            tk.Radiobutton(controles, text=texto, variable=self.kernel, value=valor, command=self._dibujar,
                           bg="white", activebackground="white", font=("Segoe UI", 10)).pack(anchor="w", pady=2)
        tk.Button(controles, text="Entrenar modelo", command=self.entrenar, bg="#167a45", fg="white",
                  activebackground="#0f6034", relief="flat", padx=10, pady=8, cursor="hand2").pack(fill="x", pady=(8, 0))

        ttk.Separator(controles).pack(fill="x", pady=14)
        tk.Label(controles, text="4. Probar una predicción", font=("Segoe UI", 11, "bold"), bg="white", fg="#123526").pack(anchor="w")
        campos = tk.Frame(controles, bg="white")
        campos.pack(fill="x", pady=(6, 5))
        tk.Label(campos, text="x:", bg="white").grid(row=0, column=0, sticky="w")
        ttk.Entry(campos, textvariable=self.pred_x, width=8).grid(row=0, column=1, padx=(4, 10))
        tk.Label(campos, text="y:", bg="white").grid(row=0, column=2, sticky="w")
        ttk.Entry(campos, textvariable=self.pred_y, width=8).grid(row=0, column=3, padx=4)
        tk.Button(controles, text="Predecir punto", command=self.predecir, bg="#416c9e", fg="white",
                  activebackground="#345981", relief="flat", padx=10, pady=7, cursor="hand2").pack(fill="x")

        self.resultado = tk.Label(controles, textvariable=self.estado, justify="left", wraplength=285,
                                  bg="#eff8f1", fg="#1d4c30", font=("Segoe UI", 9), padx=9, pady=9)
        self.resultado.pack(fill="x", pady=(14, 0))

        self.figura = Figure(figsize=(7.1, 5.6), dpi=100, facecolor="white")
        self.ax = self.figura.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figura, master=grafica)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def agregar_punto(self):
        if not np.any(np.all(self.x == [5, 5], axis=1)):
            self.x = np.vstack((self.x, [5, 5]))
            self.y = np.append(self.y, 0)
            self.modelo = None
            self.boton_punto.config(state="disabled", text="Punto [5, 5] añadido")
            self.estado.set("Punto [5, 5] añadido a la clase 0. Entrena primero con kernel lineal y luego con RBF.")
            self._actualizar_tabla()
            self._dibujar()

    def restablecer(self):
        self.x, self.y, self.modelo = BASE_X.copy(), BASE_Y.copy(), None
        self.boton_punto.config(state="normal", text="Agregar [5, 5] como clase 0")
        self.estado.set("Conjunto inicial restablecido.")
        self._actualizar_tabla()
        self._dibujar()

    def entrenar(self):
        self.modelo = SVC(kernel=self.kernel.get(), C=1.0, gamma="scale")
        self.modelo.fit(self.x, self.y)
        soportes = self.modelo.support_vectors_
        texto = ", ".join(f"[{p[0]:g}, {p[1]:g}]" for p in soportes)
        nombre = "lineal" if self.kernel.get() == "linear" else "RBF"
        self.estado.set(f"Modelo {nombre} entrenado.\nVectores de soporte: {texto}")
        self._dibujar()

    def predecir(self):
        if self.modelo is None:
            messagebox.showinfo("Primero entrena", "Selecciona un kernel y pulsa «Entrenar modelo» antes de predecir.")
            return
        try:
            punto = np.array([[float(self.pred_x.get()), float(self.pred_y.get())]])
        except ValueError:
            messagebox.showerror("Coordenadas inválidas", "Escribe valores numéricos para x e y.")
            return
        clase = int(self.modelo.predict(punto)[0])
        etiqueta = "Clase 0 (A)" if clase == 0 else "Clase 1 (B)"
        self.estado.set(f"El punto [{punto[0,0]:g}, {punto[0,1]:g}] pertenece a {etiqueta}.")
        self._dibujar(punto[0])

    def _actualizar_tabla(self):
        self.tabla.delete(0, tk.END)
        self.tabla.insert(tk.END, "  Coordenadas       Clase")
        for punto, clase in zip(self.x, self.y):
            self.tabla.insert(tk.END, f"  [{punto[0]:>3.0f}, {punto[1]:>3.0f}]          {clase}")

    def _dibujar(self, punto_predicho=None):
        self.ax.clear()
        colores = ["#e68a2e", "#3176b7"]
        for clase in (0, 1):
            datos = self.x[self.y == clase]
            self.ax.scatter(datos[:, 0], datos[:, 1], s=70, c=colores[clase], edgecolors="white", linewidths=1.2,
                            label=f"Clase {clase} ({'A' if clase == 0 else 'B'})", zorder=3)
        if self.modelo is not None:
            xx, yy = np.meshgrid(np.linspace(0.8, 9.2, 300), np.linspace(0.8, 9.2, 300))
            zz = self.modelo.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            self.ax.contourf(xx, yy, zz, levels=[-20, 0, 20], colors=["#fcebd9", "#dceafa"], alpha=.75)
            self.ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["#6d6d6d", "#202020", "#6d6d6d"],
                            linestyles=["--", "-", "--"], linewidths=[1, 1.8, 1])
            sv = self.modelo.support_vectors_
            self.ax.scatter(sv[:, 0], sv[:, 1], s=190, facecolors="none", edgecolors="#151515", linewidths=1.5,
                            label="Vector de soporte", zorder=4)
        if punto_predicho is not None:
            self.ax.scatter(*punto_predicho, marker="*", s=260, c="#d12f5b", edgecolors="white", linewidths=1,
                            label="Punto consultado", zorder=5)
        nombre = "lineal" if self.kernel.get() == "linear" else "RBF"
        self.ax.set_title(f"Frontera de decisión · kernel {nombre}", fontsize=13, fontweight="bold", pad=12)
        self.ax.set_xlabel("Coordenada X")
        self.ax.set_ylabel("Coordenada Y")
        self.ax.set_xlim(.8, 9.2); self.ax.set_ylim(.8, 9.2)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(alpha=.2)
        self.ax.legend(loc="upper left", fontsize=8, frameon=True)
        self.figura.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    TallerSVM().mainloop()
