from tkinter import Button
from tkinter import DoubleVar
from tkinter import Entry
from tkinter import Label
from tkinter import N, W, E, S
from tkinter import TclError
from tkinter import Tk
from tkinter import font
from tkinter import messagebox
from tkinter import ttk
import numpy as np


class LinearToolbox:
    @staticmethod
    def clear():
        for widget in mainframe.winfo_children():
            widget.destroy()

    @staticmethod
    def column_width(column):
        max_width = 0
        for c in column:
            if len("%2g" % c) > max_width:
                max_width = len("%2g" % c)

        return max([4, max_width])

    @staticmethod
    def vector_display(vector, row, column):
        Label(mainframe, text="[", font=LABEL_FONT).grid(column=column, row=row)

        for i in range(len(vector)):
            if i < len(vector) - 1:
                Label(mainframe, text='%2g' % vector[i] + ", ", font=LABEL_FONT).grid(column=column + i + 1, row=row)
            else:
                Label(mainframe, text='%2g' % vector[i], font=LABEL_FONT).grid(column=column + i + 1, row=row)
        Label(mainframe, text="]", font=LABEL_FONT).grid(column=column + len(vector) + 1, row=row)

    def vectors_input(self, num_vectors, dimensions):
        self.clear()
        Label(mainframe, text="Enter vectors:", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
        entries = []

        def process_input():
            vectors = []
            try:
                for r in range(num_vectors):
                    vectors.append([])
                    for c in range(dimensions):
                        vectors[r].append(entries[r][c].get())
            except TclError:
                messagebox.showerror("Error", "Invalid input")
                return
            add_vectors_solver = AddVectorsSolver(vectors)
            add_vectors_solver.calculate()

        for r in range(num_vectors):
            entries.append([])
            Label(mainframe, text="[", font=LABEL_FONT).grid(column=1, row=r + 2, sticky=W)
            for c in range(dimensions):
                entries[r].append(DoubleVar())
                entries[r][c].set("")
                Entry(mainframe, width=4, textvariable=entries[r][c], font=MATRIX_FONT).grid(column=c + 2, row=r + 2, sticky=W)
            Label(mainframe, text="]", font=LABEL_FONT).grid(column=dimensions + 2, row=r + 2, sticky=W)

        Button(mainframe, text="Submit", command=process_input, font=LABEL_FONT).grid(column=1, row=len(entries) + 2, columnspan=FILL, sticky=W)

    def matrix_input(self, rows, columns):
        self.clear()
        Label(mainframe, text="Enter matrix:", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
        entries = []

        def process_input():
            matrix = []
            try:
                for r in range(rows):
                    matrix.append([])
                    for c in range(columns):
                        matrix[r].append(entries[r][c].get())
            except TclError:
                messagebox.showerror("Error", "Invalid input")
                return
            rref_solver = RrefSolver(matrix)
            rref_solver.calculate()

        for r in range(rows):
            entries.append([])
            for c in range(columns):
                entries[r].append(DoubleVar())
                entries[r][c].set("")
                Entry(mainframe, width=4, textvariable=entries[r][c], font=MATRIX_FONT).grid(column=c + 1, row=r + 2, sticky=W)
        Button(mainframe, text="Submit", command=process_input, font=LABEL_FONT).grid(column=1, row=len(entries) + 2, columnspan=FILL, sticky=W)

    def rref(self):
        self.clear()
        rows = DoubleVar()
        columns = DoubleVar()
        rows.set(2)
        columns.set(2)

        row_entry = Entry(mainframe, width=5, textvariable=rows, font=MATRIX_FONT)
        row_entry.grid(column=1, row=2, sticky=W)
        column_entry = Entry(mainframe, width=5, textvariable=columns, font=MATRIX_FONT)
        column_entry.grid(column=1, row=3, sticky=W)

        row_label = Label(mainframe, text="rows", font=LABEL_FONT)
        row_label.grid(column=2, row=2, sticky=E)
        column_label = Label(mainframe, text="columns", font=LABEL_FONT)
        column_label.grid(column=2, row=3, sticky=E)

        def rref_input():
            try:
                row_value = rows.get()
                column_value = columns.get()
                if 2 <= row_value <= 15 and 2 <= column_value <= 15 and row_value % 1 == 0 and column_value % 1 == 0:
                    self.matrix_input(int(row_value), int(column_value))
                else:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Matrix must be between 2x2 and 15x15")
            except TclError:
                messagebox.showerror("Error", "Invalid input")
                return

        submit = Button(mainframe, text="Submit", command=rref_input, font=LABEL_FONT)
        submit.grid(column=1, row=4, columnspan=2, sticky=W)
        dimension_header = Label(mainframe, text="Enter the dimensions of the matrix (2x2 - 15x15):", font=LABEL_FONT)
        dimension_header.grid(column=1, row=1, columnspan=4, sticky=W)

    def add_vectors(self):
        self.clear()
        num_vectors = DoubleVar()
        dimension = DoubleVar()
        num_vectors.set(2)
        dimension.set(2)

        num_vectors_entry = Entry(mainframe, width=5, textvariable=num_vectors, font=MATRIX_FONT)
        num_vectors_entry.grid(column=1, row=2, sticky=W)
        dimension_entry = Entry(mainframe, width=5, textvariable=dimension, font=MATRIX_FONT)
        dimension_entry.grid(column=1, row=3, sticky=W)

        num_vectors_label = Label(mainframe, text="vectors", font=LABEL_FONT)
        num_vectors_label.grid(column=2, row=2, sticky=E)
        dimension_label = Label(mainframe, text="dimensions", font=LABEL_FONT)
        dimension_label.grid(column=2, row=3, sticky=E)

        def add_vectors_input():
            try:
                num_vectors_value = num_vectors.get()
                dimension_value = dimension.get()
                if 2 <= num_vectors_value <= 15 and 2 <= dimension_value <= 15 and num_vectors_value % 1 == 0 and dimension_value % 1 == 0:
                    self.vectors_input(int(num_vectors_value), int(dimension_value))
                else:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Number of vectors or dimensions must be between 2 and 15")
            except TclError:
                messagebox.showerror("Error", "Invalid input")
                return

        submit = Button(mainframe, text="Submit", command=add_vectors_input, font=LABEL_FONT)
        submit.grid(column=1, row=4, columnspan=2, sticky=W)
        dimension_header = Label(mainframe, text="Enter the number of vectors to add and their dimensions:", font=LABEL_FONT)
        dimension_header.grid(column=1, row=1, columnspan=FILL, sticky=W)

    def dot_product(self):
        self.clear()
        dimension = DoubleVar()
        dimension.set(2)

        dimension_entry = Entry(mainframe, width=5, textvariable=dimension, font=MATRIX_FONT)
        dimension_entry.grid(column=1, row=2, sticky=W)

        dimension_label = Label(mainframe, text="dimensions", font=LABEL_FONT)
        dimension_label.grid(column=2, row=2, sticky=E)

        def dot_product_input():
            try:
                dimension_value = dimension.get()
                if 2 <= dimension_value <= 15 and dimension_value % 1 == 0:
                    self.vectors_input(2, int(dimension_value))
                else:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Number of dimensions must be between 2 and 15")
            except TclError:
                messagebox.showerror("Error", "Invalid input")
                return

        submit = Button(mainframe, text="Submit", command=dot_product_input, font=LABEL_FONT)
        submit.grid(column=1, row=4, columnspan=FILL, sticky=W)
        dimension_header = Label(mainframe, text="Enter the number of dimensions for the vectors:", font=LABEL_FONT)
        dimension_header.grid(column=1, row=1, columnspan=FILL, sticky=W)

    def menu(self):
        self.clear()
        header = Label(mainframe, text="Linear Algebra Toolbox", font=LABEL_FONT)
        header.grid(column=1, row=1, sticky=W)

        rref_button = Button(mainframe, text="Convert a matrix into reduced row echelon form", command=self.rref, font=LABEL_FONT)
        rref_button.grid(column=1, row=2, sticky=W)

        add_vectors_button = Button(mainframe, text="Add vectors", command=self.add_vectors, font=LABEL_FONT)
        add_vectors_button.grid(column=1, row=3, sticky=W)

        dot_product_button = Button(mainframe, text="Take the dot product of two vectors", command=self.dot_product, font=LABEL_FONT)
        dot_product_button.grid(column=1, row=4, sticky=W)


class RrefSolver(LinearToolbox):
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.result = self.matrix.copy()

    def calculate(self):
        self.clear()
        self.result.astype(float)
        completed = False
        bold = []
        r = 0
        c = 0

        while r < len(self.result) and c < len(self.result[0]):
            # If a[r][c] == 0, swap the r-th row with some other row below so that a[r][c] != 0
            if self.result[r][c] == 0:
                swapped = False
                for pivot in range(r + 1, len(self.result)):
                    if self.result[pivot][c] != 0:  # Swap rows so that a[r][c] != 0
                        temp = self.result[c].copy()
                        self.result[c] = self.result[pivot]
                        self.result[pivot] = temp
                        Label(mainframe, text="Swap row " + str(pivot + 1) + " with row " + str(r + 1), font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
                        bold = [pivot, r]
                        swapped = True
                        break

                if swapped:
                    break
                else:  # If all entries in the column are zero, increase c by 1
                    c += 1
                    continue

            else:
                if self.result[r][c] != 1:  # Divide the r-th row by a[r][c] to make the pivot entry 1
                    Label(mainframe, text="Divide row " + str(r + 1) + " by " '%.2g' % self.result[r][c], font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
                    self.result[r] *= 1 / self.result[r][c]
                    bold = [r]
                    break

                pivot = 0
                pivoted = False
                # Eliminate all other entries in the c-th column by subtracting multiples of the r-th row from the other rows
                while pivot < len(self.result):
                    if pivot != r and self.result[pivot][c] != 0:
                        Label(mainframe,
                              text="Add " + '%.2g' % (-1 * self.result[pivot][c]) + " times row " + str(r + 1) + " to row " + str(pivot + 1), font=LABEL_FONT).grid(column=1, row=1,
                                                                                                                                                                    columnspan=FILL,
                                                                                                                                                                    sticky=W)
                        self.result[pivot] -= self.result[r] * self.result[pivot][c]
                        bold = [pivot, r]
                        pivoted = True
                        break
                    pivot += 1
                # Increase r by 1 and c by 1 to choose the new pivot element
                if pivot == len(self.result):
                    r += 1
                    c += 1
                if pivoted:
                    break
        else:
            completed = True
            Label(mainframe, text="The matrix is now in reduced row echelon form.", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
            Button(mainframe, text="Back to menu", command=self.menu, font=LABEL_FONT).grid(column=1, row=len(self.matrix) + 2, columnspan=FILL, sticky=W)

        for r in range(len(self.result)):
            for c in range(len(self.result[r])):
                if self.result[r][c] == -0:
                    self.result[r][c] = 0

                if r in bold:
                    Label(mainframe, text='%2g' % self.matrix[r][c], font=MATRIX_BOLD, width=self.column_width(self.matrix[:, c]), borderwidth=1, relief="solid").grid(column=c + 1,
                                                                                                                                                                       row=r + 2,
                                                                                                                                                                       sticky=E)
                else:
                    Label(mainframe, text='%2g' % self.matrix[r][c], font=MATRIX_FONT, width=self.column_width(self.matrix[:, c]), borderwidth=1, relief="solid").grid(column=c + 1,
                                                                                                                                                                       row=r + 2,
                                                                                                                                                                       sticky=E)

                if not completed:
                    if r in bold:
                        Label(mainframe, text='%2g' % self.result[r][c], font=MATRIX_BOLD, width=self.column_width(self.result[:, c]), borderwidth=1, relief="solid").grid(
                            column=len(self.result[r]) + c + 2,
                            row=r + 2,
                            sticky=E)
                    else:
                        Label(mainframe, text='%2g' % self.result[r][c], font=MATRIX_FONT, width=self.column_width(self.result[:, c]), borderwidth=1, relief="solid").grid(
                            column=len(self.result[r]) + c + 2,
                            row=r + 2,
                            sticky=E)

        if not completed:
            Label(mainframe, text='->', font=LABEL_FONT).grid(column=len(self.result[0]) + 1, row=len(self.result) // 2 + 1, sticky=W)

        if not completed:
            Button(mainframe, text="Next", command=lambda: self.calculate(), font=LABEL_FONT).grid(column=1, row=len(self.matrix) + 2, columnspan=FILL, sticky=W)

        self.matrix = self.result.copy()
        return self.result


class AddVectorsSolver(LinearToolbox):
    def __init__(self, vectors):
        self.vectors = np.array(vectors)

    def calculate(self):
        self.clear()
        for i in range(len(self.vectors)):
            LinearToolbox.vector_display(self.vectors[i], 2 + i, 1)

        Label(mainframe, text="Add each of the components together.", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
        result = np.sum(self.vectors, axis=0)
        for i in range(len(self.vectors[0])):
            sum = ''
            for j in range(len(self.vectors)):
                sum += '%2g' % self.vectors[j][i]
                if j < len(self.vectors) - 1:
                    sum += ' + '
            sum += ' = ' + '%2g' % result[i]
            Label(mainframe, text=sum, font=LABEL_FONT).grid(column=1, row=2 + len(self.vectors) + i, columnspan=FILL, sticky=W)
        LinearToolbox.vector_display(result, 2 + 2 * len(self.vectors[0]), 1)

        Button(mainframe, text="Back to menu", command=self.menu, font=LABEL_FONT).grid(column=1, row=3 + 2 * len(self.vectors[0]), columnspan=FILL, sticky=W)


# Initialization of main frame
root = Tk()
root.title("Linear Algebra Toolbox")
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Fonts
LABEL_FONT = font.Font(family="Arial", size=18)
MATRIX_FONT = font.Font(family="Consolas", size=18)
MATRIX_BOLD = font.Font(family="Consolas", size=18, weight=font.BOLD)

# Constants
FILL = 999

# Start Application
LinearToolbox().menu()
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)
root.mainloop()
