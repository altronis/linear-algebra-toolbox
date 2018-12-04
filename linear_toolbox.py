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
from constants import *


def format(n):
    if n == -0:
        n = 0

    if n > 99999 or n < 10 ** -4:
        return '%.3G' % n
    return ('%.2f' % n).rstrip('0').rstrip('.')


class LinearToolbox:
    @staticmethod
    def clear():
        for widget in mainframe.winfo_children():
            widget.destroy()

    @staticmethod
    def column_width(column):
        max_width = 0
        for c in column:
            if len(format(c)) > max_width:
                max_width = len(format(c))

        return max([4, max_width])

    def back_button(self, row, column):
        Button(mainframe, text="Back to menu", command=self.menu, font=LABEL_FONT).grid(column=column, row=row, columnspan=FILL, sticky=W)

    @staticmethod
    def vector_display(vector, row, column):
        Label(mainframe, text="[", font=MATRIX_FONT).grid(column=column, row=row)

        for i in range(len(vector)):
            if i < len(vector) - 1:
                Label(mainframe, text=format(vector[i]) + ", ", font=MATRIX_FONT).grid(column=column + i + 1, row=row)
            else:
                Label(mainframe, text=format(vector[i]), font=MATRIX_FONT).grid(column=column + i + 1, row=row)
        Label(mainframe, text="]", font=MATRIX_FONT).grid(column=column + len(vector) + 1, row=row)

    def matrix_display(self, matrix, row, column, bold):
        for r in range(len(matrix)):
            for c in range(len(matrix[r])):
                if matrix[r][c] == -0:
                    matrix[r][c] = 0

                if r in bold:
                    Label(mainframe, text=format(matrix[r][c]), font=MATRIX_BOLD, width=self.column_width(matrix[:, c]), borderwidth=1, relief="solid").grid(column=column + c,
                                                                                                                                                             row=row + r, sticky=E)
                else:
                    Label(mainframe, text=format(matrix[r][c]), font=MATRIX_FONT, width=self.column_width(matrix[:, c]), borderwidth=1, relief="solid").grid(column=column + c,
                                                                                                                                                             row=row + r, sticky=E)

    def vectors_input(self, num_vectors, dimensions, func):
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

            if func == ADD_VECTORS:
                add_vectors_solver = AddVectorsSolver(vectors)
                add_vectors_solver.calculate()
            elif func == DOT_PRODUCT:
                dot_product_solver = DotProductSolver(vectors)
                dot_product_solver.calculate()
            elif func == CROSS_PRODUCT:
                cross_product_solver = CrossProductSolver(vectors)
                cross_product_solver.calculate()

        for r in range(num_vectors):
            entries.append([])
            Label(mainframe, text="[", font=MATRIX_FONT).grid(column=1, row=r + 2, sticky=W)
            for c in range(dimensions):
                entries[r].append(DoubleVar())
                entries[r][c].set("")
                Entry(mainframe, width=4, textvariable=entries[r][c], font=MATRIX_FONT).grid(column=c + 2, row=r + 2, sticky=W)
            Label(mainframe, text="]", font=MATRIX_FONT).grid(column=dimensions + 2, row=r + 2, sticky=W)

        Button(mainframe, text="Submit", command=process_input, font=LABEL_FONT).grid(column=1, row=len(entries) + 2, columnspan=FILL, sticky=W)

    def matrix_input(self, rows, columns, func):
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

            if func == INVERSE:
                matrix = np.concatenate((matrix, np.eye(rows)), axis=1)

            rref_solver = RrefSolver(matrix)
            rref_solver.calculate(func)

        for r in range(rows):
            entries.append([])
            for c in range(columns):
                entries[r].append(DoubleVar())
                entries[r][c].set("")

                if func == SYSTEMS:
                    Entry(mainframe, width=4, textvariable=entries[r][c], font=MATRIX_FONT).grid(column=c * 2 + 1, row=r + 2, sticky=W)
                    if c < columns - 2:
                        Label(mainframe, text="x%d +" % (c + 1), font=MATRIX_FONT).grid(column=(c + 1) * 2, row=r + 2, sticky=W)
                    elif c == columns - 2:
                        Label(mainframe, text="x%d =" % (c + 1), font=MATRIX_FONT).grid(column=(c + 1) * 2, row=r + 2, sticky=W)
                else:
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
                    self.matrix_input(int(row_value), int(column_value), RREF)
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

    def inverse(self, func):
        self.clear()
        rows = DoubleVar()
        rows.set(2)

        row_entry = Entry(mainframe, width=5, textvariable=rows, font=MATRIX_FONT)
        row_entry.grid(column=1, row=2, sticky=W)

        row_label = Label(mainframe, text="rows and columns", font=LABEL_FONT)
        row_label.grid(column=2, row=2, sticky=E)

        def inverse_input():
            try:
                row_value = rows.get()
                if 2 <= row_value <= 7 and row_value % 1 == 0:
                    self.matrix_input(int(row_value), int(row_value), func)
                else:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Matrix must be between 2x2 and 7x7")
            except TclError:
                messagebox.showerror("Error", "Invalid input")
                return

        submit = Button(mainframe, text="Submit", command=inverse_input, font=LABEL_FONT)
        submit.grid(column=1, row=3, columnspan=2, sticky=W)
        dimension_header = Label(mainframe, text="Enter the dimensions of the matrix (2x2 - 7x7):", font=LABEL_FONT)
        dimension_header.grid(column=1, row=1, columnspan=4, sticky=W)

    def systems(self):
        self.clear()
        var = DoubleVar()
        eq = DoubleVar()
        var.set(2)
        eq.set(2)

        var_entry = Entry(mainframe, width=5, textvariable=var, font=MATRIX_FONT)
        var_entry.grid(column=1, row=2, sticky=W)
        eq_entry = Entry(mainframe, width=5, textvariable=eq, font=MATRIX_FONT)
        eq_entry.grid(column=1, row=3, sticky=W)

        var_label = Label(mainframe, text="variables (2-10)", font=LABEL_FONT)
        var_label.grid(column=2, row=2, sticky=E)
        eq_label = Label(mainframe, text="equations (2-10)", font=LABEL_FONT)
        eq_label.grid(column=2, row=3, sticky=E)

        def systems_input():
            try:
                var_value = var.get()
                eq_value = eq.get()
                if 2 <= var_value <= 10 and 2 <= eq_value <= 10 and var_value % 1 == 0 and eq_value % 1 == 0:
                    self.matrix_input(int(eq_value), int(var_value) + 1, SYSTEMS)
                else:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Number of variables or equations must be between 2 and 10")
            except TclError:
                messagebox.showerror("Error", "Invalid input")
                return

        submit = Button(mainframe, text="Submit", command=systems_input, font=LABEL_FONT)
        submit.grid(column=1, row=4, columnspan=2, sticky=W)
        dimension_header = Label(mainframe, text="Enter the number of variables and equations:", font=LABEL_FONT)
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
                    self.vectors_input(int(num_vectors_value), int(dimension_value), ADD_VECTORS)
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
                    self.vectors_input(2, int(dimension_value), DOT_PRODUCT)
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

    def cross_product(self):
        self.vectors_input(2, 3, CROSS_PRODUCT)

    def menu(self):
        self.clear()
        header = Label(mainframe, text="Linear Algebra Toolbox", font=LABEL_FONT)
        header.grid(column=1, row=1, sticky=W)

        Button(mainframe, text="Add vectors", command=self.add_vectors, font=LABEL_FONT).grid(column=1, row=2, sticky=W)
        Button(mainframe, text="Take the dot product of two vectors", command=self.dot_product, font=LABEL_FONT).grid(column=1, row=3, sticky=W)
        Button(mainframe, text="Take the cross product of two vectors", command=self.cross_product, font=LABEL_FONT).grid(column=1, row=4, sticky=W)
        Button(mainframe, text="Convert a matrix into reduced row echelon form", command=self.rref, font=LABEL_FONT).grid(column=1, row=5, sticky=W)
        Button(mainframe, text="Find the inverse of a matrix", command=lambda: self.inverse(INVERSE), font=LABEL_FONT).grid(column=1, row=6, sticky=W)
        Button(mainframe, text="Find the determinant of a matrix", command=lambda: self.inverse(DETERMINANT), font=LABEL_FONT).grid(column=1, row=7, sticky=W)
        Button(mainframe, text="Solve a system of equations", command=self.systems, font=LABEL_FONT).grid(column=1, row=8, sticky=W)


class RrefSolver(LinearToolbox):
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.result = self.matrix.copy()
        self.det = 1
        self.det_expr = 'Determinant = 1'

    def calculate(self, func):
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
                        self.det *= -1
                        self.det_expr += ' * -1'
                        break

                if swapped:
                    break
                else:  # If all entries in the column are zero, increase c by 1
                    c += 1
                    continue

            else:
                if self.result[r][c] != 1:  # Divide the r-th row by a[r][c] to make the pivot entry 1
                    Label(mainframe, text="Divide row " + str(r + 1) + " by " + format(self.result[r][c]), font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
                    self.result[r] *= 1 / self.result[r][c]
                    bold = [r]
                    self.det *= self.matrix[r][c]
                    self.det_expr += ' * ' + format(self.matrix[r][c])
                    break

                pivot = 0
                pivoted = False
                # Eliminate all other entries in the c-th column by subtracting multiples of the r-th row from the other rows
                while pivot < len(self.result):
                    if pivot != r and self.result[pivot][c] != 0:
                        Label(mainframe,
                              text="Add " + format(-1 * self.result[pivot][c]) + " times row " + str(r + 1) + " to row " + str(pivot + 1), font=LABEL_FONT).grid(column=1, row=1,
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
            if func == RREF:
                self.back_button(len(self.matrix) + 2, 1)

        self.matrix_display(self.matrix, 2, 1, bold)

        if func == DETERMINANT:
            Label(mainframe, text=self.det_expr + ' = ' + format(self.det), font=MATRIX_FONT).grid(column=1, row=len(self.matrix) + 2, columnspan=FILL, sticky=W)

        if not completed:
            self.matrix_display(self.result, 2, len(self.matrix[0]) + 2, bold)
            Label(mainframe, text='->', font=LABEL_FONT).grid(column=len(self.result[0]) + 1, row=len(self.result) // 2 + 1, sticky=W)

            if func == DETERMINANT:
                Button(mainframe, text="Next", command=lambda: self.calculate(func), font=LABEL_FONT).grid(column=1, row=len(self.matrix) + 3, columnspan=FILL, sticky=W)
            else:
                Button(mainframe, text="Next", command=lambda: self.calculate(func), font=LABEL_FONT).grid(column=1, row=len(self.matrix) + 2, columnspan=FILL, sticky=W)

        elif not func == RREF:
            solver = None
            if func == INVERSE:
                solver = InverseSolver(self.matrix)
            elif func == SYSTEMS:
                solver = SystemsSolver(self.matrix)
            elif func == DETERMINANT:
                solver = DeterminantSolver(self.matrix, self.det_expr + " = " + format(self.det))

            solver.calculate()
        self.matrix = self.result.copy()
        return self.result


class InverseSolver(LinearToolbox):
    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def calculate(self):
        self.clear()

        if np.array_equal(self.matrix[:, :len(self.matrix)], np.eye(len(self.matrix))):
            Label(mainframe, text="The second half of the augmented matrix is the inverse.", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
            self.matrix_display(self.matrix[:, len(self.matrix):], 2, 1, [])
        else:
            Label(mainframe, text="The matrix does not have an inverse, since the first half of the augmented matrix is not the identity matrix.", font=LABEL_FONT).grid(column=1, row=1,
                                                                                                                                                                         columnspan=FILL, sticky=W)
        self.back_button(2 + len(self.matrix), 1)


class DeterminantSolver(LinearToolbox):
    def __init__(self, matrix, det_expr):
        self.matrix = np.array(matrix)
        self.det_expr = det_expr

    def calculate(self):
        self.clear()
        self.matrix_display(self.matrix, 2, 1, [])

        if np.array_equal(self.matrix, np.eye(len(self.matrix))):
            Label(mainframe, text=self.det_expr, font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
        else:
            Label(mainframe, text="The determinant of the matrix is 0, since the rref of the matrix is not the identity matrix.", font=LABEL_FONT).grid(column=1, row=1,
                                                                                                                                                                         columnspan=FILL, sticky=W)
        self.back_button(2 + len(self.matrix), 1)


class SystemsSolver(LinearToolbox):
    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def no_solutions(self):
        for row in self.matrix:
            if np.array_equal(row[:-1], np.zeros(len(row) - 1)) and not row[-1] == 0:
                return True
        return False

    def calculate(self):
        self.clear()
        self.matrix_display(self.matrix, 2, 1, [])
        r = 0
        c = 0
        non_basic = list(range(len(self.matrix[0]) - 1))

        if self.no_solutions():
            Label(mainframe, text="There are no solutions to this system, since 0 = 1 is never true.", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
        else:
            while r < len(self.matrix) and c < len(self.matrix[0]) - 1:
                if self.matrix[r][c] == 1:
                    result = 'x%d = ' % (c + 1)
                    for var in range(c + 1, len(self.matrix[0]) - 1):
                        if not self.matrix[r][var] == 0:
                            result += '%sx%d + ' % (format(-1 * self.matrix[r][var]), var + 1)

                    if not self.matrix[r][len(self.matrix[0]) - 1] == 0:
                        result += format(self.matrix[r][len(self.matrix[0]) - 1])
                    Label(mainframe, text=result.rstrip('+ '), font=MATRIX_FONT).grid(column=1, row=2 + len(self.matrix) + c, columnspan=FILL, sticky=W)
                    non_basic.remove(c)
                    r += 1
                    c += 1
                else:
                    c += 1

            for var in non_basic:
                Label(mainframe, text='x%d = x%d' % (var + 1, var + 1), font=MATRIX_FONT).grid(column=1, row=2 + len(self.matrix) + var, columnspan=FILL, sticky=W)

            if not non_basic:
                Label(mainframe, text="There is a unique solution to this system.", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)
            else:
                Label(mainframe, text="There is an infinite number of solutions to this system.", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)

        self.back_button(2 + len(self.matrix[0]) - 1 + len(self.matrix), 1)


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
                sum += format(self.vectors[j][i])
                if j < len(self.vectors) - 1:
                    sum += ' + '
            sum += ' = ' + format(result[i])
            Label(mainframe, text=sum, font=MATRIX_FONT).grid(column=1, row=2 + len(self.vectors) + i, columnspan=FILL, sticky=W)
        LinearToolbox.vector_display(result, 2 + len(self.vectors) + len(self.vectors[0]), 1)

        self.back_button(3 + len(self.vectors) + len(self.vectors[0]), 1)


class DotProductSolver(LinearToolbox):
    def __init__(self, vectors):
        self.vectors = np.array(vectors)

    def calculate(self):
        self.clear()
        Label(mainframe, text="Multiply the corresponding components together and add them.", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)

        LinearToolbox.vector_display(self.vectors[0], 2, 1)
        Label(mainframe, text="•", font=MATRIX_FONT).grid(row=2, column=3 + len(self.vectors[0]), sticky=W)
        LinearToolbox.vector_display(self.vectors[1], 2, 4 + len(self.vectors[0]))

        result = ''
        for i in range(len(self.vectors[0])):
            result += format(self.vectors[0][i]) + '(' + format(self.vectors[1][i]) + ')'
            if i < len(self.vectors[0]) - 1:
                result += ' + '
        result += ' = ' + format(np.dot(self.vectors[0], self.vectors[1]))
        Label(mainframe, text=result, font=MATRIX_FONT).grid(row=3, column=1, columnspan=FILL, sticky=W)

        self.back_button(3 + len(self.vectors) + len(self.vectors[0]), 1)


class CrossProductSolver(LinearToolbox):
    def __init__(self, vectors):
        self.vectors = np.array(vectors)

    def calculate(self):
        self.clear()
        Label(mainframe, text="Cross Product:", font=LABEL_FONT).grid(column=1, row=1, columnspan=FILL, sticky=W)

        LinearToolbox.vector_display(self.vectors[0], 2, 1)
        Label(mainframe, text="X", font=MATRIX_FONT).grid(row=2, column=3 + len(self.vectors[0]), sticky=W)
        LinearToolbox.vector_display(self.vectors[1], 2, 4 + len(self.vectors[0]))

        result = np.cross(self.vectors[0], self.vectors[1])
        vx = 'Vx = y1z2 - y2z1 = %s(%s) - %s(%s) = %s' % (format(self.vectors[0][1]), format(self.vectors[1][2]), format(self.vectors[1][1]), format(self.vectors[0][2]), format(result[0]))
        vy = 'Vy = z1x2 - z2x1 = %s(%s) - %s(%s) = %s' % (format(self.vectors[0][2]), format(self.vectors[1][0]), format(self.vectors[1][2]), format(self.vectors[0][0]), format(result[1]))
        vz = 'Vz = x1y2 - x2y1 = %s(%s) - %s(%s) = %s' % (format(self.vectors[0][0]), format(self.vectors[1][1]), format(self.vectors[1][0]), format(self.vectors[0][1]), format(result[2]))

        Label(mainframe, text=vx, font=MATRIX_FONT).grid(row=3, column=1, columnspan=FILL, sticky=W)
        Label(mainframe, text=vy, font=MATRIX_FONT).grid(row=4, column=1, columnspan=FILL, sticky=W)
        Label(mainframe, text=vz, font=MATRIX_FONT).grid(row=5, column=1, columnspan=FILL, sticky=W)
        self.vector_display(result, 6, 1)

        self.back_button(3 + len(self.vectors) + len(self.vectors[0]), 1)


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

# Start Application
LinearToolbox().menu()
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)
root.mainloop()
