import tkinter as tk
from view.TMView import TMView

class TMWindow(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        self.view = TMView(self, *args, **kwargs)

        self.view.pack(fill="both")

    def get_view(self) -> TMView:
        return self.view

    def start(self):
        self.view.start()
        self.mainloop()