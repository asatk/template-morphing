#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

import tkinter as tk
from tkinter import ttk
from typing import Callable

from PIL import ImageTk
from matplotlib.pyplot import text

class TMSimView(tk.Toplevel):

    def __init__(self, root: tk.Tk, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.__frames: dict[str,ttk.Frame] = {}
        self.__labels: dict[str,ttk.Label] = {}
        self.__buttons: dict[str,ttk.Button] = {}
        self.initGUI()

    def initGUI(self):
        self.title("TM")
        self.geometry("700x500")

        self.__create_frame("control")
        self.__pack_frame("control",pady=5)
        self.__create_label("control: ",self.get_frame("control"))
        self.__pack_label("control: ", fill=tk.X,side='left',padx=5)
        self.__create_button("quit",self.get_frame("control"))
        self.__pack_button("quit",fill=tk.X,side='left')

        self.__create_frame("filter")
        self.__pack_frame("filter")
        self.__create_label("filter text: ",self.get_frame("filter"))
        self.__pack_label("filter text: ",fill=tk.BOTH,expand=True,side='left')
        self.filter_text = ttk.Entry(self.get_frame("filter"))
        self.filter_text.pack(fill=tk.BOTH,expand=True,side='left')

        self.__create_frame("image")
        self.__pack_frame("image",fill=tk.BOTH,expand=True)

        self.image_label = ttk.Label(self.get_frame("image"))
        self.image_label.pack(fill=tk.NONE,expand=False,side='left',padx=5,pady=5)

        self.__create_frame("sim control",self.get_frame("image"),width=250,height=300)
        self.__pack_frame("sim control")

        self.__frames["sim control"].pack_propagate(False)

        self.text_files_npy = tk.Listbox(self.get_frame("sim control"),
                selectmode=tk.SINGLE,exportselection=False, width=30)
        self.text_files_npy.pack(fill=tk.BOTH,expand=False,side='top',padx=5,pady=5,anchor='c')
        
        self.num_samples_text = ttk.Entry(self.get_frame("sim control"))
        self.num_samples_text.pack(fill=tk.X,side='left',padx=5,pady=5)

        self.__create_button("generate",self.get_frame("sim control"))
        self.__pack_button("generate",fill=tk.X,side='left')

    def start(self):
        '''
        no-op for now
        '''
        pass

    # general widget creation methods used internally (model will not be adding any components)

    def __create_frame(self,name: str, parent: tk.Frame=None, *args, **kwargs):
        if parent == None:
            parent = self
        self.__frames[name] = ttk.Frame(parent, *args, **kwargs)

    def __create_label(self,name: str, parent: tk.Frame=None, *args, **kwargs):
        if parent == None:
            parent = self
        self.__labels[name] = ttk.Label(parent,*args, text=name, **kwargs)

    def __create_button(self,name: str, parent: tk.Frame=None, *args, **kwargs):
        if parent == None:
            parent = self
        self.__buttons[name] = ttk.Button(parent, *args, text=name, **kwargs)

    def __pack_frame(self, name: str, **kwargs):
        self.__frames[name].pack(**kwargs)

    def __pack_label(self, name: str, **kwargs):
        self.__labels[name].pack(**kwargs)

    def __pack_button(self, name: str, **kwargs):
        self.__buttons[name].pack(**kwargs)

    def get_frame(self,name:str) -> ttk.Frame:
        return self.__frames[name]

    def button_cmd(self,name: str, cmd: Callable):
        self.__buttons[name].configure(command=cmd)

    def filter_cmd(self, cmd: Callable):
        self.filter_text.bind("<KeyRelease>",cmd)

    def display_image(self, image: ImageTk.PhotoImage):
        self.image_label.image = image
        self.image_label['image'] = self.image_label.image

    # data methods
    def display_file_lists(self, files_npy: list[str] = None):
        # print(files_npy)
        if files_npy == None:
            files_npy = []
        files_npy.sort()
        print(files_npy)
        self.text_files_npy.configure(listvariable=tk.StringVar(value=files_npy))
        print(self.text_files_npy.get(0))
        # if len(files_npy) > 0:
        #     self.text_files_npy.selection_set(0)

    def get_selected_file_converted_files(self, event: tk.Event=None) -> tuple[str,int,int]:
        text_files_npy = self.text_files_npy
        if tk.Event != None:
            text_files_npy = event.widget
        return (text_files_npy.get(text_files_npy.curselection()),
            self.get_frame("image").winfo_width() - self.get_frame("image control").winfo_width() - 12,
            self.get_frame("image").winfo_height() - 12)

    def get_filter_text(self, event: tk.Event) -> str:
        return event.widget.get()

    def get_samples_info(self) -> tuple[str,int]:
        return (self.text_files_npy.get(self.text_files_npy.curselection()),
                int(self.num_samples_text.get()))