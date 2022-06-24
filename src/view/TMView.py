#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Callable

from PIL import ImageTk
from matplotlib.pyplot import text

from defs import PROJECT_DIR

class TMView(tk.Toplevel):

    def __init__(self, root: tk.Tk, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.__frames: dict[str,ttk.Frame] = {}
        self.__labels: dict[str,ttk.Label] = {}
        self.__buttons: dict[str,ttk.Button] = {}
        self.initGUI()

    def initGUI(self):
        self.title("Display Window")
        self.geometry("700x500")

        self.__create_frame("control")
        self.__pack_frame("control")
        self.__create_label("control: ",self.get_frame("control"))
        self.__pack_label("control: ", fill=tk.X,side='left',padx=5)
        self.__create_button("quit",self.get_frame("control"))
        self.__pack_button("quit",fill=tk.X,side='left')
        self.__create_button("convert",self.get_frame("control"))
        self.__pack_button("convert",fill=tk.X,side='left')

        self.__create_frame("filter")
        self.__pack_frame("filter")
        self.__create_label("filter text: ",self.get_frame("filter"))
        self.__pack_label("filter text: ",fill=tk.BOTH,expand=True,side='left')
        self.filter_text = ttk.Entry(self.get_frame("filter"))
        self.filter_text.pack(fill=tk.BOTH,expand=True,side='left')

        self.__create_frame("data")
        self.__pack_frame("data",fill=tk.BOTH,expand=False,padx=5,pady=5)

        self.text_files_all = tk.Listbox(self.get_frame("data"),selectmode=tk.MULTIPLE, exportselection=False)
        self.text_files_all.pack(fill=tk.BOTH,expand=True,side='left',padx=5)

        self.__create_frame("data buttons", self.get_frame("data"))
        self.__pack_frame("data buttons",fill=None,expand=False,side='left')

        self.__create_button("add file(s)",self.get_frame("data buttons"))
        self.__pack_button("add file(s)",fill=tk.X, expand=False,side='top')
        self.__create_button("remove file(s)",self.get_frame("data buttons"))
        self.__pack_button("remove file(s)",fill=tk.X, expand=False, side='bottom')

        self.text_files_added = tk.Listbox(self.get_frame("data"),selectmode=tk.MULTIPLE, exportselection=False)
        self.text_files_added.pack(fill=tk.BOTH,expand=True,side='left',padx=5)

        self.__create_frame("image")
        self.__pack_frame("image",fill=tk.BOTH,expand=True)

        self.image_label = ttk.Label(self.get_frame("image"))
        self.image_label.pack(fill=tk.NONE,expand=False,side='left',padx=5,pady=5)

        self.__create_frame("image control",self.get_frame("image"),width=250,height=300)
        self.__pack_frame("image control")

        self.__frames["image control"].pack_propagate(False)

        self.text_files_converted = tk.Listbox(self.get_frame("image control"),
                selectmode=tk.SINGLE,exportselection=False, width=30)
        self.text_files_converted.pack(fill=tk.Y,expand=False,side='top',padx=5,anchor='c')

        # self.file_type_combobox = ttk.Combobox(self.get_frame("image control"),state='readonly')
        # self.file_type_combobox.pack(fill=tk.Y,side='top',pady=5)

        self.__create_button("image dir",self.get_frame("image control"))
        self.__pack_button("image dir",fill=tk.X,expand=False,side='top')

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

    def display_cmd(self, cmd: Callable):
        self.text_files_converted.bind("<<ListboxSelect>>",cmd)

    def display_prev_cmd(self, cmd: Callable):
        self.text_files_converted.bind("<Left>",cmd)
        self.text_files_converted.bind("<Up>",cmd)

    def display_next_cmd(self, cmd: Callable):
        self.text_files_converted.bind("<Right>",cmd)
        self.text_files_converted.bind("<Down>",cmd)

    def display_first_cmd(self, cmd: Callable):
        self.text_files_converted.bind("<Home>",cmd)

    def display_last_cmd(self, cmd: Callable):
        self.text_files_converted.bind("<End>",cmd)

    # def set_file_types(self, file_types: list[str]):
    #     self.file_type_combobox.configure(values=file_types)
    #     self.file_type_combobox.current(2)

    # def file_type_cmd(self, cmd: Callable):
    #     self.file_type_combobox.bind("<<ComboboxSelected>>",cmd)

    def display_image(self, image: ImageTk.PhotoImage):
        self.image_label.image = image
        self.image_label['image'] = self.image_label.image

    # data methods
    def display_file_lists(self, files_all: 'list[str]' = None, files_added: 'list[str]' = None, files_converted: 'list[str]' = None):
        if files_all == None:
            files_all = []
        if files_added == None:
            files_added = []
        if files_converted == None:
            files_converted = []
        files_all.sort()
        files_added.sort()
        files_converted.sort()
        self.text_files_all.configure(listvariable=tk.StringVar(value=files_all))
        self.text_files_added.configure(listvariable=tk.StringVar(value=files_added))
        self.text_files_converted.configure(listvariable=tk.StringVar(value=files_converted))

    def get_selected_files_all_files(self) -> 'list[str]':
        indices_all = self.text_files_all.curselection()
        files_all = [self.text_files_all.get(i) for i in indices_all]
        return files_all

    def get_selected_files_added_files(self) -> 'list[str]':
        indices_added = self.text_files_added.curselection()
        files_added = [self.text_files_added.get(i) for i in indices_added]
        return files_added

    def get_selected_file_converted_files(self, event: tk.Event=None, offset=0, idx=None) -> 'tuple[str,int,int]':
        text_files_converted = self.text_files_converted
        selection = text_files_converted.curselection()
        if idx is None:
            if selection == tuple():
                selection = (0, )
                if offset == -1:
                    idx = tk.END
                elif offset == 1:
                    idx = 0
            else:
                idx = selection[0] + offset

        text_files_converted.selection_clear(0,tk.END)
        text_files_converted.selection_set(idx)
        text_files_converted.see(idx)
        text_files_converted.activate(idx)
        text_files_converted.selection_anchor(idx)

        if event is not None:
            text_files_converted = event.widget
        return (text_files_converted.get(selection),
            self.get_frame("image").winfo_width() - self.get_frame("image control").winfo_width() - 12,
            self.get_frame("image").winfo_height() - 12)

    def get_filter_text(self, event: tk.Event) -> str:
        return event.widget.get()

    def get_image_directory(self):
        val = filedialog.askdirectory(initialdir=PROJECT_DIR + "/out",title='select image dir')
        self.text_files_converted.focus()
        return val
