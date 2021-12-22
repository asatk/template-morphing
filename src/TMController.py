from view.TMView import TMView
from view.V2MAdapter import V2MAdapter
from model.TMModel import TMModel
from model.M2VAdapter import M2VAdapter

import tkinter as tk

class TMController():

    class m2v(M2VAdapter):
        def init_file_lists(self, files_all: list[str], files_added: list[str]):
            pass

    class v2m(V2MAdapter):
        def update_file_lists(self):
            return self.model.get_file_lists()

    def __init__(self):
        self.view = TMView(self.v2m())
        self.model = TMModel(self.m2v())
    
    def start(self):
        self.model.start()
        self.view.start()
        self.init_buttons()
        
    def init_buttons(self):
        view = self.view
        model = self.model

        #define commands that get event from view, process data in model, and update view
        update_cmd = lambda:view.update_label(model.update_label(view.input.get()))
        quit_cmd = view.quit
        add_files_cmd = lambda:view.update_file_lists(*model.update_file_lists(view.get_selected_files()))

        #set buttons to respective commands
        view.button_cmd("press",update_cmd)
        view.button_cmd("quit",quit_cmd)
        view.button_cmd("add file(s)",add_files_cmd)

        view.update_file_lists(model.get_file_lists())

if __name__ == '__main__':
    controller = TMController()
    controller.start()