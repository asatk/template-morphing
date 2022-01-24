#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

import sys

from model.TMModel import TMModel
from model.TMSimModel import TMSimModel
from view.TMView import TMView
from view.TMSimView import TMSimView
import tkinter as tk

class TMController():

    def __init__(self, mode: str):
        self.__root = tk.Tk()
        self.__root.withdraw()
        self.mode = mode
        if int(mode[-1]):
            self.model = TMModel()
            self.view = TMView(self.__root)

        if int(mode[-2]):
            self.model_sim = TMSimModel()
            self.view_sim = TMSimView(self.__root)
    
    def start(self):
        if int(self.mode[-1]):
            self.model.start()
            self.init_cmds()
            self.init_display()
            self.view.start()

        if int(self.mode[-2]):
            self.model_sim.start()
            self.init_cmds_sim()
            self.init_display_sim()
            self.view_sim.start()
        
        self.__root.mainloop()
        
    def init_cmds(self):
        view = self.view
        model = self.model

        #define commands that get event from view, process data in model, and update view
        quit_cmd = view.quit
        convert_cmd = lambda:view.display_file_lists(*model.convert_files())
        display_cmd = lambda event:view.display_image(model.display(*view.get_selected_file_converted_files(event=event)))
        add_files_cmd = lambda:view.display_file_lists(*model.add_files(view.get_selected_files_all_files()))
        remove_files_cmd = lambda:view.display_file_lists(*model.remove_files(view.get_selected_files_added_files()))
        filter_files_cmd = lambda event:view.display_file_lists(*model.filter_files(view.get_filter_text(event)))
        image_dir_cmd = lambda:view.display_file_lists(*model.set_image_directory(view.get_image_directory()))
        # file_type_cmd = lambda event:view.display_file_lists(*model.set_image_file_mode(event.widget.get()))

        # set buttons to corresponding commands
        view.button_cmd("quit", quit_cmd)
        view.button_cmd("convert", convert_cmd)
        view.button_cmd("add file(s)", add_files_cmd)
        view.button_cmd("remove file(s)", remove_files_cmd)
        view.filter_cmd(filter_files_cmd)
        view.display_cmd(display_cmd)
        view.button_cmd("image dir", image_dir_cmd)
        # view.file_type_cmd(file_type_cmd)

    def init_display(self):
        view = self.view
        model = self.model
        # view.set_file_types(model.get_file_types())
        view.display_file_lists(*model.get_file_lists())
        
    def init_cmds_sim(self):
        view = self.view_sim
        model = self.model_sim

        #define commands that get event from view, process data in model, and update view
        quit_cmd = view.quit
        generate_cmd = lambda:model.generate_samples(*view.get_samples_info())
        experiment_cmd = model.train_GAN

        # set buttons to corresponding commands
        view.button_cmd("quit", quit_cmd)
        view.button_cmd("generate",generate_cmd)
        view.button_cmd("run experiment",experiment_cmd)

    def init_display_sim(self):
        view = self.view_sim
        model = self.model_sim
        view.display_file_lists(*model.get_file_lists())

if __name__ == '__main__':
    #the mode is a base-10 num representing the bit-string for which apps (bits) one wishes to run
    #3 -> 11 (both); 2 -> 10 (analysis app); 1 -> 01 (main app)
    mode = ("{:02b}".format(int(sys.argv[1])) if len(sys.argv) > 1 else bin(3))
    controller = TMController(mode)
    controller.start()