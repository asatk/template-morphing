#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

from model.TMModel import TMModel
from view.TMView import TMView
from view.TMWindow import TMWindow

class TMController():

    def __init__(self):
        self.model = TMModel()
        self.window = TMWindow()
    
    def start(self):
        self.model.start()
        self.init_cmds()
        self.init_display()
        self.window.start()
        
    def init_cmds(self):
        view = self.window.get_view()
        model = self.model

        #define commands that get event from view, process data in model, and update view
        quit_cmd = view.quit
        convert_cmd = lambda:view.display_file_lists(*model.convert_files())
        display_cmd = lambda event:view.display_image(model.display(*view.get_selected_file_converted_files(event=event)))
        add_files_cmd = lambda:view.display_file_lists(*model.add_files(view.get_selected_files_all_files()))
        remove_files_cmd = lambda:view.display_file_lists(*model.remove_files(view.get_selected_files_added_files()))
        filter_files_cmd = lambda event:view.display_file_lists(*model.filter_files(view.get_filter_text(event)))
        file_type_cmd = lambda event:view.display_file_lists(*model.set_image_file_mode(event.widget.get()))

        # set buttons to respective commands
        view.button_cmd("quit", quit_cmd)
        view.button_cmd("convert",convert_cmd)
        view.button_cmd("add file(s)", add_files_cmd)
        view.button_cmd("remove file(s)", remove_files_cmd)
        view.filter_cmd(filter_files_cmd)
        view.display_cmd(display_cmd)
        view.file_type_cmd(file_type_cmd)

    def init_display(self):
        view = self.window.get_view()
        model = self.model
        view.display_file_lists(*model.get_file_lists())

if __name__ == '__main__':
    controller = TMController()
    controller.start()