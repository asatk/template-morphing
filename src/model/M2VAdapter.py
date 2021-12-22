from abc import abstractmethod


from abc import ABC, abstractmethod

class M2VAdapter(ABC):

    @abstractmethod
    def init_file_lists(self, files_all: list[str], files_added: list[str]):
        pass