from abc import ABC, abstractmethod

class V2MAdapter(ABC):
    @abstractmethod
    def update_file_lists(self):
        pass