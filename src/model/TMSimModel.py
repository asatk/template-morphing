#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

from defs import PROJECT_DIR

import os
import numpy as np
from PIL import Image, ImageTk

class TMSimModel():
    
    def __init__(self):
        self.files_npy = []

    def filter_files(self, filter_str: str) -> tuple[list[str]]:
        files_npy_filtered = [x for x in self.files_npy if filter_str in x]
        return (files_npy_filtered,)

    def get_file_lists(self) -> tuple[list[str]]:
        return (self.files_npy,)

    def start(self):
        self.files_npy = os.listdir(PROJECT_DIR+"/out/npy")

    def generate_samples(self, file_name: str, num_samples: int):
        samples: np.ndarray = np.load(PROJECT_DIR+"/out/npy/"+file_name)
        samples_flat = samples.flatten()
        print(np.sum(samples_flat))
        random_indices_flat = np.random.choice(a=samples_flat.size,p=samples_flat,size=num_samples)
        random_indices = np.unravel_index(random_indices_flat, samples.shape)

    def display(self, image_name: str, width: int, height: int) -> ImageTk.PhotoImage:
        image_jpg = Image.open(PROJECT_DIR+"/out/%s/"%(self.image_file_mode)+image_name)
        image_jpg = image_jpg.resize((width,height),Image.ANTIALIAS)
        # print((width,height))
        image_tk = ImageTk.PhotoImage(image_jpg)
        return image_tk