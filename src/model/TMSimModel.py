#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

from defs import PROJECT_DIR

import os
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import subprocess

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

    def generate_samples(self, file_name: str, num_samples: int, seed: int = 100) -> np.ndarray:
        # figure out how to set seed in numpy
        # np.random.seed(100)
        data: np.ndarray = np.load(PROJECT_DIR+"/out/npy/"+file_name)
        samples_flat = data.flatten()
        print(np.sum(samples_flat))
        random_indices_flat = np.random.choice(a=samples_flat.size,p=samples_flat,size=num_samples)
        random_indices = np.unravel_index(random_indices_flat, data.shape)

        samples = np.zeros(data.shape)
        samples[random_indices] = data[random_indices]

        out_file_png = PROJECT_DIR+"/out/png/SAMPLE_"+file_name[:-3]+"png"
        out_file_npy = PROJECT_DIR+"/out/npy/SAMPLE_"+file_name

        max_val = np.amax(samples)

        plt.imsave(out_file_png,samples.T,cmap="gray",vmin=0.,vmax=max_val,format="png",origin="lower")
        np.save(out_file_npy,samples,allow_pickle=False)

        return samples

    def display(self, image_name: str, width: int, height: int) -> ImageTk.PhotoImage:
        image_jpg = Image.open(PROJECT_DIR+"/out/%s/"%(self.image_file_mode)+image_name)
        image_jpg = image_jpg.resize((width,height),Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(image_jpg)
        return image_tk

    def train_GAN(self):
        cmd_str = "model/scripts/run_train.sh "

        file_output = open(PROJECT_DIR+"/out/log/train_out.txt",'w')
        file_error = open(PROJECT_DIR+"/out/log/train_err.txt",'w')

        # print("os environ path")
        # print(os.environ["PATH"])
        # new_path = os.environ["PATH"].replace("py3CCGAN","cern2.7")
        # new_environ = os.environ.copy()
        # new_environ["PATH"] = new_path

        process = subprocess.Popen(cmd_str, shell=True,
                stdout=file_output, stderr=file_error)
        output, error = process.communicate()
        
        file_output.close()
        file_error.close()

    def gen_GAN(self):
        pass