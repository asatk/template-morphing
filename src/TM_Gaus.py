'''

2D gaussian grid phi x omega ML Generation

'''
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from defs import PROJECT_DIR

plt.ion()

# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = 8

# seeds
seed = 100
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(seed)

# labels
labels_phi_real = np.linspace(300,3000,num=10,endpoint=True)
labels_phi_all = np.linspace(0,3000,num=1000+1,endpoint=True)
labels_omega_real = np.linspace(0.2,2.0,num=10,endpoint=True)
labels_omega_all = np.linspace(0,2.0,num=1000+1,endpoint=True)

# # load numpy histogram data
# gaus_grid = []
# for f in os.listdir(PROJECT_DIR+"/out/hist_npy/"):
#     arr_temp = np.load(PROJECT_DIR+"/out/hist_npy/"+f)
#     gaus_grid.append(arr_temp)

# load json hyperparameters
hpars_path = PROJECT_DIR+"/src/hyperparams.json"
with open(hpars_path) as hpars_json:
    hpars = json.load(hpars_json)
    vars = hpars['vars']
    kappas = hpars['kappas']
    sigmas = hpars['sigmas']

# learning parameters
n_vars = 2
assert n_vars == len(vars)
n_samples = len(os.listdir(PROJECT_DIR+"/out/hist_npy/"))
n_epochs = 10000
lr_g = 5e-5
lr_d = 5e-5
batch_size_D = 5
batch_size_G = 128
threshold_type = "hard"
soft_threshold = 1e-3

# load data and labels in
train_data = [0]*n_samples
train_labels = np.zeros((n_samples,n_vars))

# associate each file's data with its label
data_mapping_path = "data_mapping.json"
with open(data_mapping_path) as json_file:
    data_label_map = json.load(json_file)
    # load each file from the data mapping
    for i,f in enumerate(data_label_map.keys()):
        arr = np.load(f)
        train_data[i] = np.load(f)
        train_labels[i] = data_label_map[f]

# generate a batch of target labels from all real labels
batch_target_labels_raw = np.ndarray((batch_size_D,n_vars))
for v in range(n_vars):
    batch_target_labels_raw[:,v] = np.random.choice(train_labels[:,v],size=batch_size_D,replace=True).transpose()

# give each target label some noise
batch_epsilons = np.ndarray((batch_size_D,n_vars))
for v in range(n_vars):
    batch_epsilons[:,v] = np.random.normal(0,sigmas[v],batch_size_D).transpose()

batch_target_labels = batch_target_labels_raw + batch_epsilons

# find real data within bounds
batch_real_index = np.zeros((batch_size_D,n_vars))
batch_fake_labels = np.zeros((batch_size_D,n_vars))

'''
train_labels_transpose = train_labels.transpose()
print "transpose",train_labels_transpose[0]
target_label = batch_target_labels[:,0]
print "target label 0",target_label
diff = train_labels_transpose - batch_target_labels[:,0]
print "diff",diff[0]
diff_div = np.divide(diff,kappas)
print "diff_div",diff_div[0]
diff_sq = np.square(diff_div)
print "diff squared",diff_sq[0]
diff_sum = np.apply_over_axes(np.sum,diff_sq,1)
print "diff_sum",diff_sum[0]

# print diff_sum<1

# print "WHERE",np.where(diff_sum<=1,train_labels_transpose,0)

print "kappas",kappas
'''

def vicinity_hard(x,sample):
    return np.sum(np.square(np.divide(x-sample,kappas)))

def vicinity_soft(x,sample):
    return np.exp(-1*vicinity_hard(x,sample))

# iterate over the whole batch
j=0
reshuffle_count = 0
while j < batch_size_D:
    print(j)
    
    batch_target_label = batch_target_labels[j]
    print("\nbatch target label",batch_target_label)


    hard_labels = []
    soft_labels = []

    print("TRAIN LABELS",train_labels)

    # print("HARD -> ",np.apply_along_axis(
    #             vicinity_hard,0,train_labels,sample=batch_target_label))
    # print("HARD TEST ->",np.apply_along_axis(
    #             vicinity_hard,0,train_labels,sample=batch_target_label) <= 1)

    # print(np.where(np.apply_along_axis(
    #     vicinity_hard,0,train_labels,sample=batch_target_label) <= 1))

    label_vicinity = np.ndarray(train_labels.shape)

    # iterate over every training label to find which are within the vicinity of the batch labels
    for i in range(len(train_labels)):
        train_label = train_labels[i]
        # print "train label",train_label
        # print "\nSAMPLE",sample
        hard = np.apply_along_axis(vicinity_hard,0,train_label,sample=batch_target_label)
        label_vicinity[i] = hard
        # print "HARD",hard
        # soft = np.apply_along_axis(vicinity_soft,0,train_label,sample=batch_target_label)
        # print "SOFT",soft

        # if train_label[0] == batch_target_labels_raw[j,0]:
        #     print("%04.0f,%01.3f|H\t%2.2f\t|S\t%1.4f\t|"%(train_label[0],train_label[1],hard,soft))
        # if hard <= 1:
        #     hard_labels.append(train_label)
        # if soft >= soft_threshold:
        #     soft_labels.append(train_label)
    
    where = set(np.where(label_vicinity <= 1)[0])
    print("where\n")
    for i in where:
        print(i)
        print(train_labels[i])
        print(train_data[i])

    # reshuffle the batch target labels, redo that sample
    if len(hard_labels) < 1 or len(soft_labels) < 1:
        reshuffle_count += 1
        print("RESHUFFLE COUNT",reshuffle_count)
        batch_epsilons_j = np.zeros((n_vars))
        for v in range(n_vars):
            batch_epsilons_j[v] = np.random.normal(0,sigmas[v],1).transpose()
        batch_target_labels[j] = batch_target_labels_raw[j] + batch_epsilons_j
        continue

    print("RESHUFFLE COUNT",reshuffle_count)
    print("BATCH SAMPLE",batch_target_label)
    print("HARD LABELS",hard_labels)
    print("SOFT LABELS",soft_labels)

    # set the bounds for random draw of possible fake labels
    if threshold_type == "hard":
        lb = batch_target_labels[j] - kappas
        ub = batch_target_labels[j] + kappas
    else:
        lb = batch_target_labels[j] - np.sqrt(-1*np.log(soft_threshold)*kappas)
        ub = batch_target_labels[j] + np.sqrt(-1*np.log(soft_threshold)*kappas)


    # generate fake labels
    for v in range(n_vars):
        batch_fake_labels[j,v] = np.random.uniform(lb[v],ub[v],size=1)[0]
    
    print("HARD LABELS",hard_labels)
    print("HARD LABELS INDEX",hard_labels[0])
    print("BATCH FAKE LABELS:\n",batch_fake_labels)
    print("BATCH TARGET LABELS:\n",batch_target_labels)
    j += 1
    reshuffle_count = 0