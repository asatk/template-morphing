import os
import re
import json

d = "/home/asatk/Documents/code/cern/TM/out/npy/"
file_list = os.listdir(d)

p_samp = re.compile(r"^SAMP.*$")
p = re.compile(r"(?!SAMP).+PH-(\d{4}).*OM-(\d)p(\d{3}).*")

data = {}
for f in file_list:
    if p_samp.match(f) is not None:
        continue
    m = p.match(f)
    phi = int(m.group(1))
    omega = int(m.group(2)) + float(m.group(3))/1000
    data[f] = [phi, omega]
    print(phi,omega)

with open("/home/asatk/Documents/code/cern/TM/src/model/data_mapping.json",'w') as json_file:
    json.dump(data,json_file, indent=4)