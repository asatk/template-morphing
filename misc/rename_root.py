import os

d = "/home/asatk/Documents/code/cern/TM/root/"
print("Current Working Directory:" + d)
l = os.listdir(d)


run_name = ""
phi = ""
omega = ""
for f in l:
    print d+f
    if raw_input("skip? y/n") == "y":
        continue
    
    temp = raw_input("name this run: ")
    if temp != "":
        run_name = temp
    temp = raw_input("phi mass (ex: 100): ")
    if temp != "":
        phi = temp
    temp = raw_input("omega mass (ex: 0p550): ")
    if temp != "":
        omega = temp
    file_name = "{0}_PH-{1}_OM-{2}.root".format(run_name,phi,omega)
    print file_name
    os.rename(d+f,d+file_name)
