#todo: Use regex to only have the thigns I need. (find "not")

# write a bash script.

# run the bash script.

# make "processed" using read_kgmls.py

file = open("HTML.txt")
bash = open("get_kgml.sh", mode='w')

for line in file:
    name = line.strip()
    command = "curl -s \"https://rest.kegg.jp/get/hsa{}/kgml\" -o hsa{}.kgml\n".format(name, name)
    bash.write(command)