import csv

path = "/Users/saeki-takaaki/Documents/LAB/research_data/10-16/Ar02/1_94us/data1.asc"
with open(path) as f:
    while True:
        s_line = f.readline()
        s_list = s_line.split(',')
        with open('data1.csv', 'a') as g:
            writer = csv.writer(g)
            writer.writerow(s_list)
        if not s_line:
            break
