import csv

file1 = "competition_result_0420_135106_90836.csv"
file2 = "/root/app/competition_result_0423_065807.csv"
    
def csv2dict(path):
    ret_dict = dict()
    with open(path, "r") as fh:
        reader = csv.reader(fh)
        for row in reader: ret_dict[row[0]] = row[1]
    return ret_dict

def comp_dict(dict1, dict2, ratio=False):
    count = 0
    for key in dict1:
        if dict1[key] == dict2[key]: count += 1
    if ratio: count /= len(dict1)
    return count
    

def comp_file(file1, file2, ratio=False):
    dict1 = csv2dict(file1)
    dict2 = csv2dict(file2)
    return comp_dict(dict1, dict2, ratio=ratio)

acc = comp_file(file1, file2, ratio=True)
print(acc)

        
    