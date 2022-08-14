import os


def split(path,name,lines_per_file=3000):
    smallfile = None
    with open(path,'r',encoding='utf-8') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = name.format(lineno + lines_per_file)
                smallfile = open(small_filename, "w",encoding="utf-8")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

def splitInDomains(path,name,domains):
    for domain in domains:
        fileName = name.format(domain)
        domainFile = open(fileName,"w",encoding="utf-8")
        with open(path,'r',encoding='utf-8') as file:
            for line in file:
                elements = line.split('\t')
                if elements[0].split('-')[0]==domain:
                    domainFile.write(line)
            domainFile.close()

split("time/train/train.tsv","time/train/train-{}.tsv")