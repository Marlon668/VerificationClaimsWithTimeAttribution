import matplotlib.pyplot as plt
import numpy as np


def calculate(path):

    domains = ["abbc","afck","bove","chct","clck","faan","faly","fani","farg",
               "hoer","huca","mpws","obry","pomt","pose","para","peck","ranz","snes","goop",
            "thet","thal","vees","vogo","wast","tron"]
    gewichtDomain = {"abbc" : 43,"afck":43,"bove":29,"chct":36,"clck":3,"faan":11,
                     "faly" : 7,"fani":1,"farg":42,"goop":295,"hoer":131,"huca":3,"mpws":5,
                     "obry" : 7,"para":23,"peck":6,"pomt":1538,"pose":137,"ranz":2,"snes":45,
                     "thal":10,"thet":8,"tron":332,"vees":50,"vogo":65,"wast":19}
    #domains = ["chct"]
    #domains = ["abbc","afck","bove","chct","clck","faan","faly","fani","farg"]
    #domains = ["farg"]
    #domains = ["goop","pomt","snes","tron"]
    #domains = ["tron"]
    #domains = ["farg"]
    print(len(domains))
    lossPreprocessing = []
    lossFinetuning = []
    #for domain in domains:
    lossPreprocessing = []
    lossFinetuning = []
    for i in {1}:
        pathI = path+"/Loss"
        for domain in domains:
            iterationPreprocessing = 0
            iterationTrain = 0
            pathDomain = pathI + domain
            file = open(pathDomain, 'r', encoding='utf-8')
            lines = file.readlines()
            for entry in lines:
                if entry.find("Preprocessing stap") != -1 and iterationPreprocessing<100:
                    if iterationPreprocessing < len(lossPreprocessing):
                        if iterationPreprocessing == 54:
                            print("Domain : " + domain)
                            print(float(entry.split(" - ")[1]) / gewichtDomain[domain])
                        lossPreprocessing[iterationPreprocessing] = (lossPreprocessing[iterationPreprocessing] + float(entry.split(" - ")[1])/gewichtDomain[domain])/2
                    else:
                        lossPreprocessing.append(float(entry.split(" - ")[1]))
                    iterationPreprocessing += 1
                    print(iterationPreprocessing)
                else:
                    if entry.find("Stap") != -1 and iterationTrain <100:
                        if iterationTrain < len(lossFinetuning) :
                            lossFinetuning[iterationTrain] = (lossFinetuning[iterationTrain]+float(entry.split(" - ")[1]))/2
                        else:
                            lossFinetuning.append(float(entry.split(" - ")[1]))
                        iterationTrain += 1
            #x = np.linspace(0, len(lossFinetuning), num=len(lossFinetuning))
            #plt.plot(x, lossFinetuning)
            #plt.title("Loss finetuning ")
            #plt.savefig(pathI + str(domain) + ".png")
            #plt.show()
    print(min(lossPreprocessing))
    print(sum(lossPreprocessing[50:])/len(lossPreprocessing[50:]))
    x = np.linspace(0,len(lossPreprocessing),num=len(lossPreprocessing))
    plt.plot(x,lossPreprocessing)
    plt.title("Loss preprocessing ")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(pathI + "loss" + ".png")
    plt.show()
    x = np.linspace(0,len(lossFinetuning),num=len(lossFinetuning))
    plt.plot(x,lossFinetuning)
    plt.title("Loss finetuning ")
    #plt.savefig(pathI + str(domain) + ".png")
    plt.show()



basispath = "C:/Users/Marlon/Documents/losses/"
calculate(basispath + "lossBert75")
