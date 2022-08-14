import statistics

def calculate(path):

    domains = ["abbc","afck","bove","chct","clck","faan","faly","fani","farg",
           "goop", "hoer","huca","mpws","obry","para","peck","pomt","pose","ranz",
            "snes","thet","thal","tron","vees","vogo","wast"]

    print(len(domains))
    resultsMicro = dict()
    resultsMacro = dict()
    for domain in domains:
        resultsMicro[domain] = []
        resultsMacro[domain] = []
    for i in ["A","B","E"]:
        pathI = path+i+ "/"
        for domain in domains:
            pathDomain = pathI + domain
            file = open(pathDomain, 'r', encoding='utf-8')
            lines = file.readlines()
            for entry in lines:
                if entry.find("Micro") != -1:
                    resultsMicro[domain].append(float(entry.split("Micro - ")[1]))
                if entry.find("Macro") != -1:
                    resultsMacro[domain].append(float(entry.split("Macro - ")[1]))
            file.close()
    totalResultsMicro = [0.0,0.0,0.0]
    totalResultsMacro = [0.0,0.0,0.0]
    print("Average results for " + path)
    for domain in domains:
        print(domain)
        print("Micro : " + str(statistics.mean(resultsMicro[domain])))
        print("Macro : " + str(statistics.mean(resultsMacro[domain])))
        print("Standaardafwijking")
        print("Micro : " + str(statistics.stdev(resultsMicro[domain])))
        print("Macro : " + str(statistics.stdev(resultsMacro[domain])))
        totalResultsMicro = [x + y for x, y in zip(totalResultsMicro, resultsMicro[domain])]
        totalResultsMacro = [x + y for x, y in zip(totalResultsMacro, resultsMacro[domain])]

    print("Voor alle modellen")
    print("Micro : " + str(statistics.mean([x/26 for x in totalResultsMicro])))
    print("Macro : " + str(statistics.mean([x/26 for x in totalResultsMacro])))
    print("Maximum")
    print("Micro : " + str(max([x / 26 for x in totalResultsMicro])))
    print("Macro : " + str(max([x / 26 for x in totalResultsMacro])))
    print("Standaardafwijking")
    print("Micro : " + str(statistics.stdev([x/26 for x in totalResultsMicro])))
    print("Macro : " + str(statistics.stdev([x/26 for x in totalResultsMacro])))

#basispath = "C:/Users/Marlon/Documents/accuracyFinal/basis/"
basispath = "resultsNormalText/"
calculate(basispath + "epochsEverything2040")


