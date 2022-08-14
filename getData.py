def getText(domains,pathToFile):
    for domain in domains:
        print(domain)
        with open(pathToFile,'r',encoding='utf-8') as faults:
            with open("faults/"+domain+".tsv","w",encoding='utf-8') as domainF:
                lines = faults.readlines()
                for line in lines:
                    if line.split('-')[0] == domain:
                        done = False
                        with open("time/train/train-"+domain+".tsv", 'r', encoding='utf-8') as train:
                            linesT = train.readlines()
                            for lineT in linesT:
                                elements = lineT.split('\t')
                                if elements[0] == line.replace('\n',''):
                                    domainF.write(lineT)
                                    done = True
                                    break
                        if not done:
                            with open("time/dev/dev-" + domain + ".tsv", 'r', encoding='utf-8') as train:
                                linesT = train.readlines()
                                for lineT in linesT:
                                    elements = lineT.split('\t')
                                    if elements[0] == line.replace('\n',''):
                                        domainF.write(lineT)
                                        done = True
                                        break
                        if not done:
                            with open("time/test/test-" + domain + ".tsv", 'r', encoding='utf-8') as train:
                                linesT = train.readlines()
                                for lineT in linesT:
                                    elements = lineT.split('\t')
                                    if elements[0] == line.replace('\n',''):
                                        domainF.write(lineT)
                                        done = True
                                        break

domains = {"abbc","afck"}
getText(domains,"faultDataSetIteratie2.txt")