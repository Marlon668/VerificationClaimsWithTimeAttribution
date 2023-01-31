import sys

import os

import baseModel.Claim as Claim

'''
    This function writes for each evidence the possible optional data (only if 
    there is no date available at the beginning of the evidence text in the 
    form of month day, year) to a file with as name the snippet number 
    in the folder processedSnippets/$claimId
    @param mode: Dev, Train or Test
    @param path: path to the given dataset
'''
def writeOptionalTimeSnippet(mode,path):
    if not (os.path.exists(os.pardir + "/processedSnippets")):
        os.mkdir(os.pardir + "/processedSnippets")
    if mode != 'Test':
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5], elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                    "snippets/",'non', 'non', 'non', "None")
                for snippet in claim.snippets:
                    claim.processDateSnippet(snippet)
    else:
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], 'None',elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11],
                                    "snippets/", 'non', 'non', 'non', "None")
                for snippet in claim.snippets:
                    claim.processDateSnippet(snippet)

'''
    This function writes the raw publication date of each claim in the given dataset to
    a file named tenses-§mode.txt
    @params mode: mode of the given dataset (Dev, Train or Test°
    @params path: path to the dataset
'''
def writeClaimDate(mode,path):
    if mode != 'Test':
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5], elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                    "snippets/",'non', 'non', 'non', "None")
                time = claim.claimDate
                f = open("tenses-" + mode + ".txt", "a",
                         encoding="utf-8")
                f.write(
                    claim.claimID+"\t"+str(time) + "\n")
                f.close()
    else:
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], 'None',elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11],
                                    "snippets/", 'non', 'non', 'non', "None")
                time = claim.claimDate
                f = open("tenses-"+mode+".txt", "a",
                         encoding="utf-8")
                f.write(
                    claim.claimID + "\t" + str(time) + "\n")
                f.close()

'''
    This function is needed to normalise the timexes in the text by Heideltime
    It saves the publication date of each claim together with that of its evidence
    in a file, named "data/data.txt" together with the IDs (claimId and evidenceNumber)
    saved in the file "data/indices.txt"
    @params mode: mode of the given dataset (Dev, Train or Test)
    @params path: path to the dataset
'''
def writeDateToFile(mode,path):
    if not (os.path.exists(os.pardir + "/data")):
        os.mkdir(os.pardir + "/data")
    if mode != 'Test':
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5], elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                    "snippets/",'non', 'non', 'non', "None")
                date = str(claim.claimID) + '\t'
                claim.readPublicationDate()
                indices = str(claim.claimID) + '\t'
                if claim.claimDate != None:
                    date += 'Claim -D ' + str(claim.claimDate) + '\t'
                else:
                    date += 'Claim -D ' + 'None' +'\t'
                for snippet in claim.snippets:
                    snippet.readPublicationDate()
                    indices += snippet.number +'\t'
                    if snippet.publishTime != None:
                        date += str(snippet.publishTime) + '\t'
                    else:
                        date += 'None' + '\t'
                date = date[:len(date) - 1]
                indices = indices[:len(indices) - 1]
                f = open("data/data.txt" , "a",
                         encoding="utf-8")
                f.write(
                    date + "\n")
                f.close()
                f = open("data/indices.txt", "a",
                         encoding="utf-8")
                f.write(
                    indices + "\n")
                f.close()
    else:
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], 'None',elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11],
                                    "snippets/", 'non', 'non', 'non', "None")
                date = str(claim.claimID) + '\t'
                indices = str(claim.claimID) + '\t'
                claim.readPublicationDate()
                if claim.claimDate!=None:
                    date += 'Claim -D ' + str(claim.claimDate) + '\t'
                else:
                    date += 'Claim -D ' + 'None' +'\t'
                for snippet in claim.snippets:
                    snippet.readPublicationDate()
                    indices += snippet.number + '\t'
                    if snippet.publishTime!=None:
                        date += str(snippet.publishTime) + '\t'
                    else:
                        date += 'None' + '\t'
                date = date[:len(date) - 2]
                indices = indices[:len(indices) - 2]
                f = open("data/data.txt", "a",
                         encoding="utf-8")
                f.write(
                    date + "\n")
                f.close()
                f = open("data/indices.txt", "a",
                         encoding="utf-8")
                f.write(
                    indices + "\n")
                f.close()

if __name__ == "__main__":
    if sys.argv[1] == "writeClaimDate":
        writeClaimDate(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == "writeDateToFile":
        writeClaimDate(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "writeOptionalTimeSnippet":
        writeOptionalTimeSnippet(sys.argv[2], sys.argv[3])
    #writeDateToFile('Dev','dev.tsv')
    #writeDateToFile('Train','train.tsv')
    #writeDateToFile('Test','test.tsv')