import sys

import spacy
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.predictor import Predictor as pred
import torch
import multiprocessing

import baseModel.Claim as Claim

'''
Extract open information from the text of the claim and the evidence snippets
Argument 1 is the type of dataset: Dev, Train and Test
Argument 2 is the path the dataset
The openextraction of the claims and evidence snippets are saved in the OpenInformation folder or is created 
if this folder doesn't already exist 
'''
def openInformationExtraction(mode,path):
    nlp = spacy.load("en_core_web_sm")
    predictorOIE = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",cuda_device=0)
    predictorNER = pred.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz",cuda_device=0)
    if mode != 'Test':
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5], elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                    "snippets/",predictorOIE,predictorNER,nlp,"None")
                claim.processOpenInformation()
                for snippet in claim.snippets:
                    snippet.processOpenInformation()
    else:
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], 'None',elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11],
                                    "snippets/", predictorOIE, predictorNER, nlp, "none")
                claim.processOpenInformation()
                for snippet in claim.snippets:
                    snippet.processOpenInformation()
    print('Done OpenInformation ' + ' - ' + path)

def readOpenIE(mode,path):
    if mode!='Test':
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                    "snippets/", 'non', 'non', 'non', "None")
                openInfo = claim.readOpenInformationExtraction()
                print('claim - ' + claim.claimID)
                for element in openInfo:
                    for verb in element['verbs']:
                        print(verb['description'])
                        print(verb['tags'])
                        print(element['words'])

                for snippet in claim.snippets:
                    print('snippet : ' + str(snippet.number))
                    openInfo = snippet.readOpenInformationExtraction()
                    for element in openInfo:
                        for verb in element['verbs']:
                            print(verb['description'])
                            print(verb['tags'])
                            print(element['words'])
    else:
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], 'None', elements[2], elements[3], elements[4],
                                    elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11],
                                    "snippets/", 'non', 'non', 'non', "none")
                openInfo = claim.readOpenInformationExtraction()
                print('claim')
                for element in openInfo:
                    for verb in element['verbs']:
                        print(verb['description'])
                        print(verb['tags'])
                        print(element['words'])

                for snippet in claim.snippets:
                    print('snippet : ' + str(snippet.number))
                    openInfo = snippet.readOpenInformationExtraction()
                    for element in openInfo:
                        for verb in element['verbs']:
                            print(verb['description'])
                            print(verb['tags'])
                            print(element['words'])

if __name__ == "__main__":
    openInformationExtraction(sys.argv[1], sys.argv[2])