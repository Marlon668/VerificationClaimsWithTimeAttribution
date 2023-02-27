import os
import sys

import spacy
from allennlp.predictors.predictor import Predictor as pred
import baseModel.Claim as Claim

def editArticleTextWithUppercaseEditing(mode,path,withTitle=True,withPretext=True,pathToSave="text"):
    nlp = spacy.load("en_core_web_sm")
    predictorOIE = "None"
    predictorNER = pred.from_path(
        "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz",cuda_device=-1)
    coreference = "None"

    if not (os.path.exists(pathToSave)):
        os.mkdir(pathToSave)
    if withPretext and not (os.path.exists("pretext")):
        os.mkdir("pretext")
    if mode != 'Test':
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                    "snippets/", predictorOIE, predictorNER, nlp, coreference)
                if not (os.path.exists(pathToSave + "/" + claim.claimID)):
                    os.mkdir(pathToSave + "/" + claim.claimID)
                if withPretext and not (os.path.exists("pretext" + "/" + claim.claimID)):
                    os.mkdir("pretext" + "/" + claim.claimID)
                f = open(pathToSave + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                f.write(claim.processDocument(withTitle))
                f.write("\n")
                f.close()
                f = open("pretext" + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                f.write(claim.getPretext())
                f.write("\n")
                f.close()
                for snippet in claim.getSnippets():
                    doc = snippet.getSnippetText(withTitle,withPretext)
                    f = open(pathToSave + "/" + claim.claimID + "/" + snippet.number, "w", encoding="utf-8")
                    f.write(doc)
                    f.write("\n")
                    f.close()
                    f = open("pretext" + "/" + claim.claimID + "/" + snippet.number, "w", encoding="utf-8")
                    f.write(snippet.getPretext())
                    f.write("\n")
                    f.close()
    else:
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], 'None',elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11],
                                    "snippets/", predictorOIE, predictorNER, nlp, coreference)
                if not (os.path.exists(pathToSave + "/" + claim.claimID)):
                    os.mkdir(pathToSave + "/" + claim.claimID)
                if withPretext and not (os.path.exists("pretext" + "/" + claim.claimID)):
                    os.mkdir("pretext" + "/" + claim.claimID)
                f = open(pathToSave + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                f.write(claim.processDocument(withTitle))
                f.write("\n")
                f.close()
                f = open("pretext" + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                f.write(claim.getPretext())
                f.write("\n")
                f.close()
                for snippet in claim.getSnippets():
                    doc = snippet.getSnippetText(withTitle, withPretext)
                    f = open(pathToSave + "/" + claim.claimID + "/" + snippet.number, "w", encoding="utf-8")
                    f.write(doc)
                    f.write("\n")
                    f.close()
                    f = open("pretext" + "/" + claim.claimID + "/" + snippet.number, "w", encoding="utf-8")
                    f.write(snippet.getPretext())
                    f.write("\n")
                    f.close()
    print('Done complete ' + str(path))

def editArticleText(mode,path,withTitle=True,withPretext=True,pathToSave="Text"):
    nlp = spacy.load("en_core_web_sm")
    predictorOIE = "None"
    predictorNER = "None"
    coreference = "None"
    if not (os.path.exists(pathToSave)):
        os.mkdir(pathToSave)
    if withPretext and not (os.path.exists("pretext")):
        os.mkdir("pretext")
    if mode != 'Test':
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                    "snippets/", predictorOIE, predictorNER, nlp, coreference)
                if not (os.path.exists(pathToSave + "/" + claim.claimID)):
                    os.mkdir(pathToSave + "/" + claim.claimID)
                if withPretext and not (os.path.exists("pretext" + "/" + claim.claimID)):
                    os.mkdir("pretext" + "/" + claim.claimID)
                f = open(pathToSave + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                f.write(claim.processDocument2(withTitle))
                f.write("\n")
                f.close()
                f = open("pretext" + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                f.write(claim.getPretext(False))
                f.write("\n")
                f.close()
                for snippet in claim.getSnippets():
                    doc = snippet.getSnippetText2(withTitle,withPretext)
                    f = open(pathToSave + "/" + claim.claimID + "/" + snippet.number, "w", encoding="utf-8")
                    f.write(doc)
                    f.write("\n")
                    f.close()
                    f = open("pretext" + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                    f.write(claim.getPretext(False))
                    f.write("\n")
                    f.close()
    else:
        with open(path, 'r', encoding='utf-8') as file:
            for claim in file:
                elements = claim.split('\t')
                claim = Claim.claim(elements[0], elements[1], 'None',elements[2], elements[3], elements[4], elements[5],
                                    elements[6],
                                    elements[7], elements[8], elements[9], elements[10], elements[11],
                                    "snippets/", predictorOIE, predictorNER, nlp, coreference)
                if not (os.path.exists(pathToSave + "/" + claim.claimID)):
                    os.mkdir(pathToSave + "/" + claim.claimID)
                if withPretext and not (os.path.exists("pretext" + "/" + claim.claimID)):
                    os.mkdir("pretext" + "/" + claim.claimID)
                f = open(pathToSave + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                f.write(claim.processDocument2(withTitle))
                f.write("\n")
                f.close()
                f = open("pretext" + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                f.write(claim.getPretext(False))
                f.write("\n")
                f.close()
                for snippet in claim.getSnippets():
                    doc = snippet.getSnippetText2(withTitle,withPretext)
                    f = open(pathToSave + "/" + claim.claimID + "/" + snippet.number, "w", encoding="utf-8")
                    f.write(doc)
                    f.write("\n")
                    f.close()
                    f = open("pretext" + "/" + claim.claimID + "/" + "claim", "w", encoding="utf-8")
                    f.write(claim.getPretext(False))
                    f.write("\n")
                    f.close()

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    if sys.argv[1] == "editArticleTextWithUppercaseEditing":
        if len(sys.argv) == 6:
            editArticleTextWithUppercaseEditing(sys.argv[2],sys.argv[3],str2bool(sys.argv[4]),str2bool(sys.argv[5]))
        else:
            editArticleTextWithUppercaseEditing(sys.argv[2], sys.argv[3], str2bool(sys.argv[4]), str2bool(sys.argv[5]),sys.argv[6])
    else:
        if len(sys.argv) == 6:
            editArticleText(sys.argv[2], sys.argv[3], str2bool(sys.argv[4]), str2bool(sys.argv[5]))
        else:
            editArticleText(sys.argv[2], sys.argv[3], str2bool(sys.argv[4]), str2bool(sys.argv[5]),sys.argv[6])