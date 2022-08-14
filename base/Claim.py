import json
import re
import xml.etree.ElementTree as ET
import datetime
import os
import Snippet


class claim:

    def __init__(self, claimID, claim, label, claimURL, reason, categories, speaker, checker, tags, articleTitle,
                 publishDate, claimDate, entities, directorySnippets,OIEpredictor,NERpredictor,spacy,coreference):
        self.coreference = coreference
        self.spacy = spacy
        self.claimID = claimID
        self.predictorOIE = OIEpredictor
        self.predictorNER = NERpredictor
        self.claim = claim
        self.label = label
        self.claimURl = claimURL
        self.reason = reason
        self.speaker = speaker
        self.checker = checker
        # self.tags = tags
        self.articleTitle = articleTitle
        self.publishDate = publishDate
        self.claimDate = claimDate
        if (self.claimDate == "None"):
            self.claimDate = publishDate
        # print(self.claimDate)
        self.readTags(tags)
        self.readEntities(entities)
        self.readCategories(categories)
        self.readSnippets(directorySnippets + "/" + claimID)

    def processSentence(self,sentence):
        sentence = str(sentence)
        if sentence.strip():
            ner = self.predictorNER.predict(sentence)
            tags = ner.get('tags')
            words = ner.get('words')
            for i in range(len(tags)):
                if i == 0:
                    words[i] = words[i][0].upper() + words[i][1:]
                else:
                    if tags[i] != "O":
                        if words[i] != "'s":
                            words[i] = words[i][0].upper() + words[i][1:]
                    else:
                        words[i] = words[i].lower()
            sent = ""
            index = 0
            parts = sentence.split(' ')
            for j in range(len(parts)):
                # print(segmentation)
                indexNew = index
                lastWord = ""
                for i in range(index, len(words)):
                    if words[i].lower() in parts[j].lower():
                        # print(words[i])
                        indexNew += 1
                        sent += words[i]
                        lastWord += words[i]
                        if lastWord.lower() == parts[j].lower():
                            index = indexNew
                            sent += " "
                            break
            return sent

    def processDocument(self):
        claim = []
        document = self.claim.split('...')
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                part.append(self.processSentence(sentence))
            claim.append(part)
        document = self.getClaimText(title, claim)
        return document

    def processDocument2(self):
        claim = []
        document = self.claim.split('...')
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += str(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                part.append(str(sentence))
            claim.append(part)
        document = self.getClaimText(title, claim)
        return document

    def processCoreference(self):
        claim = []
        document = self.claim.split('...')
        print(document)

        # article title
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                part.append(self.processSentence(sentence))
            claim.append(part)
        document = self.getClaimText(title,claim)
        for ev in self.snippets:
            document += ev.getSnippetText()
        print(document)
        f = open("Coreference" + "/" + "Versie1" + "/" + self.claimID, "a", encoding="utf-8")

        perform = self.coreference.predict(document)
        print(perform)

    def processCoreference2(self):
        claim = []
        document = self.claim.split('...')

        # article title
        title = ""
        if self.articleTitle != "None":
            sentences = self.spacy(self.articleTitle)
            for sentence in sentences.sents:
                title += self.processSentence(sentence)
        for sen in document:
            sentences = self.spacy(sen)
            part = []
            for sentence in sentences.sents:
                part.append(self.processSentence(sentence))
            claim.append(part)
        document = self.getClaimText(title,claim)
        for ev in self.snippets:
            document += ev.getSnippetText()
        print(document)
        f = open("Coreference" + "/" "Versie2" + "/" + self.claimID, "a", encoding="utf-8")
        perform = self.coreference.perform_coreference(document)
        print(perform)
        Wordtokens = perform['tokenized_doc']['orig_tokens']
        subWordTokens = perform['tokenized_doc']['subtoken_map']
        coreferenceOutput = dict()
        coreferenceOutput['Wordtokens'] = Wordtokens
        coreferenceOutput['SubWordTokens'] = subWordTokens
        coreferenceOutput['Clusters'] = perform['clusters']
        json.dump(coreferenceOutput,f)

    def readCoreference(self):
        path = "Coreference" + "/" + self.claimID
        f = open(path, "r", encoding="utf-8")
        coreferenceOutput = json.load(f)
        wordTokens = coreferenceOutput['Wordtokens']
        subWordTokens =  coreferenceOutput['SubWordTokens']
        clusters = coreferenceOutput['Clusters']
        return wordTokens,subWordTokens,clusters


    def processOpenInformation(self):
        document = self.claim.split('...')
        if not (os.path.exists("OpenInformation" + "/" + self.claimID)):
            os.mkdir("OpenInformation" + "/" + self.claimID)
        if self.articleTitle != "None":
            document.insert(0,self.articleTitle)
        openInformation = []
        for sen in document:
            sentences = self.spacy(sen)
            for sentence in sentences.sents:
                sent = self.processSentence(sentence)
                if sent is not None:
                    openInformation.append(self.predictorOIE.predict(sent))
                '''
                for verb in openInformationExtraction['verbs']:
                    print(verb['description'])
                    print(verb['tags'])
                    print(openInformationExtraction['words'])
                '''
        f = open("OpenInformation" + "/" + self.claimID + "/" + "claim", "w", encoding="utf-8")
        json.dump(openInformation, f)
        f.close()

    def readOpenInformationExtraction(self):
        path = "OpenInformation" + "/" + self.claimID + "/" + "claim"
        f = open(path, "r", encoding="utf-8")
        openInfo = json.load(f)
        return openInfo

    '''
    def processDateSnippet(self,snippet):
        try:
            datetime.datetime.strptime(snippet.publishTime[:-1], '%b %d, %Y')
            self.TimeOK += 1
        except:
            if (snippet.url != "None"):
                partsURL = snippet.url.split("/")
                index = 0
                while not (partsURL[index].isdigit() and int(partsURL[index]) >= 1900 and int(
                        partsURL[index]) <= 2021):

                    index += 1
                    if index == len(partsURL):
                        break

                if index != len(partsURL):
                    self.TimeOK += 1
                    months = ["January", "February", "March", "April", "May", "June", "July", "August",
                              "September", "October", "November", "December"]
                    date = ""
                    if index + 2 != len(partsURL) and len(partsURL[index + 2]) <= 2 and partsURL[
                        index + 2].isdigit():
                        date += str(int(partsURL[index + 2])) + " "
                    if index + 1 != len(partsURL) and len(partsURL[index + 1]) <= 2 and partsURL[
                        index + 1].isdigit() and int(partsURL[index + 1]) >= 1 and int(partsURL[index + 1]) <= 12:
                        date += months[int(partsURL[index + 1]) - 1] + " "
                    else:
                        if len(partsURL[index + 1]) <= 10:
                            date += partsURL[index + 1].strip() + " "
                    date += partsURL[index]
                    if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                        os.mkdir("SnippetDates" + "/" + self.claimID)
                    f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w", encoding="utf-8")
                    f.write(date + "\n")
                    f.close()
                else:
                    if snippet.optional != None:
                        # print(self.claimID + " - " + snippet.number + snippet.optional + "\n")
                        self.TimeOK += 1
                        if not (os.path.exists("SnippetDates" + "/" +self.claimID)):
                            os.mkdir("SnippetDates" + "/" + self.claimID)
                        f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w", encoding="utf-8")
                        f.write(snippet.optional + "\n")
                        f.close()
                    else:
                        word_list = snippet.publishTime.split()
                        if len(word_list) <= 6:
                            if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                os.mkdir("SnippetDates" + "/" + self.claimID)
                            f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w", encoding="utf-8")
                            f.write(snippet.publishTime.replace("Nob", "November").replace("Se", "September") + "\n")
                            f.close()
                        else:
                            word_list = snippet.publishTime.split("|")[0].split()
                            if len(word_list) <= 6:
                                if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                    os.mkdir("SnippetDates" + "/" + self.claimID)
                                f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                         encoding="utf-8")
                                f.write(snippet.publishTime.split("|")[0] + "\n")
                                f.close()
                            else:
                                word_list = snippet.publishTime.split(u"\u2022")[0].split()  # bullet
                                if len(word_list) <= 6:
                                    if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                        os.mkdir("SnippetDates" + "/" + self.claimID)
                                    f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                             encoding="utf-8")
                                    f.write(snippet.publishTime.split(u"\u2022")[0] + "\n")
                                    f.close()
                                else:
                                    word_list = snippet.publishTime.split("-")[0].split()  # bullet
                                    if len(word_list) <= 6:
                                        if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                            os.mkdir("SnippetDates" + "/" + self.claimID)
                                        f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                                 encoding="utf-8")
                                        f.write(snippet.publishTime.split("-")[0] + "\n")
                                        f.close()
                                    else:
                                        word_list = snippet.publishTime.split(":")[0].split()  # bullet
                                        if len(word_list) <= 6:
                                            if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                                os.mkdir("SnippetDates" + "/" + self.claimID)
                                            f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                                     encoding="utf-8")
                                            f.write(
                                                snippet.publishTime.split(":")[0] + "\n")
                                            f.close()

            else:
                if snippet.optional != None:
                    self.TimeOK += 1
                    if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                        os.mkdir("SnippetDates" + "/" + self.claimID)
                    f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w", encoding="utf-8")
                    f.write(snippet.optional + "\n")
                    f.close()
                else:
                    word_list = snippet.publishTime.split()
                    if len(word_list) <= 6:
                        if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                            os.mkdir("SnippetDates" + "/" + self.claimID)
                        f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w", encoding="utf-8")
                        f.write(snippet.publishTime.replace("Nob", "November").replace("Se", "September") + "\n")
                        f.close()
                    else:
                        word_list = snippet.publishTime.split("|")[0].split()
                        if len(word_list) <= 6:
                            if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                os.mkdir("SnippetDates" + "/" + self.claimID)
                            f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                     encoding="utf-8")
                            f.write(snippet.publishTime.split("|")[0] + "\n")
                            f.close()
                        else:
                            word_list = snippet.publishTime.split(u"\u2022")[0].split()  # bullet
                            if len(word_list) <= 6:
                                if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                    os.mkdir("SnippetDates" + "/" + self.claimID)
                                f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                         encoding="utf-8")
                                f.write(snippet.publishTime.split(u"\u2022")[0] + "\n")
                                f.close()
                            else:
                                word_list = snippet.publishTime.split("-")[0].split()  # bullet
                                if len(word_list) <= 6:
                                    if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                        os.mkdir("SnippetDates" + "/" + self.claimID)
                                    f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                             encoding="utf-8")
                                    f.write(snippet.publishTime.split("-")[0] + "\n")
                                    f.close()

                                else:
                                    word_list = snippet.publishTime.split(":")[0].split()  # bullet
                                    if len(word_list) <= 6:
                                        if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                            os.mkdir("SnippetDates" + "/" + self.claimID)
                                        f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                                 encoding="utf-8")
                                        f.write(
                                            snippet.publishTime.split(":")[0] + "\n")
                                        f.close()
    '''

    def processDateSnippet(self, snippet):
        try:
            datetime.datetime.strptime(snippet.publishTime[:-1], '%b %d, %Y')
        except:
            if snippet.optional != None:
                if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                    os.mkdir("SnippetDates" + "/" + self.claimID)
                f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w", encoding="utf-8")
                f.write(snippet.optional + "\n")
                f.close()
            else:
                word_list = snippet.publishTime.split()
                if len(word_list) <= 6:
                    if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                        os.mkdir("SnippetDates" + "/" + self.claimID)
                    f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w", encoding="utf-8")
                    f.write(snippet.publishTime.replace("Nob", "November").replace("Se", "September") + "\n")
                    f.close()
                else:
                    word_list = snippet.publishTime.split("|")[0].split()
                    if len(word_list) <= 6:
                        if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                            os.mkdir("SnippetDates" + "/" + self.claimID)
                        f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                 encoding="utf-8")
                        f.write(snippet.publishTime.split("|")[0] + "\n")
                        f.close()
                    else:
                        word_list = snippet.publishTime.split(u"\u2022")[0].split()  # bullet
                        if len(word_list) <= 6:
                            if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                os.mkdir("SnippetDates" + "/" + self.claimID)
                            f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                     encoding="utf-8")
                            f.write(snippet.publishTime.split(u"\u2022")[0] + "\n")
                            f.close()
                        else:
                            word_list = snippet.publishTime.split("-")[0].split()  # bullet
                            if len(word_list) <= 6:
                                if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                    os.mkdir("SnippetDates" + "/" + self.claimID)
                                f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                         encoding="utf-8")
                                f.write(snippet.publishTime.split("-")[0] + "\n")
                                f.close()

                            else:
                                word_list = snippet.publishTime.split(":")[0].split()  # bullet
                                if len(word_list) <= 6:
                                    if not (os.path.exists("SnippetDates" + "/" + self.claimID)):
                                        os.mkdir("SnippetDates" + "/" + self.claimID)
                                    f = open("SnippetDates" + "/" + self.claimID + "/" + snippet.number, "w",
                                             encoding="utf-8")
                                    f.write(
                                        snippet.publishTime.split(":")[0] + "\n")
                                    f.close()

    def processDate(self):
        path = 'ProcessedDates'+'/'+self.claimID+'.xml'
        if os.path.exists(path):
            tree = ET.parse(path)

            root = tree.getroot()
            if len(root) == 1:
                if root[0].attrib['value'].find(':') == -1:
                    if len(root[0].attrib['value']) == 7:
                        # print('Claim ' + self.claimID)
                        self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m')
                    else:
                        if len(root[0].attrib['value']) == 4:
                            # print('Claim ' + self.claimID)
                            self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y')
                        else:
                            self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m-%d')
                else:
                    if root[0].attrib['value'][root[0].attrib['value'].find(':') + 1:].find(':') == -1:
                        self.claimDate = datetime.datetime.strptime(root[0].attrib['value'].replace('24:', '12:'),
                                                                    '%Y-%m-%dT%H:%M')
                    else:
                        self.claimDate = datetime.datetime.strptime(root[0].attrib['value'].replace('24:', '12:'),
                                                                    '%Y-%m-%dT%H:%M:%S')
            else:
                if root[0].attrib['value'].find('X') != -1:
                    # correct fault of assigning 12pm to 24 in HeidelTime
                    self.claimDate = datetime.datetime.strptime(
                        root[1].attrib['value'] + root[0].attrib['value'][root[0].attrib['value'].find('T'):].replace('24:',
                                                                                                                      '12:'),
                        '%Y-%m-%dT%H:%M')
                else:
                    if root[0].attrib['value'].find('T') == -1 and root[1].attrib['value'].find('T') != -1:
                        # correct fault of assigning 12pm to 24 in HeidelTime
                        self.claimDate = datetime.datetime.strptime(root[1].attrib['value'].replace('24:', '12:'),
                                                                    '%Y-%m-%dT%H:%M')
                    else:
                        # we choose publication date as claim date not last updated date
                        if root[0].attrib['value'].find('T') != -1:
                            self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m-%dT%H:%M')
                        else:
                            if len(root[0].attrib['value']) == 7:
                                # print('Claim ' + self.claimID)
                                self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m')
                            else:
                                if len(root[0].attrib['value']) == 4:
                                    # print('Claim ' + self.claimID)
                                    self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y')
                                else:
                                    self.claimDate = datetime.datetime.strptime(root[0].attrib['value'], '%Y-%m-%d')
        else:
            print('Not found -' +  path)
            self.claimDate = None

    def readSnippets(self, path):
        self.snippets = []
        # print('Claim: ' + self.claimID)
        try:
            with open(path, 'r', encoding='utf-8') as file:
                for snippet in file:
                    parts = snippet.split("\t")
                    self.snippets.append(Snippet.snippet(self.claimID, parts[0], parts[1], parts[2], parts[3],self.predictorOIE,self.predictorNER,self.spacy,self.coreference))
        except:
            print(path + " not exist")



    def readEntities(self, entities):
        self.entities = re.findall("'([^']*)'", entities)
    def getSnippets(self):
        return self.snippets

    def readTags(self, tags):
        if (len(tags) == 0):
            self.tags = ['None']
        else:
            if (tags.lstrip()[0] == "["):
                self.tags = re.findall("'([^']*)'", tags)
            else:
                self.tags = tags.split(",")

    def readCategories(self, categories):
        if (categories.lstrip()[0] == "["):
            self.categories = re.findall("'([^']*)'", categories)
        else:
            self.categories = categories.split()

    def getDomain(self):
        return self.claimID.split('-')[1]

    def getClaimId(self):
        return self.claimID

    def getClaim(self):
        return self.label

    def getClaimURL(self):
        return self.claimURl

    def getReason(self):
        return self.reason

    def getCategories(self):
        return self.categories

    def getSpeaker(self):
        return self.speaker

    def getChecker(self):
        return self.checker

    def getTags(self):
        return self.tags

    def getArticle(self):
        return self.articleTitle

    def getPublishDate(self):
        return self.publishDate

    def getClaimDate(self):
        return self.claimDate

    def getEntities(self):
        return self.entities

    def getClaimText(self,title,parts):
        string = ""
        string += "The "
        string += "claim "
        if len(title)>0:
            string += "with the title '"
            string += title
            string += "' "
        string += "says "
        for part in parts:
            for sen in part:
                if sen is not None:
                    string += sen
            string += " ... "
        string = string[:-5]
        return string

    def deriveOpenInformation(self):
        self.openInformation = []

    def getOpenInformation(self):
        return self.openInformation
