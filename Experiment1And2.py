import math
import os
import sys
import random
import scipy.stats as ss
import numpy
import numpy as np
import torch.nn.functional as F
from scipy import stats
from division1DifferencePublication.encoderGlobal import encoder as encoderPublicatie
from division2DifferenceTimeText.encoderGlobal import encoder as encoderTekst
from division1And2.encoderGlobal import encoder as encoderEverything
from division1DifferencePublication.verificationModelGlobal import verifactionModel as verificationPublicatie
from division2DifferenceTimeText.verificationModelGlobal import verifactionModel as verificationTekst
from dataset import NUS
from division1And2.verificationModelGlobal import verifactionModel as verificationEverything
from baseModel import OneHotEncoderBasis, labelEmbeddingLayerBasis, verificationModelBase, encoderBase, encoderMetadataBasis, \
    instanceEncoderBasis, evidence_rankerBasis, labelMaskDomainBasis
import torch
from torch.utils.data import DataLoader


'''
Calculate Spearman correlationCoefficient for an example x for the ranking of the evidences (experiment 1) and 
the ranking of the labels (experiment 2) between (basis-U1),(basis-U2),(basis-U1AndU2),(U1-U2),(U1-U1AndU2),(U2-U1AndU2)
'''
def spearmanRanking(loaders,basisModels,models,timeTextModels,modelsEverything):
    pearonsEvidenceRanking = {}
    numberEqual = {}
    spearmanAll = []
    labelsDomain = []
    standaardDomain = []
    standaardLabelDomain = []
    labelsAll = []
    standaardLabelAllDomain = []
    spearmanAlltimeText = []
    labelsDomaintimeText = []
    standaardDomaintimeText = []
    standaardLabelDomaintimeText = []
    labelsAlltimeText = []
    standaardLabelAllDomaintimeText = []
    spearmanAllEverything = []
    labelsDomainEverything = []
    standaardDomainEverything = []
    standaardLabelDomainEverything = []
    labelsAllEverything = []
    standaardLabelAllDomainEverything = []
    for loader in loaders:
        pearonsEvidenceRanking[loader[1]] = []
        numberEqual[loader[1]] = {}
        for data in loader[0]:
            for i in range(len(data[0])):
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
                metadata_encoding = basisModels[0].metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = data[0][0].split('-')[0]
                rankingB1, labelsAllB1, labelsDomainB1,allEqualB1 = basisModels[0].getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                        metadata_encoding, domain)
                if allEqualB1:
                    if "Base" in numberEqual[domain]:
                        numberEqual[domain]["Base"] += 1
                    else:
                        numberEqual[domain]["Base"] = 1
                rankB1 = [sorted(rankingB1,reverse=True).index(x)+1 for x in rankingB1]
                labelB1 = [sorted(labelsDomainB1,reverse=True).index(x)+1 for x in labelsDomainB1]
                labelsAllB1 = [sorted(labelsAllB1,reverse=True).index(x)+1 for x in labelsAllB1]
                rankingB2, labelsAllB2, labelsDomainB2, allEqualB2 = basisModels[1].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding, domain)
                if allEqualB2:
                    if "Base2" in numberEqual[domain]:
                        numberEqual[domain]["Base2"] += 1
                    else:
                        numberEqual[domain]["Base2"] = 1
                rankB2 = [sorted(rankingB2, reverse=True).index(x) + 1 for x in rankingB2]
                labelB2 = [sorted(labelsDomainB2, reverse=True).index(x) + 1 for x in labelsDomainB2]
                labelsAllB2 = [sorted(labelsAllB2, reverse=True).index(x) + 1 for x in labelsAllB2]
                rankingB3, labelsAllB3, labelsDomainB3, allEqualB3 = basisModels[2].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding, domain)
                if allEqualB3:
                    if "Base3" in numberEqual[domain]:
                        numberEqual[domain]["Base3"] += 1
                    else:
                        numberEqual[domain]["Base3"] = 1
                rankB3 = [sorted(rankingB3, reverse=True).index(x) + 1 for x in rankingB3]
                labelB3 = [sorted(labelsDomainB3, reverse=True).index(x) + 1 for x in labelsDomainB3]
                labelsAllB3 = [sorted(labelsAllB3, reverse=True).index(x) + 1 for x in labelsAllB3]
                index = 1
                for model in models:
                    rankingM, labelsAllM, labelsDomainM,allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                               metadata_encoding,
                                                                                               domain,data[5][i],data[6][i])
                    if allEqualM:
                        if "time"+str(index) in numberEqual[domain]:
                            numberEqual[domain]["time"+str(index)] += 1
                        else:
                            numberEqual[domain]["time"+str(index)] = 1

                    rankM = [sorted(rankingM,reverse=True).index(x)+1 for x in rankingM]
                    corref = stats.spearmanr(rankB1,rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB3, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB3, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB3, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry)>0:
                    standaardDomain.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomain.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomain.append(numpy.std(labelsAllEntry))
                pearonsEvidenceRanking[loader[1]].extend(spearmanRankEntry)
                spearmanAll.extend(spearmanRankEntry)
                labelsDomain.extend(labelsEntry)
                labelsAll.extend(labelsAllEntry)
                index = 1
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in timeTextModels:
                    rankingM, labelsAllM, labelsDomainM,allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i],data[17][i],data[18][i])
                    if allEqualM:
                        if "timeText"+str(index) in numberEqual[domain]:
                            numberEqual[domain]["timeText"+str(index)] += 1
                        else:
                            numberEqual[domain]["timeText"+str(index)] = 1

                    rankM = [sorted(rankingM,reverse=True).index(x)+1 for x in rankingM]
                    corref = stats.spearmanr(rankB1,rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB3, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB3, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB3, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry)>0:
                    standaardDomaintimeText.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomaintimeText.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomaintimeText.append(numpy.std(labelsAllEntry))
                spearmanAlltimeText.extend(spearmanRankEntry)
                labelsDomaintimeText.extend(labelsEntry)
                labelsAlltimeText.extend(labelsAllEntry)
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in modelsEverything:
                    rankingM, labelsAllM, labelsDomainM, allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i],data[17][i],data[18][i])
                    if allEqualM:
                        if "everything" + str(index) in numberEqual[domain]:
                            numberEqual[domain]["everything" + str(index)] += 1
                        else:
                            numberEqual[domain]["everything" + str(index)] = 1

                    rankM = [sorted(rankingM, reverse=True).index(x) + 1 for x in rankingM]
                    corref = stats.spearmanr(rankB1, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB3, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB3, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB3, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry) > 0:
                    standaardDomainEverything.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomainEverything.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomainEverything.append(numpy.std(labelsAllEntry))
                spearmanAllEverything.extend(spearmanRankEntry)
                labelsDomainEverything.extend(labelsEntry)
                labelsAllEverything.extend(labelsAllEntry)
    return spearmanAll,standaardDomain,labelsAll,standaardLabelAllDomain,labelsDomain,standaardLabelDomain,numberEqual,\
           spearmanAlltimeText, standaardDomaintimeText,labelsDomaintimeText,standaardLabelDomaintimeText,\
           labelsAlltimeText,standaardLabelAllDomaintimeText,spearmanAllEverything,standaardDomainEverything,\
           labelsDomainEverything,standaardLabelDomainEverything,labelsAllEverything,standaardLabelAllDomainEverything

def spearmanRankingTime(loaders,basisModels,timeTextModels,modelsEverything):
    pearonsEvidenceRanking = {}
    numberEqual = {}
    spearmanAlltimeText = []
    labelsDomaintimeText = []
    standaardDomaintimeText = []
    standaardLabelDomaintimeText = []
    labelsAlltimeText = []
    standaardLabelAllDomaintimeText = []
    spearmanAllEverything = []
    labelsDomainEverything = []
    standaardDomainEverything = []
    standaardLabelDomainEverything = []
    labelsAllEverything = []
    standaardLabelAllDomainEverything = []
    for loader in loaders:
        pearonsEvidenceRanking[loader[1]] = []
        numberEqual[loader[1]] = {}
        for data in loader[0]:
            for i in range(len(data[0])):
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
                metadata_encoding = basisModels[0].metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = data[0][0].split('-')[0]
                rankingB1, labelsAllB1, labelsDomainB1,allEqualB1 = basisModels[0].getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                               metadata_encoding,
                                                                                               domain,data[5][i],data[6][i])
                if allEqualB1:
                    if "Time" in numberEqual[domain]:
                        numberEqual[domain]["Time"] += 1
                    else:
                        numberEqual[domain]["Time"] = 1
                rankB1 = [sorted(rankingB1,reverse=True).index(x)+1 for x in rankingB1]
                labelB1 = [sorted(labelsDomainB1,reverse=True).index(x)+1 for x in labelsDomainB1]
                labelsAllB1 = [sorted(labelsAllB1,reverse=True).index(x)+1 for x in labelsAllB1]
                rankingB2, labelsAllB2, labelsDomainB2, allEqualB2 = basisModels[1].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding,
                    domain, data[5][i], data[6][i])
                if allEqualB2:
                    if "Time2" in numberEqual[domain]:
                        numberEqual[domain]["Time2"] += 1
                    else:
                        numberEqual[domain]["Time2"] = 1
                rankB2 = [sorted(rankingB2, reverse=True).index(x) + 1 for x in rankingB2]
                labelB2 = [sorted(labelsDomainB2, reverse=True).index(x) + 1 for x in labelsDomainB2]
                labelsAllB2 = [sorted(labelsAllB2, reverse=True).index(x) + 1 for x in labelsAllB2]
                rankingB3, labelsAllB3, labelsDomainB3, allEqualB3 = basisModels[2].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding,
                    domain, data[5][i], data[6][i])
                if allEqualB3:
                    if "Time3" in numberEqual[domain]:
                        numberEqual[domain]["Time3"] += 1
                    else:
                        numberEqual[domain]["Time3"] = 1
                rankB3 = [sorted(rankingB3, reverse=True).index(x) + 1 for x in rankingB3]
                labelB3 = [sorted(labelsDomainB3, reverse=True).index(x) + 1 for x in labelsDomainB3]
                labelsAllB3 = [sorted(labelsAllB3, reverse=True).index(x) + 1 for x in labelsAllB3]
                index = 1

                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in timeTextModels:
                    rankingM, labelsAllM, labelsDomainM,allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i],data[17][i],data[18][î])
                    if allEqualM:
                        if "timeText"+str(index) in numberEqual[domain]:
                            numberEqual[domain]["timeText"+str(index)] += 1
                        else:
                            numberEqual[domain]["timeText"+str(index)] = 1

                    rankM = [sorted(rankingM,reverse=True).index(x)+1 for x in rankingM]
                    corref = stats.spearmanr(rankB1,rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB3, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB3, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB3, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry)>0:
                    standaardDomaintimeText.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomaintimeText.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomaintimeText.append(numpy.std(labelsAllEntry))
                spearmanAlltimeText.extend(spearmanRankEntry)
                labelsDomaintimeText.extend(labelsEntry)
                labelsAlltimeText.extend(labelsAllEntry)
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in modelsEverything:
                    rankingM, labelsAllM, labelsDomainM, allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i],data[17][i],data[18][i])
                    if allEqualM:
                        if "everything" + str(index) in numberEqual[domain]:
                            numberEqual[domain]["everything" + str(index)] += 1
                        else:
                            numberEqual[domain]["everything" + str(index)] = 1

                    rankM = [sorted(rankingM, reverse=True).index(x) + 1 for x in rankingM]
                    corref = stats.spearmanr(rankB1, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB3, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB3, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB3, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry) > 0:
                    standaardDomainEverything.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomainEverything.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomainEverything.append(numpy.std(labelsAllEntry))
                spearmanAllEverything.extend(spearmanRankEntry)
                labelsDomainEverything.extend(labelsEntry)
                labelsAllEverything.extend(labelsAllEntry)
    return numberEqual,\
           spearmanAlltimeText, standaardDomaintimeText,labelsDomaintimeText,standaardLabelDomaintimeText,\
           labelsAlltimeText,standaardLabelAllDomaintimeText,spearmanAllEverything,standaardDomainEverything,\
           labelsDomainEverything,standaardLabelDomainEverything,labelsAllEverything,standaardLabelAllDomainEverything


def spearmanRankingtimeText(loaders, basisModels, modelsEverything):
    pearonsEvidenceRanking = {}
    numberEqual = {}
    spearmanAllEverything = []
    labelsDomainEverything = []
    standaardDomainEverything = []
    standaardLabelDomainEverything = []
    labelsAllEverything = []
    standaardLabelAllDomainEverything = []
    for loader in loaders:
        pearonsEvidenceRanking[loader[1]] = []
        numberEqual[loader[1]] = {}
        for data in loader[0]:
            for i in range(len(data[0])):

                metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
                metadata_encoding = basisModels[0].metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = data[0][0].split('-')[0]
                rankingB1, labelsAllB1, labelsDomainB1, allEqualB1 = basisModels[0].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding, domain,
                    data[5][i], data[6][i],
                    data[7][i],
                    data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                    data[14][i], data[15][i], data[16][i],data[17][i],data[18][i])
                if allEqualB1:
                    if "timeText" in numberEqual[domain]:
                        numberEqual[domain]["timeText"] += 1
                    else:
                        numberEqual[domain]["timeText"] = 1
                rankB1 = [sorted(rankingB1, reverse=True).index(x) + 1 for x in rankingB1]
                labelB1 = [sorted(labelsDomainB1, reverse=True).index(x) + 1 for x in labelsDomainB1]
                labelsAllB1 = [sorted(labelsAllB1, reverse=True).index(x) + 1 for x in labelsAllB1]
                rankingB2, labelsAllB2, labelsDomainB2, allEqualB2 = basisModels[1].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding, domain,
                    data[5][i], data[6][i],
                    data[7][i],
                    data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                    data[14][i], data[15][i], data[16][i],data[17][i],data[18][i])
                if allEqualB2:
                    if "timeText2" in numberEqual[domain]:
                        numberEqual[domain]["timeText2"] += 1
                    else:
                        numberEqual[domain]["timeText2"] = 1
                rankB2 = [sorted(rankingB2, reverse=True).index(x) + 1 for x in rankingB2]
                labelB2 = [sorted(labelsDomainB2, reverse=True).index(x) + 1 for x in labelsDomainB2]
                labelsAllB2 = [sorted(labelsAllB2, reverse=True).index(x) + 1 for x in labelsAllB2]
                rankingB3, labelsAllB3, labelsDomainB3, allEqualB3 = basisModels[2].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding, domain,
                    data[5][i], data[6][i],
                    data[7][i],
                    data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                    data[14][i], data[15][i], data[16][i],data[17][i],data[18][i])
                if allEqualB3:
                    if "timeText3" in numberEqual[domain]:
                        numberEqual[domain]["timeText3"] += 1
                    else:
                        numberEqual[domain]["timeText3"] = 1
                rankB3 = [sorted(rankingB3, reverse=True).index(x) + 1 for x in rankingB3]
                labelB3 = [sorted(labelsDomainB3, reverse=True).index(x) + 1 for x in labelsDomainB3]
                labelsAllB3 = [sorted(labelsAllB3, reverse=True).index(x) + 1 for x in labelsAllB3]
                index = 1

                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in modelsEverything:
                    rankingM, labelsAllM, labelsDomainM, allEqualM = model.getRankingEvidencesLabels(data[1][i],
                                                                                                     data[2][i],
                                                                                                     metadata_encoding,
                                                                                                     domain,
                                                                                                     data[5][i],
                                                                                                     data[6][i],
                                                                                                     data[7][i],
                                                                                                     data[8][i],
                                                                                                     data[9][i],
                                                                                                     data[10][i],
                                                                                                     data[11][i],
                                                                                                     data[12][i],
                                                                                                     data[13][i],
                                                                                                     data[14][i],
                                                                                                     data[15][i],
                                                                                                     data[16][i],
                                                                                                     data[17][i],
                                                                                                     data[18][i])
                    if allEqualM:
                        if "everything" + str(index) in numberEqual[domain]:
                            numberEqual[domain]["everything" + str(index)] += 1
                        else:
                            numberEqual[domain]["everything" + str(index)] = 1

                    rankM = [sorted(rankingM, reverse=True).index(x) + 1 for x in rankingM]
                    corref = stats.spearmanr(rankB1, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB3, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB3, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB3, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry) > 0:
                    standaardDomainEverything.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomainEverything.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomainEverything.append(numpy.std(labelsAllEntry))
                spearmanAllEverything.extend(spearmanRankEntry)
                labelsDomainEverything.extend(labelsEntry)
                labelsAllEverything.extend(labelsAllEntry)
    return numberEqual, \
           spearmanAllEverything, standaardDomainEverything, \
           labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything
def getLabelIndicesDomain(domainPath,labelPath,weightsPath):
    domainsIndices = dict()
    domainsLabels = dict()
    domainLabelIndices = dict()
    domainWeights = dict()
    labelSequence = []
    file = open(labelPath,'r')
    lines = file.readlines()
    for line in lines:
        labelSequence.append(line.replace('\n',''))
    file = open(domainPath,'r')
    lines = file.readlines()
    for line in lines:
        parts = line.split("\t")
        labelsDomain = parts[1].split(",")
        labelsDomain[-1] = labelsDomain[-1].replace('\n','')
        labelIndices = []
        for label in labelsDomain:
            labelIndices.append(labelSequence.index(label.replace('\n','')))
        labelIndicesDomainM = sorted(labelIndices)
        labelIndicesDomain = []
        for index in labelIndices:
            labelIndicesDomain.append(labelIndicesDomainM.index(index))
        domainsIndices[parts[0]] = labelIndices
        domainsLabels[parts[0]] = labelsDomain
        domainLabelIndices[parts[0]] = labelIndicesDomain
    file = open(weightsPath, 'r')
    lines = file.readlines()
    for line in lines:

        parts = line.split("\t")
        weightsDomainNormal = parts[1:]
        weightsDomainNormal[-1] = weightsDomainNormal[-1].replace('\n','')
        domainWeights[parts[0]] = torch.zeros(len(weightsDomainNormal))
        for i in range(len(weightsDomainNormal)):
            domainWeights[parts[0]][domainLabelIndices[parts[0]][i]] = float(weightsDomainNormal[i])
    return domainsIndices,domainsLabels,domainLabelIndices,domainWeights



'''
    argument 1 path of first model basismodel
    argument 2 path of second model basismodel
    argument 3 path of third model basismodel
    argument 4 path of first model division1DifferencePublication
    argument 5 path of second model division1DifferencePublication
    argument 6 path of third model division1DifferencePublication
    argument 7 path of first model division2DifferenceTimeText
    argument 8 path of second model division2DifferenceTimeText
    argument 9 path of third model division2DifferenceTimeText
    argument 10 path of first model division1And2
    argument 11 path of second model division1And2
    argument 12 path of third model division1And2
    argument 13 name of domain to take examples from to calculate spearmanCorrleationCoeeficient of evidence rankings and labelrankings
    argument 14 parameter alpha for models division1DifferencePublication
    argument 15 parameter alpha for models division2DifferenceTimeText
    argument 16 parameter alpha for models division1And2
    argument 17 parameter beta for models division1And2
    argument 18 withPretext
'''
numpy.seterr(divide='ignore', invalid='ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
domainIndices, domainLabels, domainLabelIndices, domainWeights = getLabelIndicesDomain(
    'labels/labels.tsv', 'labels/labelSequence', 'labels/weights.tsv')
domains = domainIndices.keys()
models = []
for domain in domains:
    test_set = NUS(mode='Test', path='test/test-' + domain + '.tsv', domain=domain)
    test_loader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=False)
    models.append([test_loader,domain])
with torch.no_grad():
    spearmanAll = []
    standaard = []
    labelsAllE = []
    standaardLabelsAll = []
    labelsDomainE = []
    standaardDomainLabelsAll = []
    spearmanAlltimeTextD = []
    standaardtimeText = []
    labelsDomainEtimeText = []
    standaardDomainLabelsAlltimeText = []
    labelsAllEtimeText = []
    standaardLabelsAlltimeText = []
    spearmanAllEverythingD = []
    standaardEverything = []
    labelsDomainEEverything = []
    standaardDomainLabelsAllEverything = []
    labelsAllEEverything = []
    standaardLabelsAllEverything = []
    numberEqual1 = {}
    numberEqual2 = {}
    numberEqual3 = {}
    spearmanAlltimeText2 = []
    standaardtimeText2 = []
    labelsDomainEtimeText2 = []
    standaardDomainLabelsAlltimeText2 = []
    labelsAllEtimeText2 = []
    standaardLabelsAlltimeText2 = []
    spearmanAllEverything2 = []
    standaardEverything2 = []
    labelsDomainEEverything2 = []
    standaardDomainLabelsAllEverything2 = []
    labelsAllEEverything2 = []
    standaardLabelsAllEverything2 = []
    spearmanAlltimeText3 = []
    standaardtimeText3 = []
    labelsDomainEtimeText3 = []
    standaardDomainLabelsAlltimeText3 = []
    labelsAllEtimeText3 = []
    standaardLabelsAlltimeText3 = []
    spearmanAllEverything3 = []
    standaardEverything3 = []
    labelsDomainEEverything3 = []
    standaardDomainLabelsAllEverything3 = []
    labelsAllEEverything3 = []
    standaardLabelsAllEverything3 = []
    for model in models:
        oneHotEncoderBasis = OneHotEncoderBasis.oneHotEncoder('Metadata_sequence/metadata')
        labelEmbeddingLayerBasis = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderBasis = encoderBasis.encoder(300, 128).to(device)
        encoderMetadataBasis = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderBasis = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerBasis = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainBasis = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                    len(domainIndices[model[1]])).to(device)
        basisModel = verificationModelBase.verifactionModel(encoderBasis, encoderMetadataBasis, instanceEncoderBasis,
                                                             evidenceRankerBasis,
                                                             labelEmbeddingLayerBasis, labelMaskDomainBasis,
                                                             domainIndices,
                                                             model[1]).to(device)
        basisModel.loading_NeuralNetwork(sys.argv[1])
        labelEmbeddingLayerBasis = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderBasis = encoderBasis.encoder(300, 128).to(device)
        encoderMetadataBasis = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderBasis = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerBasis = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainBasis = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                    len(domainIndices[model[1]])).to(device)
        basisModel2 = verificationModelBase.verifactionModel(encoderBasis, encoderMetadataBasis, instanceEncoderBasis,
                                                             evidenceRankerBasis,
                                                             labelEmbeddingLayerBasis, labelMaskDomainBasis,
                                                             domainIndices,
                                                             model[1]).to(device)
        basisModel2.loading_NeuralNetwork(sys.argv[2])
        labelEmbeddingLayerBasis = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderBasis = encoderBasis.encoder(300, 128).to(device)
        encoderMetadataBasis = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderBasis = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerBasis = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainBasis = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                    len(domainIndices[model[1]])).to(device)
        basisModel3 = verificationModelBase.verifactionModel(encoderBasis, encoderMetadataBasis, instanceEncoderBasis,
                                                             evidenceRankerBasis,
                                                             labelEmbeddingLayerBasis, labelMaskDomainBasis,
                                                             domainIndices,
                                                             model[1]).to(device)
        basisModel3.loading_NeuralNetwork(sys.argv[3])
        labelEmbeddingLayerTime = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderTime = encoderPublicatie(300, 128, sys.argv[14]).to(device)
        encoderMetadataTime = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderTime = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerTime = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainTime = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                    len(domainIndices[model[1]])).to(device)
        verificationModelTimeA = verificationPublicatie(encoderTime, encoderMetadataTime, instanceEncoderTime,
                                                             evidenceRankerTime,
                                                             labelEmbeddingLayerTime, labelMaskDomainTime,
                                                             domainIndices,
                                                             model[1],sys.argv[14]).to(device)
        verificationModelTimeA.loading_NeuralNetwork(sys.argv[4])
        labelEmbeddingLayerTime = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderTime = encoderPublicatie(300, 128, sys.argv[14]).to(device)
        encoderMetadataTime = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderTime = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerTime = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainTime = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                   len(domainIndices[model[1]])).to(device)
        verificationModelTimeB = verificationPublicatie(encoderTime, encoderMetadataTime, instanceEncoderTime,
                                                          evidenceRankerTime,
                                                          labelEmbeddingLayerTime, labelMaskDomainTime,
                                                          domainIndices,
                                                          model[1],sys.argv[14]).to(device)
        verificationModelTimeB.loading_NeuralNetwork(sys.argv[5])
        labelEmbeddingLayerTime = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderTime = encoderPublicatie(300, 128, sys.argv[14]).to(device)
        encoderMetadataTime = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderTime = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerTime = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainTime = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                   len(domainIndices[model[1]])).to(device)
        verificationModelTimeC = verificationPublicatie(encoderTime, encoderMetadataTime, instanceEncoderTime,
                                                          evidenceRankerTime,
                                                          labelEmbeddingLayerTime, labelMaskDomainTime,
                                                          domainIndices,
                                                          model[1],sys.argv[14]).to(device)
        verificationModelTimeC.loading_NeuralNetwork(sys.argv[6])
        labelEmbeddingLayerTimeText = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderTimeText = encoderTekst(300, 128,sys.argv[15]).to(device)
        encoderMetadataTimeText = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderTimeText = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerTimeText = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainTimeText = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                   len(domainIndices[model[1]])).to(device)
        verificationModelTimeTextA = verificationTekst(
            encoderTimeText, encoderMetadataTimeText, instanceEncoderTimeText,
            evidenceRankerTimeText,
            labelEmbeddingLayerTimeText, labelMaskDomainTimeText,
            domainIndices,
            model[1],sys.argv[15],sys.argv[18]).to(device)
        verificationModelTimeTextA.loading_NeuralNetwork(sys.argv[7])
        labelEmbeddingLayerTimeText = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderTimeText = encoderTekst(300, 128, sys.argv[15]).to(device)
        encoderMetadataTimeText = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderTimeText = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerTimeText = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainTimeText = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                       len(domainIndices[model[1]])).to(device)
        verificationModelTimeTextB = verificationTekst(
            encoderTimeText, encoderMetadataTimeText, instanceEncoderTimeText,
            evidenceRankerTimeText,
            labelEmbeddingLayerTimeText, labelMaskDomainTimeText,
            domainIndices,
            model[1],sys.argv[15],sys.argv[18]).to(device)
        verificationModelTimeTextB.loading_NeuralNetwork(sys.argv[8])
        labelEmbeddingLayerTimeText = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderTimeText = encoderTekst(300, 128, sys.argv[15]).to(device)
        encoderMetadataTimeText = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderTimeText = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerTimeText = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainTimeText = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                       len(domainIndices[model[1]])).to(device)
        verificationModelTimeTextC = verificationTekst(
            encoderTimeText, encoderMetadataTimeText, instanceEncoderTimeText,
            evidenceRankerTimeText,
            labelEmbeddingLayerTimeText, labelMaskDomainTimeText,
            domainIndices,
            model[1],sys.argv[15],sys.argv[18]).to(device)
        verificationModelTimeTextC.loading_NeuralNetwork(sys.argv[9])
        labelEmbeddingLayerEverything = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderEverything = encoderEverything(300, 128, sys.argv[16],sys.argv[17]).to(device)
        encoderMetadataEverything = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderEverything = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerEverything = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainEverything = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                       len(domainIndices[model[1]])).to(device)
        verificationModelEverythingA = verificationEverything(
            encoderEverything, encoderMetadataEverything, instanceEncoderEverything,
            evidenceRankerEverything,
            labelEmbeddingLayerEverything, labelMaskDomainEverything,
            domainIndices,
            model[1], sys.argv[16],sys.argv[17],sys.argv[18]).to(device)
        verificationModelEverythingA.loading_NeuralNetwork(sys.argv[10])
        labelEmbeddingLayerEverything = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderEverything = encoderEverything(300, 128, sys.argv[16],sys.argv[17]).to(device)
        encoderMetadataEverything = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderEverything = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerEverything = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainEverything = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                         len(domainIndices[model[1]])).to(device)
        verificationModelEverythingB = verificationEverything(
            encoderEverything, encoderMetadataEverything, instanceEncoderEverything,
            evidenceRankerEverything,
            labelEmbeddingLayerEverything, labelMaskDomainEverything,
            domainIndices,
            model[1],sys.argv[16],sys.argv[17],sys.argv[18]).to(device)
        verificationModelEverythingB.loading_NeuralNetwork(sys.argv[11])
        labelEmbeddingLayerEverything = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderEverything = encoderEverything(300, 128, sys.argv[16],sys.argv[17]).to(device)
        encoderMetadataEverything = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderEverything = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerEverything = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainEverything = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, model[1],
                                                                         len(domainIndices[model[1]])).to(device)
        verificationModelEverythingC = verificationEverything(
            encoderEverything, encoderMetadataEverything, instanceEncoderEverything,
            evidenceRankerEverything,
            labelEmbeddingLayerEverything, labelMaskDomainEverything,
            domainIndices,
            model[1], sys.argv[16],sys.argv[17],sys.argv[18]).to(device)
        verificationModelEverythingC.loading_NeuralNetwork(sys.argv[12])
        basisModels = [basisModel,basisModel2,basisModel3]
        referenceModels = [verificationModelTimeA,verificationModelTimeB,verificationModelTimeC]
        modelstimeText = [verificationModelTimeTextA,verificationModelTimeTextB,verificationModelTimeTextC]
        modelsEverything = [verificationModelEverythingA, verificationModelEverythingB,
                          verificationModelEverythingC]
        spearmanDomain,standaardDomain,labelsAll,standaardAll,labelsDomain,standaardLabelDomain,numberEqual, \
            spearmanAlltimeText, standaardDomaintimeText, labelsDomaintimeText, standaardLabelDomaintimeText, \
            labelsAlltimeText, standaardLabelAllDomaintimeText, spearmanAllEverything, standaardDomainEverything, \
            labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything \
            = spearmanRanking([model],basisModels,referenceModels,modelstimeText,modelsEverything)
        numberEqual1.update(numberEqual)
        file = open('resultspearman/biLSTM/versusBase/'+model[1],"w")
        '''
        if len(spearmanDomain) > 0:
            print("Ranking Time")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanDomain)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanDomain)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomain)))
        if len(labelsDomain) >0:
            print("Labels domain Time")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomain)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomain)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomain)))
            print("Labels all Time")
        if  len(labelsAll) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAll)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAll)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardAll)))
        if len(spearmanAlltimeText) > 0:
            print("Ranking timeText")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAlltimeText)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAlltimeText)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomaintimeText)))
        if len(labelsDomaintimeText) >0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomaintimeText)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomaintimeText)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomaintimeText)))
            print("Labels all")
        if  len(labelsAlltimeText) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAlltimeText)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAlltimeText)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomaintimeText)))
        if len(spearmanAllEverything) > 0:
            print("Ranking Everything")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything)))
        if len(labelsDomainEverything) >0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything)))
            print("Labels all")
        if  len(labelsAllEverything) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything)))
        '''
        if len(spearmanDomain) > 0:
            file.write("Ranking Time"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanDomain)) +"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanDomain))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomain))+"\n")
        if len(labelsDomain) >0:
            file.write("Labels domain Time"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomain))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomain))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomain))+"\n")
            file.write("Labels all Time"+"\n")
        if  len(labelsAll) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAll))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAll))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardAll))+"\n")
        if len(spearmanAlltimeText) > 0:
            file.write("Ranking timeText"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAlltimeText))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAlltimeText))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomaintimeText))+"\n")
        if len(labelsDomaintimeText) >0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomaintimeText))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomaintimeText))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomaintimeText))+"\n")
            file.write("Labels all"+"\n")
        if  len(labelsAlltimeText) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAlltimeText))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAlltimeText))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomaintimeText))+"\n")
        if len(spearmanAllEverything) > 0:
            file.write("Ranking Everything"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything))+"\n")
        if len(labelsDomainEverything) >0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything))+"\n")
            file.write("Labels all"+"\n")
        if  len(labelsAllEverything) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything))+"\n")
        file.write(str(numberEqual))
        file.close()
        spearmanAll.extend(spearmanDomain)
        standaard.extend(standaardDomain)
        labelsAllE.extend(labelsAll)
        standaardLabelsAll.extend(standaardAll)
        labelsDomainE.extend(labelsDomain)
        standaardDomainLabelsAll.extend(standaardLabelDomain)
        spearmanAlltimeTextD.extend(spearmanAlltimeText)
        standaardtimeText.extend(standaardDomaintimeText)
        labelsDomainEtimeText.extend(labelsDomaintimeText)
        standaardDomainLabelsAlltimeText.extend(standaardLabelDomaintimeText)
        labelsAllEtimeText.extend(labelsAlltimeText)
        standaardLabelsAlltimeText.extend(standaardLabelAllDomaintimeText)
        spearmanAllEverythingD.extend(spearmanAllEverything)
        standaardEverything.extend(standaardDomainEverything)
        labelsDomainEEverything.extend(labelsDomainEverything)
        standaardDomainLabelsAllEverything.extend(standaardLabelDomainEverything)
        labelsAllEEverything.extend(labelsAllEverything)
        standaardLabelsAllEverything.extend(standaardLabelAllDomainEverything)
        numberEqualtimeText, spearmanAlltimeText, standaardDomaintimeText, labelsDomaintimeText, standaardLabelDomaintimeText, \
        labelsAlltimeText, standaardLabelAllDomaintimeText, spearmanAllEverything, standaardDomainEverything, \
        labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything \
            = spearmanRankingTime([model],referenceModels, modelstimeText, modelsEverything)
        numberEqual2.update(numberEqualtimeText)
        #print("Time with timeText and everything")
        file = open('resultspearman/biLSTM/versusTime/' + model[1], "w")
        ''''
        if len(spearmanAlltimeText) > 0:
            print("Ranking timeText")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAlltimeText)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAlltimeText)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomaintimeText)))
        if len(labelsDomaintimeText) >0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomaintimeText)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomaintimeText)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomaintimeText)))
            print("Labels all")
        if  len(labelsAlltimeText) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAlltimeText)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAlltimeText)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomaintimeText)))
        if len(spearmanAllEverything) > 0:
            print("Ranking Everything")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything)))
        if len(labelsDomainEverything) >0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything)))
            print("Labels all")
        if  len(labelsAllEverything) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything)))
        '''
        if len(spearmanAlltimeText) > 0:
            file.write("Ranking timeText"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAlltimeText))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAlltimeText))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomaintimeText))+"\n")
        if len(labelsDomaintimeText) >0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomaintimeText))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomaintimeText))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomaintimeText))+"\n")
            file.write("Labels all"+"\n")
        if  len(labelsAlltimeText) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAlltimeText))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAlltimeText))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomaintimeText))+"\n")
        if len(spearmanAllEverything) > 0:
            file.write("Ranking Everything"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything))+"\n")
        if len(labelsDomainEverything) >0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything))+"\n")
            file.write("Labels all"+"\n")
        if  len(labelsAllEverything) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything))+"\n")
        file.write(str(numberEqualtimeText))
        file.close()
        spearmanAlltimeText2.extend(spearmanAlltimeText)
        standaardtimeText2.extend(standaardDomaintimeText)
        labelsDomainEtimeText2.extend(labelsDomaintimeText)
        standaardDomainLabelsAlltimeText2.extend(standaardLabelDomaintimeText)
        labelsAllEtimeText2.extend(labelsAlltimeText)
        standaardLabelsAlltimeText2.extend(standaardLabelAllDomaintimeText)
        spearmanAllEverything2.extend(spearmanAllEverything)
        standaardEverything2.extend(standaardDomainEverything)
        labelsDomainEEverything2.extend(labelsDomainEverything)
        standaardDomainLabelsAllEverything2.extend(standaardLabelDomainEverything)
        labelsAllEEverything2.extend(labelsAllEverything)
        standaardLabelsAllEverything2.extend(standaardLabelAllDomainEverything)
        numberEqualEverything,spearmanAllEverything, standaardDomainEverything, \
        labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything \
            = spearmanRankingtimeText([model],modelstimeText, modelsEverything)
        numberEqual3.update(numberEqualEverything)
        file = open('resultspearman/biLSTM/versustimeText/' + model[1], "w")
        '''
        print("timeText with  everything")
        if len(spearmanAllEverything) > 0:
            print("Ranking Everything")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything)))
        if len(labelsDomainEverything) > 0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything)))
            print("Labels all")
        if len(labelsAllEverything) > 0:
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything)))
        '''
        if len(spearmanAllEverything) > 0:
            file.write("Ranking Everything"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything))+"\n")
        if len(labelsDomainEverything) > 0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything))+"\n")
            file.write("Labels all"+"\n")
        if len(labelsAllEverything) > 0:
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything))+"\n")
        file.write(str(numberEqualEverything))
        file.close()
        spearmanAllEverything3.extend(spearmanAllEverything)
        standaardEverything3.extend(standaardDomainEverything)
        labelsDomainEEverything3.extend(labelsDomainEverything)
        standaardDomainLabelsAllEverything3.extend(standaardLabelDomainEverything)
        labelsAllEEverything3.extend(labelsAllEverything)
        standaardLabelsAllEverything3.extend(standaardLabelAllDomainEverything)
    file = open("resultspearman/biLSTM/versusBase/AllDomains","w")
    '''
    print("All domains for base versus time,timeText and everything")
    print("Ranking")
    print("Gemiddelde all : " + str(np.mean(spearmanAll)))
    print("Deviatie all : "  + str(np.std(spearmanAll)))
    print("Deviation all : " + str(np.mean(standaard)))
    print("stand mean deviation all : " + str(np.std(standaard)) )
    print("Labels all")
    print("Gemiddelde all : " + str(np.mean(labelsAllE)))
    print("Deviatie all : " + str(np.std(labelsAllE)))
    print("Deviation all : " + str(np.mean(standaardAll)))
    print("stand mean deviation all : " + str(np.std(standaardAll)))
    print("Labels domain")
    print("Gemiddelde all : " + str(np.mean(labelsDomainE)))
    print("Deviatie all : " + str(np.std(labelsDomainE)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAll)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAll)))
    print("Ranking timeText")
    print("Gemiddelde all : " + str(np.mean(spearmanAlltimeText)))
    print("Deviatie all : " + str(np.std(spearmanAlltimeText)))
    print("Deviation all : " + str(np.mean(standaardtimeText)))
    print("stand mean deviation all : " + str(np.std(standaardtimeText)))
    print("Labels all timeText")
    print("Gemiddelde all : " + str(np.mean(labelsAllEtimeText)))
    print("Deviatie all : " + str(np.std(labelsAllEtimeText)))
    print("Deviation all : " + str(np.mean(standaardLabelsAlltimeText)))
    print("stand mean deviation all : " + str(np.std(standaardLabelsAlltimeText)))
    print("Labels domain timeText")
    print("Gemiddelde all : " + str(np.mean(labelsDomainEtimeText)))
    print("Deviatie all : " + str(np.std(labelsDomainEtimeText)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAlltimeText)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAlltimeText)))
    print("Ranking everything")
    print("Gemiddelde all : " + str(np.mean(spearmanAllEverything)))
    print("Deviatie all : " + str(np.std(spearmanAllEverything)))
    print("Deviation all : " + str(np.mean(standaardEverything)))
    print("stand mean deviation all : " + str(np.std(standaardEverything)))
    print("Labels all everything")
    print("Gemiddelde all : " + str(np.mean(labelsAllEEverything)))
    print("Deviatie all : " + str(np.std(labelsAllEEverything)))
    print("Deviation all : " + str(np.mean(standaardLabelsAllEverything)))
    print("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything)))
    print("Labels domain everything")
    print("Gemiddelde all : " + str(np.mean(labelsDomainEEverything)))
    print("Deviatie all : " + str(np.std(labelsDomainEEverything)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything)))
    print(numberEqual1)
    '''
    file.write("All domains for base versus time,timeText and everything"+"\n")
    file.write("Ranking"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAll))+"\n")
    file.write("Deviatie all : " + str(np.std(spearmanAll))+"\n")
    file.write("Deviation all : " + str(np.mean(standaard))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaard))+"\n")
    file.write("Labels all"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllE))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllE))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardAll))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardAll))+"\n")
    file.write("Labels domain"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainE))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainE))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAll))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAll))+"\n")
    file.write("Ranking timeText"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAlltimeTextD))+"\n")
    file.write("Deviatie all : " + str(np.std(spearmanAlltimeTextD))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardtimeText))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardtimeText))+"\n")
    file.write("Labels all timeText"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEtimeText))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEtimeText))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAlltimeText))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAlltimeText))+"\n")
    file.write("Labels domain timeText"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEtimeText))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEtimeText))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAlltimeText))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAlltimeText))+"\n")
    file.write("Ranking everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAllEverythingD))+"\n")
    file.write("Deviatie all : " + str(np.std(spearmanAllEverythingD))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardEverything))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardEverything))+"\n")
    file.write("Labels all everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEEverything))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEEverything))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllEverything))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything))+"\n")
    file.write("Labels domain everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEEverything))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEEverything))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything))+"\n")
    file.write(str(numberEqual1))
    file.close()

    file = open("resultspearman/biLSTM/versusTime/AllDomains", "w")
    '''
    print("All domains for time versus timeText and everything")
    print("Ranking timeText")
    print("Gemiddelde all : " + str(np.mean(spearmanAlltimeText2)))
    print("Deviatie all : " + str(np.std(spearmanAlltimeText2)))
    print("Deviation all : " + str(np.mean(standaardtimeText2)))
    print("stand mean deviation all : " + str(np.std(standaardtimeText2)))
    print("Labels all timeText")
    print("Gemiddelde all : " + str(np.mean(labelsAllEtimeText2)))
    print("Deviatie all : " + str(np.std(labelsAllEtimeText2)))
    print("Deviation all : " + str(np.mean(standaardLabelsAlltimeText2)))
    print("stand mean deviation all : " + str(np.std(standaardLabelsAlltimeText2)))
    print("Labels domain timeText")
    print("Gemiddelde all : " + str(np.mean(labelsDomainEtimeText2)))
    print("Deviatie all : " + str(np.std(labelsDomainEtimeText2)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAlltimeText2)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAlltimeText2)))
    print("Ranking everything")
    print("Gemiddelde all : " + str(np.mean(spearmanAllEverything2)))
    print("Deviatie all : " + str(np.std(spearmanAllEverything2)))
    print("Deviation all : " + str(np.mean(standaardEverything2)))
    print("stand mean deviation all : " + str(np.std(standaardEverything2)))
    print("Labels all everything")
    print("Gemiddelde all : " + str(np.mean(labelsAllEEverything2)))
    print("Deviatie all : " + str(np.std(labelsAllEEverything2)))
    print("Deviation all : " + str(np.mean(standaardLabelsAllEverything2)))
    print("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything2)))
    print("Labels domain everything")
    print("Gemiddelde all : " + str(np.mean(labelsDomainEEverything2)))
    print("Deviatie all : " + str(np.std(labelsDomainEEverything2)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything2)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything2)))
    print(numberEqual2)
    '''
    file.write("All domains for time versus timeText and everything"+"\n")
    file.write("Ranking timeText"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAlltimeText2))+"\n")
    file.write("Deviatie all : " + str(np.std(spearmanAlltimeText2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardtimeText2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardtimeText2))+"\n")
    file.write("Labels all timeText"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEtimeText2))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEtimeText2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAlltimeText2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAlltimeText2))+"\n")
    file.write("Labels domain timeText"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEtimeText2))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEtimeText2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAlltimeText2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAlltimeText2))+"\n")
    file.write("Ranking everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAllEverything2))+"\n")
    file.write("Deviatie all : " + str(np.std(spearmanAllEverything2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardEverything2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardEverything2))+"\n")
    file.write("Labels all everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEEverything2))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEEverything2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllEverything2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything2))+"\n")
    file.write("Labels domain everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEEverything2))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEEverything2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything2))+"\n")
    file.write(str(numberEqual2))
    file.close()

    file = open("resultspearman/biLSTM/versustimeText/AllDomains", "w")
    file.write("All domains for timeText versus everything"+"\n")
    file.write("Ranking everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAllEverything3))+"\n")
    file.write("Deviatie all : " + str(np.std(spearmanAllEverything3))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardEverything3))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardEverything3))+"\n")
    file.write("Labels all everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEEverything3))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEEverything3))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllEverything3))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything3))+"\n")
    file.write("Labels domain everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEEverything3))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEEverything3))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything3))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything3))+"\n")
    file.write(str(numberEqual3))