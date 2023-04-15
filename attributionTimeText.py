import io
import math
import os
import statistics
import sys
import random

import matplotlib
import scipy.stats as ss
import numpy
import numpy as np
import torch.nn.functional as F
from fontTools.merge import cmap
from matplotlib import pyplot as plt, colors
from division2DifferenceTimeText.encoderGlobal import encoder as encoderText
from division2DifferenceTimeText.verificationModelGlobal import verifactionModel as verificationText
from division2DifferenceTimeText import OneHotEncoder, labelEmbeddingLayer, encoderMetadata, \
    instanceEncoder, evidence_ranker, labelMaskDomain
from dataset import NUS
import torch
from torch.utils.data import DataLoader

'''
Calculate attribution for text and time for division2DifferenceTimeText
code based on https://github.com/ankurtaly/Integrated-Gradients
'''
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

def calculate_outputs_and_gradients(data, model):
    # do the pre-processing
    predict_idx = None
    gradientsEncoding = []
    gradientsTime = []
    for i in range(len(data[0])):
        metaDataClaim = oneHotEncoder.encode(data[3][i], device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = data[0][0].split('-')[0]
        output,claim_encodingFull,claim_EncodingWithoutTime,times,evidenceEncodings = model.forwardAttribution(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i],data[17][i],data[18][i])
        output = F.softmax(output, dim=0)
        target_label_idx = torch.argmax(output).item()
        index = np.ones((1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        output = output.gather(0, index)
        # clear grad
        model.zero_grad()
        claim_encodingFull.retain_grad()
        claim_EncodingWithoutTime.retain_grad()
        for time in times:
            time.retain_grad()
        for evidenceEncoding in evidenceEncodings:
            evidenceEncoding[0].retain_grad()
            evidenceEncoding[1].retain_grad()
            for time in evidenceEncoding[2]:
                time.retain_grad()
        output.backward()
        gradient = claim_encodingFull.grad.detach().cpu().numpy()[0]
        gradientsEncoding.append(gradient)
        claim_EncodingWithoutTime.retain_grad()
        gradient = claim_EncodingWithoutTime.grad.detach().cpu().numpy()[0]
        gradientsEncoding.append(gradient)
        gradientsTime.append([])
        for time in times:
            gradientsTime[-1].append(time.grad.detach().cpu().numpy()[0])
        for evidenceEncoding in evidenceEncodings:
            gradient = evidenceEncoding[0].grad.detach().cpu().numpy()[0]
            gradientsEncoding.append(gradient)
            gradient = evidenceEncoding[1].grad.detach().cpu().numpy()[0]
            gradientsEncoding.append(gradient)
            gradientsTime.append([])
            for time in evidenceEncoding[2]:
                gradientsTime[-1].append(time.grad.detach().cpu().numpy()[0])
        invoer = [claim_encodingFull,claim_EncodingWithoutTime,times,evidenceEncodings]
    gradientsEncoding = np.array(gradientsEncoding)
    return gradientsEncoding,gradientsTime, target_label_idx,invoer,metadata_encoding

def getInputs(data, model):
    for i in range(len(data[0])):
        metaDataClaim = oneHotEncoder.encode(data[3][i], device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = data[0][0].split('-')[0]
        output,claim_encodingFull,claim_EncodingWithoutTime,times,evidenceEncodings = model.forwardAttribution(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i],data[17][i],data[18][i])
        invoer = [claim_encodingFull,claim_EncodingWithoutTime,times,evidenceEncodings]
    return invoer,metadata_encoding

def calculate_outputs_and_gradientsIntegrated(inputs,metadata_encoding, model,target_label_idx):
    # do the pre-processing
    predict_idx = None
    gradientsEncoding = []
    gradientsTime = []
    for input in inputs:
        gradientsEncodingEntry = []
        gradientsTimeEntry = []
        #print(input[2])
        input[0].requires_grad = True
        input[0].retain_grad()
        for time in input[1]:
            time.requires_grad = True
            time.retain_grad()
        for i in range(len(input[2])):
            input[2][i][0].requires_grad = True
            input[2][i][0].retain_grad()
            for time in input[2][i][1]:
                time.requires_grad = True
                time.retain_grad()

        output = model.forwardIntegrated(input,metadata_encoding)
        output = F.softmax(output, dim=0)
        index = np.ones((1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        output = output.gather(0, index)
        # clear grad
        model.zero_grad()
        output.backward()
        gradient = input[0].grad.detach().cpu().numpy()[0]
        gradientsEncodingEntry.append(gradient)
        gradientsTimeEntry.append([])
        for time in input[1]:
            gradientsTimeEntry[-1].append(time.grad.detach().cpu().numpy()[0])
        for i in range(len(input[2])):
            gradient = input[2][i][0].grad.detach().cpu().numpy()[0]
            gradientsEncodingEntry.append(gradient)
            gradientsTimeEntry.append([])
            for time in input[2][i][1]:
                gradientsTimeEntry[-1].append(time.grad.detach().cpu().numpy()[0])
        gradientsTime.append(gradientsTimeEntry)
        gradientsEncoding.append(gradientsEncodingEntry)
    gradientsEncoding = np.array(gradientsEncoding)
    return gradientsEncoding,gradientsTime


# integrated gradients
def integrated_gradients(inputs,metadata_encoding, model, target_label_idx, predict_and_gradients, baseline, steps=50, cuda=False):
    # scale inputs and compute gradients
    baselineClaim = torch.zeros([256])
    baselineEvidence = torch.zeros([256])
    baselineClaim, baselineEvidence = model.getBaseLine(baselineClaim,baselineEvidence,metadata_encoding.squeeze(0))
    scaled_inputs = []
    for i in range(0, steps + 1):
        entry = []
        entry.append(baselineClaim + (float(i)/ steps) * (inputs[1]-baselineClaim))
        times = []
        for k in range(len(inputs[2])):
            times.append(baselineEvidence + (float(i) / steps) * (inputs[2][k]-baselineClaim))

        entry.append(times)
        evidences = []
        for j in range(len(inputs[3])):
            times = []
            for k in range(len(inputs[3][j][2])):
                times.append(baselineEvidence + (float(i) / steps) * (inputs[3][j][2][k]-baselineEvidence))

            evidences.append((baselineEvidence + (float(i) / steps) * (inputs[3][j][1]-baselineEvidence),times))
        entry.append(evidences)
        scaled_inputs.append(entry)

    gradientsEncoding,gradientsTime = predict_and_gradients(scaled_inputs,metadata_encoding, model, target_label_idx)

    sumGradsEncoding = [np.zeros(len(inputs[3])+1)]
    sumGradsTime = []
    for k in range(len(gradientsTime[0])):
        if len(gradientsTime[0][k])>0:
            sumGradsTime.append(np.zeros(len(gradientsTime[0][k])))
        else:
            sumGradsTime.append(np.zeros(1))
    for i in range(0,steps +1):
        sumGradsEncoding += gradientsEncoding[i]
        for k in range(len(gradientsTime[i])):
            if len(gradientsTime[i][k])>0:
                sumGradsTime[k] += np.array(gradientsTime[i][k])
    avg_gradientsEncoding = [element/(steps+1) for element in sumGradsEncoding]
    avg_gradientsTime = sumGradsTime
    for k in range(len(sumGradsTime)):
        if len(sumGradsTime[k])>0:
            avg_gradientsTime[k] = [element / (steps + 1) for element in sumGradsTime[k]]
        else:
            avg_gradientsTime[k] = np.zeros(1)
    integrated_gradEncoding = []
    integrated_gradTime = []
    integrated_gradEncoding.append((inputs[1].numpy()-baselineClaim.numpy())*avg_gradientsEncoding[0][0])
    integrated_gradTimeClaim = []
    for k in range(len(inputs[2])):
        integrated_gradTimeClaim.append((inputs[2][k].numpy()-baselineClaim.numpy()) * avg_gradientsTime[0][k])
    integrated_gradTime.append(integrated_gradTimeClaim)
    for i in range(len(inputs[3])):
        integrated_gradEncoding.append((inputs[3][i][1].numpy()-baselineEvidence.numpy())*avg_gradientsEncoding[0][i+1])
        integrated_gradTimeEvidence = []
        for k in range(len(inputs[3][i][2])):
            integrated_gradTimeEvidence.append((inputs[3][i][2][k].numpy()-baselineEvidence.numpy()) * avg_gradientsTime[i + 1][k])
        integrated_gradTime.append(integrated_gradTimeEvidence)
    return integrated_gradEncoding,integrated_gradTime

def random_baseline_integrated_gradients(inputs,metadata_encoding, model, target_label_idx, predict_and_gradients, steps, num_random_trials):
    all_intgradsEncoding = []
    all_intgradsTime = []
    integrated_gradEncoding,integrated_gradTime = integrated_gradients(inputs,metadata_encoding, model, target_label_idx, predict_and_gradients, \
                                            baseline=torch.zeros(256), steps=steps)
    all_intgradsEncoding.append(integrated_gradEncoding)
    all_intgradsTime.append(integrated_gradTime)
    return integrated_gradEncoding,integrated_gradTime

if __name__ == '__main__':
    '''
    argument 1 path of model
    argument 2 name of domain to take examples from to calculate attribution
    argument 3 alpha
    argument 4 withPretext
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    domainIndices, domainLabels, domainLabelIndices, domainWeights = getLabelIndicesDomain(
        'labels/labels.tsv', 'labels/labelSequence', 'labels/weights.tsv')
    domains = {sys.argv[2]}
    datas= []
    for domain in domains:
        test_set = NUS(mode='Test', path='test/test-' + domain + '.tsv', domain=domain)
        dev_loader = DataLoader(test_set,
                                 batch_size=1,
                                 shuffle=False)
        datas.append([dev_loader, domain])


    for data in datas:
        oneHotEncoder = OneHotEncoder.oneHotEncoder('Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer.labelEmbeddingLayer(772, domainIndices)
        encoderM = encoderText(300, 128,sys.argv[3]).to(device)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoder).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(772, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(772, domainIndices, data[1],
                                                           len(domainIndices[data[1]])).to(device)
        verificationModelTimeAdding = verificationText(
            encoderM, encoderMetadataM, instanceEncoderM,
            evidenceRankerM,
            labelEmbeddingLayerM, labelMaskDomainM,
            domainIndices,
            data[1],sys.argv[3],bool(sys.argv[5])).to(device)
        verificationModelTimeAdding.loading_NeuralNetwork(sys.argv[1])
        for entry in data[0]:
            print(entry)
            gradientsEncoding,gradientsTime, predictedLabel,_,_ = calculate_outputs_and_gradients(entry, verificationModelTimeAdding)
            with torch.no_grad():
                inputs, metadata_encoding = getInputs(entry, verificationModelTimeAdding)
                label_index = domainLabelIndices[data[1]][domainLabels[data[1]].index(entry[4][0])]
            attributionsEncoding,attributionsTime = random_baseline_integrated_gradients(inputs,metadata_encoding, verificationModelTimeAdding, label_index,
                                                                calculate_outputs_and_gradientsIntegrated, \
                                                                steps=1000, num_random_trials=1)

            print("Summation of the attributions encoding")
            print(numpy.sum(attributionsEncoding,axis=1))
            print("Summation of the attributions time")
            attributionsTimeSum = []
            for k in range(len(attributionsTime)):
                if len(attributionsTime[k])==0:
                    attributionsTimeSum.append(None)
                else:
                    attributionsTimeSum.append(numpy.sum(attributionsTime[k],axis=1))

            print(attributionsTimeSum)
            print("Summation of absolute text encoding")
            print(numpy.sum(np.abs(attributionsEncoding), axis=1))
            attributionsTimeSumAbsolute = []
            for k in range(len(attributionsTime)):
                if len(attributionsTime[k])==0:
                    attributionsTimeSumAbsolute.append(None)
                else:
                    attributionsTimeSumAbsolute.append(numpy.sum(np.abs(attributionsTime[k]),axis=1))
            print("Summation of absolute time ")
            print(attributionsTimeSumAbsolute)

