import argparse
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.util import bigrams
from collections import Counter
import random
from math import log
import pickle

#extract dat from URLS in txt file into individual txt files
def directoryExtract(directories_list, directories_loc):
    with open(directories_list, 'r') as directoryFile:
        for directoryURL in directoryFile:
            try:
                r = requests.get(directoryURL)
                html = r.content
            except Exception as e:
                print(e)

            try:
                soup = BeautifulSoup(html, 'html.parser')
                cleanURL = re.sub('[\W_]+', '', directoryURL)
                fileLocation = directories_loc + cleanURL
                text = soup.get_text()
                text = text.replace('\r', ' ').replace('\n', ' ')
                with open(fileLocation, "w+") as newDir:
                    newDir.write(text)
            except Exception as e:
                print(e)

#extract dat from URLS in txt file into individual txt files     
def universityExtract(universities_list, universities_loc):
    with open(universities_list, 'r') as universityFile:
        for universityURL in universityFile:
            try:
                universityURL = universityURL.replace('\r', '').replace('\n', '')
                r = requests.get(universityURL)
                html = r.content
            except Exception as e:
                print(e)
                continue
            try:
                soup = BeautifulSoup(html, 'html.parser')
                cleanURL = re.sub('[\W_]+', '', universityURL)
                fileLocation = universities_loc + cleanURL
                text = soup.get_text()
                text = text.replace('\r', ' ').replace('\n', ' ')
                with open(fileLocation, "w+") as newDir:
                    newDir.write(text)
            except Exception as e:
                print(e)

#initalize webpage for testing
def initTest(testPage, lowerCase, stemming):
    try:
        r = requests.get(testPage)
        html = r.content
    except:
        print("Bad URL", testPage)

    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    text = text.replace('\r', ' ').replace('\n', ' ')
    page = RegexpTokenizer('\w+').tokenize(text)
    if lowerCase:
        for i, word in enumerate(page):
            page[i] = PorterStemmer().stem(word)
    if stemming:
        for i, word in enumerate(page):
            page[i] = PorterStemmer().stem(word)
    page = bigrams(page)
    return page
    

#initalize words for classifier
def initWords(stemming, lowerCase, directoryFolder, universityFolder, testModel):
    trainDirectory = []
    testDirectory = []
    for file in os.listdir(directoryFolder):
        page = []
        with open(directoryFolder + file, "r") as text:
            data = text.read()
            page += RegexpTokenizer('\w+').tokenize(data)
            if lowerCase:
                for i, word in enumerate(page):
                    page[i] = word.lower()
            if stemming:
                for i, word in enumerate(page):
                    page[i] = PorterStemmer().stem(word)
        if testModel:
            trainBool = random.randint(0, 1)
            if (trainBool):
                trainDirectory.append(page)
            else:
                testDirectory.append(page)
        else:
            trainDirectory.append(page)


    trainUniversity = []
    testUniversity = []
    for file in os.listdir(universityFolder):
        page = []
        with open(universityFolder + file, "r") as text:
            data = text.read()
            page += RegexpTokenizer('\w+').tokenize(data)
            if lowerCase:
                for i, word in enumerate(page):
                    page[i] = word.lower()
            if stemming:
                for i, word in enumerate(page):
                    page[i] = PorterStemmer().stem(word)
        if testModel:
            trainBool = random.randint(0, 1)
            if (trainBool):
                trainUniversity.append(page)
            else:
                testUniversity.append(page)
        else:
            trainUniversity.append(page)

    return trainDirectory, testDirectory, trainUniversity, testUniversity

#train bigram bag of words model
def modelTrain(trainDirectory, trainUniversity, modelLoc):
    dirBigramCount = Counter()
    bigramDirectoryNum = 0
    for document in trainDirectory:
        docBigrams = bigrams(document)
        docBigramCount = Counter(docBigrams)
        dirBigramCount += docBigramCount
        bigramDirectoryNum += sum(docBigramCount.values())
    dirBigramDict = dict(dirBigramCount)

    uniBigramCount = Counter()
    bigramUniversityNum = 0
    for document in trainUniversity:
        docBigrams = bigrams(document)
        docBigramCount = Counter(docBigrams)
        uniBigramCount += docBigramCount
        bigramUniversityNum += sum(docBigramCount.values())
    uniBigramDict = dict(uniBigramCount)
    
    for bigram in dirBigramDict:
        uniBigramDict[bigram] = uniBigramDict.get(bigram, 0)

    for bigram in uniBigramCount:
        dirBigramDict[bigram] = dirBigramDict.get(bigram, 0)

    diffBigramDict = {}
    for bigram in dirBigramDict:
        bigramDirValue = log(dirBigramDict[bigram] + 1) / (uniBigramCount[bigram] + 1)
        bigramUniValue = log((len(trainUniversity)/len(trainDirectory)) * (uniBigramCount[bigram] + 1) / (dirBigramDict[bigram] + 1))
        dirBigramDict[bigram] = bigramDirValue 
        uniBigramDict[bigram] = bigramUniValue
        diffBigramDict[bigram] = bigramDirValue - bigramUniValue

    with open(modelLoc + ".pickle", "wb+") as newFile:
        pickle.dump(diffBigramDict, newFile)

#test webpage on trained model
def modelTest(testDirectory, testUniversity, modelLoc):
    model = {}
    with open(modelLoc + ".pickle", 'rb') as fileLoc:
        model = pickle.load(fileLoc)

    if len(testUniversity) != 0 and len(testDirectory) != 1:
        return
    
    DirectoryScore = 0
    for word in testDirectory:
        DirectoryScore += model.get(word, 0)

    if DirectoryScore > 0:
        print("Classified as Directory Page")
    else:
        print("Classified as NOT a Directory Page")

#work alongside options to call correct functions
def main(testPage, testDir, stemming, lowerCase, createModel, testModel, extractDirectory, extractUniversity, universityURLs, directoryURLs, universityFolder, directoryFolder, modelLoc):
    if (extractDirectory):
        directoryExtract(directoryURLs, directoryFolder)
    if (extractUniversity):
        universityExtract(universityURLs, universityFolder)
    if (createModel):
        trainDirectory, testDirectory, trainUniversity, testUniversity = initWords(stemming, lowerCase, directoryFolder, universityFolder, testModel)
        modelTrain(trainDirectory, trainUniversity, modelLoc)
        if (testModel):
            modelTest(testDirectory, testUniversity, modelLoc)
        
    if (testDir):
        testPage = initTest(testPage, lowerCase, stemming)
        modelTest(testPage, [], modelLoc)
        
#option set
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('testPage', metavar='testPage', type=str, nargs='?', default='https://www.google.com/', help='page to test against the model')
    parser.add_argument('--testDir', action="store_false", help = 'do not test a directory page against the model')
    parser.add_argument('--stemming', action="store_false", help = 'word stemming')
    parser.add_argument('--lowerCase', action="store_false", help = 'convert all words to lower case')
    parser.add_argument('--createModel', action="store_true", help = 'create classification model for directory pages')
    parser.add_argument('--testModel', action="store_true", help = 'rnadomely split data into a test and train group to test the models accuracy')
    parser.add_argument('--extractDirectory', action="store_true", help = 'extract directory data for use with training/testing')
    parser.add_argument('--extractUniversity', action="store_true", help = 'extract misc. university page data for use with training/testing')
    parser.add_argument('-universityURLs', type=str, default='university_pages.txt', help='Text file to extract university urls from')
    parser.add_argument('-directoryURLs', type=str, default='faculty_directories.txt', help='Text file to extract directory urls from')
    parser.add_argument('-universityFolder', type=str, default='university_files/', help='Folder to save extracted university pages to')
    parser.add_argument('-directoryFolder', type=str, default='directory_files/', help='Folder to save extracted directory pages to')
    parser.add_argument('-modelLoc', type=str, default='model', help='location of model')
    args = parser.parse_args()
    main(args.testPage, args.testDir, args.stemming, args.lowerCase, args.createModel, args.testModel, args.extractDirectory, args.extractUniversity, args.universityURLs, args.directoryURLs, args.universityFolder, args.directoryFolder, args.modelLoc)