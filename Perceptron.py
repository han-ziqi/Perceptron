# -*- coding = utf-8 -*-
# @Time: 25/02/2022 16:28
# @Author: Ziqi Han
# @Student ID: 201568748
# @File: Perceptron.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import random as rd

# The first part is to format the file, converting the two data files provided into an array format
# that can be easily processed by python
trainData = pd.read_csv('train.data', header=None)  # Read train.data using pandas
trainDataList = trainData.values.tolist()  # Converting the read data into List format
trainDataArray = np.array(trainDataList)  # Converting the List into Array format

# The goal is to combine three classes into three lists, so initialize three lists
Data_1 = []
Data_2 = []
Data_3 = []
Data_1_2 = []
Data_2_3 = []
Data_1_3 = []

for row in trainDataArray:
    if row[4] == 'class-1':
        Data_1.append(row)
        Data_1_2.append(row)
        Data_1_3.append(row)
    if row[4] == 'class-2':
        Data_2.append(row)
        Data_1_2.append(row)
        Data_2_3.append(row)
    if row[4] == 'class-3':
        Data_3.append(row)
        Data_1_3.append(row)
        Data_2_3.append(row)

# The following procedure formats test.data in much the same way as before
testData = pd.read_csv('test.data', header=None)
testDataList = testData.values.tolist()
testDataArray = np.array(testDataList)

Data_1_t = []
Data_2_t = []
Data_3_t = []
Data_1_2_t = []
Data_2_3_t = []
Data_1_3_t = []

for row in testDataArray:
    if row[4] == 'class-1':
        Data_1_t.append(row)
        Data_1_2_t.append(row)
        Data_1_3_t.append(row)
    if row[4] == 'class-2':
        Data_2_t.append(row)
        Data_1_2_t.append(row)
        Data_2_3_t.append(row)
    if row[4] == 'class-3':
        Data_3_t.append(row)
        Data_1_3_t.append(row)
        Data_2_3_t.append(row)


# The following def section shows the training and testing data using perceptron
def PerceptronBinary(sourceData, w, b, status):
    # First, we define the status, if NOT equal to test, then it will be regard as train
    Status = status.upper()
    if Status != "TEST":
        iterations = 20
        w = [0.0, 0.0, 0.0, 0.0]
        b = 0
        bestWeight = 0
        bestBias = 0
        bestAcc = 0
    else:
        iterations = 1
        bestWeight = w
        bestBias = b
        bestAcc = 0

    # Divide the data into two parts: feature and classNumber
    data = np.hsplit((np.array(sourceData)), np.array([4, 8]))
    feature = data[0].astype(float)
    classNumber = np.array(np.unique(data[1], return_inverse=True))
    classNumber = np.array(classNumber[1])
    classNumber[classNumber < 1] = -1  # change classNumber from 0 to -1

    # Random disruptions required in each episode
    for episode in range(iterations):
        acc = 0
        zipList = list(zip(feature, classNumber))
        rd.shuffle(zipList)
        feature, classNumber = zip(*zipList)

        for i in range(len(feature)):
            a = 0  # Set activation value to 0
            for j in range(len(feature[i])):
                a += (w[j] * feature[i][j]) + b
            if a > 0:
                a = 1
            else:
                a = -1
            if (a * classNumber[i]) <= 0:
                if status != "TEST":
                    for j in range(len(w)):
                        w[j] = w[j] + (classNumber[i] * feature[i][j])
                    b += classNumber[i]  # change the bias value
            else:
                acc += 1
        # Record best results
        if bestAcc < acc:
            bestAcc = acc
            if status != "TEST":
                bestWeight = w.copy()
                bestBias = b
    # Dashboard
    print("Current state is:", status, ". correctness is: ", bestAcc, "/", len(classNumber))
    print("Current Weight is:", w, "and Bias is:", b)
    print("Currency is:", bestAcc / len(feature) * 100, "% Percentage")

    # Return the best weight and bias at the end of the training
    if status != "TEST":
        return bestWeight, bestBias
    else:
        return


'''
w = 0
b = 0
print("Result of Class 1 and 2 ")
weight, bias = PerceptronBinary(Data_1_2, w, b, "Train")
b=PerceptronBinary(Data_1_2_t, weight, bias, "Test")
print("---------------------------------------------------------------------------")
print("Result of Class 2 and 3 ")
weight, bias = PerceptronBinary(Data_2_3, w, b, "Train")
PerceptronBinary(Data_2_3_t, weight, bias, "Test")
print("---------------------------------------------------------------------------")
print("Result of Class 1 and 3 ")
weight, bias = PerceptronBinary(Data_1_3, w, b, "Train")
PerceptronBinary(Data_1_3_t, weight, bias, "Test")
'''


def PerceptronMultiClass(sourceData, w, b, status, classValue):

    # Define coefficient for l2 regularisation
    # coefficient = 0.01

    # First, we define the status, if NOT equal to test, then it will be regard as train
    Status = status.upper()
    if Status != "TEST":
        iterations = 20
        w = [0.0, 0.0, 0.0, 0.0]
        b = 0
        bestWeight = 0
        bestBias = 0
        bestAcc = 0
    else:
        iterations = 1
        bestWeight = w
        bestBias = b
        bestAcc = 0

    # Divide the data into two parts: feature and classNumber
    data = np.hsplit((np.array(sourceData)), np.array([4, 8]))
    feature = data[0].astype(float)
    classNumber = np.array(np.unique(data[1], return_inverse=True))
    classNumber = np.array(classNumber[1])
    classNumber[classNumber != (classValue - 1)] = -1  # change classNumber from 0 to -1
    classNumber[classNumber == (classValue - 1)] = 1

    # Random disruptions required in each episode
    for episode in range(iterations):
        acc = 0
        zipList = list(zip(feature, classNumber))
        rd.shuffle(zipList)
        feature, classNumber = zip(*zipList)

        for i in range(len(feature)):
            a = 0  # Set activation value to 0
            for j in range(len(feature[i])):
                a += (w[j] * feature[i][j]) + b
            if a > 0:
                a = 1
            else:
                a = -1
            if (a * classNumber[i]) <= 0:
                if status != "TEST":
                    for j in range(len(w)):
                        w[j] = w[j] + (classNumber[i] * feature[i][j])
                        # w[j] = w[j] + (classNumber[i] * feature[i][j]) - (2 * coefficient *w[j])
                    b += classNumber[i]  # change the bias value
            else:
                acc += 1
        # Record best results
        if bestAcc < acc:
            bestAcc = acc
            if status != "TEST":
                bestWeight = w.copy()
                bestBias = b
    # Dashboard
    print("Current state is:", status, ". correctness is: ", bestAcc, "/", len(classNumber))
    print("Current Weight is:", w, "and Bias is:", b)
    print("Currency is:", bestAcc / len(feature) * 100, "% Percentage")

    # Return the best weight and bias at the end of the training
    if status != "TEST":
        return bestWeight, bestBias
    else:
        return

'''
w = 0
b = 0
print("Result of 1-vs-rest Class 1 vs 2 & 3 ")
weight, bias = PerceptronMultiClass(trainDataArray, w, b, "Train", 1)
b = PerceptronMultiClass(testDataArray, weight, bias, "Test", 1)
print("---------------------------------------------------------------------------")
print("Result of 1-vs-rest Class 2 vs 1 & 3 ")
weight, bias = PerceptronMultiClass(trainDataArray, w, b, "Train", 2)
PerceptronMultiClass(testDataArray, weight, bias, "Test", 2)
print("---------------------------------------------------------------------------")
print("Result of 1-vs-rest Class 3 vs 1 & 2 ")
weight, bias = PerceptronMultiClass(trainDataArray, w, b, "Train2", 3)
PerceptronMultiClass(testDataArray, weight, bias, "Test", 3)
'''
