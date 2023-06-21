import json

#relative imports of Isotomics
import sys
import os
from pathlib import Path
cwd = Path().resolve()

import dataScreen
import DataAnalyzerWithPeakInteg

#Folder with FT Statistic-ified files. All the files need to be processed using the same metrics.
toRun = ['M1','M2','M3','M4']

with open(str(cwd) + '/Reference Files/MasterMNDict.json') as f:
    masterMNDict = json.load(f)

#Used for the peak drift tests
mostAbundantDict = {'M1':['13C','13C','13C','13C','13C','Unsub','Unsub','13C'],
                    'M2':['34S','34S','18O','Unsub','Unsub','Unsub','Unsub','Unsub'],
                    'M3':['13C-34S','13C-34S','13C','34S','13C-18O','13C','34S','Unsub'],
                    'M4': ['36S','18O-34S','36S','18O','36S','13C','Unsub','34S','13C']}
 
for experiment in toRun:
    folderPath = str(cwd) + '/' + experiment

    #Source code says "gc elution", but this is just culling by time. 
    cullByTime = True
    #Time frames for the GC elution; different time for each fragment, hence *9
    cullTimes = [(15.00,75.00)] * 9

    metMNDict = masterMNDict[experiment]
    fragmentMostAbundant = mostAbundantDict[experiment]

    #Repackage MetMNDict as a list.
    massStr = []
    fragmentIsotopeList = []
    for i, v in metMNDict.items():
        massStr.append(i)
        fragmentIsotopeList.append(v)
        fragmentDict = metMNDict

    #Any specific properties you want to cull on
    cullOn = "TIC*IT"
    #Multiple of SD you want to cull beyond for the cullOn property
    cull_amount = 3
    #Whether you want to cull zero scans
    cullZeroScans = False
    #Whether you want to calculate weighted averages based on NL height (specifically designed for GC elution but widely applicable!)
    weightByNLHeight = False
    #Whether you want to output each file as you process it, and where you want it to go:
    fileOutputPath = None
    #Set file extension
    fileExt = '.csv'

    #Use this to get M+N relative abundances
    MNRelativeAbundance = True

    rtnAllFilesDF, mergedDict, allOutputDict = DataAnalyzerWithPeakInteg.calc_Folder_Output(folderPath, cullOn=cullOn, cullAmount=cull_amount,
                                                cullByTime=cullByTime, cullTimes = cullTimes, 
                                                fragmentIsotopeList = fragmentIsotopeList, 
                                                fragmentMostAbundant = fragmentMostAbundant, debug = True, 
                                                MNRelativeAbundance = MNRelativeAbundance, fileExt = fileExt, 
                                                massStrList = list(fragmentDict.keys()),
                                                Microscans = 1)

    #Screen to confirm peak location, data quality
    dataScreen.peakDriftScreen(folderPath, fragmentDict, fragmentMostAbundant, mergedDict, driftThreshold = 2, fileExt = fileExt)
    dataScreen.RSESNScreen(allOutputDict)
    dataScreen.zeroCountsScreen(folderPath, fragmentDict, mergedDict, fileExt = fileExt)

    #for merged in mergedList:
    #    cDf = merged[0]
    #    ticMean = cDf['tic'].mean()
    #    print(ticMean)
    #    print(len(cDf))

    sampleOutputDict = DataAnalyzerWithPeakInteg.folderOutputToDict(rtnAllFilesDF)

    with open(folderPath + 'Results.json', 'w', encoding='utf-8') as f:
        json.dump(sampleOutputDict, f, ensure_ascii=False, indent=4)