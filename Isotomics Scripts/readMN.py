import json

import dataScreen
import DataAnalyzerWithPeakInteg

#Folder with FT Statistic-ified files. All the files need to be processed using the same metrics.
toRun = ['Met_M4']

#"OMIT" included for peaks which were extracted but later rejected; we pass over these. Give the fragments, then isotopes of extracted peaks, in order they appear.  
masterMNDict = {'Met_M1':{'133':['D','13C','33S','Unsub'],
                    '104':['OMIT','13C','33S','15N','Unsub'],
                    '102':['D','13C','15N','Unsub'],
                    '87':['D','13C','33S','Unsub'],
                    '74High':['OMIT','13C','15N','Unsub'],
                    '74Low':['OMIT','OMIT','13C','15N','Unsub'],
                    '61':['D','13C','33S','Unsub'],
                    '56':['D','13C','15N','Unsub']},

            'Met_M2':{'133':['OMIT','18O','34S','OMIT'],
                    '104':['13C-13C','OMIT','34S','13C','Unsub'],
                    '102':['13C-13C','18O','OMIT','13C','Unsub'],
                    '87':['13C-13C','OMIT','34S','13C','Unsub'],
                    '74High':['OMIT','18O','OMIT','Unsub'],
                    '74Low':['18O','13C','Unsub'],
                    '61':['13C-13C','13C-33S','34S','OMIT','13C','33S','Unsub'],
                    #'61':['OMIT','OMIT','34S','OMIT','13C','33S','Unsub'],
                    '56':['OMIT','13C-13C','13C-15N','OMIT','13C','15N','Unsub']},
                    #'56':['13C-D','13C-13C','13C-15N','OMIT','OMIT','15N','OMIT']},

            'Met_M3':{#'Full':['OMIT','OMIT'],
                    #'133':['13C-13C-13C','13C-18O','18O-33S','OMIT','13C-34S','18O','34S'],
                    '133':['OMIT','OMIT','13C-13C-13C','OMIT','18O-33S','OMIT','13C-34S','OMIT','34S'],
                    #'104':['13C-13C-13C','34S-D','13C-34S','34S-15N','34S','13C','33S','15N','Unsub'],
                    '104':['OMIT','OMIT','13C-34S','34S-15N','34S','13C','33S','15N','Unsub'],
                    #'102':['13C-13C-13C','OMIT','13C-13C','18O','D','13C','15N','Unsub'],
                    '102':['OMIT','OMIT','OMIT','18O','D','13C','15N','Unsub'],
                    '87':['34S-D','13C-34S','34S','13C','Unsub'],
                    '74High':['13C-18O','13C','15N','Unsub'],
                    '74Low':['13C-18O','18O','13C','15N','Unsub'],
                    '61':['34S-D','13C-34S','34S','13C','33S','Unsub'],
                    #'56':['13C-13C-13C','13C-13C','D','13C','15N','Unsub']},
                    '56':['OMIT','13C-13C','OMIT','13C','15N','Unsub']},


            'Met_M4':{'Full':['13C-13C-34S','18O-34S','36S'],
                    #'Full':['OMIT','18O-34S','36S'],
                    #'133':['13C-13C-18O','18O-18O','13C-18O-33S','13C-34S-D','13C-13C-34S','18O-34S','36S','13C-18O',
                    #    '13C-34S'],
                    '133':['13C-13C-18O','18O-18O','OMIT','13C-34S-D','13C-13C-34S','18O-34S','36S','OMIT',
                        '13C-34S'],
                    
                    #'104':['13C-34S-D','13C-13C-34S','36S','13C-34S','34S-15N','13C-13C','13C-33S',
                    #    '34S','13C','Unsub'],
                    '104':['13C-34S-D','13C-13C-34S','36S','13C-34S','34S-15N','13C-13C','OMIT',
                        '34S','13C','Unsub'],
                    '102':['13C-18O','13C-13C','18O','13C-15N','13C','Unsub'],
                    '87':['13C-13C-34S','36S','13C-34S','34S'],
                    '74High':['13C-13C','18O','13C','Unsub'],
                    #'74Low':['18O','13C','Unsub'],
                    '74Low':['18O','13C','Unsub'],
                    #'61':['13C-13C-34S','36S','13C-34S','34S','13C','Unsub'],
                    '61':['13C-13C-34S','36S','13C-34S','34S','OMIT','OMIT'],
                    '56':['13C-D','13C-13C','13C-15N','13C','15N','Unsub']}}

#Used for the peak drift tests
mostAbundantDict = {'Met_M1':['13C','13C','13C','13C','13C','Unsub','Unsub','13C'],
                    'Met_M2':['34S','34S','18O','Unsub','Unsub','Unsub','Unsub','Unsub'],
                    #'Met_M3':['13C-34S','13C-34S','13C-34S','13C','34S','13C-18O','13C','34S','Unsub'],
                    'Met_M3':['13C-34S','13C-34S','13C','34S','13C-18O','13C','34S','Unsub'],
                    'Met_M4': ['36S','18O-34S','36S','18O','36S','13C','Unsub','34S','13C']}
 
for experiment in toRun:
    folderPath = experiment

    #Source code says "gc elution", but this routine just culls based on time; so it is ok for these measurements. 
    gc_elution_on = True
    #Time frames for the GC elution; different time for each fragment, hence *9
    gcElutionTimes = [(15.00,75.00)] * 9

    metMNDict = masterMNDict[folderPath]
    fragmentMostAbundant = mostAbundantDict[folderPath]

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

    rtnAllFilesDF, mergedList, allOutputDict = DataAnalyzerWithPeakInteg.calc_Folder_Output(folderPath, cullOn=cullOn, cullAmount=cull_amount,
                                                gcElutionOn=gc_elution_on, gcElutionTimes = gcElutionTimes, 
                                                fragmentIsotopeList = fragmentIsotopeList, 
                                                fragmentMostAbundant = fragmentMostAbundant, debug = True, 
                                                MNRelativeAbundance = MNRelativeAbundance, fileExt = fileExt, 
                                                massStrList = list(fragmentDict.keys()),
                                                Microscans = 1)

    #Screen to confirm peak location, data quality
    #dataScreen.peakDriftScreen(folderPath, fragmentDict, fragmentMostAbundant, mergedList, driftThreshold = 2, fileExt = fileExt)
    #dataScreen.RSESNScreen(allOutputDict)
    #dataScreen.zeroCountsScreen(folderPath, fragmentDict, mergedList, fileExt = fileExt)

    #for merged in mergedList:
    #    cDf = merged[0]
    #    ticMean = cDf['tic'].mean()
    #    print(ticMean)
    #    print(len(cDf))

    sampleOutputDict = {}
    fragmentList = []
    for i, info in rtnAllFilesDF.iterrows():
        fragment = info['Fragment']
        file = info['FileNumber']
        ratio = info['IsotopeRatio']
        avg = info['Average']
        std = info['StdDev']
        stderr = info['StdError']
        rse = info['RelStdError']
        ticvar = info['TICVar']
        ticitvar = info['TIC*ITVar']
        ticitmean = info['TIC*ITMean']
        SN = info['ShotNoise']
        
        if file not in sampleOutputDict:
            sampleOutputDict[file] = {}
            
        if fragment not in sampleOutputDict[file]:
            sampleOutputDict[file][fragment] = {}
            
        sampleOutputDict[file][fragment][ratio] = {'Average':avg,'StdDev':std,'StdError':stderr,'RelStdError':rse,
                                'TICVar':ticvar, 'TIC*ITVar':ticitvar,'TIC*ITMean':ticitmean,
                                'ShotNoise':SN}

    with open(folderPath + 'Results.json', 'w', encoding='utf-8') as f:
        json.dump(sampleOutputDict, f, ensure_ascii=False, indent=4)