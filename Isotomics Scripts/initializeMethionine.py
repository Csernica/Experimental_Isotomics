import copy
import json

import numpy as np
import pandas as pd

import basicDeltaOperations as op
import calcIsotopologues as ci
import fragmentAndSimulate as fas
import solveSystem as ss

'''
This is a set of functions to quickly initalize methionine molecules based on input delta values and to simulate its fragmentation.
'''

def initializeMethionine(deltas, fragSubset = ['full','133','104','102','61','56'], printHeavy = True, intuitiveFrags = False, smallFrags = False):
    '''
    Initializes methionine, returning a dataframe with basic information about the molecule as well as information about fragmentation.

    There are three possible site definitions we use. The default has combined Calpha and Cbeta and combined Halpha and Hbeta. The 'intuitiveFrags' separates out the Calpha and Cbeta sites. The 'smallFrags' also splits up the Ocarboxyl site into two, Ocarboxylretained and Ocarboxyllost, necessary for 74High because it contains one oxygen. Only the 'smallfrags' entry includes fragment geometries for 74High, 74Low, and 87. 

    Inputs:
        deltas: A list of 13 M1 delta values, giving the delta values by site for the 13C, 17O, 15N, 33S, and 2H isotopes. The sites are defined in the IDList variable, below.
        fragSubset: A list giving the subset of fragments to observe. If you are not observing all fragments, you may input only those you do observe. 
        printHeavy: The user manually specifies delta 17O, and delta 18O is set via mass scaling (see basicDeltaOperations). If True, this will print out delta 18O, 34S, & 36S.
        intuitiveFrags: Use the intuitiveFrags site definitions
        smallFrags: Use the smallFrags site definitions.

    Outputs:
        molecularDataFrame: A dataframe containing basic information about the molecule. 
        expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
        fragSubgeometryKeys: A list of strings, e.g. 133_01, 133_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
        fragmentationDictionary: A dictionary like the allFragments variable, but only including the subset of fragments selected by fragSubset.
    '''
    if intuitiveFrags and smallFrags:
        raise Exception("Must choose either 'intuitive' frags or 'small' frags or neither")
    ##### INITIALIZE SITES #####
    IDList = ['Cmethyl','Cgamma','Calphabeta','Ccarboxyl','Ocarboxyl','Ssulfur','Namine','Hmethyl','Hgamma','Halphabeta','Hamine','Hhydroxyl','Hprotonated']
    elIDs = ['C','C','C','C','O','S','N','H','H','H','H','H','H']
    numberAtSite = [1,1,2,1,2,1,1,3,2,3,2,1,1]

    l = [elIDs, numberAtSite, deltas]
    cols = ['IDS','Number','deltas']
    condensedFrags =[]
    fragKeys = []
    
    #87 and both 74 are conjecture. 74 High has only one oxygen, so we generally do not use it. 
    allFragments = {'full':{'01':{'subgeometry':[1,1,1,1,1,1,1,1,1,1,1,1,1],'relCont':1}},
              '133':{'01':{'subgeometry':[1,1,1,1,1,1,'x',1,1,1,'x',1,'x'],'relCont':1}},
              '104':{'01':{'subgeometry':[1,1,1,'x','x',1,1,1,1,1,1,'x','x'],'relCont':1}},
              '102':{'01':{'subgeometry':['x',1,1,1,1,'x',1,'x',1,1,1,1,'x'],'relCont':1}},
              '61':{'01':{'subgeometry':[1,1,'x','x','x',1,'x',1,1,'x','x','x','x'],'relCont':1}},
              '56':{'01':{'subgeometry':['x',1,1,'x','x','x',1,'x',1,1,'x',1,'x'],'relCont':1}}}
    
    if intuitiveFrags:
        IDList = ['Cmethyl','Cgamma','Calpha','Cbeta','Ccarboxyl','Ocarboxyl','Ssulfur','Namine','Hmethyl','Hgamma','Halpha','Hbeta','Hamine','Hhydroxyl','Hprotonated']
        elIDs = ['C','C','C','C','C','O','S','N','H','H','H','H','H','H','H']
        numberAtSite = [1,1,1,1,1,2,1,1,3,2,1,2,2,1,1]
        l = [elIDs, numberAtSite, deltas]

        allFragments = {'full':{'01':{'subgeometry':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],'relCont':1}},
                    '133':{'01':{'subgeometry':[1,1,1,1,1,1,1,'x',1,1,1,1,'x',1,'x'],'relCont':1}},
                    '104':{'01':{'subgeometry':[1,1,1,1,'x','x',1,1,1,1,1,1,1,'x','x'],'relCont':1}},
                    '102':{'01':{'subgeometry':['x',1,1,1,1,1,'x',1,'x',1,1,1,1,1,'x'],'relCont':1}},
                    '61':{'01':{'subgeometry':[1,1,'x','x','x','x',1,'x',1,1,'x','x','x','x','x'],'relCont':1}},
                    '56':{'01':{'subgeometry':['x',1,1,1,'x','x','x',1,'x',1,1,1,'x',1,'x'],'relCont':1}}}
        
    if smallFrags:
        IDList = ['Cmethyl','Cgamma','Calpha','Cbeta','Ccarboxyl','Ocarboxylretained','Ocarboxyllost','Ssulfur','Namine','Hmethyl','Hgamma',
                'Halphabetadouble','Halphabetasingle','Hamine','Hhydroxyl','Hprotonated']
        elIDs = ['C','C','C','C','C','O','O','S','N','H','H','H','H','H','H','H']
        numberAtSite = [1,1,1,1,1,1,1,1,1,3,2,2,1,2,1,1]

        l = [elIDs, numberAtSite, deltas]
        cols = ['IDS','Number','deltas']
        condensedFrags =[]
        fragKeys = []

        #With additional O loss
        allFragments = {'full':{'01':{'subgeometry':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],'relCont':1}},
                    '133':{'01':{'subgeometry':[1,1,1,1,1,1,1,1,'x',1,1,1,1,'x',1,'x'],'relCont':1}},
                    '104':{'01':{'subgeometry':[1,1,1,1,'x','x','x',1,1,1,1,1,1,1,'x','x'],'relCont':1}},
                    '102':{'01':{'subgeometry':['x',1,1,1,1,1,1,'x',1,'x',1,1,1,1,1,'x'],'relCont':1}},
                    '87':{'01':{'subgeometry':[1,1,1,1,'x','x','x',1,'x',1,1,1,'x',1,'x','x'],'relCont':1}},
                    '74High':{'01':{'subgeometry':['x','x',1,1,1,1,'x','x',1,'x',1,1,1,1,'x',1],'relCont':1}},
                    '74Low':{'01':{'subgeometry':['x','x',1,'x',1,1,1,'x',1,'x','x','x',1,1,1,'x'],'relCont':1}},
                    '61':{'01':{'subgeometry':[1,1,'x','x','x','x','x',1,'x',1,1,'x','x','x','x','x'],'relCont':1}},
                    '56':{'01':{'subgeometry':['x',1,1,1,'x','x','x','x',1,'x',1,1,1,'x',1,'x'],'relCont':1}}}

    fragmentationDictionary = {key: value for key, value in allFragments.items() if key in fragSubset}
    
    for fragKey, subFragDict in fragmentationDictionary.items():
        for subFragNum, subFragInfo in subFragDict.items():
            l.append(subFragInfo['subgeometry'])
            cols.append(fragKey + '_' + subFragNum)
            condensedFrags.append(subFragInfo['subgeometry'])
            fragKeys.append(fragKey + '_' + subFragNum)
    
    try:
        molecularDataFrame = pd.DataFrame(l, columns = IDList)
        molecularDataFrame = molecularDataFrame.transpose()
        molecularDataFrame.columns = cols
    except:
        raise Exception("Could not construct molecular dataframe:\nLength IDs: " + str(len(IDList)) + "\nLength numAtSite: " + str(len(numberAtSite)) + "\nLength deltas: " + str(len(deltas)))

    expandedFrags = [fas.expandFrag(x, numberAtSite) for x in condensedFrags]

    if printHeavy:
        SConc = op.deltaToConcentration('S',deltas[5])
        del34 = op.ratioToDelta('34S',SConc[2]/SConc[0])
        del36 = op.ratioToDelta('36S',SConc[3]/SConc[0])

        OConc = op.deltaToConcentration('O',deltas[4])
        del18 = op.ratioToDelta('18O',OConc[2]/OConc[0])
        print("Delta 34S")
        print(del34)
        print("Delta 36S")
        print(del36)

        print("Delta 18O")
        print(del18)

    
    methionineData = {'molecularDataFrame':molecularDataFrame,
                      'expandedFrags':expandedFrags,
                      'fragKeys':fragKeys,
                      'fragmentationDictionary':fragmentationDictionary}
    
    return methionineData

def simulateMeasurement(methionineData, overrideIsoDict = {}, abundanceThreshold = 0, UValueList = [],
                        massThreshold = 4, clumpD = {}, outputPath = None, disableProgress = False, calcFF = False, fractionationFactors = {}, omitMeasurements = {}, ffstd = 0.05, unresolvedDict = {}, outputFull = False):
    '''
    Simulates M+N measurements of a methionine molecule with input deltas specified by the input dataframe molecularDataFrame. 

    Inputs:
        methionineData: A dictionary with the following entries:
            molecularDataFrame: A dataframe containing basic information about the molecule. 
            expandedFrags: An ATOM depiction of each fragment, where an ATOM depiction has one entry for each atom (rather than for each site). See fragmentAndSimulate for details.
            fragSubgeometryKeys: A list of strings, e.g. 133_01, 133_02, corresponding to each subgeometry of each fragment. A fragment will have multiple subgeometries if there are multiple fragmentation pathways to form it.
            fragmentationDictionary: A dictionary like the allFragments variable from initalizeMethionine, but only including the subset of fragments selected by fragSubset.
        abundanceThreshold: A float; Does not include measurements below this M+N relative abundance, i.e. assuming they will not be  measured due to low abundance. 
        UValueList: A list giving specific substitutions to calculate molecular average U values for ('13C', '15N', etc.)
        massThreshold: An integer; will calculate M+N relative abundances for N <= massThreshold
        clumpD: Specifies information about clumps to add; otherwise the isotome follows the stochastic assumption. Currently works only for mass 1 substitutions (e.g. 1717, 1317, etc.) See ci.introduceClump for details.
        outputPath: A string, e.g. 'output', or None. If it is a string, outputs the simulated spectrum as a json. 
        disableProgress: Disables tqdm progress bars when True.
        calcFF: When True, computes a new set of fractionation factors for this measurement.
        fractionationFactors: A dictionary, specifying a fractionation factor to apply to each ion beam. This is used to apply fractionation factors calculated previously to this predicted measurement (e.g. for a sample/standard comparison with the same experimental fractionation)
        omitMeasurements: omitMeasurements: A dictionary, {}, specifying measurements which I will not observed. For example, omitMeasurements = {'M1':{'61':'D'}} would mean I do not observe the D ion beam of the 61 fragment of the M+1 experiment, regardless of its abundance. 
        ffstd: A float; if new fractionation factors are calculated, they are pulled from a normal distribution centered around 1, with this standard deviation.
        unresolvedDict: A dictionary, specifying which unresolved ion beams add to each other.
        outputFull: A boolean. Typically False, in which case beams that are not observed are culled from the dictionary. If True, includes this information; this should only be used for debugging, and will likely break the solver routine. 
        
    Outputs:
        predictedMeasurement: A dictionary giving information from the M+N measurements. 
        MN: A dictionary where keys are mass selections ("M1", "M2") and values are dictionaries giving information about the isotopologues of each mass selection.
        fractionationFactors: The calculated fractionation factors for this measurement (empty unless calcFF == True)
    '''
    molecularDataFrame = methionineData['molecularDataFrame']
    expandedFrags = methionineData['expandedFrags']
    fragKeys = methionineData['fragKeys']
    fragmentationDictionary = methionineData['fragmentationDictionary']

    M1Only = False
    if massThreshold == 1:
        M1Only = True
        
    byAtom = ci.inputToAtomDict(molecularDataFrame, overrideIsoDict = overrideIsoDict, disable = disableProgress, M1Only = M1Only)
    
    #Introduce any clumps of interest with clumps
    if clumpD == {}:
        bySub = ci.calcSubDictionary(byAtom, molecularDataFrame, atomInput = True)
    else:
        print("Adding clumps")
        stochD = copy.deepcopy(byAtom)
        
        for clumpNumber, clumpInfo in clumpD.items():
            byAtom = ci.introduceClump(byAtom, clumpInfo['Sites'], clumpInfo['Amount'], molecularDataFrame)
            
        for clumpNumber, clumpInfo in clumpD.items():
            ci.checkClumpDelta(clumpInfo['Sites'], molecularDataFrame, byAtom, stochD)
            
        bySub = ci.calcSubDictionary(byAtom, molecularDataFrame, atomInput = True)
    
    #Initialize Measurement output
    if disableProgress == False:
        print("Simulating Measurement")
    allMeasurementInfo = {}
    allMeasurementInfo = fas.UValueMeasurement(bySub, allMeasurementInfo, massThreshold = massThreshold,
                                              subList = UValueList)

    MN = ci.massSelections(byAtom, massThreshold = massThreshold)
    MN = fas.trackMNFragments(MN, expandedFrags, fragKeys, molecularDataFrame, unresolvedDict = unresolvedDict)
        
    predictedMeasurement, FF = fas.predictMNFragmentExpt(allMeasurementInfo, MN, expandedFrags, fragKeys, molecularDataFrame, 
                                                 fragmentationDictionary,
                                                 abundanceThreshold = abundanceThreshold, calcFF = calcFF, ffstd = ffstd, fractionationFactors = fractionationFactors, omitMeasurements = omitMeasurements, unresolvedDict = unresolvedDict, outputFull = outputFull)
    
    if outputPath != None:
        output = json.dumps(predictedMeasurement)

        f = open(outputPath + ".json","w")
        f.write(output)
        f.close()
        
    return predictedMeasurement, MN, FF