import json
import copy

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from scipy import stats

#relative imports of Isotomics
import sys
import os
from pathlib import Path
cwd = Path().resolve()

sys.path.insert(1, os.path.join(cwd, 'Isotomics Scripts'))

import readInput as ri
import initializeMethionine as metInit

def standardizeLR(standardValues):
    '''
    Perform a linear regression across all 4 standards to get predicted standard values at the timepoint of each sample. 
    '''
    n = len(standardValues)
    xs = np.arange(0,n,1)
    slope, intercept, r_value, p_value, slope_serr = stats.linregress(xs,standardValues)

    #calculate useful intermediates
    predictions = slope * xs + intercept 
    yerr = standardValues - predictions 
    s_err = np.sum(yerr**2)
    mean_x = np.mean(xs)       
    # appropriate t value (where n=4, two tailed 68%)            
    t = stats.t.ppf(1-0.16, n-2)              

    #Define a function to calculate the error in this regression at a certain x value
    def errorAtX(thisX):
        pointErr = np.sqrt(s_err / (n-2)) * np.sqrt(1.0/n + (thisX - mean_x)**2 / np.sum((xs-mean_x)**2))
        return pointErr

    #Standardize sample values
    smpxs = np.arange(0.5,3.5,1)
    smpPreds = slope * smpxs + intercept
    smpErr = errorAtX(smpxs)

    return smpPreds, smpErr

def prettyLabel(string):
    '''
    Add superscripts for to plot the labels for various substitutitons, e.g. '13C' goes to '$^{13}C$'

    Inputs:
        string: the isotopic substitution written like 13C or 15N

    Outputs:
        The same with a superscript
    '''
    if string == 'Unsub':
        return "Unsub"
    split = string.split('-')
    combined = []
    for sub in split:
        if sub == 'D':
            combined.append('$^{2}H$')
        else:
            num = sub[:-1]
            pretty = '$^{' + num + '}' + sub[-1] + "$"
            combined.append(pretty)
    return ''.join(combined)

def prettyYLabel(string):
    '''
    Only does something to 74High and 74Low, rewriting them with a subscript. 
    '''
    if string == '74High':
        return '$74_{High}$'
    elif string == '74Low':
        return "$74_{Low}$"
    else:
        return string

#Help guide the ri.readObservedData function through the experimental data. This has a purpose if the fragments were named automatically (i.e., by pulling out a string), and you could get something like '134.1': '133' (if 134.1 was the mass of an observed peak of the 133 fragment). Anyway, that's dealt with earlier in the data processing. 
processFragKeys = {'133':'133',
                   '104':'104',
                   '102':'102',
                   '74High':'74High',
                   '74Low':'74Low',
                   '87':'87',
                   '61':'61',
                   '56':'56'}

smpData = {}
#One iteration gives one plot for each MNKey
for MNKey in ['M1','M2','M3','M4']:
    with open(str(cwd) + '/Read Process CSV/' + MNKey + 'Results.json') as f:
        sampleOutputDict = json.load(f)
        
    #If standard is true, add predicted data for that peak; because we aren't employing the standardization process, no need to do here. 
    replicateData = ri.readObservedData(sampleOutputDict, MNKey = MNKey, theory = {},
                                    standard = [False] * 7,
                                    processFragKeys = processFragKeys)

    replicateDataKeys = list(replicateData.keys())

    #Standardize and average data from all replicates to get a one number summary for the plot
    observedAvg = {}
    linearStandardization = {}
    linear = True
    for repIdx, (repKey, repData) in enumerate(replicateData.items()):
        #pull out the relevant data and sort by whether it is a sample or standard
        if repIdx % 2 == 1:
            for fragKey, fragData in repData[MNKey].items():
                if fragKey not in linearStandardization:
                    linearStandardization[fragKey] = {}

                #calculate linear prediction of standard composition once
                if repIdx == 1:
                    thisFragLinearStandardization = {'Values':[],'PredSmp':[],'PredSmpErr':[]}
                    for stdFileIdx in [0,2,4,6]:
                        thisFragLinearStandardization['Values'].append(np.array(replicateData[replicateDataKeys[stdFileIdx]][MNKey][fragKey]['Observed Abundance']))

                    thisFragLinearStandardization['Values'] = np.array(thisFragLinearStandardization['Values'])
                        
                    #Each substitution is a column of the resulting array
                    for subCol in thisFragLinearStandardization['Values'].T:

                        smpPreds, smpErr = standardizeLR(subCol)

                        #fill in dictionary
                        thisFragLinearStandardization['PredSmp'].append(smpPreds)
                        thisFragLinearStandardization['PredSmpErr'].append(smpErr)

                    #Make them numpy arrays
                    linearStandardization[fragKey]['PredSmp'] = np.array(thisFragLinearStandardization['PredSmp'])
                    linearStandardization[fragKey]['PredSmpErr'] = np.array(thisFragLinearStandardization['PredSmpErr'])

                if fragKey not in observedAvg:
                    observedAvg[fragKey] = {}
                    observedAvg[fragKey]['Delta Comparison'] = []
                    observedAvg[fragKey]['Observed RSE'] = []
                    observedAvg[fragKey]['ER'] = []

                smpNmIdx = repIdx // 2
                smpAbund = np.array(fragData['Observed Abundance'])
                predStdForThisSmp = linearStandardization[fragKey]['PredSmp'].T[smpNmIdx]
                standardizedAbund = smpAbund / predStdForThisSmp
                
                predStdErrForThisSmp = linearStandardization[fragKey]['PredSmpErr'].T[smpNmIdx] /  predStdForThisSmp
                smpError = np.array(fragData['Error']) / smpAbund
                standardizedError = np.sqrt(smpError**2 + predStdErrForThisSmp**2)
                
                observedAvg[fragKey]['Subs'] = fragData['Subs']
                observedAvg[fragKey]['Delta Comparison'].append(standardizedAbund)
                observedAvg[fragKey]['Observed RSE'].append(standardizedError)

    #Take the averages of the observed abundances
    for fragKey, fragData in observedAvg.items():
        fragData['ER'] = 1000* (np.array(fragData['Delta Comparison']).std(axis = 0))
        fragData['Delta Comparison'] = 1000* (np.array(fragData['Delta Comparison']).mean(axis = 0) - 1)
        fragData['Observed RSE'] = 1000*np.array(fragData['Observed RSE']).mean(axis = 0)

        print(MNKey + '_' + fragKey)
        print((fragData['ER'] / fragData['Observed RSE']).max())
        
    #Store the standardized & averaged data
    smpData[MNKey] = copy.deepcopy(observedAvg)

    for fragKey, fragData in observedAvg.items():
        fragData['Delta Comparison'] = list(fragData['Delta Comparison'])
        fragData['Observed RSE'] = list(fragData['Observed RSE'])
        fragData['ER'] = list(fragData['ER'])

    with open('Fig3_Direct_Comparison/Fig3 ' + MNKey + 'direct Comparison.json', 'w', encoding='utf-8') as f:
        json.dump(observedAvg, f, ensure_ascii=False, indent=4)

    #Prepare for plotting by setting formatting & colorbar limits
    cBarLims = (-50,50)
    sns.set(style='whitegrid', rc = {'legend.labelspacing': 2.5})
    RdBu = cm.get_cmap('RdBu_r', 256)
    figSizes = {'M1':(3,3.7),'M2':(5,3.7),'M3':(6,3.7),'M4':(7,3.7)}
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figSizes[MNKey], dpi = 300)
    ax.grid(True, zorder = 0)

    thisDict = smpData[MNKey]

    #Sets the x positions of each substitution
    placementDict = {'M1': {'Unsub':0,'15N':1,'33S':2,'13C':3,'D':4},
                     'M2': {'Unsub':0,'15N':1,'33S':2,'13C':3,'18O':4,'34S':5,'13C-13C':6,'13C-15N':7,'13C-D':8,'13C-33S':9},
                     'M3': {'Unsub':0,'15N':1,'33S':2,'13C':3,'D':4,'18O':5,'34S':6,'13C-13C':7,'13C-34S':8,'34S-15N':9,'18O-33S':10,'34S-D':11,'13C-18O':12,'13C-13C-13C':13},
                     'M4': {'Unsub':0,'15N':1,'13C':2,'18O':3,'34S':4,'36S':5,'13C-13C':6,'13C-15N':7,'13C-18O':8,'13C-D':9,'18O-34S':10,'13C-34S':11,'13C-13C-34S':12,'13C-34S-D':13,'34S-15N':14,'13C-13C-18O':15,'18O-18O':16}}
    thisPlacementDict = placementDict[MNKey]
    #Sets the y position of each substitution.
    fragPlacementDict = {'133':0,'104':1,'102':2,'87':3,'74High':4,'74Low':5,'61':6,'56':7}

    #Generate proper labels with subscripts/superscripts for each fragment & substitution
    yticksUgly = ['133','104','102','87','74High','74Low','61','56']
    yticks = [prettyYLabel(y) for y in yticksUgly]
    xticksUgly = list(thisPlacementDict.keys())
    xticks = [prettyLabel(x) for x in xticksUgly]

    if (MNKey == 'M2') or (MNKey == 'M3'):
        yticks = [''] * 8


    #each observation is plotted individually via this loop
    for fragKey, fragData in thisDict.items():    
        for subIdx, subKey in enumerate(fragData['Subs']):
            ErrorBar = fragData['Observed RSE'][subIdx]
            #Compute the size of the circle based on the error
            thisSize = max(100,100 + min(200 / 2.7 * (3-ErrorBar),200))
            
            #Compute the color of the circle based on the delta value
            scaledDelta = np.abs(cBarLims[0]) + fragData['Delta Comparison'][subIdx]
            scaleHeight = np.abs(cBarLims[0]) + np.abs(cBarLims[1])
            fraction = scaledDelta / scaleHeight
            thisColor = RdBu(fraction)
            
            #Plot the point
            ax.scatter(thisPlacementDict[subKey], fragPlacementDict[fragKey], s = thisSize, color = thisColor, edgecolors = 'k',zorder = 3)

    #Optionally add a legend
    legend = True
    if legend:
        #Show these error setpoints in the legend
        errorSetpoints = [0,1,2,3]
        for errorSet in errorSetpoints: 
            print("Adding Legend")
            thisSize = max(100,100 + min(200 / 2.7 * (3-errorSet),200))
            #Plot points of this size way off the scale, so we can use them in the legend
            ax.scatter(100,100,s = thisSize, color = 'w', edgecolors = 'k', label = str(errorSet))

        #Generate the legend for this dummy plot
        leg = ax.legend(loc=(1.04,0.1), fontsize = 20, labelspacing = 0.8)
        leg.set_title('Error in ‰',prop={'size':'20'})

    #Set x & y ticks, title
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, fontsize = 16, rotation = 90)

    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks, fontsize = 16)

    ax.set_title("M+" + MNKey[1], fontsize = 18)

    #ax.grid(color = 'k')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')

    #Set x & y limits
    ax.set_xlim(-0.5,len(xticks)-0.5)
    ax.set_ylim(-0.5,7.5)

    ax.invert_yaxis()

    #Export image
    fig.savefig("Fig3_Direct_Comparison/Fig3" + MNKey + ".jpeg", bbox_inches = 'tight', dpi = 1000)