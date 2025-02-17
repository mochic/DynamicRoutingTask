# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:35:34 2023

@author: svc_ccg
"""

import copy
import glob
import os
import itertools
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from statsmodels.stats.multitest import multipletests
from DynamicRoutingAnalysisUtils import DynRoutData,getPerformanceStats


baseDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting"

summarySheets = pd.read_excel(os.path.join(baseDir,'Sam','BehaviorSummary.xlsx'),sheet_name=None)
summaryDf = pd.concat((summarySheets['not NSB'],summarySheets['NSB']))

drSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTraining.xlsx'),sheet_name=None)
nsbSheets = pd.read_excel(os.path.join(baseDir,'DynamicRoutingTask','DynamicRoutingTrainingNSB.xlsx'),sheet_name=None)

hitThresh = 100
dprimeThresh = 1.5


def getSessionData(mouseId,dataDir,sessionStartTimes):
    d = []
    for t in sessionStartTimes:
        fileName = 'DynamicRouting1_' + str(mouseId) + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5'
        if str(mouseId) in drSheets:
            filePath = os.path.join(dataDir,str(mouseId),fileName)
        else:
            filePath = glob.glob(os.path.join(dataDir,'**',fileName))[0]
        obj = DynRoutData()
        obj.loadBehavData(filePath)
        d.append(obj)
    return d


def plotLearning(mice,stage):
    hitCount = {lbl:[] for lbl in mice}
    dprime = copy.deepcopy(hitCount)
    sessionsToPass = copy.deepcopy(hitCount)
    for lbl,mouseIds in mice.items():
        for mid in mouseIds:
            df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
            sessions = np.where([str(stage) in task for task in df['task version']])[0]
            hitCount[lbl].append([])
            dprime[lbl].append([])
            passed = False
            for sessionInd in sessions:
                hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
                hitCount[lbl][-1].append(hits[0][0])
                dprime[lbl][-1].append(dprimeSame[0][0])
                if sessionInd > sessions[0] and not passed:
                    hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
                    if all(h[0] >= hitThresh for h in hits) and all(d[0] >= dprimeThresh for d in dprimeSame):
                        sessionsToPass[lbl].append(np.where(sessions==sessionInd)[0][0] + 1)
                        passed = True
            if not passed:
                if mid==614910:
                    sessionsToPass[lbl].append(np.where(sessions==sessionInd)[0][0]+ 1)
                else:
                    sessionsToPass[lbl].append(np.nan)
                    
    xlim = (0.5,max(np.nanmax(ps) for ps in sessionsToPass.values())+0.5)
    xticks = np.arange(0,100,5) if xlim[1]>10 else np.arange(10)
                
    for data,thresh,ylbl in zip((hitCount,dprime),(hitThresh,dprimeThresh),('Hit count','d\'')):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(xlim,[thresh]*2,'k--')
        for lbl,clr in zip(mice.keys(),'gm'):
            m = np.full((len(data[lbl]),int(np.nanmax(sessionsToPass[lbl]))),np.nan)
            for i,d in enumerate(data[lbl]):
                d = d[:sessionsToPass[lbl][i]]
                m[i,:len(d)] = d
                ax.plot(np.arange(len(d))+1,d,color=clr,alpha=0.25,zorder=2)
                ax.plot(sessionsToPass[lbl][i],d[sessionsToPass[lbl][i]-1],'o',ms=12,color=clr,alpha=0.5,zorder=0)
            lbl += ' (n='+str(np.sum(~np.isnan(sessionsToPass[lbl])))+')'
            # ax.plot(np.arange(m.shape[1])+1,np.nanmean(m,axis=0),clr,lw=2,zorder=1)   
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xticks(xticks)
        ax.set_xlim(xlim)
        ax.set_xlabel('Session',fontsize=14)
        ax.set_ylabel(ylbl,fontsize=14)
        plt.tight_layout()
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for lbl,clr in zip(mice.keys(),'gm'):
        dsort = np.sort(np.array(sessionsToPass[lbl])[~np.isnan(sessionsToPass[lbl])])
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        lbl += ' (n='+str(dsort.size)+')'
        ax.plot(dsort,cumProb,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Sessions to pass',fontsize=14)
    ax.set_ylabel('Cumalative fraction',fontsize=14)
    plt.legend()
    plt.tight_layout()   


## stage 1, stationary gratings, timeouts with noise vs no timeouts, no reward click or wheel fixed
ind = summaryDf['stage 1 pass'] & summaryDf['stat grating'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'stationary, timeouts with noise': np.array(summaryDf[ind & summaryDf['timeout noise']]['mouse id']),
        'stationary, no timeouts': np.array(summaryDf[ind & ~summaryDf['timeouts']]['mouse id'])}
plotLearning(mice,stage=1)


## stage 1, stationary vs moving gratings, both with noise timeouts
ind = summaryDf['stage 1 pass'] & summaryDf['timeout noise'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'moving, timeouts with noise':  np.array(summaryDf[ind & summaryDf['moving grating']]['mouse id']),
        'stationary, timeouts with noise': np.array(summaryDf[ind & summaryDf['stat grating']]['mouse id'])}
plotLearning(mice,stage=1)


## stage 1 moving gratings, timeouts with vs without noise
ind = summaryDf['stage 1 pass'] & summaryDf['moving grating'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'moving, timeouts with noise': np.array(summaryDf[ind & summaryDf['timeout noise']]['mouse id']),
        'moving, timeouts without noise':  np.array(summaryDf[ind & summaryDf['timeouts'] & ~summaryDf['timeout noise']]['mouse id'])}
plotLearning(mice,stage=1)


## stage 1, stationary with noise timeouts vs moving with noiseless timeouts
ind = summaryDf['stage 1 pass'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'moving, timeouts without noise':  np.array(summaryDf[ind & summaryDf['moving grating'] & summaryDf['timeouts'] & ~summaryDf['timeout noise'] ]['mouse id']),
        'stationary, timeouts with noise': np.array(summaryDf[ind & summaryDf['stat grating'] & summaryDf['timeout noise'] ]['mouse id'])}
plotLearning(mice,stage=1)


## stage 1 moving gratings, timeout without noise, with vs without reward clicks
ind = summaryDf['stage 1 pass'] & summaryDf['moving grating'] & summaryDf['timeouts'] & ~summaryDf['timeout noise'] & ~summaryDf['wheel fixed']
mice = {'moving, reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'moving, no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotLearning(mice,stage=1)
 

## stage 2, tones, timeouts with noise vs no timeouts
ind = summaryDf['stage 2 pass'] & summaryDf['tone'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'tones, timeouts with noise': np.array(summaryDf[ind & summaryDf['timeout noise']]['mouse id']),
        'tones, no timeouts':  np.array(summaryDf[ind  & ~summaryDf['timeouts']]['mouse id'])}
plotLearning(mice,stage=2)


## stage 2, tones with noise timeouts vs AMN with noiseless timeouts
ind = summaryDf['stage 2 pass'] & ~summaryDf['reward click'] & ~summaryDf['wheel fixed']
mice = {'tones, timeouts with noise': np.array(summaryDf[ind & summaryDf['tone'] & summaryDf['timeout noise']]['mouse id']),
        'AM noise, timeouts without noise':  np.array(summaryDf[ind & summaryDf['AM noise'] & summaryDf['timeouts'] & ~summaryDf['timeout noise']]['mouse id'])}
plotLearning(mice,stage=2)


## stage 2 AMN, with vs without reward clicks
ind = summaryDf['stage 2 pass'] & summaryDf['AM noise'] & summaryDf['timeouts'] & ~summaryDf['timeout noise'] & ~summaryDf['wheel fixed']
mice = {'AM noise, reward click': np.array(summaryDf[ind & summaryDf['reward click']]['mouse id']),
        'AM noise, no reward click':  np.array(summaryDf[ind & ~summaryDf['reward click']]['mouse id'])}
plotLearning(mice,stage=2)


## stationary vs moving gratings and tone vs AMN after stage 2
ind = summaryDf['stage 5 pass']
miceVis = {'moving':  np.array(summaryDf[ind & summaryDf['moving grating']]['mouse id']),
           'stationary': np.array(summaryDf[ind & summaryDf['stat grating']]['mouse id'])}

miceAud = {'tone': np.array(summaryDf[ind & summaryDf['tone']]['mouse id']),
           'AM noise':  np.array(summaryDf[ind & summaryDf['AM noise']]['mouse id'])}

for mice in (miceVis,miceAud):
    dprime = {lbl:[] for lbl in mice}
    for lbl,mouseIds in mice.items():
        for mid in mouseIds:
            df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
            sessions = np.array(['ori' in task and ('stage 3' in task or 'stage 4' in task or 'stage variable' in task or 'stage 5' in task) for task in df['task version']])
            firstExperimentSession = np.where(['multimodal' in task
                                               or 'contrast'in task
                                               or 'opto' in task
                                               or 'nogo' in task
                                               or 'noAR' in task
                                               # or 'NP' in rig 
                                               for task,rig in zip(df['task version'],df['rig name'])])[0]
            if len(firstExperimentSession)>0:
                sessions[firstExperimentSession[0]:] = False
            sessions = np.where(sessions)[0]
            dprime[lbl].append([])
            for sessionInd in sessions:
                hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
                dprimeSame = dprimeSame[0]
                if len(dprimeSame) > 1:
                    task = df.loc[sessionInd,'task version']
                    visFirst = 'ori tone' in task or 'ori AMN' in task
                    if ('moving' in mice and visFirst) or ('tone' in mice and not visFirst):
                        dprime[lbl][-1].append(np.nanmean(dprimeSame[0:6:2]))
                    else:
                        dprime[lbl][-1].append(np.nanmean(dprimeSame[1:6:2]))
                else:
                    dprime[lbl][-1].append(dprimeSame[0])
    
    maxSessions = max(len(d) for lbl in dprime for d in dprime[lbl])
    minMice = 8
                
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xmax = 1e6
    for lbl,clr in zip(mice.keys(),'gm'):
        y = np.full((len(dprime[lbl]),maxSessions+1),np.nan)
        for i,d in enumerate(dprime[lbl]):
            y[i,:len(d)] = d
        lbl += ' (n='+str(len(dprime[lbl]))+')'
        x = np.arange(y.shape[1])+1
        n = np.sum(~np.isnan(y),axis=0)
        xmax = min(xmax,x[n>=minMice][-1])
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=lbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    # ax.set_xlim([0,xmax])
    ax.set_xlim([0,36])
    ax.set_ylim([0,4])
    ax.set_xlabel('Session after stage 2',fontsize=14)
    ax.set_ylabel('d\'',fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
    

## moving to stationary grating switch
preSessions = 1
postSessions = 1
dprime = []
for mid in summaryDf[summaryDf['moving to stat']]['mouse id']:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    prevTask = None
    dprime.append([])
    for i,task in enumerate(df['task version']):
        if prevTask is not None and 'stage 5' in prevTask and 'stage 5' in task and 'moving' in prevTask and 'moving' not in task:
            for j in range(i-preSessions,i+postSessions+1):
                hits,dprimeSame,dprimeOther = getPerformanceStats(df,[j])
                if 'ori tone' in df.loc[j,'task version'] or 'ori AMN' in df.loc[j,'task version']:
                    dprime[-1].append(np.mean(dprimeSame[0][0:2:6]))
                else:
                    dprime[-1].append(np.mean(dprimeSame[0][1:2:6]))
            break
        prevTask = task

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(-preSessions,postSessions+1)
for dp in dprime:
    ax.plot(xticks,dp,'k',alpha=0.25)
mean = np.mean(dprime,axis=0)
sem = np.std(dprime,axis=0)/(len(dprime)**0.5)
ax.plot(xticks,mean,'ko-',lw=2,ms=12)
for x,m,s in zip(xticks,mean,sem):
    ax.plot([x,x],[m-s,m+s],'k',lw=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xticks(xticks)
ax.set_xticklabels(['-1\nmoving','0\nstationary','1\nmoving'])
ax.set_xlim([-preSessions-0.5,postSessions+0.5])
ax.set_ylim([0,4.1])
ax.set_xlabel('Session',fontsize=14)
ax.set_ylabel('d\'',fontsize=14)
plt.tight_layout()


## training after stage 2
hasIndirectRegimen = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var']
mice = {'direct': np.array(summaryDf[~hasIndirectRegimen & summaryDf['stage 2 pass']]['mouse id']),
        'indirect': np.array(summaryDf[hasIndirectRegimen & summaryDf['stage 2 pass']]['mouse id'])}

sessionsToPass = {lbl:[] for lbl in mice.keys()}
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        firstExperimentSession = np.where(['multimodal' in task
                                           or 'contrast'in task
                                           or 'opto' in task
                                           or 'nogo' in task
                                           or 'noAR' in task
                                           # or 'NP' in rig 
                                           for task,rig in zip(df['task version'],df['rig name'])])[0]
        if len(firstExperimentSession)>0:
            sessions[firstExperimentSession[0]:] = False
        sessions = np.where(sessions)[0]
        passed = False
        for sessionInd in sessions[1:]:
            hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
            if not passed:
                if np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                    firstSession = np.where(np.array([('stage 3' in task and 'distract' in task) or 
                                                       'stage 4' in task or 
                                                       'stage variable' in task or
                                                       'stage 5' in task for task in df['task version']]))[0][0]
                    sessionsToPass[lbl].append(sessionInd - firstSession + 1)
                    passed = True
                    break
        if not passed:
            sessionsToPass[lbl].append(np.nan)
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(mice.keys(),'gm'):
    dsort = np.sort(np.array(sessionsToPass[lbl])[~np.isnan(sessionsToPass[lbl])])
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    lbl += ' to 6-block training'+' (n='+str(dsort.size)+')'
    ax.plot(dsort,cumProb,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlabel('Sessions to pass (after stage 2)',fontsize=14)
ax.set_ylabel('Cumalative fraction',fontsize=14)
plt.legend(loc='lower right')
plt.tight_layout()   


## training in stage 5
hasIndirectRegimen = summaryDf['stage 3 alt'] | summaryDf['stage 3 distract'] | summaryDf['stage 4'] | summaryDf['stage var']
mice = np.array(summaryDf[~hasIndirectRegimen & summaryDf['stage 5 pass']]['mouse id'])

dprime = {comp: {mod: [] for mod in ('all','vis','sound')} for comp in ('same','other')}
sessionsToPass = []
sessionData = []
for mid in mice:
    df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
    sessions = np.array(['stage 5' in task for task in df['task version']])
    firstExperimentSession = np.where(['multimodal' in task
                                       or 'contrast'in task
                                       or 'opto' in task
                                       or 'nogo' in task
                                       or 'noAR' in task
                                       # or 'NP' in rig 
                                       for task,rig in zip(df['task version'],df['rig name'])])[0]
    if len(firstExperimentSession)>0:
        sessions[firstExperimentSession[0]:] = False
    sessions = np.where(sessions)[0]
    passed = False
    for sessionInd in sessions:
        hits,dprimeSame,dprimeOther = getPerformanceStats(df,[sessionInd])
        for dp,comp in zip((dprimeSame,dprimeOther),('same','other')):
            if sessionInd == sessions[0]:
                for mod in ('all','vis','sound'):
                    dprime[comp][mod].append([])
            dp = dp[0]
            dprime[comp]['all'][-1].append(dp)
            task = df.loc[sessionInd,'task version']
            visFirst = 'ori tone' in task or 'ori AMN' in task
            if visFirst:
                dprime[comp]['vis'][-1].append(dp[0:6:2])
                dprime[comp]['sound'][-1].append(dp[1:6:2])
            else:
                dprime[comp]['sound'][-1].append(dp[0:6:2])
                dprime[comp]['vis'][-1].append(dp[1:6:2])
                
        if not passed and sessionInd > sessions[0]:
            hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
            if np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                sessionsToPass.append(sessionInd - sessions[0] + 1)
                passed = True
    sessionStartTimes = list(df['start time'][sessions])
    dataDir = summaryDf.loc[summaryDf['mouse id']==mid,'data path'].values[0]
    sessionData.append(getSessionData(mid,dataDir,sessionStartTimes))
                
mouseClrs = plt.cm.tab20(np.linspace(0,1,len(sessionsToPass)))

for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    dp = np.full((len(dprime[comp]['all']),max(len(d) for d in dprime[comp]['all'])),np.nan)
    for i,(d,clr) in enumerate(zip(dprime[comp]['all'],mouseClrs)):
        y = np.nanmean(d,axis=1)
        ax.plot(np.arange(len(y))+1,y,color=clr,alpha=0.25,zorder=2)
        ax.plot(sessionsToPass[i],y[sessionsToPass[i]-1],'o',ms=12,color=clr,alpha=0.5,zorder=0)
        dp[i,:len(y)] = y
    m = np.nanmean(dp,axis=0)
    ax.plot(np.arange(len(m))+1,m,color='k',lw=2,zorder=1)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,max(sessionsToPass)+2])
    ax.set_ylim([-0.5,4])
    ax.set_xlabel('Session',fontsize=14)
    ax.set_ylabel('d\' '+comp+' modality',fontsize=14)
    plt.tight_layout()

for comp in ('same','other'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for mod,clr in zip(('vis','sound'),'gm'):
        dp = np.full((len(dprime[comp][mod]),max(len(d) for d in dprime[comp][mod])),np.nan)
        for i,d in enumerate(dprime[comp][mod]):
            y = np.nanmean(d,axis=1)
            ax.plot(np.arange(len(y))+1,y,color=clr,alpha=0.25,zorder=2)
            dp[i,:len(y)] = y
        m = np.nanmean(dp,axis=0)
        ax.plot(np.arange(len(m))+1,m,color=clr,lw=2,zorder=1,label=mod)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,max(sessionsToPass)+2])
    ax.set_ylim([-3,4])
    ax.set_xlabel('Session',fontsize=14)
    ax.set_ylabel('d\' '+comp+' modality',fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
# compare early, late, and after learning
nSessions = 5
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)  
for phase in ('initial training','late training','after learning'):
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
            y = []
            for exps,s in zip(sessionData,sessionsToPass):
                if len(exps)>0:
                    if phase=='initial training':
                        exps = exps[:nSessions]
                    elif phase=='late training':
                        exps = exps[s-2-nSessions:s-2]
                    else:
                        exps = exps[s:s+nSessions]
                    y.append(np.full((len(exps),preTrials+postTrials+1),np.nan))
                    for i,obj in enumerate(exps):
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                trials = (obj.trialStim==stim) & ~obj.autoRewarded 
                                pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                j = min(preTrials,pre.size)
                                y[-1][i][preTrials-j:preTrials] = pre[-j:]
                                post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                                j = min(postTrials,post.size)
                                y[-1][i][preTrials+1:preTrials+1+j] = post[:j]
                    y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(-20,20,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch (excluding auto-rewards)',fontsize=12)
        ax.set_ylabel('Response Rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),fontsize=12)
        ax.set_title(phase+'\n'+blockLabel,fontsize=12)
        plt.tight_layout()
    
# d' correlations by session
passOnly = False
combos = list(itertools.combinations(itertools.product(('same','other'),('vis','sound')),2))
r = {c: [] for c in combos}
p = {c: [] for c in combos}
fig = plt.figure(figsize=(6,8))
for i,c in enumerate(combos):
    ax = fig.add_subplot(3,2,i+1)
    alim = [10,-10]
    (compX,modX),(compY,modY) = c
    for j,clr in enumerate(mouseClrs):
        dx,dy = [np.nanmean(dprime[comp][mod][j],axis=1) for comp,mod in zip((compX,compY),(modX,modY))]
        if passOnly:
            ind = slice(sessionsToPass[j]-2,None)
            dx = dx[ind]
            dy = dy[ind]
        dmin = min(np.nanmin(dx),np.nanmin(dy))
        dmax = max(np.nanmax(dx),np.nanmax(dy))
        alim = [min(alim[0],dmin),max(alim[1],dmax)]
        ax.plot(dx,dy,'o',color=clr,alpha=0.25)
        slope,yint,rval,pval,stderr = scipy.stats.linregress(dx[~np.isnan(dx)],dy[~np.isnan(dy)])
        x = np.array([dmin,dmax])
        ax.plot(x,slope*x+yint,'-',color=clr)
        r[c].append(rval)
        p[c].append(pval)
    p[c] = multipletests(p[c],alpha=0.05,method='fdr_bh')[1]
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    offset = 0.05*(alim[1]-alim[0])
    alim = [alim[0]-offset,alim[1]+offset]
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('d\' '+compX+', '+modX,fontsize=14)
    ax.set_ylabel('d\' '+compY+', '+modY,fontsize=14)
plt.tight_layout()
    
fig = plt.figure(figsize=(6,6))
for i,(d,xlbl) in enumerate(zip((r,p),('d\' correlation across sessions','corrected p value'))):
    ax = fig.add_subplot(2,1,i+1)
    x = 0.05 if i==1 else 0
    ax.plot([x,x],[0,1],'--',color='0.5')
    for c,clr in zip(combos,'grmcbk'):
        dsort = np.sort(d[c])
        cumProb = np.array([np.sum(dsort<=j)/dsort.size for j in dsort])
        ax.plot(dsort,cumProb,color=clr,label=c)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    xmin = 0 if i==1 else -1
    ax.set_xlim([xmin,1])
    ax.set_ylim([0,1.01])
    ax.set_xlabel(xlbl,fontsize=14)
    ax.set_ylabel('Cumulative fraction',fontsize=14)
    if i==0:
        plt.legend()
plt.tight_layout()

# d' correlations by block
passOnly = False
r = {mod: [] for mod in ('vis','sound')}
p = {mod: [] for mod in ('vis','sound')}
fig = plt.figure(figsize=(6,6))
for i,mod in enumerate(('vis','sound')):
    ax = fig.add_subplot(2,1,i+1)
    alim = [10,-10]
    for j,clr in enumerate(mouseClrs):
        if passOnly:
            dx,dy = [np.ravel(dprime[comp][mod][j][sessionsToPass[j]-2:]) for comp in ('same','other')]
        else:
            dx,dy = [np.ravel(dprime[comp][mod][j]) for comp in ('same','other')]
        dmin = min(np.nanmin(dx),np.nanmin(dy))
        dmax = max(np.nanmax(dx),np.nanmax(dy))
        alim = [min(alim[0],dmin),max(alim[1],dmax)]
        ax.plot(dx,dy,'o',color=clr,alpha=0.25)
        slope,yint,rval,pval,stderr = scipy.stats.linregress(dx[~np.isnan(dx)],dy[~np.isnan(dy)])
        x = np.array([dmin,dmax])
        ax.plot(x,slope*x+yint,'-',color=clr)
        r[mod].append(rval)
        p[mod].append(pval)
    p[mod] = multipletests(p[mod],alpha=0.05,method='fdr_bh')[1]
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    offset = 0.05*(alim[1]-alim[0])
    alim = [alim[0]-offset,alim[1]+offset]
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('d\' same'+', '+mod,fontsize=14)
    ax.set_ylabel('d\' other'+', '+mod,fontsize=14)
    plt.tight_layout()
    
fig = plt.figure(figsize=(6,6))
for i,(d,xlbl) in enumerate(zip((r,p),('d\' correlation across blocks','corrected p value'))):
    ax = fig.add_subplot(2,1,i+1)
    x = 0.05 if i==1 else 0
    ax.plot([x,x],[0,1],'--',color='0.5')
    for mod,clr in zip(('vis','sound'),'rb'):
        dsort = np.sort(d[mod])
        cumProb = np.array([np.sum(dsort<=j)/dsort.size for j in dsort])
        ax.plot(dsort,cumProb,color=clr,label=mod)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    xmin = -0.05 if i==1 else -1
    ax.set_xlim([xmin,1])
    ax.set_ylim([0,1.01])
    ax.set_xlabel(xlbl,fontsize=14)
    ax.set_ylabel('Cumulative fraction',fontsize=14)
    if i==0:
        plt.legend()
plt.tight_layout()

# response rate correlations
r = {}
p = {}
for combo in ((('same','vis'),('other','sound')),
              (('same','sound'),('other','vis')),
              ('catch',('same','vis')),
              ('catch',('same','sound')),
              ('catch',('other','vis')),
              ('catch',('other','sound'))):
    r[combo] = []
    p[combo] = []
    for exps,sp in zip(sessionData,sessionsToPass):
        if passOnly:
            exps = exps[sp-2:]
        respRate = [[],[]]
        for obj in exps:
            for i,c in enumerate(combo):
                j = 0 if i==1 else 1
                if (('same' in c and 'vis' in c) or ('other' in c and 'sound' in c) or
                    (c=='catch' and (('same' in c[j] and 'vis' in c[j]) or ('other' in c[j] and 'sound' in c[j])))):
                    blocks = obj.blockStimRewarded=='vis1'
                else:
                    blocks = obj.blockStimRewarded=='sound1'
                if c=='hit':
                    respRate[i].append(np.array(obj.hitRate)[blocks])
                elif c=='catch':
                    respRate[i].append(np.array(obj.catchResponseRate)[blocks])
                elif 'same' in c:
                    respRate[i].append(np.array(obj.falseAlarmSameModal)[blocks])
                elif 'other' in c:
                    respRate[i].append(np.array(obj.falseAlarmOtherModalGo)[blocks])
        x,y = [np.ravel(rr) for rr in respRate]
        notNan = ~np.isnan(x) & ~np.isnan(y)
        slope,yint,rval,pval,stderr = scipy.stats.linregress(x[notNan],y[notNan])
        r[combo].append(rval)
        p[combo].append(pval)
    p[combo] = multipletests(p[combo],alpha=0.05,method='fdr_bh')[1]
    
fig = plt.figure(figsize=(6,6))
for i,(d,xlbl) in enumerate(zip((r,p),('False alarm rate correlation across blocks','corrected p value'))):
    ax = fig.add_subplot(2,1,i+1)
    x = 0.05 if i==1 else 0
    ax.plot([x,x],[0,1],'--',color='0.5')
    for combo,clr in zip(r,'rbgmck'):
        dsort = np.sort(d[combo])
        cumProb = np.array([np.sum(dsort<=j)/dsort.size for j in dsort])
        ax.plot(dsort,cumProb,color=clr,label=combo)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    xmin = -0.05 if i==1 else -1
    ax.set_xlim([xmin,1])
    ax.set_ylim([0,1.01])
    ax.set_xlabel(xlbl,fontsize=14)
    ax.set_ylabel('Cumulative fraction',fontsize=14)
    if i==0:
        plt.legend()
plt.tight_layout()

# block switch plot by performance quantiles
stimNames = ('vis1','vis2','sound1','sound2','catch')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target','catch')
postTrials = 15
x = np.arange(postTrials)+1
nQuantiles = 3
quantiles = [(i/nQuantiles,(i+1)/nQuantiles) for i in range(nQuantiles)]
for q in quantiles:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmmk',('-','--','-','--','-')):
            y = []
            for exps,sp in zip(sessionData,sessionsToPass):
                exps = exps[sp-2:]
                dp = np.ravel([obj.dprimeOtherModalGo for obj in exps])
                lower,upper = np.quantile(dp,q)
                inQuantile = (dp>lower) & (dp<=upper) if lower>0 else (dp>=lower) & (dp<=upper)
                qBlocks = np.where(inQuantile)[0]
                blockCount = 0
                y.append(np.full((len(exps),postTrials),np.nan))
                for i,obj in enumerate(exps):
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if rewStim==rewardStim and blockCount in qBlocks:
                            trials = (obj.trialBlock==blockInd+1) & (obj.trialStim==stim) & ~obj.autoRewarded 
                            j = min(postTrials,trials.sum())
                            y[-1][i][:j] = obj.trialResponse[trials][:j]
                        blockCount += 1
                y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(0,20,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([0,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch\n(excluding auto-rewards)',fontsize=12)
        ax.set_ylabel('Response Rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),fontsize=12)
        ax.set_title(str(q)+blockLabel,fontsize=12)
        plt.tight_layout()
    

## performance after passing
ind = summaryDf['stage 5 pass']
mice = {'stationary, tones': np.array(summaryDf[ind & summaryDf['stat grating'] & summaryDf['tone']]['mouse id']),
        'moving, tones':  np.array(summaryDf[ind & summaryDf['moving grating'] & summaryDf['tone']]['mouse id']),
        'moving, AM noise': np.array(summaryDf[ind & summaryDf['moving grating'] & summaryDf['AM noise']]['mouse id'])}

sessionStartTimes = {lbl: [] for lbl in mice}
sessionData = {lbl: [] for lbl in mice}
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array(['stage 5' in task for task in df['task version']])
        firstExperimentSession = np.where(['multimodal' in task
                                           or 'contrast'in task
                                           or 'opto' in task
                                           or 'nogo' in task
                                           or 'noAR' in task
                                           # or 'NP' in rig 
                                           for task,rig in zip(df['task version'],df['rig name'])])[0]
        if len(firstExperimentSession)>0:
            sessions[firstExperimentSession[0]:] = False
        sessions = np.where(sessions)[0]
        passSession = None
        for i,sessionInd in enumerate(sessions[1:]):
            hits,dprimeSame,dprimeOther = getPerformanceStats(df,(sessionInd-1,sessionInd))
            if np.all(np.sum((np.array(dprimeSame) >= dprimeThresh) & (np.array(dprimeOther) >= dprimeThresh),axis=1) > 3):
                passSession = i+1
                break
        sessions = sessions[passSession+1:]
        sessions = [i for i in sessions if 'repeats' not in df.loc[i,'task version']]
        sessionStartTimes = list(df['start time'][sessions])
        dataDir = summaryDf.loc[summaryDf['mouse id']==mid,'data path'].values[0]
        sessionData[lbl].append(getSessionData(mid,dataDir,sessionStartTimes))
          
# block switch plot, all stimuli
stimNames = ('vis1','vis2','sound1','sound2')
stimLabels = ('visual target','visual non-target','auditory target','auditory non-target')
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)   
for lbl in sessionData:
    for rewardStim,blockLabel in zip(('vis1','sound1'),('visual rewarded blocks','auditory rewarded blocks')):
        fig = plt.figure(figsize=(8,4.5))
        ax = fig.add_subplot(1,1,1)
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stim,stimLbl,clr,ls in zip(stimNames,stimLabels,'ggmm',('-','--','-','--')):
            y = []
            for exps in sessionData[lbl]:
                if len(exps)>0:
                    y.append(np.full((len(exps),preTrials+postTrials+1),np.nan))
                    for i,obj in enumerate(exps):
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0 and rewStim==rewardStim:
                                trials = (obj.trialStim==stim) & ~obj.autoRewarded 
                                pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                j = min(preTrials,pre.size)
                                y[-1][i][preTrials-j:preTrials] = pre[-j:]
                                post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                                j = min(postTrials,post.size)
                                y[-1][i][preTrials+1:preTrials+1+j] = post[:j]
                    y[-1] = np.nanmean(y[-1],axis=0)
            m = np.nanmean(y,axis=0)
            s = np.nanstd(y,axis=0)/(len(y)**0.5)
            ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(-20,20,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch\n(excluding auto-rewards)',fontsize=12)
        ax.set_ylabel('Response Rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),fontsize=12)
        ax.set_title(lbl+' ('+str(len(mice[lbl]))+' mice)\n'+blockLabel,fontsize=12)
        plt.tight_layout()
        
# block switch plot, target stimuli only
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)    
ax.plot([0,0],[0,1],'--',color='0.5')
for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
    y = []
    for exps in [exps for lbl in sessionData for exps in sessionData[lbl]]:
        if len(exps)>0:
            y.append(np.full((len(exps),preTrials+postTrials+1),np.nan))
            for i,obj in enumerate(exps):
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0:
                        stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                        trials = (obj.trialStim==stim) & ~obj.autoRewarded
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        j = min(preTrials,pre.size)
                        y[-1][i][preTrials-j:preTrials] = pre[-j:]
                        post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                        j = min(postTrials,post.size)
                        y[-1][i][preTrials+1:preTrials+1+j] = post[:j]
            y[-1] = np.nanmean(y[-1],axis=0)
    m = np.nanmean(y,axis=0)
    s = np.nanstd(y,axis=0)/(len(y)**0.5)
    ax.plot(x,m,color=clr,label=stimLbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(np.arange(-20,21,5))
ax.set_yticks([0,0.5,1])
ax.set_xlim([-preTrials-0.5,postTrials+0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Trials of indicated type after block switch (excluding auto-rewards)',fontsize=12)
ax.set_ylabel('Response rate',fontsize=12)
ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
ax.set_title(str(len(y))+' mice',fontsize=12)
plt.tight_layout()

# block switch plot, target stimuli only, delayed auto-rewards
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
preTrials = 15
postTrials = 15
x = np.arange(-preTrials,postTrials+1)    
ax.plot([0,0],[0,1],'--',color='0.5')
for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
    y = []
    for exps in [[obj for obj in exps if obj.autoRewardOnsetFrame>=obj.responseWindow[1]] for lbl in sessionData for exps in sessionData[lbl]]:
        if len(exps)>0:
            y.append(np.full((len(exps),preTrials+postTrials+1),np.nan))
            for i,obj in enumerate(exps):
                for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                    if blockInd > 0:
                        stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                        trials = (obj.trialStim==stim)
                        pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                        j = min(preTrials,pre.size)
                        y[-1][i][preTrials-j:preTrials] = pre[-j:]
                        post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                        j = min(postTrials,post.size)
                        y[-1][i][preTrials+1:preTrials+1+j] = post[:j]
            y[-1] = np.nanmean(y[-1],axis=0)
    m = np.nanmean(y,axis=0)
    s = np.nanstd(y,axis=0)/(len(y)**0.5)
    ax.plot(x,m,color=clr,label=stimLbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xticks(np.arange(-20,21,5))
ax.set_yticks([0,0.5,1])
ax.set_xlim([-preTrials-0.5,postTrials+0.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
ax.set_ylabel('Response rate',fontsize=12)
ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
ax.set_title(str(len(y))+' mice',fontsize=12)
plt.tight_layout()

# probability of response since last reward, response, or same stimulus
stimType = ('rewarded target','unrewarded target','non-target (rewarded modality)','non-target (unrewarded modality)')
resp = {s: [] for s in stimType}
trialsSincePrevReward = copy.deepcopy(resp)
trialsSincePrevNonReward = copy.deepcopy(resp)
trialsSincePrevResp = copy.deepcopy(resp)
trialsSincePrevStim = copy.deepcopy(resp)
for obj in [obj for lbl in sessionData for exps in sessionData[lbl] for obj in exps]:
    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
        otherModalTarget = np.setdiff1d(obj.blockStimRewarded,rewStim)[0]
        blockTrials = (obj.trialBlock==blockInd+1) & ~obj.catchTrials
        rewTrials = np.where(blockTrials & obj.trialRewarded)[0]
        nonRewTrials = np.where(blockTrials & obj.trialResponse & ~obj.trialRewarded)[0]
        respTrials = np.where(blockTrials & obj.trialResponse)[0]
        for s in stimType:
            if s=='rewarded target':
                stim = rewStim
            elif s=='unrewarded target':
                stim = otherModalTarget
            elif s=='non-target (rewarded modality)':
                stim = rewStim[:-1]+'2'
            else:
                stim = otherModalTarget[:-1]+'2'
            stimTrials = np.where(blockTrials & (obj.trialStim==stim))[0]
            prevRewardTrial = rewTrials[np.searchsorted(rewTrials,stimTrials) - 1]
            prevRespTrial = respTrials[np.searchsorted(respTrials,stimTrials) - 1]
            trialsSincePrevReward[s].extend(stimTrials - prevRewardTrial)
            trialsSincePrevResp[s].extend(stimTrials - prevRespTrial)
            trialsSincePrevStim[s].extend(np.concatenate(([np.nan],np.diff(stimTrials))))
            if len(nonRewTrials) > 0:
                prevNonRewardTrial = nonRewTrials[np.searchsorted(nonRewTrials,stimTrials) - 1]
                trialsSincePrevNonReward[s].extend(stimTrials - prevNonRewardTrial)
            else:
                trialsSincePrevNonReward[s].extend(np.full(len(stimTrials),np.nan))
            resp[s].extend(obj.trialResponse[stimTrials])
for d in (trialsSincePrevReward,trialsSincePrevNonReward,trialsSincePrevResp,trialsSincePrevStim,resp):
    for s in stimType:
        d[s] = np.array(d[s])

for trialsSince,lbl in zip((trialsSincePrevReward,trialsSincePrevNonReward,trialsSincePrevResp,trialsSincePrevStim),
                           ('reward','non-reward','response','same stimulus')):
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot(1,1,1)
    x = np.arange(20)
    for s,clr,ls in zip(stimType,'gmgm',('-','-','--','--')):
        n = np.zeros(x.size)
        p = np.zeros(x.size)
        ci = np.zeros((x.size,2))
        for i in x:
            if i>0:
                j = trialsSince[s]==i
                n[i] += j.sum()
                p[i] += resp[s][j].sum()
        p /= n
        ci = np.array([[b/n[i] for b in scipy.stats.binom.interval(0.95,n[i],p[i])] for i in x])
        ax.plot(x,p,color=clr,ls=ls,label=s)
        ax.fill_between(x,ci[:,0],ci[:,1],color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,14])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('trials since last '+lbl)
    ax.set_ylabel('response rate')
    ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()


## nogo and noAR
ind = summaryDf['stage 5 pass']
mice = {'nogo': np.array(summaryDf[ind & summaryDf['nogo']]['mouse id']),
        'noAR': np.array(summaryDf[ind & summaryDf['noAR']]['mouse id'])}

sessionStartTimes = {lbl: [] for lbl in mice}
sessionData = {lbl: [] for lbl in mice}
for lbl,mouseIds in mice.items():
    for mid in mouseIds:
        df = drSheets[str(mid)] if str(mid) in drSheets else nsbSheets[str(mid)]
        sessions = np.array(['stage 5' in task and lbl in task for task in df['task version']])
        sessionStartTimes = list(df['start time'][sessions])
        dataDir = summaryDf.loc[summaryDf['mouse id']==mid,'data path'].values[0]
        sessionData[lbl].append([])
        for t in sessionStartTimes:
            fileName = 'DynamicRouting1_' + str(mid) + '_' + t.strftime('%Y%m%d_%H%M%S') + '.hdf5'
            if str(mid) in drSheets:
                filePath = os.path.join(dataDir,str(mid),fileName)
            else:
                filePath = glob.glob(os.path.join(dataDir,'**',fileName))[0]
            obj = DynRoutData()
            obj.loadBehavData(filePath)
            sessionData[lbl][-1].append(obj)
            
# block switch plot
for lbl,title in zip(sessionData,('block switch cued with non-rewarded target trials','no block switch cues')):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)    
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
        y = []
        for exps in sessionData[lbl]:
            if len(exps)>0:
                y.append(np.full((len(exps),preTrials+postTrials+1),np.nan))
                for i,obj in enumerate(exps):
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockInd > 0:
                            stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                            trials = (obj.trialStim==stim)
                            pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                            j = min(preTrials,pre.size)
                            y[-1][i][preTrials-j:preTrials] = pre[-j:]
                            post = obj.trialResponse[(obj.trialBlock==blockInd+1) & trials]
                            j = min(postTrials,post.size)
                            y[-1][i][preTrials+1:preTrials+1+j] = post[:j]
                y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(np.arange(-20,21,5))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
    ax.set_ylabel('Response rate',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1),loc='upper left')
    ax.set_title(title+' ('+str(len(y))+' mice)',fontsize=12)
    plt.tight_layout()

# block switch plot aligned to first reward
for lbl,title in zip(sessionData,('block switch cued with non-rewarded target trials','no block switch cues')):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    preTrials = 15
    postTrials = 15
    x = np.arange(-preTrials,postTrials+1)    
    ax.plot([0,0],[0,1],'--',color='0.5')
    for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
        y = []
        for exps in sessionData[lbl]:
            if len(exps)>0:
                y.append(np.full((len(exps),preTrials+postTrials+1),np.nan))
                for i,obj in enumerate(exps):
                    for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                        if blockInd > 0:
                            stim = np.setdiff1d(obj.blockStimRewarded,rewStim) if 'unrewarded' in stimLbl else rewStim
                            stimTrials = np.where(obj.trialStim==stim)[0]
                            blockTrials = np.where(obj.trialBlock==blockInd+1)[0]
                            firstReward = blockTrials[obj.trialRewarded[blockTrials]][0]
                            lastPreTrial = np.where(stimTrials<firstReward)[0][-1]
                            pre = obj.trialResponse[stimTrials[lastPreTrial-preTrials:lastPreTrial+1]]
                            j = min(preTrials,pre.size)
                            y[-1][i][preTrials-j:preTrials] = pre[-j:]
                            firstPostTrial = np.where(stimTrials>firstReward)[0][0]
                            post = obj.trialResponse[stimTrials[firstPostTrial:max(firstPostTrial+postTrials,blockTrials[-1])]]
                            j = min(postTrials,post.size)
                            y[-1][i][preTrials+1:preTrials+1+j] = post[:j]
                y[-1] = np.nanmean(y[-1],axis=0)
        m = np.nanmean(y,axis=0)
        s = np.nanstd(y,axis=0)/(len(y)**0.5)
        ax.plot(x,m,color=clr,label=stimLbl)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xticks(np.arange(-20,21,5))
    ax.set_yticks([0,0.5,1])
    ax.set_xlim([-preTrials-0.5,postTrials+0.5])
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Trials of indicated type after first reward',fontsize=12)
    ax.set_ylabel('Response rate',fontsize=12)
    ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
    ax.set_title(title+' ('+str(len(y))+' mice)',fontsize=12)
    plt.tight_layout()
    
# block switch plots by first trial stim/reward type
for firstTrialRewStim,blockLbl in zip((True,False),('rewarded target first','non-rewarded target first')):
    for firstTrialLick,lickLbl in zip((True,False),('lick','no lick')):
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)
        preTrials = 15
        postTrials = 15
        x = np.arange(-preTrials,postTrials+1)    
        ax.plot([0,0],[0,1],'--',color='0.5')
        for stimLbl,clr in zip(('rewarded target stim','unrewarded target stim'),'gm'):
            n = 0
            y = []
            for exps in sessionData['noAR']:
                if len(exps)>0:
                    y.append(np.full((len(exps),preTrials+postTrials+1),np.nan))
                    for i,obj in enumerate(exps):
                        for blockInd,rewStim in enumerate(obj.blockStimRewarded):
                            if blockInd > 0:
                                nonRewStim = np.setdiff1d(obj.blockStimRewarded,rewStim)
                                blockTrials = obj.trialBlock==blockInd+1
                                firstRewStim = np.where(blockTrials & (obj.trialStim==rewStim))[0][0]
                                firstNonRewStim = np.where(blockTrials & (obj.trialStim==nonRewStim))[0][0]
                                if ((firstTrialRewStim and firstRewStim > firstNonRewStim) or
                                    (not firstTrialRewStim and firstRewStim < firstNonRewStim)):
                                    continue
                                firstTargetTrial = firstRewStim if firstTrialRewStim else firstNonRewStim
                                if obj.trialResponse[firstTargetTrial] != firstTrialLick:
                                    continue
                                stim = nonRewStim if 'unrewarded' in stimLbl else rewStim
                                trials = obj.trialStim==stim
                                pre = obj.trialResponse[(obj.trialBlock==blockInd) & trials]
                                j = min(preTrials,pre.size)
                                y[-1][i][preTrials-j:preTrials] = pre[-j:]
                                post = obj.trialResponse[blockTrials & trials]
                                k = min(postTrials,post.size)
                                y[-1][i][preTrials+1:preTrials+1+j] = post[:j]
                    n += np.sum(~np.isnan(y[-1])[:,0])
                    y[-1] = np.nanmean(y[-1],axis=0)
            if len(y)>0:
                print(len(y))
                m = np.nanmean(y,axis=0)
                s = np.nanstd(y,axis=0)/(len(y)**0.5)
                ax.plot(x,m,color=clr,label=stimLbl)
                ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
          
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xticks(np.arange(-20,21,5))
        ax.set_yticks([0,0.5,1])
        ax.set_xlim([-preTrials-0.5,postTrials+0.5])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('Trials of indicated type after block switch',fontsize=12)
        ax.set_ylabel('Response rate',fontsize=12)
        ax.legend(bbox_to_anchor=(1,1),loc='upper left',fontsize=12)
        ax.set_title(blockLbl+', '+lickLbl+', '+str(n)+' blocks')
        plt.tight_layout()









