# -*- coding: utf-8 -*-
"""
Superclass for behavioral task control

"""

from __future__ import division
import json, math, os, sys, time
from threading import Timer
import h5py
import numpy as np
import scipy.signal
from psychopy import monitors, visual, event
from psychopy.visual.windowwarp import Warper
import psychtoolbox.audio
import sounddevice
import nidaqmx
import serial
import zmq
import datetime
import time


class TaskControl():
    
    def __init__(self,params=None):
        self.configPath = None
        self.rigName = None
        self.subjectName = None
        self.startTime = None
        self.maxFrames = None # max number of frames before task terminates
        self.saveParams = True # if True, saves all attributes not starting with underscore
        self.saveFrameIntervals = True
        self.monBackgroundColor = 0 # gray; can adjust this for luminance measurement
        self.minWheelAngleChange = 0 # radians per frame
        self.maxWheelAngleChange = 0.5 # radians per frame
        self.spacebarRewardsEnabled = True
        self.soundMode = 'sound card' # 'sound card' or 'daq'
        self.soundLibrary = 'psychtoolbox' # 'psychtoolbox' or 'sounddevice'
        self.soundSampleRate = 48000 # Hz
        self.soundHanningDur = 0.005 # seconds
        
        # rig specific settings
        self.frameRate = 60
        self.screen = 0 # monitor to present stimuli on
        self.monWidth = 52.0 # cm
        self.monDistance = 15.3 # cm
        self.monGamma = 2.3 # float or None
        self.monSizePix = (1920,1200)
        self.warp = None # 'spherical', 'cylindrical', 'warpfile', None
        self.warpFile = None
        self.drawDiodeBox = False
        self.wheelRadius = 4.69 # cm
        self.wheelPolarity = -1
        self.rotaryEncoder = 'digital' # 'digital', 'analog', or None
        self.rotaryEncoderCh = 1 # nidaq analog input channel
        self.rotaryEncoderSerialPort = None # serial input from arduino for digital encoder
        self.rotaryEncoderCountsPerRev = 8192 # digital pulses per revolution of encoder
        self.microphoneCh = None
        self.digitalSolenoidTrigger = True
        self.behavNidaqDevice = 'Dev1'
        self.rewardLine = (0,7)
        self.rewardSoundLine = (2,0)
        self.lickLine = (0,0)
        self.syncNidaqDevice = None
        self.frameSignalLine = (1,4)
        self.acquisitionSignalLine = (1,7)
        self.optoNidaqDevice = None
        self.galvoNidaqDevice = None
        self.soundNidaqDevice = None
        self.solenoidOpenTime = 0.03
        self.rewardSoundDeviceOpenTime = 0.01
        
        if params is not None:
            self.rigName = params['rigName']
            if 'configPath' in params:
                self.startTime = params['startTime']
                self.saveDir = None
                self.savePath = params['savePath']
                self.configPath = params['configPath']
                self.rotaryEncoderSerialPort = params['rotaryEncoderSerialPort']
                self.behavNidaqDevice = params['behavNidaqDevice']
                self.rewardLine = params['rewardLines'][0]
                self.rewardSoundLine = params['rewardLines'][1]
                self.lickLine = params['lickLines'][0]
                self.rewardVol = 0.005 # uL
                self.waterCalibrationSlope = params['waterCalibrationSlope']
                self.waterCalibrationIntercept = params['waterCalibrationIntercept']
                self.solenoidOpenTime = self.waterCalibrationSlope * self.rewardVol + self.waterCalibrationIntercept
                self.soundCalibrationFit = params['soundCalibrationFit']
            else:
                self.saveDir = r"\\allen\programs\mindscope\workgroups\dynamicrouting\DynamicRoutingTask\Data"
                self.configPath = None
                self.rewardVol = None
                self.waterCalibrationSlope = None
                self.waterCalibrationIntercept = None
                self.soundCalibrationFit = None
                if self.rigName in ('NP2','NP3'):
                    self.drawDiodeBox = True
                    self.diodeBoxSize = 120
                    self.diodeBoxPosition = (900,540)
                    self.behavNidaqDevice = 'Dev0'
                    self.syncNidaqDevice = 'Dev1'
                    self.optoNidaqDevice = 'Dev2'
                    if self.rigName == 'NP2':
                        self.rotaryEncoderSerialPort = 'COM5'
                        self.galvoNidaqDevice = 'Dev4'
                        self.solenoidOpenTime = 0.06 # 2.6 uL
                        self.soundCalibrationFit = (33.17940258725825,-5.040610266883152,56.936135475568065)
                    elif self.rigName == 'NP3':
                        self.rotaryEncoderSerialPort = 'COM3'
                        # self.soundMode = 'daq'
                        # self.soundNidaqDevice = 'zDAQ9185-213AB43Mod4'
                        self.galvoNidaqDevice = 'GalvoDAQ'
                elif self.rigName in ('B1','B2','B3','B4','B5','B6'):
                    if self.rigName == 'B1':
                        self.rotaryEncoderSerialPort = 'COM3'
                        self.solenoidOpenTime = 0.02 # 2.56 uL 5/24/2023
                        self.soundCalibrationFit = (25.943102352592554,-1.7225414088360975,59.4889757694944)
                    elif self.rigName == 'B2':
                        self.rotaryEncoderSerialPort = 'COM3'
                        self.solenoidOpenTime = 0.03 # 2.54 uL 5/24/2023
                        self.soundCalibrationFit = (25.87774455245642,-2.5151852106916355,57.58077780177194)
                    elif self.rigName == 'B3':
                        self.rotaryEncoderSerialPort = 'COM3'
                        self.solenoidOpenTime = 0.035 # 2.48 uL 5/24/2023
                        self.soundCalibrationFit = (25.773538946631238,-2.4069019340061995,57.65570739632032)
                    elif self.rigName == 'B4':
                        self.rotaryEncoderSerialPort = 'COM3'
                        self.solenoidOpenTime = 0.025 # 2.62 uL 5/24/2023
                        self.soundCalibrationFit = (27.723495908673165,-2.8409439349143746,56.05978764386811)
                    elif self.rigName == 'B5':
                        self.rotaryEncoderSerialPort = 'COM3'
                        self.solenoidOpenTime = 0.02 # 3.21 uL 5/24/2023
                        self.soundCalibrationFit = (25.399041813825953,-1.624962406018245,62.1366870220353)
                    elif self.rigName == 'B6':
                        self.rotaryEncoderSerialPort = 'COM3'
                        self.solenoidOpenTime = 0.035 # 2.77 uL 5/24/2023
                        self.soundCalibrationFit = (26.184874388495313,-2.397480288683932,59.6253081914033,)
                elif self.rigName in ('E1','E2','E3','E4','E5','E6'):
                    if self.rigName == 'E1':
                        self.rotaryEncoderSerialPort = 'COM6'
                        self.soundCalibrationFit = (28.676264670218284,-3.5404140940509587,61.98218469422576)
                    elif self.rigName == 'E2':
                        self.rotaryEncoderSerialPort = 'COM6'
                        self.soundCalibrationFit = (31.983188322031314,-4.643575999625382,56.72811699132991)
                    elif self.rigName == 'E3':
                        self.rotaryEncoderSerialPort = 'COM6'
                        self.soundCalibrationFit = (32.3885667779314,-4.757139011008818,55.730111844845254)
                    elif self.rigName == 'E4':
                        self.rotaryEncoderSerialPort = 'COM7'
                        self.soundCalibrationFit = (32.14419775571485,-4.83179517041608,56.003815715642524)
                    elif self.rigName == 'E5':
                        self.rotaryEncoderSerialPort = 'COM9'
                        self.soundCalibrationFit = (30.1311066394785,-3.868157939967758,58.0042625794081)
                    elif self.rigName == 'E6':
                        self.rotaryEncoderSerialPort = 'COM6'
                        self.soundCalibrationFit = (26.666445962440992,-2.8916289462120144,64.65830226417953)
                elif self.rigName in ('F1','F2','F3','F4','F5','F6'):
                    if self.rigName == 'F1':
                        self.rotaryEncoderSerialPort = 'COM4'
                        self.soundCalibrationFit = (28.56806078789988,-3.5156341154859634,61.625654083217164)
                    elif self.rigName == 'F2':
                        self.rotaryEncoderSerialPort = 'COM4'
                        self.soundCalibrationFit = (29.152737840643113,-3.4784089950469115,60.67264755690783)
                    elif self.rigName == 'F3':
                        self.rotaryEncoderSerialPort = 'COM4'
                        self.soundCalibrationFit = (32.580177659520615,-4.803418185877209,56.06508525277285)
                    elif self.rigName == 'F4':
                        self.rotaryEncoderSerialPort = 'COM4'
                        self.soundCalibrationFit = (28.14111954492041,-3.576562269222293,60.82925878937895)
                    elif self.rigName == 'F5':
                        self.rotaryEncoderSerialPort = 'COM4'
                        self.soundCalibrationFit = (28.187343339799433,-3.259256871294408,61.01815522295702)
                    elif self.rigName == 'F6':
                        self.rotaryEncoderSerialPort = 'COM4'
                        self.soundCalibrationFit = (28.655615630746905,-3.5166732104004796,61.36404105849515)
                else:
                    raise ValueError(self.rigName + ' is not a recognized rig name')
        
        self.accumulator_interface_attr_name = "accumulator_interface"
        accumulator_params = params.get("accumulator")
        if accumulator_params:
            setattr(
                self,
                self.accumulator_interface_attr_name,
                AccumulatorInterface(
                    **accumulator_params,
                ),
            ) 
            
    def prepareSession(self,window=True):
        self._win = None
        self._nidaqTasks = []
        
        if self.startTime is None:
            startTime = time.localtime()
            self.startTime = time.strftime('%Y%m%d_%H%M%S',startTime)
            print('start time was: ' + time.strftime('%I:%M',startTime))
        
        self.pixelsPerDeg = 0.5 * self.monSizePix[0] / math.degrees(math.atan(0.5 * self.monWidth / self.monDistance))
        
        if window:
            self.prepareWindow()
        
        self.startNidaqDevice()

        self.initSound()

        self.initOpto()

        if self.rotaryEncoder == 'digital':
            self.initDigitalEncoder()
        
        self.rotaryEncoderVolts = [] # rotary encoder analog input each frame
        self.rotaryEncoderIndex = [] # rotary encoder digital input read index
        self.rotaryEncoderCount = [] # rotary encoder digital input count
        self.wheelPosRadians = []
        self.deltaWheelPos = []
        self.microphoneData = []
        self.lickFrames = []
        
        self._continueSession = True
        self._sessionFrame = 0 # index of frame since start of session
        self._trialFrame = 0 # index of frame since start of trial
        self._reward = False # reward delivered at next frame flip if True
        self.rewardFrames = [] # index of frames at which reward delivered
        self.manualRewardFrames = [] # index of frames at which reward manually delivered
        self.rewardSize = [] # size (solenoid open time) of each reward
        self._rewardSound = False # trigger reward device (external clicker) at next frame flip if True
        self._sound = False # sound triggered at next frame flip if True
        self._galvo = False # False or galvo voltage waveform applied next frame flip
        self._opto = False # False or opto voltage waveform applied next frame flip
        
    
    def prepareWindow(self):
        self._mon = monitors.Monitor('monitor1',
                                     width=self.monWidth,
                                     distance=self.monDistance,
                                     gamma=self.monGamma)
        self._mon.setSizePix(self.monSizePix)
        self._mon.saveMon()
        self._win = visual.Window(monitor=self._mon,
                                  screen=self.screen,
                                  fullscr=True,
                                  units='pix',
                                  color=self.monBackgroundColor)
        self._warper = Warper(self._win,warp=self.warp,warpfile=self.warpFile)
        for _ in range(10):
            self._win.flip()
        self._win.setRecordFrameIntervals(self.saveFrameIntervals)
        
        if self.drawDiodeBox:
            self._diodeBox = visual.Rect(self._win,
                                         units='pix',
                                         width=self.diodeBoxSize,
                                         height=self.diodeBoxSize,
                                         lineColor=0,
                                         fillColor=1, 
                                         pos=self.diodeBoxPosition)
        
        
    def start(self,subjectName=None):
        try:
            if subjectName is not None:
                self.subjectName = str(subjectName)
            
            self.prepareSession()

            accumulator_interface = getattr(
                self,
                self.accumulator_interface_attr_name,
            )
            if accumulator_interface:
                accumulator_interface.publish_header()
            
            self.taskFlow()
        
        except:
            raise
            
        finally:
            self.completeSession()
    
    
    def taskFlow(self):
        # override this method in subclass
        
        while self._continueSession:
            # get rotary encoder and digital input states
            self.getInputData()
            
            # do stuff, for example:
            # check for licks and/or wheel movement
            # update/draw stimuli
            
            self.showFrame()


    def getInputData(self):
        self.getNidaqData()
        if self.rotaryEncoder == 'digital':
            self.readDigitalEncoder()
    
    
    def showFrame(self):
        if self.syncNidaqDevice is not None:
            self._frameSignalOutput.write(True)
            if self._sessionFrame == 0:
                self._acquisitionSignalOutput.write(True)

        if self.spacebarRewardsEnabled and 'space' in event.getKeys(['space']) and not self._reward:
            self._reward = self.solenoidOpenTime
            self.manualRewardFrames.append(self._sessionFrame)
        
        escape = event.getKeys(['escape'],modifiers=True)
        if (len(escape) > 0 and escape[0][1]['shift']) or (self.maxFrames is not None and self._sessionFrame == self.maxFrames - 1):   
            self._continueSession = False

        # show new frame
        if self.drawDiodeBox:
            self._diodeBox.fillColor = -self._diodeBox.fillColor
            self._diodeBox.draw()
        self._win.flip()

        if self.syncNidaqDevice is not None:
            self._frameSignalOutput.write(False)
            if not self._continueSession:
                self._acquisitionSignalOutput.write(False)

        if self._sound:
            if self.soundMode == 'sound card':
                self.playSound(self._sound[0])
            self._sound = False

        if self._galvo:
            self.applyGalvoWaveform(self._galvo[0])
            self._galvo = False
        
        if self._opto:
            self.applyOptoWaveform(self._opto[0])
            self._opto = False
        
        if self._reward:
            self.triggerReward(self._reward)
            self.rewardFrames.append(self._sessionFrame)
            rewardSize = self._reward if self.rewardVol is None else (self._reward - self.waterCalibrationIntercept) / self.waterCalibrationSlope
            self.rewardSize.append(rewardSize)
            self._reward = False
            
        if self._rewardSound:
            self.triggerRewardSound()
            self._rewardSound = False
           
        self._sessionFrame += 1
        self._trialFrame += 1
                                               
    
    def completeSession(self):
        try:
            if self._win is not None:
                self._win.close()
            self.stopNidaqDevice()
            if hasattr(self,'_audioStream'):
                self._audioStream.close()
            if hasattr(self,'_digitalEncoder'):
                self._digitalEncoder.close()
        except:
            raise
        finally:
            if self.saveParams:
                if self.subjectName is None:
                    subjName = ''
                else:
                    subjName = self.subjectName + '_'
                    if self.saveDir is not None:
                        saveDir = os.path.join(self.saveDir,self.subjectName)
                        if not os.path.exists(saveDir):
                            os.makedirs(saveDir)
                        self.savePath = os.path.join(saveDir,self.__class__.__name__ + '_' + subjName + self.startTime + '.hdf5')
                fileOut = h5py.File(self.savePath,'w')
                saveParameters(fileOut,self.__dict__)
                if self.saveFrameIntervals and self._win is not None:
                    fileOut.create_dataset('frameIntervals',data=self._win.frameIntervals)
                fileOut.close()
            self.startTime = None
        
    
    def startNidaqDevice(self):
        # rotary encoder and microphone
        if self.rotaryEncoder == 'analog' or self.microphoneCh is not None:
            aiSampleRate = 2000 if self._win.monitorFramePeriod < 0.0125 else 1000
            aiBufferSize = 16
            self._analogInput = nidaqmx.Task()

            if self.rotaryEncoder == 'analog':
                self._analogInput.ai_channels.add_ai_voltage_chan(self.behavNidaqDevice+'/ai'+str(self.rotaryEncoderCh),
                                                                  min_val=0,max_val=5)
            if self.microphoneCh is not None:
                self._analogInput.ai_channels.add_ai_voltage_chan(self.behavNidaqDevice+'/ai'+str(self.microphoneCh),
                                                                  min_val=0,max_val=1)
            
            self._analogInput.timing.cfg_samp_clk_timing(aiSampleRate,
                                                         sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                                         samps_per_chan=aiBufferSize)
                                                
            def readAnalogInput(task_handle,every_n_samples_event_type,number_of_samples,callback_data):
                self._analogInputData = self._analogInput.read(number_of_samples_per_channel=number_of_samples)
                return 0
            
            self._analogInput.register_every_n_samples_acquired_into_buffer_event(aiBufferSize,readAnalogInput)
            self._analogInputData = None
            self._analogInput.start()
            self._nidaqTasks.append(self._analogInput)
        
        # water reward solenoid
        self._rewardOutput = nidaqmx.Task()
        if self.digitalSolenoidTrigger:
            self._rewardOutput.do_channels.add_do_chan(self.behavNidaqDevice+'/port'+str(self.rewardLine[0])+'/line'+str(self.rewardLine[1]),
                                                       line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self._rewardOutput.write(False)
        else:
            self._rewardOutput.ao_channels.add_ao_voltage_chan(self.behavNidaqDevice+'/ao0',min_val=0,max_val=5)
            self._rewardOutput.write(0)
            self._rewardOutput.timing.cfg_samp_clk_timing(1000) # samples/s
        self._nidaqTasks.append(self._rewardOutput)
        
        # reward sound device
        self._rewardSoundOutput = nidaqmx.Task()
        self._rewardSoundOutput.do_channels.add_do_chan(self.behavNidaqDevice+'/port'+str(self.rewardSoundLine[0])+'/line'+str(self.rewardSoundLine[1]),
                                                        line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._rewardSoundOutput.write(False)
        self._nidaqTasks.append(self._rewardSoundOutput)
            
        # lick input
        self._lickInput = nidaqmx.Task()
        self._lickInput.di_channels.add_di_chan(self.behavNidaqDevice+'/port'+str(self.lickLine[0])+'/line'+str(self.lickLine[1]),
                                                line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        self._nidaqTasks.append(self._lickInput)
        
        # frame and acquistion signals
        if self.syncNidaqDevice is not None:
            self._frameSignalOutput = nidaqmx.Task()
            self._frameSignalOutput.do_channels.add_do_chan(self.syncNidaqDevice+'/port'+str(self.frameSignalLine[0])+'/line'+str(self.frameSignalLine[1]),
                                                            line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self._frameSignalOutput.write(False)
            self._nidaqTasks.append(self._frameSignalOutput)

            self._acquisitionSignalOutput = nidaqmx.Task()
            self._acquisitionSignalOutput.do_channels.add_do_chan(self.syncNidaqDevice+'/port'+str(self.acquisitionSignalLine[0])+'/line'+str(self.acquisitionSignalLine[1]),
                                                                  line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
            self._acquisitionSignalOutput.write(False)
            self._nidaqTasks.append(self._acquisitionSignalOutput)
    
    
    def stopNidaqDevice(self):
        if hasattr(self,'_optoVoltage'):
            self.optoOff()
        for task in self._nidaqTasks:
            task.close()
            
                
    def getNidaqData(self):
        # analog
        if hasattr(self,'_analogInput'):
            if self._analogInputData is None:
                if self.rotaryEncoder == 'analog':
                    self.rotaryEncoderVolts.append(np.nan)
                    encoderAngle = np.nan
                if self.microphoneCh is not None:
                    self.microphoneData.append(np.nan)
            else:
                if self.rotaryEncoder == 'analog':
                    encoderData = np.array(self._analogInputData) if self.microphoneCh is None else np.array(self._analogInputData[0])
                if self.microphoneCh is not None:
                    micData = self._analogInputData[1] if self.rotaryEncoder == 'analog' else self._analogInputData
                    self.microphoneData.append(np.std(micData))
                self.rotaryEncoderVolts.append(encoderData[-1])
                encoderData *= 2 * math.pi / 5
                encoderAngle = np.arctan2(np.mean(np.sin(encoderData)),np.mean(np.cos(encoderData)))
            self.wheelPosRadians.append(encoderAngle)
            self.deltaWheelPos.append(self.calculateWheelChange())
        
        # digital
        if self._lickInput.read():
            self._lick = True
            self.lickFrames.append(self._sessionFrame)
        else:
            self._lick = False


    def calculateWheelChange(self):
        # calculate angular change in wheel position
        if len(self.wheelPosRadians) < 2 or np.isnan(self.wheelPosRadians[-1]):
            angleChange = 0
        else:
            angleChange = self.wheelPosRadians[-1] - self.wheelPosRadians[-2]
            if angleChange < -math.pi:
                angleChange += 2 * math.pi
            elif angleChange > math.pi:
                angleChange -= 2 * math.pi
            if self.minWheelAngleChange < abs(angleChange) < self.maxWheelAngleChange:
                angleChange *= self.wheelPolarity
            else:
                angleChange = 0
        return angleChange
            
            
    def initDigitalEncoder(self):
        self._digitalEncoder = serial.Serial(port=self.rotaryEncoderSerialPort,baudrate=9600,timeout=0.5)

        # intialize arduino
        for message,response in zip(('7','3','8'),('MDR0','STR','MDR0')):
            self._digitalEncoder.write(message.encode('utf8'))
            for _ in range(1000):
                val = self._digitalEncoder.readline()[:-2].decode('utf-8')
                if response in val:
                    break
            else:
                raise Exception('unable to initialize digital rotary encoder')

        # reset encoder count to zero
        self._digitalEncoder.write(b'2')
        count = 0
        val = self._digitalEncoder.readline()[:-2].decode('utf-8')
        c = int(val.split(';')[-1].split(':')[-1])
        while c > 1000 or c < -1000:
            count += 1
            if count == 1000:
                break
            val = self._digitalEncoder.readline()[:-2].decode('utf-8')
            c = int(val.split(';')[-1].split(':')[-1])

    
    def readDigitalEncoder(self):
        try:
            r = self._digitalEncoder.readline()[:-2].decode('utf-8')
            self.rotaryEncoderIndex.append(int(r.split(';')[-2].split(':')[-1]))
            self.rotaryEncoderCount.append(int(r.split(';')[-1].split(':')[-1]))
        except:
            self.rotaryEncoderIndex.append(np.nan)
            self.rotaryEncoderCount.append(np.nan)


    def initSolenoid(self):
        self._solenoid = nidaqmx.Task()
        if self.digitalSolenoidTrigger:
            self._solenoid.do_channels.add_do_chan(self.behavNidaqDevice+'/port0/line7',
                                                   line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
        else:
            self._solenoid.ao_channels.add_ao_voltage_chan(self.nidaqDeviceNames[0]+'/ao0',min_val=0,max_val=5)
            
            
    def openSolenoid(self):
        self.initSolenoid()
        if self.digitalSolenoidTrigger:
            self._solenoid.write(True)
        else:
            self._solenoid.write(5)
        
    
    def closeSolenoid(self):
        if not hasattr(self,'_solenoid'):
            self.initSolenoid()
        if self.digitalSolenoidTrigger:
            self._solenoid.write(False)
        else:
            self._solenoid.write(0)
        self._solenoid.stop()
        self._solenoid.close()
        self._solenoid = None
        
        
    def triggerReward(self,openTime):
        if self.digitalSolenoidTrigger:
            t = Timer(openTime,self.endReward)
            self._rewardOutput.write(True)
            t.start()
        else:
            sampleRate = self._rewardOutput.timing.samp_clk_rate
            nSamples = int(openTime * sampleRate) + 1
            s = np.zeros(nSamples)
            s[:-1] = 5
            self._rewardOutput.stop()
            self._rewardOutput.timing.samp_quant_samp_per_chan = nSamples
            self._rewardOutput.write(s,auto_start=True)
    
    
    def endReward(self):
        if self.digitalSolenoidTrigger:
            self._rewardOutput.write(False)
            
            
    def triggerRewardSound(self):
        t = Timer(self.rewardSoundDeviceOpenTime,self.endRewardSound)
        self._rewardSoundOutput.write(True)
        t.start()
    
    
    def endRewardSound(self):
        if self.digitalSolenoidTrigger:
            self._rewardSoundOutput.write(False)


    def initSound(self):
        if self.soundMode == 'sound card':
            if self.soundLibrary == 'psychtoolbox':
                self._audioStream = psychtoolbox.audio.Stream(latency_class=[3],
                                                              freq=self.soundSampleRate,
                                                              channels=1)
            elif self.soundLibrary == 'sounddevice':
                sounddevice.default.latency = 0.016
        elif self.soundMode == 'daq':
            self._soundOutput = nidaqmx.Task()
            self._soundOutput.ao_channels.add_ao_voltage_chan(self.soundNidaqDevice+'/ao0',min_val=-10,max_val=10)
            self._soundOutput.write(0)
            self._soundOutput.timing.cfg_samp_clk_timing(self.soundSampleRate)
            self._nidaqTasks.append(self._soundOutput)
                
    
    def playSound(self,soundArray):
        if self.soundMode == 'sound card':
            if self.soundLibrary == 'psychtoolbox':
                self._audioStream.fill_buffer(soundArray)
                self._audioStream.start()
            elif self.soundLibrary == 'sounddevice':
                sounddevice.play(soundArray,self.soundSampleRate)
        elif self.soundMode == 'daq':
            self._soundOutput.stop()
            self._soundOutput.timing.samp_quant_samp_per_chan = soundArray.size
            self._soundOutput.write(soundArray,auto_start=True)


    def stopSound(self):
        if self.soundMode == 'sound card':
            if self.soundLibrary == 'psychtoolbox':
                if hasattr(self,'_audioStream'):
                    self._audioStream.stop()
            elif self.soundLibrary == 'sounddevice':
                sounddevice.stop()
        elif self.soundMode == 'daq':
            self._soundOutput.stop()
                
                
    def makeSoundArray(self,soundType,dur,vol,freq,AM=None,seed=None):
        t = np.arange(0,dur,1/self.soundSampleRate)
        if soundType == 'tone':
            soundArray = np.sin(2 * np.pi * freq * t)
        elif soundType in ('linear sweep','log sweep'):
            f = np.linspace(freq[0],freq[1],t.size)
            if soundType == 'log sweep':
                f = (2 ** f) * 1000
            soundArray = np.sin(2 * np.pi * f * t)
        elif soundType in ('noise','AM noise'):
            rng = np.random.RandomState(seed)
            soundArray = 2 * rng.random(t.size) - 1
            b,a = scipy.signal.butter(10,freq,btype='bandpass',fs=self.soundSampleRate)
            soundArray = scipy.signal.filtfilt(b,a,soundArray)
            soundArray = np.ascontiguousarray(soundArray)
        if AM is not None and ~np.isnan(AM) and AM > 0:
            soundArray *= (np.sin(1.5*np.pi + 2*np.pi*AM*t) + 1) / 2
        elif self.soundHanningDur > 0:
            # reduce onset/offset click
            hanningSamples = int(self.soundSampleRate * self.soundHanningDur)
            hanningWindow = np.hanning(2 * hanningSamples + 1)
            soundArray[:hanningSamples] *= hanningWindow[:hanningSamples]
            soundArray[-hanningSamples:] *= hanningWindow[hanningSamples+1:]
        soundArray /= np.absolute(soundArray).max()
        soundArray *= vol
        if self.soundMode == 'daq':
            soundArray *= 10
        return soundArray
    
    
    def dBToVol(self,dB,a,b,c):
        return math.log(1 - ((dB - c) / a)) / b
    
    
    def initOpto(self):
        if self.optoNidaqDevice is not None:
            self._optoOutput = nidaqmx.Task()
            self._optoOutput.ao_channels.add_ao_voltage_chan(self.optoNidaqDevice+'/ao0:1',min_val=0,max_val=5)
            self._optoVoltage = 0
            self._optoOutput.write([0,0])
            self._optoOutput.timing.cfg_samp_clk_timing(1000) # samples/s
            self._nidaqTasks.append(self._optoOutput)

        if self.galvoNidaqDevice is not None:
            self._galvoOutput = nidaqmx.Task()
            self._galvoOutput.ao_channels.add_ao_voltage_chan(self.galvoNidaqDevice+'/ao0:1',min_val=-5,max_val=5)
            self._galvoVoltage = [0,0]
            self._galvoOutput.write(self._galvoVoltage)
            self._galvoOutput.timing.cfg_samp_clk_timing(1000) # samples/s
            self._nidaqTasks.append(self._galvoOutput)


    def getOptoParams(self):
        from OptoParams import optoParams, getBregmaGalvoCalibrationData, bregmaToGalvo, getOptoPowerCalibrationData, powerToVolts
        
        self.bregmaGalvoCalibrationData = getBregmaGalvoCalibrationData(self.rigName)
        self.optoBregma = [optoParams[self.subjectName][region]['bregma'] for region in self.optoRegions]
        self.galvoVoltage = [bregmaToGalvo(self.bregmaGalvoCalibrationData,x,y) for x,y in self.optoBregma]
        
        self.optoPowerCalibrationData = getOptoPowerCalibrationData(self.rigName,self.optoDevName)
        self.optoPower = [optoParams[self.subjectName][region]['power'] for region in self.optoRegions]
        self.optoVoltage = [powerToVolts(self.optoPowerCalibrationData,pwr) for pwr in self.optoPower]

          
    def optoOn(self,amp,ramp=0):
        waveform = self.getOptoPulseWaveform(amp,onRamp=ramp,lastVal=amp)
        self.applyOptoWaveform(waveform)
    
    
    def optoOff(self,ramp=0):
        amp = self._optoVoltage if ramp > 0 else 0 
        waveform = self.getOptoPulseWaveform(amp,offRamp=ramp)
        self.applyOptoWaveform(waveform)
    
    
    def getOptoPulseWaveform(self,amp,dur=0,onRamp=0,offRamp=0,lastVal=0):
        sampleRate = self._optoOutput.timing.samp_clk_rate
        nSamples = int((dur + onRamp + offRamp) * sampleRate) + 1
        if nSamples < 2:
            nSamples = 2
        waveform = np.zeros(nSamples)
        waveform[:-1] = amp
        waveform[-1] = lastVal
        if onRamp > 0:
            ramp = np.linspace(0,amp,int(onRamp * sampleRate))
            waveform[:ramp.size] = ramp
        if offRamp > 0:
            ramp = np.linspace(amp,0,int(offRamp * sampleRate))
            waveform[-(ramp.size+1):-1] = ramp
        return waveform


    def applyOptoWaveform(self,waveform):
        self._optoOutput.stop()
        self._optoOutput.timing.samp_quant_samp_per_chan = waveform.size
        output = np.zeros((2,waveform.size))
        output[0] = waveform
        output[1,waveform>0] = 5
        self._optoOutput.write(output,auto_start=True)
        self._optoVoltage = waveform[-1]


    def getGalvoPositionWaveform(self,x=None,y=None):
        if x is None:
            x = self._galvoVoltage[0]
        if y is None:
            y = self._galvoVoltage[1]
        waveform = np.zeros((2,2))
        waveform[0] = x
        waveform[1] = y
        return waveform
    
    
    def setGalvoPosition(self,x,y):
        waveform = self.getGalvoPositionWaveform(x,y)
        self.applyGalvoWaveform(waveform)


    def applyGalvoWaveform(self,waveform):
        self._galvoOutput.stop()
        self._galvoOutput.timing.samp_quant_samp_per_chan = waveform.shape[1]
        self._galvoOutput.write(waveform,auto_start=True)
        self._galvoVoltage = waveform[:,-1]

    
        
class WaterTest(TaskControl):
                
    def __init__(self,params,openTime=None,numPulses=100,pulseInterval=120):
        TaskControl.__init__(self,params)
        self.saveParams = False
        if openTime is not None:
            self.solenoidOpenTime = openTime
        self.numPulses = numPulses
        self.pulseInterval = pulseInterval
              
    def taskFlow(self):
        while self._continueSession:
            if self._sessionFrame > 0 and not self._sessionFrame % self.pulseInterval:
                if len(self.rewardFrames) < self.numPulses:
                    self._reward = self.solenoidOpenTime
                else:
                    self._continueSession = False
            self.showFrame()
            


class LuminanceTest(TaskControl):
                
    def __init__(self,params,levels=None,framesPerLevel=300):
        TaskControl.__init__(self,params)
        self.saveParams = False
        self.levels = np.arange(-1,1.1,0.25) if levels is None else levels
        self.framesPerLevel = framesPerLevel
              
    def taskFlow(self):
        i = 0
        while self._continueSession:
            if not self._sessionFrame % self.framesPerLevel:
                if i < len(self.levels):
                    self._win.color = self.levels[i]
                else:
                    self._continueSession = False
                i += 1
            self.showFrame()



class LickTest(TaskControl):
                
    def __init__(self,params):
        TaskControl.__init__(self,params)
              
    def taskFlow(self):

        for _ in range(5):
            task.triggerRewardSound()
            time.sleep(1)

        while self._continueSession:
            self.getInputData()

            if self._lick:
                task.triggerRewardSound()
                time.sleep(0.1)

            self.showFrame()


class AccumulatorInterface:

    def __init__(
        self,
        socket_address: str,
        mouse_id: str,
        task_id: str,
        session_id: str,
        rig_id: str,
    ):
        """
        """
        self.__socket_address = socket_address
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.PUB)
        # just mirroring legacy code settings, may be unneccessary
        self.__socket.setsockopt(zmq.SNDHWM, 10)
        self.__socket.bind(self.__socket_address)
        self.__task_index = 0
        self.__header_meta = {
            "mouse_id": mouse_id,
            "task_id": task_id,
        }
        self.__packet_template = {
            'rig_name': rig_id,
            "session_id": session_id,
        }

    def publish_header(self):
        self._publish({
            **self.__packet_template,
            "init_data": self.__header_meta,
            "index": -1,
        })

    def publish_footer(self):
        self._publish({
            **self.__packet_template,
            "header": self.__header_meta,
            "init_data": self.__header_meta,
            "index": -2,
        })

    def publish(self, **values):
        self._publish({
            **self.__packet_template,
            "index": self.__task_index,
            **values,
        })
        self.__task_index += 1

    def _publish(self, packet: dict):
        # no idea why...supposedly this is debouncing something...
        time.sleep(0.5)
        timestamped_packet = {
            # this is expected datetime format
            "publish_time": str(datetime.datetime.now()),
            **packet,
        }
        self.__socket.send_pyobj(timestamped_packet)


def measureSound(params,soundVol,soundDur,soundInterval,nidaqDevName):
    
    soundVol = np.array(soundVol)

    task = TaskControl(params)
    task.initSound()

    digitalOut = nidaqmx.Task()
    digitalOut.do_channels.add_do_chan(nidaqDevName+'/port0/line0',line_grouping=nidaqmx.constants.LineGrouping.CHAN_PER_LINE)
    digitalOut.write(False)

    from nidaqmx.stream_readers import AnalogMultiChannelReader
    analogIn = nidaqmx.Task()
    analogInChannels = 3
    analogInSampleRate = 5000
    analogInBufferSize = 500
    analogIn.ai_channels.add_ai_voltage_chan(nidaqDevName+'/ai0:'+str(analogInChannels-1),
                                             terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
                                             min_val=-5,
                                             max_val=5)
    analogIn.timing.cfg_samp_clk_timing(analogInSampleRate,
                                        sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                                        samps_per_chan=analogInBufferSize)
    analogInReader = AnalogMultiChannelReader(analogIn.in_stream)
    analogInData = np.zeros((analogInChannels,analogInBufferSize))

    startTime = time.strftime('%Y%m%d_%H%M%S',time.localtime())
    savePath = os.path.join(task.saveDir,'sound','soundMeasure_' + params['rigName'] +'_' + startTime)
    h5File = h5py.File(savePath+'.hdf5','a',libver='latest')
    h5Dataset = h5File.create_dataset('AnalogInput',
                                      (0,analogInChannels),
                                      maxshape=(None,analogInChannels),
                                      dtype=np.float64,
                                      chunks=(analogInBufferSize,analogInChannels),
                                      compression='gzip',
                                      compression_opts=1)
    h5Dataset.attrs.create('channel names',('sound on','sound','SPL dB'))
    h5Dataset.attrs.create('sample rate',analogInSampleRate)
    h5Dataset.attrs.create('volume',soundVol)

    def readAnalogData(task_handle,every_n_samples_event_type,number_of_samples,callback_data):
        analogInReader.read_many_sample(analogInData,number_of_samples_per_channel=number_of_samples)
        analogInData[2] *= 100 # 10 mV / dB
        h5Dataset.resize(h5Dataset.shape[0]+number_of_samples,axis=0)
        h5Dataset[-number_of_samples:] = analogInData.T
        return 0

    analogIn.register_every_n_samples_acquired_into_buffer_event(analogInBufferSize,readAnalogData) 
    analogIn.start()
    time.sleep(1)
    
    for vol in soundVol:
        soundArray = task.makeSoundArray(soundType='noise',dur=soundDur,vol=vol,freq=[2000,20000])
        digitalOut.write(True)
        task.playSound(soundArray)
        time.sleep(soundDur)
        digitalOut.write(False)
        time.sleep(soundInterval)
    
    digitalOut.close()
    analogIn.close()
    
    os.environ['QT_API'] = 'pyside2'
    import matplotlib.pyplot as plt
    sampInt = 1/analogInSampleRate
    t = np.arange(sampInt,h5Dataset.shape[0]*sampInt+sampInt/2,sampInt)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,clr in enumerate('krb'):
        ax.plot(t,h5Dataset[:,i],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Time (s)')
    plt.savefig(savePath+'.png')
    
    soundOn = np.where((h5Dataset[:-1,0] < 0.5) & (h5Dataset[1:,0] > 0.5))[0] + 1
    soundOff = np.where((h5Dataset[:-1,0] > 0.5) & (h5Dataset[1:,0] < 0.5))[0] + 1
    soundLevel = np.array([np.mean(h5Dataset[offset-int(2*analogInSampleRate):offset,2]) for offset in soundOff])
    
    fitParams = None
    if np.sum(soundVol>0) > 1:
        try:
            fitFunc = lambda x,a,b,c: a * (1 - np.exp(x*b)) + c
            fitParams = scipy.optimize.curve_fit(fitFunc,soundVol[soundVol>0],soundLevel[soundVol>0])[0]
            fitX = np.arange(0,1.01,0.01)
        except:
            pass
        
    with open(savePath+'_sound_level.txt','w') as f:
        f.write('Volume' + '\t' + 'SPL (dB)' + '\n')
        for vol,spl in zip(soundVol,soundLevel):
            f.write(str(vol) + '\t' + str(spl) + '\n')
        if fitParams is not None:
            f.write('\nFit params: dB = a * (1 - exp(volume * b) + c\n')
            for param in fitParams:
                f.write(str(param) + '\n')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(soundVol,soundLevel,'ko')
    if fitParams is not None:
        ax.plot(fitX,fitFunc(fitX,*fitParams),'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Volume')
    ax.set_ylabel('SPL (dB)')
    if fitParams is None:
        ax.set_title('no fit')
    else:
        ax.set_title('dB = ' + str(round(fitParams[0],2)) + 
                     ' * (1 - exp(volume * ' + str(round(fitParams[1],3)) +')) + ' + 
                     str(round(fitParams[2],2)))
    plt.savefig(savePath+'_sound_level.png')
    
    t = np.arange(0,0.12,sampInt) * 1000
    fig = plt.figure(figsize=(5,8))
    for i,(onset,vol) in enumerate(zip(soundOn,soundVol)):
        ax = fig.add_subplot(len(soundOn),1,i+1)
        ax.plot(t,h5Dataset[onset:onset+t.size,1],'k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_title('Volume = ' + str(vol))
    ax.set_xlabel('Time from sound trigger (ms)')
    plt.tight_layout()
    plt.savefig(savePath+'_sound_latency.png')
            
    h5File.close()


def saveParameters(group,paramDict):
    for key,val in paramDict.items():
        if key[0] != '_':
            if isinstance(val,dict):
                saveParameters(group.create_group(key),val)
            else:
                if val is None:
                    val = np.nan
                try:
                    if isStringSequence(val):
                        if not all(isinstance(v,str) for v in val):
                            for i,v in enumerate(val):
                                val[i] = str(v)
                        group.create_dataset(key,data=np.array(val,dtype=object),dtype=h5py.special_dtype(vlen=str)) 
                    elif (isinstance(val,(list,tuple,np.ndarray)) and len(val) > 0 and
                          all(isinstance(d,(list,tuple,np.ndarray)) for d in val) and [len(d) for d in val].count(len(val[0])) != len(val)):
                        group.create_dataset(key,data=np.array(val,dtype=object),dtype=h5py.special_dtype(vlen=float))
                    else:
                        group.create_dataset(key,data=val)
                except:
                    print('\n' + 'could not save ' + key)            


def isStringSequence(obj):
    if (isinstance(obj,(tuple,list,np.ndarray)) and len(obj) > 0 and
        all((isinstance(d,str) or isStringSequence(d)) for d in obj)):
        return True
    else:
        return False          


                    
if __name__ == "__main__":
    import uuid

    paramsPath = sys.argv[1]
    with open(paramsPath,'r') as f:
        params = json.load(f)
    if params['taskVersion'] == 'open solenoid':
        task = TaskControl(params)
        task.openSolenoid()
    elif params['taskVersion'] == 'close solenoid':
        task = TaskControl(params)
        task.closeSolenoid()
    elif params['taskVersion'] == 'water test':
        task = WaterTest(params)
        task.start()
    elif params['taskVersion'] == 'luminance test':
        task = LuminanceTest(params)
        task.start()
    elif params['taskVersion'] == 'reward test':
        task = TaskControl(params)
        task._nidaqTasks = []
        task.startNidaqDevice()
        for _ in range(5):
            task.triggerReward(task.solenoidOpenTime)
            task.triggerRewardSound()
            time.sleep(1)
        task.stopNidaqDevice()
    elif params['taskVersion'] == 'lick test':
        task = LickTest(params)
        task.maxFrames = 600
        task.start()
    elif params['taskVersion'] == 'sound test':
        task = TaskControl(params)
        task.soundMode == 'sound card' # 'sound card' or 'daq'
        task._nidaqTasks = []
        task.initSound()
        soundDur = 4
        soundLevel = 68 # dB
        soundVol = 0.08 if task.soundCalibrationFit is None else task.dBToVol(soundLevel,*task.soundCalibrationFit)
        soundArray = task.makeSoundArray(soundType='noise',dur=soundDur,vol=soundVol,freq=[2000,20000])
        #soundArray = task.makeSoundArray(soundType='AMnoise',dur=soundDur,vol=soundVol,freq=[2000,20000],AM=10)
        task.playSound(soundArray)
        time.sleep(soundDur+1)
        task.stopNidaqDevice()
    elif params['taskVersion'] == 'sound measure':
        nidaqDevName = 'Dev2'
        #soundVol = [0.5]
        soundVol = [0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1]
        soundDur = 5
        soundInterval = 5
        measureSound(params,soundVol,soundDur,soundInterval,nidaqDevName)
    elif params['taskVersion'] == 'opto test':
        task = TaskControl(params)
        task._nidaqTasks = []
        task.initOpto()
        x,y,amp,dur = [float(params[key]) for key in ('galvoX','galvoY','optoAmp','optoDur')]
        task.setGalvoPosition(x,y)
        task.applyOptoWaveform(task.getOptoPulseWaveform(amp,dur))
        time.sleep(dur + 0.5)
        task.stopNidaqDevice()
    elif params['taskVersion'] == 'accumulator_test':
        ai = AccumulatorInterface(
            session_id=str(uuid.uuid4()),
            **params["accumulator"],
        )
        ai.publish_header()
        cumulative_volume = 0.0
        initial_time = time.time()
        for _ in range(10):
            ai.publish(
                starttime=time.time() - initial_time,
                cumulative_volume=cumulative_volume,
            )
            cumulative_volume += 0.05
            time.sleep(5)
        ai.publish_header()
    else:
        task = TaskControl(params)
        task.saveParams = False
        task.maxFrames = 60 * 3600
        task.start()