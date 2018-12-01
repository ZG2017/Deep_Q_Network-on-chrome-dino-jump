import time
import cv2
from PIL import Image
import numpy as np
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys

# region information base on http://wayou.github.io/t-rex-runner/
# whole game zone(without dino):[15:215,445:1245]

class WebDriver():
    def __init__(self,Location2driver,URL2Runner,id):
        self.driver = webdriver.Chrome(Location2driver)
        self.driver.get(URL2Runner)
        self.handler = self.driver.find_element_by_id(id)

    def GetByteImage(self):
        return self.driver.get_screenshot_as_png()
    
    def Quit(self):
        self.driver.quit()

class Runner_Env():
    def __init__(self,webdriver,NumberOfStates):
        self.NumberOfState = NumberOfStates
        self.web_driver = webdriver

    def GetImage(self,ImageThreshold):
        init = time.time()
        tmp = self.web_driver.GetByteImage()
        tmp = np.fromstring(tmp,np.uint8)
        tmp = cv2.imdecode(tmp,cv2.IMREAD_COLOR)[15:215,445:1245]  # whole image
        tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
        _,tmp = cv2.threshold(tmp,ImageThreshold,255,cv2.THRESH_BINARY)
        return tmp
    
    def IsDone(self,image,ThresholdRate):
        end_image = image[69:86,177:194]   # letter "G"
        return np.count_nonzero(end_image)<306*(1-ThresholdRate)
    
    def FeedbackBinaryCoding(self,Image,OneHotRate):
        track = Image[150:180,0:336]
        sections = np.array(np.split(track,self.NumberOfState,axis=1))
        value = int(sections.shape[1]*sections.shape[2])
        sections = sections.reshape(self.NumberOfState,value)
        binary_pre = np.count_nonzero(sections,axis=1).reshape(1,self.NumberOfState)
        binary = (binary_pre<int(value*(1-OneHotRate))).astype(np.uint8)
        return binary,track        

    # didnt use this method for now
    def FeedbackOneHot(self,Image,OneHotRate):
        track = Image[150:180,0:336]
        sections = np.array(np.split(track,self.NumberOfState,axis=1))
        value = int(sections.shape[1]*sections.shape[2])
        sections = sections.reshape(self.NumberOfState,value)
        one_hot = np.zeros((1,2**self.NumberOfState),dtype = np.float)
        one_hot_pow = np.arange(self.NumberOfState).reshape(1,self.NumberOfState)
        one_hot_pre = np.count_nonzero(sections,axis=1).reshape(1,self.NumberOfState)
        one_hot_pre = (one_hot_pre<int(value*(1-OneHotRate))).astype(np.float)*2
        index = np.sum(np.power(one_hot_pre,one_hot_pow))
        one_hot[0,int(index-1)] = 1.0
        return one_hot,track
    
    def Start(self):
        self.web_driver.handler.send_keys(Keys.SPACE)

    def Wait(self,Time):
        time.sleep(Time)

    def GetTime(self):
        return time.time()

    def EnvDestroy(self):
        self.web_driver.Quit()