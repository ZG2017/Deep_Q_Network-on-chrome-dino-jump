import time
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys
import cv2
import io
from PIL import Image
import numpy as np

def PIL2array(img):
    return np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 4)

def GetOneHotInput(Driver,ImageThreshold,OneHotRate,SectionNumber):
    init = time.time()
    tmp = driver.get_screenshot_as_png()
    print(time.time()-init)
    tmp = np.fromstring(tmp,np.uint8)
    tmp = cv2.imdecode(tmp,cv2.IMREAD_COLOR)[65:265,445:1245]
    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
    _,tmp = cv2.threshold(tmp,ImageThreshold,255,cv2.THRESH_BINARY)
    tmp = tmp[105:200,0:800]
    sections = np.array(np.split(tmp,SectionNumber,axis=1))
    value = int(76000/SectionNumber)
    sections = sections.reshape(SectionNumber,value)
    one_hot_pre = np.count_nonzero(sections,axis=1)
    one_hot = (one_hot_pre<int(value*(1-OneHotRate))).astype(np.uint8)
    return tmp,one_hot

def GetImage(handler):
    tmp = handler.screenshot_as_png
    tmp = np.fromstring(tmp,np.uint8)
    tmp = cv2.imdecode(tmp,cv2.IMREAD_COLOR)
    return tmp


driver = webdriver.Chrome("D:\setup\chromedriver.exe") 
driver.get('http://wayou.github.io/t-rex-runner/')  # link to github
runner = driver.find_element_by_id("t")
zone = driver.find_element_by_class_name("runner-canvas")
runner.send_keys(Keys.SPACE)
time.sleep(6.5)
init = time.time()
image = GetImage(zone)
print(time.time()-init)
game_zone,one_hot = GetOneHotInput(driver,170,0.1,20)
cv2.imshow("test",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
driver.quit()

# whole game zone(without dino):[65:265,445:1245]
# just track(without dino):[105:200,0:800]
# just score:[0:20,675:770]
# is over(game over):[45:75,185:475]
# is over(only letter "g"):[51:68,189:207] (17*18)
# is over(only letter "m"):[51:68,260:278]
# is over(only letter "o"):[51:68,350:368]
# is over(only letter "e"):[51:68,421:439]
