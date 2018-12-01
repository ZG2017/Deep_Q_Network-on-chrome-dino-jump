from selenium.webdriver.common.keys import Keys
import cv2
import io
import time
from PIL import Image
import numpy as np

class Dino:
    def __init__(self,webdriver):
        self.web_driver = webdriver

    def jump(self):
        self.web_driver.handler.send_keys(Keys.SPACE)
        time.sleep(0.355)
