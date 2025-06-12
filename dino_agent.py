from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time

class Dino:
    def __init__(self, webdriver, jump_duration=0.355):
        self.web_driver = webdriver
        self.jump_duration = jump_duration
        self.actions = ActionChains(self.web_driver.driver)

    def jump(self):
        self.actions.key_down(Keys.ARROW_UP).perform()
        time.sleep(self.jump_duration)
        self.actions.key_up(Keys.ARROW_UP).perform()
