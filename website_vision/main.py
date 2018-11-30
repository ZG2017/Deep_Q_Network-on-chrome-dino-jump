import time
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys
import cv2
import numpy as np
import os
import math
import tensorflow as tf
import win_unicode_console
win_unicode_console.enable()
from DQN import DeepQNetwork
from env import Runner_Env,WebDriver
from dino_agent import Dino

# learning parameters
lr = 0.01 # 0.012
epsilon = 0.99
gamma = 0.95

WDforDinoRunner = WebDriver("D:\setup\chromedriver.exe",'http://wayou.github.io/t-rex-runner/',"t")
environment = Runner_Env(WDforDinoRunner,12)
runner = Dino(WDforDinoRunner)
DQN_model = DeepQNetwork(lr = lr,
                        epsilon_max = epsilon,
                        epsilon_increase =  0.005,
                        gamma = gamma,
                        number_of_states = environment.NumberOfState,
                        number_of_actions= 2,
                        replace_steps = 70,
                        memory_size = 1500,
                        batch_size = 32)
DQN_model.build_net()


def OnlineTraining_1(model,agent,env,max_epoch,max_step):
    total_step = 0
    for j in range(max_epoch):
        env.Start()
        env.Wait(4)
        image = env.GetImage(170)
        s,_ = env.FeedbackBinaryCoding(image,0.05)
        r = 0
        time_stamp_1 = env.GetTime()
        for i in range(max_step):
            a = model.choose_action(s)
            if a == 1:
                agent.jump()
            image = env.GetImage(170)
            s_,_ = env.FeedbackBinaryCoding(image,0.05)
            done = env.IsDone(image,0.2)
            if done:
                r = -15
            else:
                r = (env.GetTime()-time_stamp_1)*3
            model.save_memory(s,s_,a,r)
            if done:
                print("epoch {0} done!_____step:{1}______current_epsilon:{2}".format(j,i,DQN_model.epsilon_init))
                break
            if model.memory_counter >= 70:
                model.learning()
            s = s_
            total_step += 1
            if total_step%150 == 0:
                model.SaveModel(j)
        env.Wait(1)
    DQN_model.plot()
    env.EnvDestroy()
    print("Done!")

def Image_Test(model,agent,env):
    env.Start()
    env.Wait(6.5)
    image = env.GetImage(170)
    s,test_image = env.FeedbackBinaryCoding(image,0.05)
    print(s)
    cv2.imshow("test",test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    env.EnvDestroy()
    print("Done!")

def Model_Test(model,agent,env,max_epoch,model_path):
    model.LoadModel(model_path)
    model.epsilon_init = 1
    for j in range(max_epoch):
        env.Start()
        env.Wait(3.5)
        image = env.GetImage(170)
        s,_ = env.FeedbackBinaryCoding(image,0.08)
        while True:
            a = model.choose_action(s)
            if a == 1:
                agent.jump()
            image = env.GetImage(170)
            s_,_ = env.FeedbackBinaryCoding(image,0.08)
            done = env.IsDone(image,0.2)
            if done: break
            s = s_
        env.Wait(1)
        print("epoch {0} done!".format(j))
    env.EnvDestroy()

def main():
    #OnlineTraining_1(DQN_model,runner,environment,300,70)
    #Image_Test(DQN_model,runner,environment)
    Model_Test(DQN_model,runner,environment,5,"C:\\Users\\81466\\Desktop\\DQN_Dino\\website_vision\\DQN_GOOD_section_equal_6\\model_9\\My_DQN_Model-129")

main()