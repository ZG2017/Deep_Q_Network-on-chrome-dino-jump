from DQN import DeepQNetwork
import time
import threading
from KeyEvents import PressKey,ReleaseKey  
import os
import sys
import pygame
import random
import cv2
import numpy as np

# global setting
lock = threading.Lock()
s = None
r = None
done = False

# env_start------------------------------------------------------------------------------------
# original_game_author: Shivam Shekhar
# game_updated: Ge Zhang (John)
pygame.init()
width = 600
height = 150
FPS = 60
gravity = 0.6 
background_col = (255,255,255) # (255,255,255)
screen = pygame.display.set_mode((600,150))

def Path2Res():
    return os.path.join(os.path.dirname(__file__),"sprites")

def load_image(
    name,
    sizex=-1,
    sizey=-1,
    colorkey=None,
    ):
    s_dir = Path2Res()
    fullname = os.path.join(s_dir, name)
    image = pygame.image.load(fullname)
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pygame.RLEACCEL)

    if sizex != -1 or sizey != -1:
        image = pygame.transform.scale(image, (sizex, sizey))

    return (image, image.get_rect())

def load_sprite_sheet(
        sheetname,
        nx,
        ny,
        scalex = -1,
        scaley = -1,
        colorkey = None,
        ):
    s_dir = Path2Res()
    fullname = os.path.join(s_dir,sheetname)
    sheet = pygame.image.load(fullname)
    sheet = sheet.convert()

    sheet_rect = sheet.get_rect()

    sprites = []

    sizex = sheet_rect.width/nx
    sizey = sheet_rect.height/ny

    for i in range(0,ny):
        for j in range(0,nx):
            rect = pygame.Rect((j*sizex,i*sizey,sizex,sizey))
            image = pygame.Surface(rect.size)
            image = image.convert()
            image.blit(sheet,(0,0),rect)

            if colorkey is not None:
                if colorkey is -1:
                    colorkey = image.get_at((0,0))
                image.set_colorkey(colorkey,pygame.RLEACCEL)

            if scalex != -1 or scaley != -1:
                image = pygame.transform.scale(image,(scalex,scaley))

            sprites.append(image)

    sprite_rect = sprites[0].get_rect()

    return sprites,sprite_rect

"""
def disp_gameOver_msg(retbutton_image,gameover_image):
    retbutton_rect = retbutton_image.get_rect()
    retbutton_rect.centerx = width / 2
    retbutton_rect.top = height*0.52

    gameover_rect = gameover_image.get_rect()
    gameover_rect.centerx = width / 2
    gameover_rect.centery = height*0.35

    screen.blit(retbutton_image, retbutton_rect)
    screen.blit(gameover_image, gameover_rect)
"""

def extractDigits(number):
    if number > -1:
        digits = []
        while(number/10 != 0):
            digits.append(number%10)
            number = int(number/10)

        digits.append(number%10)
        for i in range(len(digits),5):
            digits.append(0)
        digits.reverse()
        return digits

class Dino():
    def __init__(self,sizex=-1,sizey=-1):
        self.images,self.rect = load_sprite_sheet('dino.png',5,1,sizex,sizey,-1)
        self.images1,self.rect1 = load_sprite_sheet('dino_ducking.png',2,1,59,sizey,-1)
        self.rect.bottom = int(0.98*height)
        self.rect.left = width/15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0,0]
        self.jumpSpeed = 11.5
        self.s_dir = Path2Res()
        self.checkPoint_sound = pygame.mixer.Sound(os.path.join(self.s_dir,'checkPoint.wav'))

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        screen.blit(self.image,self.rect)

    def checkbounds(self):
        if self.rect.bottom > int(0.98*height):
            self.rect.bottom = int(0.98*height)
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1)%2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1)%2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2 + 2

        if self.isDead:
           self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[(self.index)%2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and self.isBlinking == False:
            self.score += 1
            if self.score % 100 == 0 and self.score != 0:
                if pygame.mixer.get_init() != None:
                    self.checkPoint_sound.play()

        self.counter = (self.counter + 1)

class Cactus(pygame.sprite.Sprite):
    def __init__(self,speed=5,sizex=-1,sizey=-1):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.images,self.rect = load_sprite_sheet('cacti-small.png',3,1,sizex,sizey,-1)
        self.rect.bottom = int(0.98*height)
        self.rect.left = width + self.rect.width
        self.image = self.images[random.randrange(0,3)]
        self.movement = [-1*speed,0]

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right < 0:
            self.kill()

# temporarily dont need Ptera
"""
class Ptera(pygame.sprite.Sprite):
    def __init__(self,speed=5,sizex=-1,sizey=-1):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.images,self.rect = load_sprite_sheet('ptera.png',2,1,sizex,sizey,-1)
        self.ptera_height = [height*0.82,height*0.75,height*0.60]
        self.rect.centery = self.ptera_height[random.randrange(0,3)]
        self.rect.left = width + self.rect.width
        self.image = self.images[0]
        self.movement = [-1*speed,0]
        self.index = 0
        self.counter = 0

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index+1)%2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()
"""

class Ground():
    def __init__(self,speed=-5):
        self.image,self.rect = load_image('ground.png',-1,-1,-1)
        self.image1,self.rect1 = load_image('ground.png',-1,-1,-1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen.blit(self.image,self.rect)
        screen.blit(self.image1,self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right

class Cloud(pygame.sprite.Sprite):
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.image,self.rect = load_image('cloud.png',int(90*30/42),30,-1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1*self.speed,0]

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()

class Scoreboard():
    def __init__(self,x=-1,y=-1):
        self.tempimages,self.temprect = load_sprite_sheet('numbers.png',12,1,11,int(11*6/5),-1)
        self.image = pygame.Surface((55,int(11*6/5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = width*0.89
        else:
            self.rect.left = x
        if y == -1:
            self.rect.top = height*0.1
        else:
            self.rect.top = y

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self,score):
        score_digits = extractDigits(score)
        self.image.fill(background_col)
        for s in score_digits:
            self.image.blit(self.tempimages[s],self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0

class T_res():
    def __init__(self,fps,section):
        self.FPS = fps
        self.section = section

    def GetImage(self,zone,image_threshold):
        image = np.transpose(pygame.surfarray.array3d(zone),(1,0,2))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)               
        _,image = cv2.threshold(image,image_threshold,255,cv2.THRESH_BINARY)
        return image

    def GetFeedback(self,track,number_sections,rate):
        sections = np.array(np.split(track,number_sections,axis=1))
        value = int(sections.shape[1]*sections.shape[2])
        sections = sections.reshape(number_sections,value)
        binary_pre = np.count_nonzero(sections,axis=1).reshape(1,number_sections)
        binary = (binary_pre<int(value*(1-rate))).astype(np.uint8)
        return binary    
        
    def GameRun(self):
        global s,r,done,lock
        self.agent = Dino(44,47)
        clock = pygame.time.Clock()
        pygame.display.set_caption("T-Rex Runner")
        s_dir = Path2Res()
        jump_sound = pygame.mixer.Sound(os.path.join(s_dir,'jump.wav'))
        die_sound = pygame.mixer.Sound(os.path.join(s_dir,'die.wav'))

        high_score = 0
        gamespeed = 5
        gameOver = False
        gameQuit = False
        speed_counter = 0
        new_ground = Ground(-1*gamespeed)
        scb = Scoreboard()
        highsc = Scoreboard(width*0.78)
        
        # -----------------------------------------------------------
        # training setting
        # change here to change the region of target zone
        image_pixel_width = 400
        # -----------------------------------------------------------

        rect = pygame.Rect(100, 110, 100+image_pixel_width, 25)
        target = screen.subsurface(rect)

        cacti = pygame.sprite.Group()
        #pteras = pygame.sprite.Group()
        clouds = pygame.sprite.Group()
        last_obstacle = pygame.sprite.Group()

        Cactus.containers = cacti
        #Ptera.containers = pteras
        Cloud.containers = clouds

        #retbutton_image,_ = load_image('replay_button.png',35,31,-1)
        #gameover_image,_ = load_image('game_over.png',190,11,-1)

        temp_images,temp_rect = load_sprite_sheet('numbers.png',12,1,11,int(11*6/5),-1)
        HI_image = pygame.Surface((22,int(11*6/5)))
        HI_rect = HI_image.get_rect()
        HI_image.fill(background_col)
        HI_image.blit(temp_images[10],temp_rect)
        temp_rect.left += temp_rect.width
        HI_image.blit(temp_images[11],temp_rect)
        HI_rect.top = height*0.1
        HI_rect.left = width*0.73

        while not gameQuit:
            while not gameOver:
                lock.acquire()
                if pygame.display.get_surface() == None:
                    print("Couldn't load display surface")
                    gameQuit = True
                    gameOver = True
                else:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            gameQuit = True
                            gameOver = True

                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                if self.agent.rect.bottom == int(0.98*height):
                                    self.agent.isJumping = True
                                    if pygame.mixer.get_init() != None:
                                        jump_sound.play()
                                    self.agent.movement[1] = -1*self.agent.jumpSpeed

                        """
                            if event.key == pygame.K_DOWN:
                                if not (self.agent.isJumping and self.agent.isDead):
                                    self.agent.isDucking = True
                        
                        if event.type == pygame.KEYUP:
                            if event.key == pygame.K_DOWN:
                                self.agent.isDucking = False
                        """
                for c in cacti:
                    c.movement[0] = -1*gamespeed
                    if pygame.sprite.collide_mask(self.agent,c):
                        self.agent.isDead = True
                        if pygame.mixer.get_init() != None:
                            die_sound.play()
                """
                for p in pteras:
                    p.movement[0] = -1*gamespeed
                    if pygame.sprite.collide_mask(self.agent,p):
                        self.agent.isDead = True
                        if pygame.mixer.get_init() != None:
                            die_sound.play()
                """
                if len(cacti) < 2:
                    if len(cacti) == 0:
                        last_obstacle.empty()
                        last_obstacle.add(Cactus(gamespeed,40,40))
                    else:
                        for l in last_obstacle:
                            if l.rect.right < width*0.8 and random.randrange(0,15) == 14:
                                last_obstacle.empty()
                                last_obstacle.add(Cactus(gamespeed, 40, 40))
                """
                if len(pteras) == 0 and random.randrange(0,200) == 10 and speed_counter > 500:
                    for l in last_obstacle:
                        if l.rect.right < width*0.8:
                            last_obstacle.empty()
                            last_obstacle.add(Ptera(gamespeed, 46, 40))
                """
                if len(clouds) < 5 and random.randrange(0,300) == 10:
                    Cloud(width,random.randrange(height/5,height/2))

                self.agent.update()
                cacti.update()
                #pteras.update()
                clouds.update()
                new_ground.update()
                scb.update(self.agent.score)
                highsc.update(high_score)

                if pygame.display.get_surface() != None:
                    screen.fill(background_col)
                    new_ground.draw()
                    clouds.draw(screen)
                    scb.draw()
                    if high_score != 0:
                        highsc.draw()
                        screen.blit(HI_image,HI_rect)
                    cacti.draw(screen)
                    #pteras.draw(screen)
                    self.agent.draw()

                    pygame.display.update()
                clock.tick(self.FPS)

                if self.agent.isDead:
                    gameOver = True
                    if self.agent.score > high_score:
                        high_score = self.agent.score

                if speed_counter%1000 == 999:
                    new_ground.speed -= 0.5
                    gamespeed += 0.5

                speed_counter += 1
                image = self.GetImage(target,170)
                s = self.GetFeedback(image,self.section,0.05)
                r = self.agent.score
                done = gameOver
                lock.release()

            if gameQuit:
                break

            time.sleep(0.8)             
            lock.acquire()
            if gameOver:
                if pygame.display.get_surface() == None:
                    print("Couldn't load display surface")
                    gameQuit = True
                    gameOver = False
                else:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            gameQuit = True
                            gameOver = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                gameQuit = True
                                gameOver = False

                    gameOver = False
                    gamespeed = 5
                    self.agent = Dino(44,47)
                    new_ground = Ground(-1*gamespeed)
                    speed_counter = 0

                    cacti = pygame.sprite.Group()
                    #pteras = pygame.sprite.Group()
                    clouds = pygame.sprite.Group()
                    last_obstacle = pygame.sprite.Group()

                    Cactus.containers = cacti
                    #Ptera.containers = pteras
                    Cloud.containers = clouds

                highsc.update(high_score)
                if pygame.display.get_surface() != None:
                    #disp_gameOver_msg(retbutton_image,gameover_image)
                    if high_score != 0:
                        highsc.draw()
                        screen.blit(HI_image,HI_rect)
                    pygame.display.update()
                clock.tick(self.FPS)
            lock.release()
        pygame.quit()

# env_end-----------------------------------------------------------------------------------------


def OnlineTraining(model,max_epoch,max_step,sleep_time,start_learning,save_path):
    global s,r,done,lock
    state = None
    state_ = None
    reward = None
    time.sleep(1)
    total_step = 0
    for j in range(max_epoch):
        while done:
            continue
        state = s
        for i in range(max_step):
            a = model.choose_action(state)
            if a == 1:
                PressKey(0x20)
                time.sleep(sleep_time)
                ReleaseKey(0x20)
            else:
                time.sleep(0.1)
            state_ = s
            if done:          
                reward = -r/2
            else:
                reward = r/8                        
            #print(state,state_,a,reward)
            model.save_memory(state,state_,a,reward)
            if done:
                print("epoch {0} done!_____step:{1}______current_epsilon:{2}".format(j,i,model.epsilon_init))
                break
            if model.memory_counter >= start_learning:
                model.learning()
            state = state_  
            total_step += 1            
            if total_step%300 == 0:
                model.SaveModel(j,save_path)
    #model.plot()
    print("Done!")

def ModelTest(model,max_epoch,max_step,sleep_time):
    global s,r,done
    state = None
    state_ = None
    a = None
    time.sleep(1)
    for j in range(max_epoch):
        while done:
            continue
        state = s
        for i in range(max_step):
            a = model.choose_action(state)
            if a == 1:
                PressKey(0x20)
                time.sleep(sleep_time)
                ReleaseKey(0x20)
            else:
                time.sleep(0.1)
            state_ = s
            #print(state,state_,a)
            state = state_  
            if done:
                break
    #model.plot()
    print("Done!")


def Training():
    global s,r,done
    max_epoch = 250
    max_step = 400
    lr = 0.012 
    epsilon = 0.90
    epsilon_incr = 0.002  
    gamma = 0.95
    sections = 20                                         
    weight = [0.8,0.2]
    layer_info = [120,30]                                  
    fps_plus_sleep_time = [60, 0.7] # (60fps, 0.7) (10fps,3.8) (100,0.45)
    start_learning = 400
    save_path = os.path.dirname(os.path.realpath(__file__))+'\\good_model\\6\\'

    env = T_res(fps_plus_sleep_time[0],sections)
    DQN_model = DeepQNetwork(layer_info = layer_info,
                             lr = lr,
                             epsilon_max = epsilon,
                             epsilon_increase = epsilon_incr,
                             gamma = gamma,                     
                             number_of_states = sections, 
                             number_of_actions= 2,
                             weight = weight,
                             replace_steps = 70,          
                             memory_size = 2500,   
                             batch_size = 128)
    DQN_model.build_net()
    T1 = threading.Thread(target = OnlineTraining, args=(DQN_model,max_epoch,max_step,fps_plus_sleep_time[1],start_learning,save_path))
    T1.daemon = True
    T1.start()  
    env.GameRun()  

def Test():
    global s,r,done
    max_epoch = 5
    max_step = 400
    lr = 0.012
    epsilon = 0.90
    epsilon_incr = 0.002
    gamma = 0.95
    sections = 20                                      
    weight = [0.8,0.2]
    layer_info = [80,20]
    fps_plus_sleep_time = [60,0.7] # (60fps, 0.7) (10fps,3.8)

    env = T_res(fps_plus_sleep_time[0],sections)

    # just for init, number won't matter for testing
    DQN_model = DeepQNetwork(layer_info = layer_info,
                             lr = lr,
                             epsilon_max = epsilon,
                             epsilon_increase = epsilon_incr,
                             gamma = gamma,
                             number_of_states = sections,
                             number_of_actions= 2,
                             weight = weight,
                             replace_steps = 70,
                             memory_size = 2500,
                             batch_size = 128)
    DQN_model.build_net()
    DQN_model.epsilon_init = 1.0
    dir_path = os.path.dirname(os.path.realpath(__file__))
    DQN_model.LoadModel(dir_path+"\\good_model\\2\\my_model-77")
    T1 = threading.Thread(target = ModelTest, args=(DQN_model,max_epoch,max_step,fps_plus_sleep_time[1]))
    T1.daemon = True
    T1.start()
    env.GameRun()  


if __name__ == "__main__":
    #Training() 
    Test()