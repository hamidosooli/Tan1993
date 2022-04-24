'''
This is the gridworld_template.py file that implements
a 1D gridworld and is part of the mid-term project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 10/19/2021
'''
import numpy as np
import pygame as pg
import time


'''If you need to use the big environment make BIG_ENV = True'''
BIG_ENV = False

# Constants
WIDTH = 800  # width of the environment (px)
HEIGHT = 800  # height of the environment (px)
TS = 10  # delay in msec
if BIG_ENV:
    Col_num = 50  # number of columns
    Row_num = 50  # number of rows
else:
    Col_num = 10  # number of columns
    Row_num = 10  # number of rows

# define colors
bg_color = pg.Color(255, 255, 255)
line_color = pg.Color(128, 128, 128)
vfdh_color = pg.Color(136, 8, 8)
vfds_color = pg.Color(255, 165, 0)

# define primary positions
goal_pos_x1 = 3*(WIDTH//Col_num)
goal_pos_y1 = 5*(HEIGHT//Row_num)

goal_pos_x2 = 2*(WIDTH//Col_num)
goal_pos_y2 = 2*(HEIGHT//Row_num)
# eleline_pos_x = WIDTH//Col_num
# eleline_pos_y = HEIGHT//Row_num
# tree_pos_x = goal_pos_x - 2*(WIDTH//Col_num)
# tree_pos_y = 0
# pud_pos_x = (np.floor(Col_num/4))*(WIDTH//Col_num)
# pud_pos_y = (Row_num-np.floor(Row_num/4))*HEIGHT//Row_num
# mount_pos_x = (Col_num-1)*(WIDTH//Col_num)
# mount_pos_y = 0


# Define the map
def map_builder(box=True):

    map = np.zeros((Row_num, Col_num))  # electricity line

    # mountains and tree_line
    # for j in range(int(np.floor(Row_num / 2)) - 1):
    #     for i in range(int(np.floor(Col_num / 2)) - 1):
    #         if i == j:
    #             map[i, Col_num - 1 - i] = 4  # trees
    #         else:
    #             map[j, Col_num - 1 - i] = 2  # mountains

    # trees and volcanoes
    # for j in range(int(np.floor(Row_num / 4)) - 1):
    #     for i in range(int(np.floor(Col_num / 4)) - 1):
    #         if BIG_ENV:
    #             if j == 2*i+2 or j == 2*i-2 or j == 2*i+5 or j == 2*i-5:
    #                 map[j, int(np.floor(Col_num / 2)) - i - 3] = 8  # volcanoes
    #             else:
    #                 map[j, int(np.floor(Col_num / 2)) - i - 3] = 4  # trees
    #         else:
    #             map[j, int(np.floor(Col_num / 2)) - i - 3] = 4  # trees

    # puddle and diamond and flame
    # for j in range(int(np.floor(Row_num / 4)) - 1):
    #     for i in range(int(np.floor(Col_num / 4)) - 1):
    #         if BIG_ENV:
    #             if i == j and not box:
    #                 map[(Row_num - 5) - j, 5 + i] = 6  # diamond
    #             elif j == 2 * i + 1 or j == 2 * i - 1:
    #                 map[(Row_num - 5) - j, 5 + i] = 7  # flame
    #             else:
    #                 map[(Row_num - 5) - j, 5 + i] = 3  # puddle
    #         else:
    #             if i == j and not box:
    #                 map[(Row_num - 5) - j, 5 + i] = 6  # diamond
    #             else:
    #                 map[(Row_num - 5) - j, 5 + i] = 3  # puddle

    # goal
    map[0, int(np.floor(Col_num / 2)) - 1] = 5  # goal

    return map


def draw_grid(scr):
    '''a function to draw gridlines and other objects'''
    # Horizontal lines
    for j in range(Row_num+1):
        pg.draw.line(scr, line_color, (0, j*HEIGHT//Row_num), (WIDTH, j*HEIGHT//Row_num), 2)
    # # Vertical lines
    for i in range(Col_num+1):
        pg.draw.line(scr, line_color, (i*WIDTH//Col_num, 0), (i*WIDTH//Col_num, HEIGHT), 2)

    for x1 in range(0, WIDTH, WIDTH//Col_num):
        for y1 in range(0, HEIGHT, HEIGHT//Row_num):
            rect = pg.Rect(x1, y1, WIDTH//Col_num, HEIGHT//Row_num)
            pg.draw.rect(scr, bg_color, rect, 1)


class Agent:
    '''the agent class '''
    def __init__(self, scr):
        self.w = WIDTH//(Row_num)
        self.h = HEIGHT//(Col_num)
        self.x1 = 0
        self.y1 = HEIGHT - self.h
        
        self.x2 = 0
        self.y2 = HEIGHT - self.h
        self.scr = scr
        self.my_rect = pg.Rect((self.x1, self.y1), (self.w, self.h))
        self.BOX_IS_FULL = True
        self.map = map_builder(self.BOX_IS_FULL)
        self.FIRST_TIME = True

    def reward(self, loc):
        if loc[1] == goal_pos_x1 and loc[0] == goal_pos_y1 and self.BOX_IS_FULL:
            return 100  # goal reward
        if loc[1] == goal_pos_x2 and loc[0] == goal_pos_y2 and self.BOX_IS_FULL:
            return 100  # goal reward
        # elif self.map[(int(loc[0] / (HEIGHT // Row_num)), int(loc[1] / (WIDTH // Col_num)))] == 3:
        #     return -25  # puddle punishment
        # elif self.map[(int(loc[0] / (HEIGHT // Row_num)), int(loc[1] / (WIDTH // Col_num)))] == 7:
        #     return -50  # flame punishment
        # elif self.map[(int(loc[0] / (HEIGHT // Row_num)), int(loc[1] / (WIDTH // Col_num)))] == 8:
        #     return -75  # volcano punishment
        else:
            return -1  # decreasing battery level

    def show(self, color):
        self.my_rect = pg.Rect((self.x1, self.y1), (self.w, self.h))
        pg.draw.rect(self.scr, color, self.my_rect)

    def h_move_valid(self, a1, a2):
        '''checking for the validity of moves'''
        if 0 <= self.x1 + a1 < WIDTH and 0 <= self.x2 + a2 < WIDTH:
            return True
        else:
            return False

    def v_move_valid(self, a1, a2):
        if 0 <= self.y1 + a1 < HEIGHT and 0 <= self.y2 + a2 < HEIGHT:
            return True
        else:
            return False

    def h_move(self, a1, a2):
        '''move the agent'''
        if self.h_move_valid(a1, a2):
            self.x1 += a1
            self.x2 += a2
            self.show(bg_color)

    def v_move(self, a1, a2):
        '''move the agent'''
        if self.v_move_valid(a1, a2):
            self.y1 += a1
            self.y2 += a2
            self.show(bg_color)


def main():
    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH+2, HEIGHT+2))   # set up the screen
    pg.display.set_caption("Hamid Osooli")              # add a caption
    bg = pg.Surface(screen.get_size())                  # get a background surface
    bg = bg.convert()

    # img_nat = pg.image.load('nature.png')
    # img_mdf_nat = pg.transform.scale(img_nat, (WIDTH//Col_num, HEIGHT//Row_num))
    # img_pud = pg.image.load('puddle.png')
    # img_mdf_pud = pg.transform.scale(img_pud, (WIDTH//Col_num, HEIGHT//Row_num))
    # img_dmd = pg.image.load('diamond.png')
    # img_mdf_dmd = pg.transform.scale(img_dmd, (WIDTH//Col_num, HEIGHT//Row_num))
    img_quad1 = pg.image.load('quad.png')
    img_mdf_quad1 = pg.transform.scale(img_quad1, (WIDTH // Col_num, HEIGHT // Row_num))
    img_home1 = pg.image.load('home.png')
    img_mdf_home1 = pg.transform.scale(img_home1, (WIDTH // Col_num, HEIGHT // Row_num))
    img_quad2 = pg.image.load('drone.png')
    img_mdf_quad2 = pg.transform.scale(img_quad2, (WIDTH // Col_num, HEIGHT // Row_num))
    img_home2 = pg.image.load('house-icon.png')
    img_mdf_home2 = pg.transform.scale(img_home2, (WIDTH // Col_num, HEIGHT // Row_num))
    # img_mnt = pg.image.load('mountain.png')
    # img_mdf_mnt = pg.transform.scale(img_mnt, (WIDTH // Col_num, HEIGHT // Row_num))
    # img_ele = pg.image.load('electric-tower.png')
    # img_mdf_ele = pg.transform.scale(img_ele, (WIDTH // Col_num, HEIGHT // Row_num))
    # img_flm = pg.image.load('flame.png')
    # img_mdf_flm = pg.transform.scale(img_flm, (WIDTH // Col_num, HEIGHT // Row_num))
    # img_vlc = pg.image.load('volcano.png')
    # img_mdf_vlc = pg.transform.scale(img_vlc, (WIDTH // Col_num, HEIGHT // Row_num))


    bg.fill(bg_color)
    screen.blit(bg, (0,0))
    clock = pg.time.Clock()
    agent = Agent(screen)  # instantiate an agent
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
                agent.show(bg_color)
                agent.h_move(WIDTH//Col_num, WIDTH//Col_num)
                print('Reward =', agent.reward((agent.y1, agent.x1)))
            elif event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
                agent.show(bg_color)
                agent.h_move(-WIDTH//Col_num, -WIDTH//Col_num)
                print('Reward =', agent.reward((agent.y1, agent.x1)))
            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                agent.show(bg_color)
                agent.v_move(-HEIGHT//Row_num, -HEIGHT//Row_num)
                print('Reward =', agent.reward((agent.y1, agent.x1)))
            elif event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                agent.show(bg_color)
                agent.v_move(HEIGHT//Row_num, HEIGHT//Row_num)
                print('Reward =', agent.reward((agent.y1, agent.x1)))
            elif (event.type == pg.KEYDOWN and event.key == pg.K_SPACE and
                  map_builder()[int((agent.y1)/(HEIGHT//Row_num)), int((agent.x1)/(WIDTH//Col_num))] == 6):
                agent.BOX_IS_FULL = True

        screen.blit(img_mdf_home1, (goal_pos_x1, goal_pos_y1))
        screen.blit(img_mdf_home2, (goal_pos_x2, goal_pos_y2))

        # agent
        screen.blit(img_mdf_quad1, (agent.x1, agent.y1))
        screen.blit(img_mdf_quad2, (agent.x2, agent.y2))

        draw_grid(screen)
        pg.display.flip()
        pg.display.update()
        if ((agent.x1 == goal_pos_x1 and agent.y1 == goal_pos_y1) or
            (agent.x2 == goal_pos_x2 and agent.y2 == goal_pos_y2)):
            run = False
    pg.quit()


# screen = pg.display.set_mode((WIDTH+2, HEIGHT+2))   # set up the screen
# pg.display.set_caption("Hamid Osooli")              # add a caption


def animate(trajectory1, trajectory2, trajectory3,
            action_history1, action_history2, action_history3,
            hunter_vfd, scout_vfd, wait_time=0):
    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH+2, HEIGHT+2))   # set up the screen
    pg.display.set_caption("Hamid Osooli")              # add a caption
    bg = pg.Surface(screen.get_size())                  # get a background surface
    bg = bg.convert()

    img_hunter_coyote = pg.image.load('hunter_coyote')
    img_mdf_h_coyote = pg.transform.scale(img_hunter_coyote, (WIDTH // Col_num, HEIGHT // Row_num))
    img_scout_coyote = pg.image.load('scout_coyote')
    img_mdf_s_coyote = pg.transform.scale(img_scout_coyote, (WIDTH // Col_num, HEIGHT // Row_num))
    img_roadrunner = pg.image.load('roadrunner.png')
    img_mdf_roadrunner = pg.transform.scale(img_roadrunner, (WIDTH // Col_num, HEIGHT // Row_num))

    bg.fill(bg_color)
    screen.blit(bg, (0,0))
    clock = pg.time.Clock()
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        for state1, state2, state3, action1, action2, action3 in zip(trajectory1, trajectory2, trajectory3,
                                                                     action_history1, action_history2, action_history3):
            # hunter visual field depth
            for j in range(int(max(state1[1]-hunter_vfd, 0)), int(min(Row_num, state1[1]+hunter_vfd+1))):
                for i in range(int(max(state1[0]-hunter_vfd, 0)), int(min(Col_num, state1[0]+hunter_vfd+1))):
                    rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                   (WIDTH // Col_num), (HEIGHT // Row_num))
                    pg.draw.rect(screen, vfdh_color, rect)

            # scout visual field depth
            for j in range(int(max(state2[1]-scout_vfd, 0)), int(min(Col_num, state2[1]+scout_vfd+1))):
                for i in range(int(max(state2[0]-scout_vfd, 0)), int(min(Col_num, state2[0]+scout_vfd+1))):
                    rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                   (WIDTH // Col_num), (HEIGHT // Row_num))
                    pg.draw.rect(screen, vfds_color, rect)

            # agents
            screen.blit(img_mdf_h_coyote, (state1[1] * (WIDTH // Col_num), state1[0] * (HEIGHT // Row_num)))
            screen.blit(img_mdf_s_coyote, (state2[1] * (WIDTH // Col_num), state2[0] * (HEIGHT // Row_num)))
            screen.blit(img_mdf_roadrunner, (state3[1] * (WIDTH // Col_num), state3[0] * (HEIGHT // Row_num)))

            draw_grid(screen)
            pg.display.flip()
            pg.display.update()
            time.sleep(wait_time)  # wait between the shows
            screen.blit(bg, (state1[1] * (WIDTH // Col_num), state1[0] * (HEIGHT // Row_num)))
            screen.blit(bg, (state2[1] * (WIDTH // Col_num), state2[0] * (HEIGHT // Row_num)))
            screen.blit(bg, (state3[1] * (WIDTH // Col_num), state3[0] * (HEIGHT // Row_num)))

            # hunter visual field depths
            for j in range(int(max(state1[1]-hunter_vfd, 0)), int(min(Row_num, state1[1]+hunter_vfd+1))):
                for i in range(int(max(state1[0]-hunter_vfd, 0)), int(min(Col_num, state1[0]+hunter_vfd+1))):
                    rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                    pg.draw.rect(screen, bg_color, rect)

            # scout visual field depth
            for j in range(int(max(state2[1]-scout_vfd, 0)), int(min(Row_num, state2[1]+scout_vfd+1))):
                for i in range(int(max(state2[0]-scout_vfd, 0)), int(min(Col_num, state2[0]+scout_vfd+1))):
                    rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                   (WIDTH // Col_num), (HEIGHT // Row_num))
                    pg.draw.rect(screen, bg_color, rect)

        screen.blit(img_mdf_h_coyote, (trajectory1[-1][1] * (WIDTH // Col_num), trajectory1[-1][0] * (HEIGHT // Row_num)))
        screen.blit(img_mdf_s_coyote, (trajectory2[-1][1] * (WIDTH // Col_num), trajectory2[-1][0] * (HEIGHT // Row_num)))
        screen.blit(img_mdf_roadrunner, (trajectory3[-1][1] * (WIDTH // Col_num), trajectory3[-1][0] * (HEIGHT // Row_num)))
        draw_grid(screen)
        pg.display.flip()
        pg.display.update()
        run = False
    pg.quit()

def Qcheck(Q, initpose1, initpose2, wait_time=0):
    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH+2, HEIGHT+2))   # set up the screen
    pg.display.set_caption("Hamid Osooli")              # add a caption
    bg = pg.Surface(screen.get_size())                  # get a background surface
    bg = bg.convert()
    agent = Agent(screen)

    # img_nat = pg.image.load('nature.png')
    # img_mdf_nat = pg.transform.scale(img_nat, (WIDTH//Col_num, HEIGHT//Row_num))
    # img_pud = pg.image.load('puddle.png')
    # img_mdf_pud = pg.transform.scale(img_pud, (WIDTH//Col_num, HEIGHT//Row_num))
    # img_dmd = pg.image.load('diamond.png')
    # img_mdf_dmd = pg.transform.scale(img_dmd, (WIDTH//Col_num, HEIGHT//Row_num))
    img_quad1 = pg.image.load('quad.png')
    img_mdf_quad1 = pg.transform.scale(img_quad1, (WIDTH // Col_num, HEIGHT // Row_num))
    img_home1 = pg.image.load('home.png')
    img_mdf_home1 = pg.transform.scale(img_home1, (WIDTH // Col_num, HEIGHT // Row_num))
    img_quad2 = pg.image.load('drone.png')
    img_mdf_quad2 = pg.transform.scale(img_quad2, (WIDTH // Col_num, HEIGHT // Row_num))
    img_home2 = pg.image.load('house-icon.png')
    img_mdf_home2 = pg.transform.scale(img_home2, (WIDTH // Col_num, HEIGHT // Row_num))

    # arr0 = pg.image.load('arrow0.png')
    # arr0_mdf =
    # arr1 = pg.image.load('arrow1.png')
    # arr1_mdf = pg.transform.scale(arr1, (WIDTH // Col_num, HEIGHT // Row_num))
    # arr2 = pg.image.load('arrow2.png')
    # arr2_mdf = pg.transform.scale(arr2, (WIDTH // Col_num, HEIGHT // Row_num))
    # arr3 = pg.image.load('arrow3.png')
    # arr3_mdf = pg.transform.scale(arr3, (WIDTH // Col_num, HEIGHT // Row_num))
    # img_mnt = pg.image.load('mountain.png')
    # img_mdf_mnt = pg.transform.scale(img_mnt, (WIDTH // Col_num, HEIGHT // Row_num))
    # img_ele = pg.image.load('electric-tower.png')
    # img_mdf_ele = pg.transform.scale(img_ele, (WIDTH // Col_num, HEIGHT // Row_num))
    # img_flm = pg.image.load('flame.png')
    # img_mdf_flm = pg.transform.scale(img_flm, (WIDTH // Col_num, HEIGHT // Row_num))
    # img_vlc = pg.image.load('volcano.png')
    # img_mdf_vlc = pg.transform.scale(img_vlc, (WIDTH // Col_num, HEIGHT // Row_num))

    bg.fill(bg_color)
    screen.blit(bg, (0,0))
    clock = pg.time.Clock()
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        # goal
        screen.blit(img_mdf_home1, (goal_pos_x1, goal_pos_y1))
        screen.blit(img_mdf_home2, (goal_pos_x2, goal_pos_y2))
        # agent
        screen.blit(img_mdf_quad1, (initpose1[1] * (WIDTH // Col_num), initpose1[0] * (HEIGHT // Row_num)))
        screen.blit(img_mdf_quad2, (initpose2[1] * (WIDTH // Col_num), initpose2[0] * (HEIGHT // Row_num)))
        draw_grid(screen)
        pg.display.flip()
        pg.display.update()
        time.sleep(wait_time)  # wait between the shows
        screen.blit(bg, (initpose1[1] * (WIDTH // Col_num), initpose1[0] * (HEIGHT // Row_num)))
        screen.blit(bg, (initpose2[1] * (WIDTH // Col_num), initpose2[0] * (HEIGHT // Row_num)))
        screen.blit(img_mdf_home1, (goal_pos_x1, goal_pos_y1))
        screen.blit(img_mdf_home2, (goal_pos_x2, goal_pos_y2))
        screen.blit(img_mdf_quad1, (initpose1[1] * (WIDTH // Col_num), initpose1[0] * (HEIGHT // Row_num)))
        screen.blit(img_mdf_quad2, (initpose2[1] * (WIDTH // Col_num), initpose2[0] * (HEIGHT // Row_num)))

        for row in range(Row_num):
            for col in range(Col_num):
                screen.blit(pg.transform.scale(pg.image.load('arrow'+str(np.argmax(Q[row, col, :]))+'.png'), (WIDTH // Col_num, HEIGHT // Row_num)),
                            (col * (WIDTH // Col_num), row * (HEIGHT // Row_num)))


        draw_grid(screen)
        pg.display.flip()
        pg.display.update()
        # run = False
    # pg.quit()

if __name__ == "__main__":
    main()
# print(map_builder())