'''
This is the gridworld_template.py file that implements
a 1D gridworld and is part of the mid-term project in
the COMP4600/5500-Reinforcement Learning course - Fall 2021
Code: Reza Ahmadzadeh
Late modified: 10/19/2021
'''
import numpy as np
import pygame
import pygame as pg
import time

pygame.init()

# Constants
WIDTH = 800  # width of the environment (px)
HEIGHT = 800  # height of the environment (px)
TS = 10  # delay in msec
Col_num = 10  # number of columns
Row_num = 10  # number of rows

# define colors
bg_color = pg.Color(255, 255, 255)
line_color = pg.Color(128, 128, 128)
vfdh_color = pg.Color(8, 136, 8, 128)
vfds_color = pg.Color(255, 165, 0, 128)

# define primary positions
goal_pos_x1 = 3 * (WIDTH // Col_num)
goal_pos_y1 = 5 * (HEIGHT // Row_num)

goal_pos_x2 = 2 * (WIDTH // Col_num)
goal_pos_y2 = 2 * (HEIGHT // Row_num)


def draw_grid(scr):
    '''a function to draw gridlines and other objects'''
    # Horizontal lines
    for j in range(Row_num + 1):
        pg.draw.line(scr, line_color, (0, j * HEIGHT // Row_num), (WIDTH, j * HEIGHT // Row_num), 2)
    # # Vertical lines
    for i in range(Col_num + 1):
        pg.draw.line(scr, line_color, (i * WIDTH // Col_num, 0), (i * WIDTH // Col_num, HEIGHT), 2)

    for x1 in range(0, WIDTH, WIDTH // Col_num):
        for y1 in range(0, HEIGHT, HEIGHT // Row_num):
            rect = pg.Rect(x1, y1, WIDTH // Col_num, HEIGHT // Row_num)
            pg.draw.rect(scr, bg_color, rect, 1)


class Agent:
    '''the agent class '''

    def __init__(self, scr):
        self.w = WIDTH // (Row_num)
        self.h = HEIGHT // (Col_num)
        self.x1 = 0
        self.y1 = HEIGHT - self.h

        self.x2 = 0
        self.y2 = HEIGHT - self.h
        self.scr = scr
        self.my_rect = pg.Rect((self.x1, self.y1), (self.w, self.h))
        self.BOX_IS_FULL = True
        self.FIRST_TIME = True

    def reward(self, loc):
        if loc[1] == goal_pos_x1 and loc[0] == goal_pos_y1 and self.BOX_IS_FULL:
            return 100  # goal reward
        if loc[1] == goal_pos_x2 and loc[0] == goal_pos_y2 and self.BOX_IS_FULL:
            return 100  # goal reward
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


def animate(rescuers_traj, scouts_traj, victim_traj, rescuer_vfd, scout_vfd, wait_time, have_scout):
    font = pg.font.SysFont('arial', 20)

    num_scouts = len(scout_vfd)
    num_rescuers = len(rescuer_vfd)
    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH + 2, HEIGHT + 2))  # set up the screen
    pg.display.set_caption("Hamid Osooli")  # add a caption
    bg = pg.Surface(screen.get_size())  # get a background surface
    bg = bg.convert()

    img_rescuer = pg.image.load('TurtleBot.png')
    img_mdf_r = pg.transform.scale(img_rescuer, (WIDTH // Col_num, HEIGHT // Row_num))
    img_victim = pg.image.load('victim.png')
    img_mdf_victim = pg.transform.scale(img_victim, (WIDTH // Col_num, HEIGHT // Row_num))

    if have_scout:
        img_scout = pg.image.load('Crazyflie.JPG')
        img_mdf_scout = pg.transform.scale(img_scout, (WIDTH // Col_num, HEIGHT // Row_num))

    bg.fill(bg_color)
    screen.blit(bg, (0, 0))
    clock = pg.time.Clock()
    pg.display.flip()
    run = True
    while run:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
        if have_scout:
            for rescuers_stt, scouts_stt, victim_stt in zip(rescuers_traj[0], scouts_traj[0], victim_traj[0]):
                # for rescuers_stt, scouts_stt, victim_stt in zip(np.moveaxis(rescuers_traj, 0, -1),
                #                                                 np.moveaxis(scouts_traj, 0, -1), victim_traj):
                for num in range(num_rescuers):
                    # rescuer visual field depth
                    for j in range(int(max(rescuers_stt[1] - rescuer_vfd[num], 0)),
                                   int(min(Row_num, rescuers_stt[1] + rescuer_vfd[num] + 1))):
                        for i in range(int(max(rescuers_stt[0] - rescuer_vfd[num], 0)),
                                       int(min(Col_num, rescuers_stt[0] + rescuer_vfd[num] + 1))):
                            rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                           (WIDTH // Col_num), (HEIGHT // Row_num))
                            pg.draw.rect(screen, vfdh_color, rect)

                for num in range(num_scouts):
                    # scout visual field depth
                    for j in range(int(max(scouts_stt[1] - scout_vfd[num], 0)),
                                   int(min(Col_num, scouts_stt[1] + scout_vfd[num] + 1))):
                        for i in range(int(max(scouts_stt[0] - scout_vfd[num], 0)),
                                       int(min(Col_num, scouts_stt[0] + scout_vfd[num] + 1))):
                            rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                           (WIDTH // Col_num), (HEIGHT // Row_num))
                            pg.draw.rect(screen, vfds_color, rect)

                # agents
                for num in range(num_rescuers):
                    screen.blit(img_mdf_r,
                                (rescuers_stt[1] * (WIDTH // Col_num), rescuers_stt[0] * (HEIGHT // Row_num)))
                    screen.blit(font.render(str(num + 1), True, (0, 0, 0)),
                                (rescuers_stt[1] * (WIDTH // Col_num), rescuers_stt[0] * (HEIGHT // Row_num)))
                for num in range(num_scouts):
                    screen.blit(img_mdf_scout, (scouts_stt[1] * (WIDTH // Col_num),
                                                scouts_stt[0] * (HEIGHT // Row_num)))
                    screen.blit(font.render(str(num + 1), True, (0, 0, 0)),
                                (scouts_stt[1] * (WIDTH // Col_num), scouts_stt[0] * (HEIGHT // Row_num)))
                screen.blit(img_mdf_victim, (victim_stt[1] * (WIDTH // Col_num), victim_stt[0] * (HEIGHT // Row_num)))

                draw_grid(screen)
                pg.display.flip()
                pg.display.update()
                time.sleep(wait_time)  # wait between the shows
                for num in range(num_rescuers):
                    screen.blit(bg, (rescuers_stt[1] * (WIDTH // Col_num), rescuers_stt[0] * (HEIGHT // Row_num)))
                screen.blit(bg, (victim_stt[1] * (WIDTH // Col_num), victim_stt[0] * (HEIGHT // Row_num)))
                for num in range(num_scouts):
                    screen.blit(bg, (scouts_stt[1] * (WIDTH // Col_num), scouts_stt[0] * (HEIGHT // Row_num)))
                    # scout visual field depth
                    for j in range(int(max(scouts_stt[1] - scout_vfd[num], 0)),
                                   int(min(Row_num, scouts_stt[1] + scout_vfd[num] + 1))):
                        for i in range(int(max(scouts_stt[0] - scout_vfd[num], 0)),
                                       int(min(Col_num, scouts_stt[0] + scout_vfd[num] + 1))):
                            rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                           (WIDTH // Col_num), (HEIGHT // Row_num))
                            pg.draw.rect(screen, bg_color, rect)

                for num in range(num_rescuers):
                    # rescuer visual field depths
                    for j in range(int(max(rescuers_stt[1] - rescuer_vfd[num], 0)),
                                   int(min(Row_num, rescuers_stt[1] + rescuer_vfd[num] + 1))):
                        for i in range(int(max(rescuers_stt[0] - rescuer_vfd[num], 0)),
                                       int(min(Col_num, rescuers_stt[0] + rescuer_vfd[num] + 1))):
                            rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                           (WIDTH // Col_num), (HEIGHT // Row_num))
                            pg.draw.rect(screen, bg_color, rect)
            # for num in range(num_rescuers):
            #     screen.blit(img_mdf_r, (rescuers_traj[-1][1, num] * (WIDTH // Col_num),
            #                             rescuers_traj[-1][0, num] * (HEIGHT // Row_num)))
            # for num in range(num_scouts):
            #     screen.blit(img_mdf_scout,
            #                 (scouts_traj[-1][1, num] * (WIDTH // Col_num),
            #                  scouts_traj[-1][0, num] * (HEIGHT // Row_num)))
            screen.blit(img_mdf_victim,
                        (victim_traj[-1][1] * (WIDTH // Col_num),
                         victim_traj[-1][0] * (HEIGHT // Row_num)))
        else:
            for rescuers_stt, victim_stt in zip(rescuers_traj, victim_traj):
                # rescuer visual field depth
                for j in range(int(max(rescuers_stt[1] - rescuer_vfd, 0)),
                               int(min(Row_num, rescuers_stt[1] + rescuer_vfd + 1))):
                    for i in range(int(max(rescuers_stt[0] - rescuer_vfd, 0)),
                                   int(min(Col_num, rescuers_stt[0] + rescuer_vfd + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, vfdh_color, rect)

                # agents
                screen.blit(img_mdf_r, (rescuers_stt[1] * (WIDTH // Col_num), rescuers_stt[0] * (HEIGHT // Row_num)))
                screen.blit(img_mdf_victim, (victim_stt[1] * (WIDTH // Col_num), victim_stt[0] * (HEIGHT // Row_num)))

                draw_grid(screen)
                pg.display.flip()
                pg.display.update()
                time.sleep(wait_time)  # wait between the shows

                screen.blit(bg, (rescuers_stt[1] * (WIDTH // Col_num), rescuers_stt[0] * (HEIGHT // Row_num)))
                screen.blit(bg, (victim_stt[1] * (WIDTH // Col_num), victim_stt[0] * (HEIGHT // Row_num)))

                # rescuer visual field depths
                for j in range(int(max(rescuers_stt[1] - rescuer_vfd, 0)),
                               int(min(Row_num, rescuers_stt[1] + rescuer_vfd + 1))):
                    for i in range(int(max(rescuers_stt[0] - rescuer_vfd, 0)),
                                   int(min(Col_num, rescuers_stt[0] + rescuer_vfd + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, bg_color, rect)

            screen.blit(img_mdf_r, (rescuers_traj[-1][1] * (WIDTH // Col_num),
                                    rescuers_traj[-1][0] * (HEIGHT // Row_num)))
            screen.blit(img_mdf_victim, (victim_traj[-1][1] * (WIDTH // Col_num),
                                         victim_traj[-1][0] * (HEIGHT // Row_num)))
        draw_grid(screen)
        pg.display.flip()
        pg.display.update()
        run = False
    pg.quit()
