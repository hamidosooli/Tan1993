import pygame as pg
import numpy as np
import pygame
import time


pygame.init()

# Constants
WIDTH = 800  # width of the environment (px)
HEIGHT = 800  # height of the environment (px)
TS = 10  # delay in msec
Col_num = 40  # number of columns
Row_num = 40  # number of rows

# define colors
bg_color = pg.Color(255, 255, 255)
line_color = pg.Color(128, 128, 128)
vfdr_color = pg.Color(8, 136, 8, 128)
vfds_color = pg.Color(255, 165, 0, 128)
vfdrs_color = pg.Color(173, 216, 230, 128)


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


def animate(rescuers_traj, rescuers_scouts_traj, scouts_traj, victims_traj,
            rescuers_vfd, scouts_vfd, rescuers_scouts_vfd,
            wait_time):

    font = pg.font.SysFont('arial', 20)

    num_rescuers = len(rescuers_vfd)
    num_scouts = len(scouts_vfd)
    num_rescuers_scouts = len(rescuers_scouts_traj)
    num_victims = len(victims_traj)

    pg.init()  # initialize pygame
    screen = pg.display.set_mode((WIDTH + 2, HEIGHT + 2))  # set up the screen
    pg.display.set_caption("gridworld")  # add a caption
    bg = pg.Surface(screen.get_size())  # get a background surface
    bg = bg.convert()

    img_rescuer = pg.image.load('TurtleBot.png')
    img_mdf_r = pg.transform.scale(img_rescuer, (WIDTH // Col_num, HEIGHT // Row_num))

    img_scout = pg.image.load('Crazyflie.JPG')
    img_mdf_scout = pg.transform.scale(img_scout, (WIDTH // Col_num, HEIGHT // Row_num))

    img_rescuer_scout = pg.image.load('typhoon.jpg')
    img_mdf_rescuer_scout = pg.transform.scale(img_rescuer_scout, (WIDTH // Col_num, HEIGHT // Row_num))

    img_victim = pg.image.load('victim.png')
    img_mdf_victim = pg.transform.scale(img_victim, (WIDTH // Col_num, HEIGHT // Row_num))

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

        for rescuers_stt, scouts_stt, rescuers_scouts_stt, victims_stt in zip(np.moveaxis(rescuers_traj, 0, -1),
                                                                             np.moveaxis(scouts_traj, 0, -1),
                                                                             np.moveaxis(rescuers_scouts_traj, 0, -1),
                                                                             np.moveaxis(victims_traj, 0, -1)):
            for num in range(num_rescuers):
                # rescuer visual field depth
                for j in range(int(max(rescuers_stt[1, num] - rescuers_vfd[num], 0)),
                               int(min(Row_num, rescuers_stt[1, num] + rescuers_vfd[num] + 1))):
                    for i in range(int(max(rescuers_stt[0, num] - rescuers_vfd[num], 0)),
                                   int(min(Col_num, rescuers_stt[0, num] + rescuers_vfd[num] + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, vfdr_color, rect)

            for num in range(num_scouts):
                # scout visual field depth
                for j in range(int(max(scouts_stt[1, num] - scouts_vfd[num], 0)),
                               int(min(Col_num, scouts_stt[1, num] + scouts_vfd[num] + 1))):
                    for i in range(int(max(scouts_stt[0, num] - scouts_vfd[num], 0)),
                                   int(min(Col_num, scouts_stt[0, num] + scouts_vfd[num] + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, vfds_color, rect)

            for num in range(num_rescuers_scouts):
                # rescuer/scout visual field depth
                for j in range(int(max(rescuers_scouts_stt[1, num] - rescuers_scouts_vfd[num], 0)),
                               int(min(Col_num, rescuers_scouts_stt[1, num] + rescuers_scouts_vfd[num] + 1))):
                    for i in range(int(max(rescuers_scouts_stt[0, num] - rescuers_scouts_vfd[num], 0)),
                                   int(min(Col_num, rescuers_scouts_stt[0, num] + rescuers_scouts_vfd[num] + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, vfdrs_color, rect)

            # agents
            for num in range(num_rescuers):
                screen.blit(img_mdf_r,
                            (rescuers_stt[1, num] * (WIDTH // Col_num), rescuers_stt[0, num] * (HEIGHT // Row_num)))
                screen.blit(font.render(str(num + 1), True, (0, 0, 0)),
                            (rescuers_stt[1, num] * (WIDTH // Col_num), rescuers_stt[0, num] * (HEIGHT // Row_num)))

            for num in range(num_scouts):
                screen.blit(img_mdf_scout, (scouts_stt[1, num] * (WIDTH // Col_num),
                                            scouts_stt[0, num] * (HEIGHT // Row_num)))
                screen.blit(font.render(str(num + 1), True, (0, 0, 0)),
                            (scouts_stt[1, num] * (WIDTH // Col_num), scouts_stt[0, num] * (HEIGHT // Row_num)))

            for num in range(num_rescuers_scouts):
                screen.blit(img_mdf_rescuer_scout,
                            (rescuers_scouts_stt[1, num] * (WIDTH // Col_num),
                             rescuers_scouts_stt[0, num] * (HEIGHT // Row_num)))
                screen.blit(font.render(str(num + 1), True, (0, 0, 0)),
                            (rescuers_scouts_stt[1, num] * (WIDTH // Col_num),
                             rescuers_scouts_stt[0, num] * (HEIGHT // Row_num)))

            for num in range(num_victims):
                screen.blit(img_mdf_victim, (victims_stt[1, num] * (WIDTH // Col_num),
                                             victims_stt[0, num] * (HEIGHT // Row_num)))
                screen.blit(font.render(str(num + 1), True, (0, 0, 0)),
                            (victims_stt[1, num] * (WIDTH // Col_num), victims_stt[0, num] * (HEIGHT // Row_num)))

            draw_grid(screen)
            pg.display.flip()
            pg.display.update()
            time.sleep(wait_time)  # wait between the shows

            for num in range(num_victims):
                screen.blit(bg, (victims_stt[1, num] * (WIDTH // Col_num),
                                 victims_stt[0, num] * (HEIGHT // Row_num)))

            for num in range(num_rescuers):
                screen.blit(bg, (rescuers_stt[1, num] * (WIDTH // Col_num),
                                 rescuers_stt[0, num] * (HEIGHT // Row_num)))
                # rescuer visual field depths
                for j in range(int(max(rescuers_stt[1, num] - rescuers_vfd[num], 0)),
                               int(min(Row_num, rescuers_stt[1, num] + rescuers_vfd[num] + 1))):
                    for i in range(int(max(rescuers_stt[0, num] - rescuers_vfd[num], 0)),
                                   int(min(Col_num, rescuers_stt[0, num] + rescuers_vfd[num] + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, bg_color, rect)

            for num in range(num_scouts):
                screen.blit(bg, (scouts_stt[1, num] * (WIDTH // Col_num),
                                 scouts_stt[0, num] * (HEIGHT // Row_num)))
                # scout visual field depth
                for j in range(int(max(scouts_stt[1, num] - scouts_vfd[num], 0)),
                               int(min(Row_num, scouts_stt[1, num] + scouts_vfd[num] + 1))):
                    for i in range(int(max(scouts_stt[0, num] - scouts_vfd[num], 0)),
                                   int(min(Col_num, scouts_stt[0, num] + scouts_vfd[num] + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, bg_color, rect)

            for num in range(num_rescuers_scouts):
                screen.blit(bg, (rescuers_scouts_stt[1, num] * (WIDTH // Col_num),
                                 rescuers_scouts_stt[0, num] * (HEIGHT // Row_num)))
                # rescuer/scout visual field depth
                for j in range(int(max(rescuers_scouts_stt[1, num] - rescuers_scouts_vfd[num], 0)),
                               int(min(Row_num, rescuers_scouts_stt[1, num] + rescuers_scouts_vfd[num] + 1))):
                    for i in range(int(max(rescuers_scouts_stt[0, num] - rescuers_scouts_vfd[num], 0)),
                                   int(min(Col_num, rescuers_scouts_stt[0, num] + rescuers_scouts_vfd[num] + 1))):
                        rect = pg.Rect(j * (WIDTH // Col_num), i * (HEIGHT // Row_num),
                                       (WIDTH // Col_num), (HEIGHT // Row_num))
                        pg.draw.rect(screen, bg_color, rect)

        screen.blit(img_mdf_r, (rescuers_traj[-1][1] * (WIDTH // Col_num),
                                rescuers_traj[-1][0] * (HEIGHT // Row_num)))
        screen.blit(img_mdf_scout, (scouts_traj[-1][1] * (WIDTH // Col_num),
                                    scouts_traj[-1][0] * (HEIGHT // Row_num)))
        screen.blit(img_mdf_rescuer_scout, (rescuers_scouts_traj[-1][1] * (WIDTH // Col_num),
                                            rescuers_scouts_traj[-1][0] * (HEIGHT // Row_num)))
        screen.blit(img_mdf_victim, (victims_traj[-1][1] * (WIDTH // Col_num),
                                     victims_traj[-1][0] * (HEIGHT // Row_num)))

        draw_grid(screen)
        pg.display.flip()
        pg.display.update()
        run = False
    pg.quit()
