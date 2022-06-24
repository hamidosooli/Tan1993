import numpy as np
import h5py

from action_selection import eps_greedy, Boltzmann
from network import Network
from agent import Agent


# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]
num_Acts = len(ACTIONS)

NUM_EPISODES = 2000

gamma = .9

# Environment dimensions
Row_num = 20
Col_num = 20
row_lim = Row_num - 1
col_lim = Col_num - 1

#                    r1 r2 s1 s2 s3 rs1 rs2 rs3
adj_mat = np.array([[0, 0, 0, 0, 0,  0,  0,  1],
                    [0, 0, 1, 0, 0,  1,  0,  0],
                    [0, 0, 0, 0, 0,  0,  0,  0],
                    [0, 0, 0, 0, 0,  0,  0,  0],
                    [0, 0, 0, 0, 0,  0,  0,  0],
                    [0, 0, 0, 1, 0,  0,  1,  0],
                    [0, 0, 0, 0, 1,  0,  0,  0],
                    [0, 0, 0, 0, 0,  1,  0,  0]], dtype=float)

wereHere = np.ones((Row_num, Col_num))


# Transition function
def movement(pos, action, speed):
    row = pos[0]
    col = pos[1]

    if action == 0:  # up
        next_pos = [max(row - speed, 0), col]
    elif action == 1:  # down
        next_pos = [min(row + speed, row_lim), col]
    elif action == 2:  # right
        next_pos = [row, min(col + speed, col_lim)]
    elif action == 3:  # left
        next_pos = [row, max(col - speed, 0)]

    return next_pos


def reward_r(sensation_prime):
    if sensation_prime[0] == 0 and sensation_prime[1] == 0:
        re = 1
    else:
        re = -.1
    return re


def reward_s(sensation_prime):
    if sensation_prime[0] == 0 and sensation_prime[1] == 0:
        re = 1
    else:
        re = -.1
    return re


def q_learning(q, old_idx, curr_idx, re, act, alpha=0.8):
    q[old_idx, act] += alpha * (re + gamma * np.max(q[curr_idx, :]) - q[old_idx, act])
    return q


def env():
    # Define the Network and the agent objects
    network = Network
    agent = Agent

    # Define the agents
    r1 = agent(0, 'r', 2, Row_num, 2, [0, 0], num_Acts, Row_num, Col_num)  # First Rescuer
    r2 = agent(1, 'r', 3, Row_num, 2, [0, col_lim], num_Acts, Row_num, Col_num)  # Second Rescuer

    s1 = agent(2, 's', 3, 3, 2, [0, int(Col_num/2)], num_Acts, Row_num, Col_num)  # First Scout
    s2 = agent(3, 's', 2, 2, 2, [int(Row_num/2), 0], num_Acts, Row_num, Col_num)  # Second Scout
    s3 = agent(4, 's', 3, 3, 2, [int(Row_num/2), col_lim], num_Acts, Row_num, Col_num)  # Third Scout

    rs1 = agent(5, 'rs', 2, Row_num, 2, [row_lim, 0], num_Acts, Row_num, Col_num)  # First Rescuer\Scout
    rs2 = agent(6, 'rs', 3, Row_num, 2, [row_lim, int(Col_num/2)], num_Acts, Row_num, Col_num)  # Second Rescuer\Scout
    rs3 = agent(7, 'rs', 4, Row_num, 2, [row_lim, col_lim], num_Acts, Row_num, Col_num)  # Third Rescuer\Scout

    v1 = agent(0, 'v', 0, 0, 1, [int(Row_num/2), int(Col_num/2)], num_Acts, Row_num, Col_num)
    v2 = agent(1, 'v', 0, 0, 1, [int(Row_num / 2) + 2, int(Col_num / 2) + 2], num_Acts, Row_num, Col_num)
    v3 = agent(2, 'v', 0, 0, 1, [int(Row_num / 2) - 2, int(Col_num / 2) - 2], num_Acts, Row_num, Col_num)
    v4 = agent(3, 'v', 0, 0, 1, [int(Row_num / 2) + 4, int(Col_num / 2) + 4], num_Acts, Row_num, Col_num)
    v5 = agent(4, 'v', 0, 0, 1, [int(Row_num / 2) - 4, int(Col_num / 2) - 4], num_Acts, Row_num, Col_num)

    for eps in range(NUM_EPISODES):
        # Reset the agents flags, positions, etc
        r1.reset()
        r2.reset()

        s1.reset()
        s2.reset()
        s3.reset()

        rs1.reset()
        rs2.reset()
        rs3.reset()

        v1.reset()
        v2.reset()
        v3.reset()
        v4.reset()
        v5.reset()

        wereHere = np.ones((Row_num, Col_num))

        t_step = 0

        while True:

            net = network(adj_mat, 8, 5)

            t_step += 1
            if r1.CanSeeIt:
                r1.t_step_seen += 1
            if r2.CanSeeIt:
                r2.t_step_seen += 1

            if s1.CanSeeIt:
                s1.t_step_seen += 1
            if s2.CanSeeIt:
                s2.t_step_seen += 1
            if s3.CanSeeIt:
                s3.t_step_seen += 1

            if rs1.CanSeeIt:
                rs1.t_step_seen += 1
            if rs2.CanSeeIt:
                rs2.t_step_seen += 1
            if rs3.CanSeeIt:
                rs3.t_step_seen += 1

            # Marking the formerly covered locations on exploration time
            # r1.cell_marker(r1.old_Pos)
            # r2.cell_marker(r2.old_Pos)
            #
            # s1.cell_marker(s1.old_Pos)
            # s2.cell_marker(s2.old_Pos)
            # s3.cell_marker(s3.old_Pos)
            #
            # rs1.cell_marker(rs1.old_Pos)
            # rs2.cell_marker(rs2.old_Pos)
            # rs3.cell_marker(rs3.old_Pos)
            wereHere[r1.old_Pos[0], r1.old_Pos[1]] = 0
            wereHere[r2.old_Pos[0], r2.old_Pos[1]] = 0
            wereHere[s1.old_Pos[0], s1.old_Pos[1]] = 0
            wereHere[s2.old_Pos[0], s2.old_Pos[1]] = 0
            wereHere[s3.old_Pos[0], s3.old_Pos[1]] = 0
            wereHere[rs1.old_Pos[0], rs1.old_Pos[1]] = 0
            wereHere[rs2.old_Pos[0], rs2.old_Pos[1]] = 0
            wereHere[rs3.old_Pos[0], rs3.old_Pos[1]] = 0

            # Keeping track of the rescue team positions
            r1.Traj.append(r1.old_Pos)
            r2.Traj.append(r2.old_Pos)

            s1.Traj.append(s1.old_Pos)
            s2.Traj.append(s2.old_Pos)
            s3.Traj.append(s3.old_Pos)

            rs1.Traj.append(rs1.old_Pos)
            rs2.Traj.append(rs2.old_Pos)
            rs3.Traj.append(rs3.old_Pos)

            # Keep track of the victims positions
            v1.Traj.append(v1.old_Pos)
            v2.Traj.append(v2.old_Pos)
            v3.Traj.append(v3.old_Pos)
            v4.Traj.append(v4.old_Pos)
            v5.Traj.append(v5.old_Pos)

            # Calculation of the distance between scouts and rescuers
            old_scouts2rescuers = net.pos2pos(np.array([r1.old_Pos, r2.old_Pos,
                                                        s1.old_Pos, s2.old_Pos, s3.old_Pos,
                                                        rs1.old_Pos, rs2.old_Pos, rs3.old_Pos]))

            # Calculation of the raw sensations for the rescue team
            old_raw_sensations = net.sensed_pos(np.array([v1.old_Pos, v2.old_Pos, v3.old_Pos, v4.old_Pos, v5.old_Pos]),
                                                np.array([r1.old_Pos, r2.old_Pos,
                                                          s1.old_Pos, s2.old_Pos, s3.old_Pos,
                                                          rs1.old_Pos, rs2.old_Pos, rs3.old_Pos]))

            # Check to see if the sensations are in the agents visual fields
            eval_old_sensations = net.is_seen(np.array([r1.VisualField, r2.VisualField,
                                                        s1.VisualField, s2.VisualField, s3.VisualField,
                                                        rs1.VisualField, rs2.VisualField, rs3.VisualField]),
                                              old_raw_sensations)

            # Calculation of the sensations for the rescue team
            r1.old_Sensation = r1.update_sensation(old_raw_sensations, eval_old_sensations,
                                                   old_scouts2rescuers, net.adj_mat, adj_mat)
            r2.old_Sensation = r2.update_sensation(old_raw_sensations, eval_old_sensations,
                                                   old_scouts2rescuers, net.adj_mat, adj_mat)

            s1.old_Sensation = s1.update_sensation(old_raw_sensations, eval_old_sensations,
                                                   old_scouts2rescuers, net.adj_mat, adj_mat)
            s2.old_Sensation = s2.update_sensation(old_raw_sensations, eval_old_sensations,
                                                   old_scouts2rescuers, net.adj_mat, adj_mat)
            s3.old_Sensation = s3.update_sensation(old_raw_sensations, eval_old_sensations,
                                                   old_scouts2rescuers, net.adj_mat, adj_mat)

            rs1.old_Sensation = rs1.update_sensation(old_raw_sensations, eval_old_sensations,
                                                     old_scouts2rescuers, net.adj_mat, adj_mat)
            rs2.old_Sensation = rs2.update_sensation(old_raw_sensations, eval_old_sensations,
                                                     old_scouts2rescuers, net.adj_mat, adj_mat)
            rs3.old_Sensation = rs3.update_sensation(old_raw_sensations, eval_old_sensations,
                                                     old_scouts2rescuers, net.adj_mat, adj_mat)

            # Calculation of the indices for the rescue team
            r1.old_Index = r1.sensation2index(r1.old_Sensation, Row_num)
            r2.old_Index = r2.sensation2index(r2.old_Sensation, Row_num)

            s1.old_Index = s1.sensation2index(s1.old_Sensation, s1.VisualField)
            s2.old_Index = s2.sensation2index(s2.old_Sensation, s2.VisualField)
            s3.old_Index = s3.sensation2index(s3.old_Sensation, s3.VisualField)

            rs1.old_Index = rs1.sensation2index(rs1.old_Sensation, Row_num)
            rs2.old_Index = rs2.sensation2index(rs2.old_Sensation, Row_num)
            rs3.old_Index = rs3.sensation2index(rs3.old_Sensation, Row_num)

            # Probabilities for the rescue team
            r1.probs = Boltzmann(r1.Q[r1.old_Index, :])
            r2.probs = Boltzmann(r2.Q[r2.old_Index, :])

            s1.probs = Boltzmann(s1.Q[s1.old_Index, :])
            s2.probs = Boltzmann(s2.Q[s2.old_Index, :])
            s3.probs = Boltzmann(s3.Q[s3.old_Index, :])

            rs1.probs = Boltzmann(rs1.Q[rs1.old_Index, :])
            rs2.probs = Boltzmann(rs2.Q[rs2.old_Index, :])
            rs3.probs = Boltzmann(rs3.Q[rs3.old_Index, :])

            # Actions for the rescue team
            r1.action = np.random.choice(ACTIONS, p=r1.probs)
            r2.action = np.random.choice(ACTIONS, p=r2.probs)

            s1.action = np.random.choice(ACTIONS, p=s1.probs)
            s2.action = np.random.choice(ACTIONS, p=s2.probs)
            s3.action = np.random.choice(ACTIONS, p=s3.probs)

            rs1.action = np.random.choice(ACTIONS, p=rs1.probs)
            rs2.action = np.random.choice(ACTIONS, p=rs2.probs)
            rs3.action = np.random.choice(ACTIONS, p=rs3.probs)

            # Next positions for the rescue team
            r1.curr_Pos = movement(r1.old_Pos, r1.action, r1.Speed)
            r2.curr_Pos = movement(r2.old_Pos, r2.action, r2.Speed)

            s1.curr_Pos = movement(s1.old_Pos, s1.action, s1.Speed)
            s2.curr_Pos = movement(s2.old_Pos, s2.action, s2.Speed)
            s3.curr_Pos = movement(s3.old_Pos, s3.action, s3.Speed)

            rs1.curr_Pos = movement(rs1.old_Pos, rs1.action, rs1.Speed)
            rs2.curr_Pos = movement(rs2.old_Pos, rs2.action, rs2.Speed)
            rs3.curr_Pos = movement(rs3.old_Pos, rs3.action, rs3.Speed)

            # Prevent random exploration when receiving no data
            # r1.curr_Pos = r1.smart_move(r1.old_Index, wereHere)
            # r2.curr_Pos = r2.smart_move(r2.old_Index, wereHere)
            #
            # s1.curr_Pos = s1.smart_move(s1.old_Index, wereHere)
            # s2.curr_Pos = s2.smart_move(s2.old_Index, wereHere)
            # s3.curr_Pos = s3.smart_move(s3.old_Index, wereHere)
            #
            # rs1.curr_Pos = rs1.smart_move(rs1.old_Index, wereHere)
            # rs2.curr_Pos = rs2.smart_move(rs2.old_Index, wereHere)
            # rs3.curr_Pos = rs3.smart_move(rs3.old_Index, wereHere)

            # Calculation of the distance between scouts and rescuers (after their movement)
            curr_scouts2rescuers = net.pos2pos(np.array([r1.curr_Pos, r2.curr_Pos,
                                                         s1.curr_Pos, s2.curr_Pos, s3.curr_Pos,
                                                         rs1.curr_Pos, rs2.curr_Pos, rs3.curr_Pos]))

            # Calculation of the new raw sensations for the rescue team (after their movement)
            curr_raw_sensations = net.sensed_pos(np.array([v1.old_Pos, v2.old_Pos, v3.old_Pos, v4.old_Pos, v5.old_Pos]),
                                                 np.array([r1.curr_Pos, r2.curr_Pos,
                                                           s1.curr_Pos, s2.curr_Pos, s3.curr_Pos,
                                                           rs1.curr_Pos, rs2.curr_Pos, rs3.curr_Pos]))

            # Check to see if the sensations are in the agents visual fields
            eval_curr_sensations = net.is_seen(np.array([r1.VisualField, r2.VisualField,
                                                         s1.VisualField, s2.VisualField, s3.VisualField,
                                                         rs1.VisualField, rs2.VisualField, rs3.VisualField]),
                                               curr_raw_sensations)

            for v_id, v_finish in enumerate([v1.Finish, v2.Finish, v3.Finish, v4.Finish, v5.Finish]):
                if v_finish:
                    eval_old_sensations[:, v_id] = False
                    eval_curr_sensations[:, v_id] = False

            # Calculation of the new sensations for the rescue team (after their movement)
            r1.curr_Sensation = r1.update_sensation(curr_raw_sensations, eval_curr_sensations,
                                                    curr_scouts2rescuers, net.adj_mat, adj_mat)
            r2.curr_Sensation = r2.update_sensation(curr_raw_sensations, eval_curr_sensations,
                                                    curr_scouts2rescuers, net.adj_mat, adj_mat)

            s1.curr_Sensation = s1.update_sensation(curr_raw_sensations, eval_curr_sensations,
                                                    curr_scouts2rescuers, net.adj_mat, adj_mat)
            s2.curr_Sensation = s2.update_sensation(curr_raw_sensations, eval_curr_sensations,
                                                    curr_scouts2rescuers, net.adj_mat, adj_mat)
            s3.curr_Sensation = s3.update_sensation(curr_raw_sensations, eval_curr_sensations,
                                                    curr_scouts2rescuers, net.adj_mat, adj_mat)

            rs1.curr_Sensation = rs1.update_sensation(curr_raw_sensations, eval_curr_sensations,
                                                      curr_scouts2rescuers, net.adj_mat, adj_mat)
            rs2.curr_Sensation = rs2.update_sensation(curr_raw_sensations, eval_curr_sensations,
                                                      curr_scouts2rescuers, net.adj_mat, adj_mat)
            rs3.curr_Sensation = rs3.update_sensation(curr_raw_sensations, eval_curr_sensations,
                                                      curr_scouts2rescuers, net.adj_mat, adj_mat)

            # Calculation of the indices for the rescue team (after their movement)
            r1.curr_Index = r1.sensation2index(r1.curr_Sensation, Row_num)
            r2.curr_Index = r2.sensation2index(r2.curr_Sensation, Row_num)

            s1.curr_Index = s1.sensation2index(s1.curr_Sensation, s1.VisualField)
            s2.curr_Index = s2.sensation2index(s2.curr_Sensation, s2.VisualField)
            s3.curr_Index = s3.sensation2index(s3.curr_Sensation, s3.VisualField)

            rs1.curr_Index = rs1.sensation2index(rs1.curr_Sensation, Row_num)
            rs2.curr_Index = rs2.sensation2index(rs2.curr_Sensation, Row_num)
            rs3.curr_Index = rs3.sensation2index(rs3.curr_Sensation, Row_num)

            # Rewarding the rescue team
            r1.reward = reward_r(r1.old_Sensation)
            r2.reward = reward_r(r2.old_Sensation)

            s1.reward = reward_s(s1.old_Sensation)
            s2.reward = reward_s(s2.old_Sensation)
            s3.reward = reward_s(s3.old_Sensation)

            rs1.reward = reward_r(rs1.old_Sensation)
            rs2.reward = reward_r(rs2.old_Sensation)
            rs3.reward = reward_r(rs3.old_Sensation)

            # Keeping track of the rewards
            if r1.CanSeeIt:
                r1.RewHist_seen.append(r1.reward)
            else:
                r1.RewHist.append(r1.reward)

            if r2.CanSeeIt:
                r2.RewHist_seen.append(r2.reward)
            else:
                r2.RewHist.append(r2.reward)

            if s1.CanSeeIt:
                s1.RewHist_seen.append(s1.reward)
            else:
                s1.RewHist.append(s1.reward)

            if s2.CanSeeIt:
                s2.RewHist_seen.append(s2.reward)
            else:
                s2.RewHist.append(s2.reward)

            if s3.CanSeeIt:
                s3.RewHist_seen.append(s3.reward)
            else:
                s3.RewHist.append(s3.reward)

            if rs1.CanSeeIt:
                rs1.RewHist_seen.append(rs1.reward)
            else:
                rs1.RewHist.append(rs1.reward)

            if rs2.CanSeeIt:
                rs2.RewHist_seen.append(rs2.reward)
            else:
                rs2.RewHist.append(rs2.reward)

            if rs3.CanSeeIt:
                rs3.RewHist_seen.append(rs3.reward)
            else:
                rs3.RewHist.append(rs3.reward)

            # Actions for the victims
            v1.action = np.random.choice(ACTIONS)
            v2.action = np.random.choice(ACTIONS)
            v3.action = np.random.choice(ACTIONS)
            v4.action = np.random.choice(ACTIONS)
            v5.action = np.random.choice(ACTIONS)

            # Victims next positions
            v1.curr_Pos = movement(v1.old_Pos, v1.action, v1.Speed)
            v2.curr_Pos = movement(v2.old_Pos, v2.action, v2.Speed)
            v3.curr_Pos = movement(v3.old_Pos, v3.action, v3.Speed)
            v4.curr_Pos = movement(v4.old_Pos, v4.action, v4.Speed)
            v5.curr_Pos = movement(v5.old_Pos, v5.action, v5.Speed)

            # Q learning for the rescue team
            r1.Q = q_learning(r1.Q, r1.old_Index, r1.curr_Index, r1.reward, r1.action, alpha=0.8)
            r2.Q = q_learning(r2.Q, r2.old_Index, r2.curr_Index, r2.reward, r2.action, alpha=0.8)

            s1.Q = q_learning(s1.Q, s1.old_Index, s1.curr_Index, s1.reward, s1.action, alpha=0.8)
            s2.Q = q_learning(s2.Q, s2.old_Index, s2.curr_Index, s2.reward, s2.action, alpha=0.8)
            s3.Q = q_learning(s3.Q, s3.old_Index, s3.curr_Index, s3.reward, s3.action, alpha=0.8)

            rs1.Q = q_learning(rs1.Q, rs1.old_Index, rs1.curr_Index, rs1.reward, rs1.action, alpha=0.8)
            rs2.Q = q_learning(rs2.Q, rs2.old_Index, rs2.curr_Index, rs2.reward, rs2.action, alpha=0.8)
            rs3.Q = q_learning(rs3.Q, rs3.old_Index, rs3.curr_Index, rs3.reward, rs3.action, alpha=0.8)
            # print(r1.Finish, r2.Finish, rs1.Finish, rs2.Finish, rs3.Finish)
            # print('//////////////////////////////////////////////////// \n',
            #       v1.Finish, v2.Finish, v3.Finish, v4.Finish, v5.Finish)
            # print('//////////////////////////////////////////////////// \n',
            #       adj_mat)
            # Check to see the team rescued any victim
            if r1.Finish and r1.First:
                r1.Steps.append(t_step)
                r1.Steps_seen.append(r1.t_step_seen)
                r1.RewSum.append(np.sum(r1.RewHist))
                r1.RewSum_seen.append(np.sum(r1.RewHist_seen))
                adj_mat[:, r1.id] = 0

                r1.First = False

            elif not r1.Finish:
                r1.rescue_accomplished()
                # print(r1.Finish, '\n', r1.old_Pos, v1.old_Pos, v2.old_Pos, v3.old_Pos, v4.old_Pos, v5.old_Pos, '\n',
                #       'others',
                #       r1.curr_Pos, v1.curr_Pos, v2.curr_Pos, v3.curr_Pos, v4.curr_Pos, v5.curr_Pos, '\n',
                #       v1.Finish, v2.Finish, v3.Finish, v4.Finish, v5.Finish)
                # print('//////////////////////////////////////////////////// \n',
                #       eval_old_sensations, '//////////////////////////////////////////////////// \n',
                #       eval_curr_sensations)

            if r2.Finish and r2.First:
                r2.Steps.append(t_step)
                r2.Steps_seen.append(r2.t_step_seen)
                r2.RewSum.append(np.sum(r2.RewHist))
                r2.RewSum_seen.append(np.sum(r2.RewHist_seen))
                adj_mat[:, r2.id] = 0
                r2.First = False

            elif not r2.Finish:
                r2.rescue_accomplished()

            if rs1.Finish and rs1.First:
                rs1.Steps.append(t_step)
                rs1.Steps_seen.append(rs1.t_step_seen)
                rs1.RewSum.append(np.sum(rs1.RewHist))
                rs1.RewSum_seen.append(np.sum(rs1.RewHist_seen))
                adj_mat[:, rs1.id] = 0
                rs1.First = False

            elif not rs1.Finish:
                rs1.rescue_accomplished()

            if rs2.Finish and rs2.First:
                rs2.Steps.append(t_step)
                rs2.Steps_seen.append(rs2.t_step_seen)
                rs2.RewSum.append(np.sum(rs2.RewHist))
                rs2.RewSum_seen.append(np.sum(rs2.RewHist_seen))
                adj_mat[:, rs2.id] = 0
                rs2.First = False

            elif not rs2.Finish:
                rs2.rescue_accomplished()

            if rs3.Finish and rs3.First:
                rs3.Steps.append(t_step)
                rs3.Steps_seen.append(rs3.t_step_seen)
                rs3.RewSum.append(np.sum(rs3.RewHist))
                rs3.RewSum_seen.append(np.sum(rs3.RewHist_seen))
                adj_mat[:, rs3.id] = 0
                rs3.First = False

            elif not rs3.Finish:
                rs3.rescue_accomplished()

            # moving the scouts
            s1.old_Pos = s1.curr_Pos
            s2.old_Pos = s2.curr_Pos
            s3.old_Pos = s3.curr_Pos

            # Check to see if the victim rescued by the team
            # Keep track of the steps
            # Remove the victim from the list
            # Update the victims position
            if v1.Finish and v1.First:
                v1.Steps.append(t_step)

                v1.First = False
            elif not v1.Finish:
                v1.victim_rescued([r1.old_Pos, r2.old_Pos, rs1.old_Pos, rs2.old_Pos, rs3.old_Pos])

            if v2.Finish and v2.First:
                v2.Steps.append(t_step)

                v2.First = False
            elif not v2.Finish:
                v2.victim_rescued([r1.old_Pos, r2.old_Pos, rs1.old_Pos, rs2.old_Pos, rs3.old_Pos])

            if v3.Finish and v3.First:
                v3.Steps.append(t_step)
                v3.First = False
            elif not v3.Finish:
                v3.victim_rescued([r1.old_Pos, r2.old_Pos, rs1.old_Pos, rs2.old_Pos, rs3.old_Pos])

            if v4.Finish and v4.First:
                v4.Steps.append(t_step)

                v4.First = False
            elif not v4.Finish:
                v4.victim_rescued([r1.old_Pos, r2.old_Pos, rs1.old_Pos, rs2.old_Pos, rs3.old_Pos])

            if v5.Finish and v5.First:
                v5.Steps.append(t_step)

                v5.First = False
            elif not v5.Finish:
                v5.victim_rescued([r1.old_Pos, r2.old_Pos, rs1.old_Pos, rs2.old_Pos, rs3.old_Pos])
            # print(v1.Finish, v2.Finish, v3.Finish, v4.Finish, v5.Finish, '\n', 'others', r1.Finish, r2.Finish,
            #                        rs1.Finish, rs2.Finish, rs3.Finish)
            rescue_flags = [r1.Finish, r2.Finish,
                            rs1.Finish, rs2.Finish, rs3.Finish]
            if all(rescue_flags):

                s1.RewSum.append(np.sum(s1.RewHist))
                s1.RewSum_seen.append(np.sum(s1.RewHist_seen))

                s2.RewSum.append(np.sum(s2.RewHist))
                s2.RewSum_seen.append(np.sum(s2.RewHist_seen))

                s3.RewSum.append(np.sum(s3.RewHist))
                s3.RewSum_seen.append(np.sum(s3.RewHist_seen))

                print(f'In episode {eps + 1} of {NUM_EPISODES}, all of the victims were rescued in {t_step} steps')

                break

    return (r1.Traj, r2.Traj,
            s1.Traj, s2.Traj, s3.Traj,
            rs1.Traj, rs2.Traj, rs3.Traj,

            r1.RewSum, r2.RewSum,
            s1.RewSum, s2.RewSum, s3.RewSum,
            rs1.RewSum, rs2.RewSum, rs3.RewSum,

            r1.Steps, r2.Steps,
            s1.Steps, s2.Steps, s3.Steps,
            rs1.Steps, rs2.Steps, rs3.Steps,

            r1.RewSum_seen, r2.RewSum_seen,
            s1.RewSum_seen, s2.RewSum_seen, s3.RewSum_seen,
            rs1.RewSum_seen, rs2.RewSum_seen, rs3.RewSum_seen,

            r1.Steps_seen, r2.Steps_seen,
            s1.Steps_seen, s2.Steps_seen, s3.Steps_seen,
            rs1.Steps_seen, rs2.Steps_seen, rs3.Steps_seen,

            r1.Q, r2.Q,
            s1.Q, s2.Q, s3.Q,
            rs1.Q, rs2.Q, rs3.Q,

            v1.Traj, v2.Traj, v3.Traj, v4.Traj, v5.Traj)


(r1_Traj, r2_Traj,
 s1_Traj, s2_Traj, s3_Traj,
 rs1_Traj, rs2_Traj, rs3_Traj,

 r1_RewSum, r2_RewSum,
 s1_RewSum, s2_RewSum, s3_RewSum,
 rs1_RewSum, rs2_RewSum, rs3_RewSum,

 r1_Steps, r2_Steps,
 s1_Steps, s2_Steps, s3_Steps,
 rs1_Steps, rs2_Steps, rs3_Steps,

 r1_RewSum_seen, r2_RewSum_seen,
 s1_RewSum_seen, s2_RewSum_seen, s3_RewSum_seen,
 rs1_RewSum_seen, rs2_RewSum_seen, rs3_RewSum_seen,

 r1_Steps_seen, r2_Steps_seen,
 s1_Steps_seen, s2_Steps_seen, s3_Steps_seen,
 rs1_Steps_seen, rs2_Steps_seen, rs3_Steps_seen,

 r1_Q, r2_Q,
 s1_Q, s2_Q, s3_Q,
 rs1_Q, rs2_Q, rs3_Q,

 v1_Traj, v2_Traj, v3_Traj, v4_Traj, v5_Traj) = env()

with h5py.File(f'multi_agent_Q_learning_2r_3s_3rs.hdf5', "w") as f:
    f.create_dataset('r1_trajectory', data=r1_Traj)
    f.create_dataset('r1_reward', data=r1_RewSum)
    f.create_dataset('r1_steps', data=r1_Steps)
    f.create_dataset('r1_reward_seen', data=r1_RewSum_seen)
    f.create_dataset('r1_steps_seen', data=r1_Steps_seen)
    f.create_dataset('r1_Q', data=r1_Q)

    f.create_dataset('r2_trajectory', data=r2_Traj)
    f.create_dataset('r2_reward', data=r2_RewSum)
    f.create_dataset('r2_steps', data=r2_Steps)
    f.create_dataset('r2_reward_seen', data=r2_RewSum_seen)
    f.create_dataset('r2_steps_seen', data=r2_Steps_seen)
    f.create_dataset('r2_Q', data=r2_Q)

    f.create_dataset('s1_trajectory', data=s1_Traj)
    f.create_dataset('s1_reward', data=s1_RewSum)
    f.create_dataset('s1_steps', data=s1_Steps)
    f.create_dataset('s1_reward_seen', data=s1_RewSum_seen)
    f.create_dataset('s1_steps_seen', data=s1_Steps_seen)
    f.create_dataset('s1_Q', data=s1_Q)

    f.create_dataset('s2_trajectory', data=s2_Traj)
    f.create_dataset('s2_reward', data=s2_RewSum)
    f.create_dataset('s2_steps', data=s2_Steps)
    f.create_dataset('s2_reward_seen', data=s2_RewSum_seen)
    f.create_dataset('s2_steps_seen', data=s2_Steps_seen)
    f.create_dataset('s2_Q', data=s2_Q)

    f.create_dataset('s3_trajectory', data=s3_Traj)
    f.create_dataset('s3_reward', data=s3_RewSum)
    f.create_dataset('s3_steps', data=s3_Steps)
    f.create_dataset('s3_reward_seen', data=s3_RewSum_seen)
    f.create_dataset('s3_steps_seen', data=s3_Steps_seen)
    f.create_dataset('s3_Q', data=s3_Q)

    f.create_dataset('rs1_trajectory', data=rs1_Traj)
    f.create_dataset('rs1_reward', data=rs1_RewSum)
    f.create_dataset('rs1_steps', data=rs1_Steps)
    f.create_dataset('rs1_reward_seen', data=rs1_RewSum_seen)
    f.create_dataset('rs1_steps_seen', data=rs1_Steps_seen)
    f.create_dataset('rs1_Q', data=rs1_Q)

    f.create_dataset('rs2_trajectory', data=rs2_Traj)
    f.create_dataset('rs2_reward', data=rs2_RewSum)
    f.create_dataset('rs2_steps', data=rs2_Steps)
    f.create_dataset('rs2_reward_seen', data=rs2_RewSum_seen)
    f.create_dataset('rs2_steps_seen', data=rs2_Steps_seen)
    f.create_dataset('rs2_Q', data=rs2_Q)

    f.create_dataset('rs3_trajectory', data=rs3_Traj)
    f.create_dataset('rs3_reward', data=rs3_RewSum)
    f.create_dataset('rs3_steps', data=rs3_Steps)
    f.create_dataset('rs3_reward_seen', data=rs3_RewSum_seen)
    f.create_dataset('rs3_steps_seen', data=rs3_Steps_seen)
    f.create_dataset('rs3_Q', data=rs3_Q)

    f.create_dataset('v1_trajectory', data=v1_Traj)
    f.create_dataset('v2_trajectory', data=v2_Traj)
    f.create_dataset('v3_trajectory', data=v3_Traj)
    f.create_dataset('v4_trajectory', data=v4_Traj)
    f.create_dataset('v5_trajectory', data=v5_Traj)
