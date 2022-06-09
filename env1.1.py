import numpy as np
import h5py

from action_selection import eps_greedy
from network import Network
from agent import Agent


# Actions
FORWARD = 0
BACKWARD = 1
RIGHT = 2
LEFT = 3
ACTIONS = [FORWARD, BACKWARD, RIGHT, LEFT]
num_Acts = len(ACTIONS)

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


# Transition function
def movement(pos, action, speed):
    row = pos[0]
    col = pos[1]
    next_pos = pos.copy()
    if action == 0:  # up
        next_pos = [max(row - speed, 0), col]
    elif action == 1:  # down
        next_pos = [min(row + speed, row_lim), col]
    elif action == 2:  # right
        next_pos = [row, min(col + speed, col_lim)]
    elif action == 3:  # left
        next_pos = [row, max(col - speed, 0)]

    return next_pos


def reward_func(sensation_prime):
    if sensation_prime[0] == 0 and sensation_prime[1] == 0:
        re = 1
    else:
        re = -.1
    return re


def q_learning(q, old_idx, curr_idx, re, act, alpha=0.8, gamma=0.9):
    q[old_idx, act] += alpha * (re + gamma * np.max(q[curr_idx, :]) - q[old_idx, act])
    return q


def env(accuracy=0.01):
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

    # List of objects
    rescue_team = [r1, r2,
                   s1, s2, s3,
                   rs1, rs2, rs3]
    victims = [v1, v2, v3, v4, v5]

    num_rescue_team = len(rescue_team)
    num_victims = len(victims)

    eps = 0

    while True:

        eps += 1

        # Reset the agents flags, positions, etc
        for agent in rescue_team:
            agent.reset()

        for victim in victims:
            victim.reset()

        t_step = -1

        while True:

            net = network(adj_mat, num_rescue_team, num_victims)

            t_step += 1

            rescue_team_VFD_list = []
            for agent in rescue_team:
                if agent.CanSeeIt:
                    agent.t_step_seen += 1
                # Keeping track of the rescue team positions
                agent.Traj.append(agent.old_Pos)
                # List of the Visual Fields
                rescue_team_VFD_list.append(agent.VisualField)
                # History of Q
                agent.Q_hist = agent.Q.copy()
            rescue_team_VFD_list = np.asarray(rescue_team_VFD_list)

            # Keep track of the victims positions
            victims_old_pos_list = []
            for victim in victims:
                victim.Traj.append(victim.old_Pos)
                victims_old_pos_list.append(victim.old_Pos)
            victims_old_pos_list = np.asarray(victims_old_pos_list)

            # Calculation of the distance between scouts and rescuers
            rescue_team_old_pos_list = []
            for agent in rescue_team:
                rescue_team_old_pos_list.append(agent.old_Pos)
            rescue_team_old_pos_list = np.asarray(rescue_team_old_pos_list)

            old_scouts2rescuers = net.pos2pos(rescue_team_old_pos_list)

            # Calculation of the raw sensations for the rescue team
            old_raw_sensations = net.sensed_pos(victims_old_pos_list, rescue_team_old_pos_list)

            # Check to see if the sensations are in the agents visual fields
            eval_old_sensations = net.is_seen(rescue_team_VFD_list,
                                              old_raw_sensations)
            rescue_team_curr_pos_list = []
            for agent in rescue_team:
                # Calculation of the sensations for the rescue team
                agent.old_Sensation = agent.update_sensation(old_raw_sensations, eval_old_sensations,
                                                             old_scouts2rescuers, net.adj_mat, adj_mat)
                # Calculation of the indices for the rescue team
                agent.old_Index = agent.sensation2index(agent.old_Sensation, agent.max_VisualField)

                # Actions for the rescue team
                agent.action = eps_greedy(agent.Q[agent.old_Index, :], ACTIONS)

                # Next positions for the rescue team
                agent.curr_Pos = movement(agent.old_Pos, agent.action, agent.Speed)

                # List of the current positions for the rescue team members
                rescue_team_curr_pos_list.append(agent.curr_Pos)

            rescue_team_curr_pos_list = np.asarray(rescue_team_curr_pos_list)

            # Calculation of the distance between scouts and rescuers (after their movement)
            curr_scouts2rescuers = net.pos2pos(rescue_team_curr_pos_list)

            # Calculation of the new raw sensations for the rescue team (after their movement)
            curr_raw_sensations = net.sensed_pos(victims_old_pos_list, rescue_team_curr_pos_list)

            # Check to see if the sensations are in the agents visual fields
            eval_curr_sensations = net.is_seen(rescue_team_VFD_list, curr_raw_sensations)

            # Removing the rescued victims from the list
            for victim in victims:
                if victim.Finish:
                    eval_old_sensations[:, victim.id] = False
                    eval_curr_sensations[:, victim.id] = False

            # Calculation of the new sensations for the rescue team (after their movement)
            rescue_flags = []
            for agent in rescue_team:
                agent.curr_Sensation = agent.update_sensation(curr_raw_sensations, eval_curr_sensations,
                                                              curr_scouts2rescuers, net.adj_mat, adj_mat)
                # Calculation of the indices for the rescue team (after their movement)
                agent.curr_Index = agent.sensation2index(agent.curr_Sensation, agent.max_VisualField)

                # Rewarding the rescue team
                agent.reward = reward_func(agent.old_Sensation)

                # Keeping track of the rewards
                if not agent.Finish:
                    agent.RewHist.append(agent.reward)
                    if agent.CanSeeIt:
                        agent.RewHist_seen.append(agent.reward)

                # Q learning for the rescue team
                agent.Q = q_learning(agent.Q, agent.old_Index, agent.curr_Index, agent.reward, agent.action, alpha=0.8)

                rescue_flags.append(agent.Finish)
                # Check to see if the team rescued any victim
                if agent.Finish and agent.First:
                    agent.Steps.append(t_step)
                    agent.Steps_seen.append(agent.t_step_seen)
                    agent.RewSum.append(np.sum(agent.RewHist))
                    agent.RewSum_seen.append(np.sum(agent.RewHist_seen))
                    adj_mat[:, agent.id] = 0
                    agent.First = False

                elif not agent.Finish:
                    agent.rescue_accomplished()

            for victim in victims:
                # Actions for the victims
                victim.action = np.random.choice(ACTIONS)
                # Victims next positions
                victim.curr_Pos = movement(victim.old_Pos, victim.action, victim.Speed)

                # Check to see if the victim rescued by the team
                # Keep track of the steps
                # Remove the victim from the list
                # Update the victims position
                if victim.Finish and victim.First:
                    victim.Steps.append(t_step)

                    victim.First = False
                elif not victim.Finish:
                    victim.victim_rescued(rescue_team_old_pos_list)

            if all(rescue_flags):
                print(f'In episode {eps}, all of the victims were rescued in {t_step} steps')
                break

        convergence_flag = []
        for agent in rescue_team:
            convergence_flag.append(np.abs(np.sum(agent.Q - agent.Q_hist) /
                                    (np.shape(agent.Q)[0]*np.shape(agent.Q)[1])) <= accuracy)
        if all(convergence_flag):
            break

    rescue_team_Traj = []
    rescue_team_RewSum = []
    rescue_team_Steps = []
    rescue_team_RewSum_seen = []
    rescue_team_Steps_seen = []
    rescue_team_Q = []
    for agent in rescue_team:
        rescue_team_Traj.append(agent.Traj)
        rescue_team_RewSum.append(agent.RewSum)
        rescue_team_Steps.append(agent.Steps)
        rescue_team_RewSum_seen.append(agent.RewSum_seen)
        rescue_team_Steps_seen.append(agent.Steps_seen)
        rescue_team_Q.append(agent.Q)

    victims_Traj = []
    for victim in victims:
        victims_Traj.append(victim.Traj)

    return (rescue_team_Traj,
            rescue_team_RewSum, rescue_team_Steps,
            rescue_team_RewSum_seen, rescue_team_Steps_seen,
            rescue_team_Q, victims_Traj, rescue_team_VFD_list)


(rescue_team_Traj,
 rescue_team_RewSum, rescue_team_Steps,
 rescue_team_RewSum_seen, rescue_team_Steps_seen,
 rescue_team_Q, victims_Traj, rescue_team_VFD_list) = env(accuracy=1e-7)

with h5py.File('multi_agent_Q_learning.hdf5', 'w') as f:
    for idx, traj in enumerate(rescue_team_Traj):
        f.create_dataset(f'RS{idx}_trajectory', data=traj)
    for idx, rew_sum in enumerate(rescue_team_RewSum):
        f.create_dataset(f'RS{idx}_reward', data=rew_sum)
    for idx, steps in enumerate(rescue_team_Steps):
        f.create_dataset(f'RS{idx}_steps', data=steps)
    for idx, rew_sum_seen in enumerate(rescue_team_RewSum_seen):
        f.create_dataset(f'RS{idx}_reward_seen', data=rew_sum_seen)
    for idx, steps_seen in enumerate(rescue_team_Steps_seen):
        f.create_dataset(f'RS{idx}_steps_seen', data=steps_seen)
    for idx, q in enumerate(rescue_team_Q):
        f.create_dataset(f'RS{idx}_Q', data=q)
    for idx, victim_traj in enumerate(victims_Traj):
        f.create_dataset(f'victim{idx}_trajectory', data=victim_traj)
    f.create_dataset('victims_num', data=[len(victims_Traj)])
    f.create_dataset('RS_VFD', data=rescue_team_VFD_list)
