import time

import numpy as np
import h5py

from network import Network
from agent_game_theory import Agent

NUM_EPISODES = 2000
NUM_RUNS = 10
Multi_Runs = False
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
global env_map
env_map = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]])

env_mat = 1000000 * env_map
walls_locations = np.argwhere(env_map)
#                          rs1  rs2  rs3  s4
adj_mat_prior = np.array([[0,   1],
                          [1,   0]], dtype=float)
exp_name = '2RS_GT'

input_file = h5py.File('scenarios/multi_agent_Q_learning_1R_6S_1V.hdf5', 'r')
Q_star = np.asarray(input_file['RS0_Q'])
input_file.close()


# Transition function
def movement(pos, action, speed):
    global env_map
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
    if env_map[next_pos[0], next_pos[1]] == 0:
        return next_pos
    else:
        return pos


def env(accuracy=1e-15):
    global env_map
    global adj_mat_prior
    # Define the Network and the agent objects
    network = Network
    agent = Agent
    '''
        _ID's should be increasingly assigned from 0 onward 
        _Roles are r: rescuer, rs: rescuer and scout, s: scout, or v: victim
        _Visual field is the radius by which the agent sees around itself. 
        _Max visual field determines the size for the Q matrix. An agent with scout/s will have a max isula field 
        with a radius as large as the environment.
        _Speed is the number of cells agent moves in one time step. Then the smallest value for speed is 1 
        (zero speed is for fixed agent). 
        _Initial location of the agent in the gridworld (y, x).
        _Number of actions
        _Number of rows
        _Number of columns 

        Make sure to put all of the rescue team members, and victims in their relevant lists after definition 
        (rescue_team and victims)
    '''
    # Define the rescue team
    rs1 = agent(0, 'rs', 3, Row_num, 1, [0, 9],
                num_Acts, Row_num, Col_num)
    rs2 = agent(1, 'rs', 3, Row_num, 1, [0, 10],
                num_Acts, Row_num, Col_num)
    # rs3 = agent(2, 'rs', 3, Row_num, 1, [np.random.choice(range(Row_num)), np.random.choice(range(Col_num))],
    #             num_Acts, Row_num, Col_num)

    rs1.Q = Q_star.copy()
    rs2.Q = Q_star.copy()
    # rs3.Q = Q_star.copy()
    # s3 = agent(2, 's', 4, 4, 1, [row_lim, 0], num_Acts, Row_num, Col_num)
    # s4 = agent(3, 's', 4, 4, 1, [0, col_lim], num_Acts, Row_num, Col_num)
    # rs5 = agent(4, 'r', 4, Row_num, 1, [row_lim, col_lim], num_Acts, Row_num, Col_num)

    # Define the victims
    v1 = agent(0, 'v', 0, 0, 1, [9, 7], num_Acts, Row_num, Col_num)
    # v2 = agent(1, 'v', 0, 0, 1, [row_lim, col_lim], num_Acts, Row_num, Col_num)
    # v3 = agent(2, 'v', 0, 0, 1, [int(Row_num / 2) - 2, int(Col_num / 2) - 2], num_Acts, Row_num, Col_num)
    # v4 = agent(3, 'v', 0, 0, 1, [int(Row_num / 2) + 4, int(Col_num / 2) + 4], num_Acts, Row_num, Col_num)
    # v5 = agent(4, 'v', 0, 0, 1, [int(Row_num / 2) - 4, int(Col_num / 2) - 4], num_Acts, Row_num, Col_num)

    # List of objects
    rescue_team = [rs1, rs2]
    victims = [v1]
    VFD_list = []
    num_just_scouts = 0
    rescue_team_roles = []
    for agent in rescue_team:
        rescue_team_roles.append(agent.Role)
        # List of the Visual Fields
        VFD_list.append(agent.VisualField)
        # Count the number of just scouts
        if agent.Role == 's':
            num_just_scouts += 1
    rescue_team_roles = np.array(rescue_team_roles, dtype=list)
    # eps = -1
    tic = time.time()

    rescue_team_Hist = rescue_team.copy()
    victims_Hist = victims.copy()
    adj_mat = adj_mat_prior.copy()

    agents_idx = []
    for agent in rescue_team:
        agents_idx.append(agent.id)

    victims_idx = []
    for victim in victims:
        victims_idx.append(victim.id)

    # eps += 1

    # Reset the agents flags, positions, etc
    for agent in rescue_team:
        agent.reset()
    # Reset the victims flags, positions, etc
    for victim in victims:
        victim.reset()

    t_step = 0

    while True:
        num_rescue_team = len(rescue_team_Hist)
        num_victims = len(victims_Hist)

        net = network(adj_mat, num_rescue_team, num_victims)

        t_step += 1

        rescue_team_VFD_list = []
        for agent in rescue_team_Hist:
            # List of the Visual Fields
            rescue_team_VFD_list.append(agent.VisualField)

            # Count the steps that agent could see a victim
            if agent.CanSeeIt:
                agent.t_step_seen += 1

            # Keeping track of the rescue team positions
            agent.Traj.append(agent.old_Pos)

            agent.action_from_others = []
            agent.sns_from_others = []
            agent.rew_from_others = []

        rescue_team_VFD_list = np.asarray(rescue_team_VFD_list)

        # Keep track of the victims positions
        # Make a list of the victims old positions
        victims_old_pos_list = []
        for victim in victims_Hist:
            victim.Traj.append(victim.old_Pos)
            victims_old_pos_list.append(victim.old_Pos)
        victims_old_pos_list = np.asarray(victims_old_pos_list)

        # Make a list of the agents old positions
        rescue_team_old_pos_list = []
        for agent in rescue_team_Hist:
            rescue_team_old_pos_list.append(agent.old_Pos)
        rescue_team_old_pos_list = np.asarray(rescue_team_old_pos_list)

        # Calculation of the distance between the agents
        old_scouts2rescuers = net.pos2pos(rescue_team_old_pos_list)

        # Calculation of the raw sensations for the rescue team
        old_raw_sensations = net.sensed_pos(victims_old_pos_list, rescue_team_old_pos_list)
        old_wall_sensations = net.sensed_pos(walls_locations, rescue_team_old_pos_list)
        victims_wall_sensations = net.sensed_pos(walls_locations, victims_old_pos_list)

        # Check to see if the sensations are in the agents visual fields
        eval_old_sensations = net.is_seen(rescue_team_VFD_list, old_raw_sensations)

        rescue_team_curr_pos_list = []
        rescue_team_role_list = []
        for agent in rescue_team_Hist:
            eval_old_sensations = net.wall_sensor(old_wall_sensations, old_raw_sensations, victims_wall_sensations,
                                                  rescue_team_Hist.index(agent), eval_old_sensations)
            # Calculation of the sensations for the rescue team
            agent.old_Sensation = agent.update_sensation(rescue_team_Hist.index(agent),
                                                         old_raw_sensations, eval_old_sensations)
            # Calculation of the indices for the rescue team
            agent.old_Index = agent.sensation2index(agent.old_Sensation, agent.max_VisualField)

            # Actions from the scouts
            '''
            calculating the suggested actions
            '''
            agent.act_from_others(rescue_team_Hist.index(agent),
                                  old_raw_sensations, eval_old_sensations,
                                  old_scouts2rescuers, net.adj_mat, adj_mat,
                                  rescue_team_Hist)

            # Actions for the rescue team
            agent.action = np.argmax(agent.Q[agent.old_Index, :])
            '''
            adding agents action in the beginning of the action list
            '''
            agent.action_from_others.insert(0, agent.action)
            # Next positions for the rescue team
            agent.curr_Pos = movement(agent.old_Pos, agent.action, agent.Speed)

            # Smart move algorithm
            # agent.smart_move(agent.old_Pos, agent.old_Index, agent.wereHere)
            # agent.random_walk(agent.old_Index, agent.old_Pos, agent.Speed)
            # agent.ant_colony_move(env_mat, agent.old_Index, env_map)
            # List of the current positions for the rescue team members
            rescue_team_curr_pos_list.append(agent.curr_Pos)

            # List of the roles for the rescue team members
            rescue_team_role_list.append(agent.Role)

        rescue_team_curr_pos_list = np.asarray(rescue_team_curr_pos_list)

        # Calculation of the distance between agents (after their movement)
        curr_scouts2rescuers = net.pos2pos(rescue_team_curr_pos_list)

        # Calculation of the new raw sensations for the rescue team (after their movement)
        curr_raw_sensations = net.sensed_pos(victims_old_pos_list, rescue_team_curr_pos_list)
        curr_wall_sensations = net.sensed_pos(walls_locations, rescue_team_curr_pos_list)

        # Check to see if the sensations are in the agents visual fields
        eval_curr_sensations = net.is_seen(rescue_team_VFD_list, curr_raw_sensations)

        # Calculation of the new sensations for the rescue team (after their movement)
        game_mat = []
        for agent in rescue_team_Hist:
            eval_curr_sensations = net.wall_sensor(curr_wall_sensations, curr_raw_sensations,
                                                   victims_wall_sensations, rescue_team_Hist.index(agent),
                                                   eval_curr_sensations)
            agent.curr_Sensation = agent.update_sensation(rescue_team_Hist.index(agent),
                                                          curr_raw_sensations, eval_curr_sensations)
            '''
            adding agents sensation in the beginning of the sensation list
            '''
            agent.sns_from_others.insert(0, agent.curr_Sensation)
            # Calculation of the indices for the rescue team (after their movement)
            agent.curr_Index = agent.sensation2index(agent.curr_Sensation, agent.max_VisualField)

            # Rewarding the rescue team
            agent.reward = agent.reward_func(agent.curr_Sensation)
            '''
            adding agents reward in the beginning of the reward list
            '''
            agent.rew_from_others.insert(0, agent.reward)
            '''
            making the game matrix
            '''
            game_mat.append(np.tile(np.asarray(agent.rew_from_others).reshape(num_rescue_team, 1), (1, num_rescue_team)))

        '''
        check to see if there are teammates 
        '''
        if num_rescue_team > 1:
            game_mat = np.add(game_mat[0], np.transpose(game_mat[1]))
            '''
            calculate the value and location of the Nash equilibrium from each agents point of view
            '''
            V_P1 = np.amax(np.amin(game_mat, axis=0))
            Loc_P1 = np.argmax(np.argmin(game_mat, axis=0))
            V_P2 = np.amax(np.amin(game_mat, axis=1))
            Loc_P2 = np.argmax(np.argmin(game_mat, axis=1))
            adloc_P1 = Loc_P1
            adloc_P2 = Loc_P2
            '''
            leveraging admissible Nash equilibrium in case of different equilibriums
            '''
            # Using admissible Nash equilibrium in case of different equilibriums
            if game_mat[Loc_P1, Loc_P2] != V_P1 and game_mat[Loc_P1, Loc_P2] != V_P2:
                if V_P1 < V_P2:
                    for k in range(num_rescue_team):
                        if game_mat[Loc_P1, k] == V_P1:
                            V_P2 = game_mat[Loc_P1, k]
                            adloc_P2 = k
                elif V_P2 < V_P1:
                    for k in range(num_rescue_team):
                        if game_mat[k, Loc_P2] == V_P2:
                            V_P1 = game_mat[k, Loc_P2]
                            adloc_P1 = k

            '''
            walking through the agents and equilibriums to calculate action, reward, next position, next sensation
            '''
            for agent, adloc in zip(rescue_team_Hist, [adloc_P1, adloc_P2]):
                agent.action = agent.action_from_others[adloc]
                agent.reward = agent.rew_from_others[adloc]
                agent.curr_Pos = movement(agent.old_Pos, agent.action, agent.Speed)

                agent.curr_Sensation = agent.sns_from_others[adloc]
                agent.curr_Index = agent.sensation2index(agent.curr_Sensation, agent.max_VisualField)
                agent.ant_colony_move(env_mat, agent.old_Index, agent.curr_Index, env_map)
            '''
            update the list of the current positions for the rescue team members
            '''
            rescue_team_curr_pos_list = []
            for agent in rescue_team_Hist:
                # List of the current positions for the rescue team members
                rescue_team_curr_pos_list.append(agent.curr_Pos)
            rescue_team_curr_pos_list = np.asarray(rescue_team_curr_pos_list)

        for agent in rescue_team_Hist:
            # Check to see if the team rescued any victim
            if not agent.Finish:
                rescue_team_Hist, adj_mat = agent.rescue_accomplished(rescue_team_Hist, agent, adj_mat)
                # Keeping track of the rewards
                agent.RewHist.append(agent.reward)
                if agent.CanSeeIt:
                    agent.RewHist_seen.append(agent.reward)
                if agent.Finish and agent.First:
                    agent.Steps.append(t_step)
                    agent.Steps_seen.append(agent.t_step_seen)
                    agent.RewSum.append(np.sum(agent.RewHist))
                    agent.RewSum_seen.append(np.sum(agent.RewHist_seen))
                    rescue_team[agent.id] = agent
                    agent.First = False
                    for victim in victims_Hist:
                        # Check to see if the victim rescued by the team
                        # Keep track of the steps
                        # Remove the victim from the list
                        if not victim.Finish:
                            victims[victim.id] = victim
                            victims_Hist = victim.victim_rescued(rescue_team_old_pos_list,
                                                                 rescue_team_curr_pos_list,
                                                                 rescue_team_role_list,
                                                                 victim, victims_Hist)
                            if victim.Finish and victim.First:
                                victim.Steps.append(t_step)
                                victim.First = False
                                break  # Rescue more than one victim by an agent
        if len(victims_Hist) == 0:
            print(f'all of the victims were rescued in {t_step} steps')
            break

        # Update the rescue team positions
        for agent in rescue_team_Hist:
            agent.old_Pos = agent.curr_Pos

        # Victims' actions and positions
        for victim in victims_Hist:
            # Actions for the victims
            victim.action = np.random.choice(ACTIONS)
            # Victims next positions
            victim.curr_Pos = movement(victim.old_Pos, victim.action, victim.Speed)
            # Update the victims position
            victim.old_Pos = victim.curr_Pos

        # Check for the proper number of episodes
        # convergence_flag = []
        # for agent in rescue_team:
        #     convergence_flag.append(agent.convergence_check(accuracy))
        # if all(convergence_flag):
        #     break

    # Add agents last pos in the trajectory
    for agent in rescue_team:
        for victim in victims:
            if agent.curr_Pos[0] == victim.old_Pos[0] and agent.curr_Pos[1] == victim.old_Pos[1]:
                agent.Traj.append(agent.curr_Pos)

    rescue_team_Traj = []
    rescue_team_RewSum = []
    rescue_team_Steps = []
    rescue_team_RewSum_seen = []
    rescue_team_Steps_seen = []
    rescue_team_Q = []
    largest = len(rescue_team[0].Traj)
    for agent in rescue_team:
        if len(agent.Traj) > largest:
            largest = len(agent.Traj)
        rescue_team_RewSum.append(agent.RewSum)
        rescue_team_Steps.append(agent.Steps)
        rescue_team_RewSum_seen.append(agent.RewSum_seen)
        rescue_team_Steps_seen.append(agent.Steps_seen)
        rescue_team_Q.append(agent.Q)
    for agent in rescue_team:
        while len(agent.Traj) < largest:
            agent.Traj.append(agent.Traj[-1])
        rescue_team_Traj.append(agent.Traj)

    victims_Traj = []
    for victim in victims:
        while len(victim.Traj) < largest:
            victim.Traj.append(victim.Traj[-1])
        victims_Traj.append(victim.Traj)
    print(f'This experiment took {time.time() - tic} seconds')
    return (rescue_team_Traj,
            rescue_team_RewSum, rescue_team_Steps,
            rescue_team_RewSum_seen, rescue_team_Steps_seen,
            rescue_team_Q, victims_Traj, VFD_list, rescue_team_roles)


if Multi_Runs:
    # Multi Runs
    rescue_team_RewSum_Run = []
    rescue_team_Steps_Run = []
    rescue_team_RewSum_seen_Run = []
    rescue_team_Steps_seen_Run = []
    for run in range(NUM_RUNS):
        print(f'Run {run + 1} of {NUM_RUNS}')
        (rescue_team_Traj,
         rescue_team_RewSum, rescue_team_Steps,
         rescue_team_RewSum_seen, rescue_team_Steps_seen,
         rescue_team_Q, victims_Traj, VFD_list, rescue_team_roles) = env(accuracy=1e-7)

        rescue_team_RewSum_Run.append(list(filter(None, rescue_team_RewSum)))
        rescue_team_Steps_Run.append(list(filter(None, rescue_team_Steps)))
        rescue_team_RewSum_seen_Run.append(list(filter(None, rescue_team_RewSum_seen)))
        rescue_team_Steps_seen_Run.append(list(filter(None, rescue_team_Steps_seen)))

    rescue_team_RewSum_Run = np.mean(np.asarray(rescue_team_RewSum_Run), axis=0)
    rescue_team_Steps_Run = np.mean(np.asarray(rescue_team_Steps_Run), axis=0)
    rescue_team_RewSum_seen_Run = np.mean(np.asarray(rescue_team_RewSum_seen_Run), axis=0)
    rescue_team_Steps_seen_Run = np.mean(np.asarray(rescue_team_Steps_seen_Run), axis=0)

    with h5py.File(f'multi_agent_Q_learning_{exp_name}_{str(NUM_RUNS)}Runs.hdf5', 'w') as f:
        for idx, rew_sum in enumerate(rescue_team_RewSum_Run):
            f.create_dataset(f'RS{idx}_reward', data=rew_sum)
        for idx, steps in enumerate(rescue_team_Steps_Run):
            f.create_dataset(f'RS{idx}_steps', data=steps)
        for idx, rew_sum_seen in enumerate(rescue_team_RewSum_seen_Run):
            f.create_dataset(f'RS{idx}_reward_seen', data=rew_sum_seen)
        for idx, steps_seen in enumerate(rescue_team_Steps_seen_Run):
            f.create_dataset(f'RS{idx}_steps_seen', data=steps_seen)
        f.create_dataset('RS_VFD', data=VFD_list)

else:
    # Single Run
    (rescue_team_Traj,
     rescue_team_RewSum, rescue_team_Steps,
     rescue_team_RewSum_seen, rescue_team_Steps_seen,
     rescue_team_Q, victims_Traj, VFD_list, rescue_team_roles) = env(accuracy=1e-7)

    with h5py.File(f'multi_agent_Q_learning_{exp_name}.hdf5', 'w') as f:
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
        f.create_dataset('RS_VFD', data=VFD_list)
        f.create_dataset('RS_ROLES', data=rescue_team_roles)