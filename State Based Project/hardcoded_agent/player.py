from os import path 
import numpy as np 
import torch

def limit_period(angle):
    # turn angle into -1 to 1 
    return angle - torch.floor(angle / 2 + 0.5) * 2 

class Team:
    agent_type = 'state'

    def __init__(self):
        self.team = None
        self.num_players = None

    def generate_single_action(self, pstate, soccer_state, player_id):
        """
        Set the Action for the low-level controller
        :param pstate: State of kart at a certain frame.
        :param soccer_state: State of soccer at a certain frame.
        :return: a pystk.Action (set acceleration, brake, steer)
        """ 
        import numpy as np
        import pystk
        team_id = self.team

        # Get kart velocity
        kart_velocity = np.linalg.norm(pstate['kart']['velocity'])

        # Get kart angle
        kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
        kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_direction = (kart_front-kart_center) / torch.norm(kart_front-kart_center)
        kart_angle = torch.atan2(kart_direction[1], kart_direction[0])

        # Get kart_to_puck_angle
        puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
        kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
        kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0])
        normalized_angle = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

        goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
        puck_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)
        puck_to_goal_line_angle = torch.atan2(puck_to_goal_line[1], puck_to_goal_line[0]) 
        goal_angle = limit_period((kart_angle - puck_to_goal_line_angle)/np.pi)
        #print(f'kart_front: {kart_front}')
        #print(f'kart_center: {kart_center}')

        player_forward = True
        drift = False
        # Compute steering
        if abs(normalized_angle) > 0.9:
            # If the kart is wildly not aligned with the puck, try going in reverse?
            steer = np.sign(normalized_angle)
            puck_aligned = False
            player_forward = False
            # And drift maybe? In reverse?? --Okay no, got really good against Geoffrey and nothing else
            #drift = True
        elif abs(normalized_angle) > 0.001:
            # If the kart is not aligned with the puck, focus on reducing the kart_to_puck_angle_difference first.
            steer = np.sign(normalized_angle)  # Positive angle -> steer right, Negative angle -> steer left
            puck_aligned = False
            player_forward = True
        else:
            # Puck is about aligned, now align with the goal
            puck_aligned = True
            if abs(goal_angle) > 0.001:
                # If not aligned with the goal, adjust steering to reduce goal_angle
                steer = np.sign(goal_angle)  # Positive angle -> steer right, Negative angle -> steer left
            else:
                # Aligned with both puck and goal
                steer = 0

        # Reverse if stuck in goal
        if (abs(float(kart_center[1])) > abs(float(goal_line_center[1]))):
            player_forward = False
                
        # Compute acceleration
        if player_forward:
            acceleration = acceleration = 1 - normalized_angle*0.1
            brake = False
        else:
            acceleration = 0
            brake = True
        
        # Compute braking
        # if steer > 0.80:
        #     acceleration = 0
        #     brake = True
        # else:
        #     brake = False

        #print(f'normalized_angle = {normalized_angle}, goal angle = {goal_angle}, \n, steer = {steer}, aligned = {puck_aligned}\n')
        
        return dict(acceleration=acceleration, steer=steer, brake=brake, drift=drift)
        

    def new_match(self, team: int, num_players: int) -> list:
        self.team, self.num_players = team, num_players
        return ['sara_the_racer'] * num_players

    def act(self, player_state, opponent_state, soccer_state):
        actions = [] 
        for player_id, pstate in enumerate(player_state):
            actions.append(self.generate_single_action(pstate, soccer_state, player_id))                        
        return actions 
