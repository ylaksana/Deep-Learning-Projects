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
        self.history = []
        self.max_hist_len = 100
        self.p_states = {0: {'state':'attacking', 'age': 0, 'reached': False} , 1: {'state':'attacking', 'age': 0, 'reached': False}}
        self.max_attacking_age = 400
        self.max_defending_age = 400
        self.max_stuck_age = 20
        self.stuck_displacement_threshold = 0.1

    def calculate_circle_center(self, goal_pos, goal_direction, turning_rad, kart_pos):
        # Normalize the goal direction vector
        goal_direction_norm = goal_direction / torch.norm(goal_direction)

        # Calculate both perpendicular directions to the goal direction
        perp_direction_1 = torch.tensor(
            [-goal_direction_norm[1], goal_direction_norm[0]])  # Rotate goal direction by 90 degrees (left)
        perp_direction_2 = torch.tensor(
            [goal_direction_norm[1], -goal_direction_norm[0]])  # Rotate goal direction by 90 degrees (right)

        # Calculate potential circle centers
        circle_center_1 = goal_pos + perp_direction_1 * turning_rad
        circle_center_2 = goal_pos + perp_direction_2 * turning_rad

        # Choose the circle center that is closest to the kart's current position
        dist_1 = torch.norm(circle_center_1 - kart_pos)
        dist_2 = torch.norm(circle_center_2 - kart_pos)

        if dist_1 < dist_2:
            return circle_center_1
        else:
            return circle_center_2

    def steer_towards_point(self, kart_center, kart_front, goal_pos):

        kart_direction = (kart_front - kart_center) / torch.norm(kart_front - kart_center)
        goal_direction = (goal_pos - kart_center) / torch.norm(kart_front - kart_center)

        kart_angle = torch.atan2(kart_direction[1], kart_direction[0])
        kart_to_goal_angle = torch.atan2(goal_direction[1], goal_direction[0])

        normalized_angle = limit_period((kart_angle - kart_to_goal_angle) / np.pi)
        if abs(normalized_angle) > 0.7:
            steer = np.sign(normalized_angle)
        else:
            steer = np.sign(normalized_angle)

        return steer, normalized_angle

    def generate_single_action(self, player_info, opponent_states, soccer_state, team_id):
        import numpy as np
        import pystk
        base_accel = 0.5
        turning_rad = 5  # experiment with this
        kart_velocity = np.linalg.norm(player_info['kart']['velocity'])
        puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
        goal_line_center = torch.tensor(soccer_state['goal_line'][self.team], dtype=torch.float32)[:, [0, 2]].mean(
            dim=0)
        opp_goal_line = torch.tensor(soccer_state['goal_line'][(self.team + 1) % 2], dtype=torch.float32)[:, [0, 2]].mean(
            dim=0)
        kart_front = torch.tensor(player_info['kart']['front'], dtype=torch.float32)[[0, 2]]
        kart_center = torch.tensor(player_info['kart']['location'], dtype=torch.float32)[[0, 2]]
        dist_to_puck = torch.norm(puck_center - kart_center).item()
        puck_to_goal_dist = torch.norm(puck_center - goal_line_center).item()
        puck_to_opp_goal_dist = torch.norm(puck_center - opp_goal_line).item()
        puck_to_goal_direction = (goal_line_center - puck_center) / torch.norm(goal_line_center - puck_center)

        acceleration = None
        steer = None
        brake = False
        brake = False
        defending = False
        go_to_goal_line = False
        for ostate in opponent_states:
            o_kart_center = torch.tensor(ostate['kart']['location'], dtype=torch.float32)[[0, 2]]
            dist_to_puck = torch.norm(puck_center - o_kart_center).item()
            if dist_to_puck < 20:
                defending = True
                go_to_goal_line = True
        if puck_to_goal_dist > puck_to_opp_goal_dist and defending:
            dist_to_opp_goal_line = torch.norm(opp_goal_line - kart_center).item()
            if dist_to_opp_goal_line > 30 and go_to_goal_line:
                goal_point = opp_goal_line
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                acceleration = 1
            elif dist_to_opp_goal_line > 10 and go_to_goal_line:
                goal_point = opp_goal_line
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                brake = True
                acceleration = 0
            elif dist_to_opp_goal_line <= 10 and go_to_goal_line:
                go_to_goal_line = False
            else:
                goal_point = puck_center
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                acceleration = 1
        else:
            if dist_to_puck < 40:
                goal_point = puck_center - puck_to_goal_direction
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                if abs(normalized_angle) > 0.8:
                    # If the kart is wildly not aligned with the puck, try going in reverse?
                    steer = -np.sign(normalized_angle)
                    acceleration = 0
                    brake = True

                elif abs(normalized_angle) > 0.01:
                    # If the kart is not aligned with the puck, focus on reducing the kart_to_puck_angle_difference first.
                    steer = np.sign(normalized_angle)  # Positive angle -> steer right, Negative angle -> steer left
                    acceleration = 0.5
                else:
                    acceleration = 1
            elif dist_to_puck < 10:
                goal_point = goal_line_center
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                acceleration = 1
            else:
                goal_point = puck_center - 10 * puck_to_goal_direction

                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                if abs(normalized_angle) > 0.5:
                    acceleration = 0.1
                else:
                    acceleration = 0.7
        return dict(acceleration=acceleration, steer=steer, brake=brake, drift=False)

    def generate_single_action_with_state(self, player_info, opponent_states, soccer_state, player_id):
        """
        Set the Action for the low-level controller
        :param pstate: State of kart at a certain frame.
        :param soccer_state: State of soccer at a certain frame.
        :return: a pystk.Action (set acceleration, brake, steer)
        """ 
        import numpy as np
        import pystk
        state_dict = self.p_states[player_id]
        state_age = state_dict['age']
        current_state = state_dict['state']
        reached = state_dict['reached']
        base_accel = 0.5
        turning_rad = 5 # experiment with this
        kart_velocity = np.linalg.norm(player_info['kart']['velocity'])
        puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
        goal_line_center = torch.tensor(soccer_state['goal_line'][(self.team+1)%2], dtype=torch.float32)[:, [0, 2]].mean(
            dim=0)
        opp_goal_line = torch.tensor(soccer_state['goal_line'][self.team], dtype=torch.float32)[:, [0, 2]].mean(
            dim=0)
        kart_front = torch.tensor(player_info['kart']['front'], dtype=torch.float32)[[0, 2]]
        kart_center = torch.tensor(player_info['kart']['location'], dtype=torch.float32)[[0, 2]]
        dist_to_puck = torch.norm(puck_center - kart_center).item()
        puck_to_goal_dist = torch.norm(puck_center - goal_line_center).item()
        kart_to_goal_dist = torch.norm(kart_center - goal_line_center).item()
        puck_to_opp_goal_dist = torch.norm(puck_center - opp_goal_line).item()
        puck_to_goal_direction = (goal_line_center - puck_center)/ torch.norm(goal_line_center-puck_center)
        opp_goal_to_our_goal = (goal_line_center - opp_goal_line) / torch.norm(goal_line_center - opp_goal_line)
        acceleration = None
        steer = None
        brake = False
        if current_state == 'attacking':
            if dist_to_puck < 10:
                goal_point = goal_line_center
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                acceleration = 1
            elif dist_to_puck < 20:
                goal_point = puck_center
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                if normalized_angle > 0.1:
                    acceleration = 0
                    brake = True
                else:
                    acceleration = 0.6
            elif dist_to_puck < 40:
                goal_point = puck_center - puck_to_goal_direction
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                if (abs(normalized_angle) > 0.1):
                    acceleration = 0.5
                    brake = True
                else:
                    acceleration = 0.5
            else:
                goal_point = puck_center - 5*puck_to_goal_direction
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                if abs(normalized_angle) > 0.5:
                    acceleration = 0.2
                else:
                    acceleration = 0.8

            self.p_states[player_id] = {'state': 'attacking', 'age':state_age + 1, 'reached': False}
            if state_age >= self.max_attacking_age:
                self.p_states[player_id] = {'state': 'defending', 'age': 0,'reached': False}
                #print(f'player {player_id} switched from attacking to defending')
        elif current_state == 'defending':
            opp_goal_point = opp_goal_line + 10*opp_goal_to_our_goal
            dist_to_opp_goal_line = torch.norm(opp_goal_point - kart_center).item()
            self.p_states[player_id] = {'state': 'defending', 'age': state_age + 1, 'reached': False}
            if dist_to_opp_goal_line > 30 and not reached:
                goal_point = opp_goal_point
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                acceleration = 1
            elif dist_to_opp_goal_line <= 30 or reached:
                goal_point = goal_line_center
                reached = True
                #print(f'goal_line_center {goal_point}')
                steer, normalized_angle = self.steer_towards_point(kart_center, kart_front, goal_point)
                #print(kart_velocity)
                brake = True
                acceleration = 0.05
                if abs(normalized_angle) < 0.3:
                    self.p_states[player_id] = {'state': 'attacking', 'age':0, 'reached': False}
                    #print(f'player {player_id} switched from defending to attacking mode')
                else:
                    self.p_states[player_id] = {'state': 'defending', 'age': state_age + 1, 'reached': reached}
        elif current_state == 'stuck':
            acceleration = 0.0
            steer = 0.0
            brake = True
            if state_age > self.max_stuck_age:
                self.p_states[player_id] = {'state': "attacking", 'age': 0,'reached': False}
            else:
                self.p_states[player_id] = {'state': 'stuck', 'age':state_age + 1, 'reached':False}

        if len(self.history) > 20 and current_state != 'stuck':
            avg_position = 0.0
            for old_player_info, _, _ in self.history[len(self.history)-20:]:
                avg_position += torch.tensor(old_player_info[player_id]['kart']['location'], dtype=torch.float32)[[0, 2]]
            avg_position = avg_position/20
            if torch.norm(kart_center - avg_position).item() < self.stuck_displacement_threshold:
                self.p_states[player_id] = {'state': 'stuck', 'age':0, 'reached':False}

        return dict(acceleration=acceleration, steer=steer, brake=brake, drift=False)



    def new_match(self, team: int, num_players: int) -> list:
        self.team, self.num_players = team, num_players
        self.history = []
        return ['sara_the_racer'] * num_players

    def act(self, player_state, opponent_state, soccer_state):
        self.history.append((player_state, opponent_state, soccer_state))

        actions = [] 
        for player_id, pstate in enumerate(player_state):
            #actions.append(self.generate_single_action(pstate, opponent_state, soccer_state, player_id))
            actions.append(self.generate_single_action_with_state(pstate, opponent_state, soccer_state, player_id))
        return actions 
