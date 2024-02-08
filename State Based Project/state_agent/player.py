from os import path
import numpy as np
import torch
from torch.distributions import Bernoulli

def extract_features_simple(player_state, opponent_state, soccer_state, team_id):
    """
    Ally kart directions and positions and angles, velocities, (8 3d Vectors)
    Direction from puck to scoreline, puck location (2 3d vector)
    Direction from ally karts to puck (2 3d vectors)
    Ally kart to opponent cart directions (4 3d vectors)
    """
    feature_vectors = []

    # Ally kart directions, positions, angle, velocities
    for pstate in player_state:
        kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
        kart_position = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_direction = (kart_front - kart_position) / torch.norm(kart_front - kart_position)
        kart_angle = torch.atan2(kart_direction[1], kart_direction[0])
        kart_velocity = torch.tensor(pstate['kart']['velocity'], dtype=torch.float32)[[0, 2]]
        feature_vectors.append(kart_position)
        feature_vectors.append(kart_direction)
        feature_vectors.append(kart_angle)
        feature_vectors.append(kart_velocity)

    # Direction from puck to scoreline, puck location (2 3d vector)
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
    puck_to_goal_line = (goal_line_center - puck_center) / torch.norm(goal_line_center - puck_center)
    feature_vectors.append(puck_center)
    feature_vectors.append(puck_to_goal_line)

    # Direction from ally karts to goal lines (2 3d vectors)
    for pstate in player_state:
        kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_to_goal_line = (goal_line_center - kart_center) / torch.norm(goal_line_center - kart_center)
        feature_vectors.append(kart_to_goal_line)

    # Ally kart to puck direction (2 3d vectors)
    for pstate in player_state:
        kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center - kart_center)
        feature_vectors.append(kart_to_puck_direction)

    # Ally kart to opponent cart directions (4 3d vectors)
    for pstate in player_state:
        a_kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        for opponent in opponent_state:
            o_kart_center = torch.tensor(opponent['kart']['location'], dtype=torch.float32)[[0, 2]]
            a_to_o_direction = (o_kart_center - a_kart_center) / torch.norm(o_kart_center - a_kart_center)
            feature_vectors.append(a_to_o_direction)

    feature_tensor = torch.hstack(feature_vectors)
    return feature_tensor

# TODO: Remove this class before submission
class Team1:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.feature_size = 52
        self.window_size = 30
        self.window_stride = 2
        self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'imitation_10.jit'))
        self.state_queue = [(torch.zeros(1, self.feature_size),torch.zeros(1, self.feature_size) )] * self.window_size * self.window_stride
        self.max_history = 100

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players
    def generate_single_action(self):
        input_sequence_1 = []
        input_sequence_2 = []
        for i in range(-self.window_size * self.window_stride, 0, self.window_stride):
            input_sequence_1.append(self.state_queue[i][0])
            input_sequence_2.append(self.state_queue[i][1])
        self.state_queue.pop(0)
        input_tensor1 = torch.vstack(input_sequence_1)
        input_tensor2 = torch.vstack(input_sequence_2)

        out_logits = [self.model(input_tensor1.to('mps')), self.model(input_tensor2.to('mps'))]
        actions = []
        for out_logits in out_logits:
            acc_mean = out_logits[-1][0]
            acc_var = out_logits[-1][1]
            steer_mean = out_logits[-1][2]
            steer_var = out_logits[-1][3]
            acc = torch.normal(acc_mean, acc_var)
            steer = torch.normal(steer_mean, steer_var)
            brake = Bernoulli(logits=out_logits[-1][4])
            drift = Bernoulli(logits=out_logits[-1][5])
            actions.append(dict(acceleration=acc, steer=steer, brake=brake.sample(), drift=drift.sample()))
        return actions
    def act(self, player_state, opponent_state, soccer_state):
        
        # (frame['team1_state'], frame['team2_state'], frame['soccer_state'], player_id)
        try:
            current_feature1, current_feature2 = (extract_single_frame_features((player_state, opponent_state, soccer_state, 0)),extract_single_frame_features((player_state, opponent_state, soccer_state, 1)))
            self.state_queue.append((current_feature1, current_feature2))
        except Exception as e:
            print(f"{e}")
        return self.generate_single_action()
        #return [dict(acceleration=0, steer=0, brake=False, drift=False)]

class Team:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'imitation_jurgen_final.jit'), map_location='cpu')

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['sara_the_racer'] * num_players

    def generate_single_action(self, player_state, opponent_state, soccer_state):
        model = self.model
        features = extract_features_simple(player_state, opponent_state, soccer_state, self.team)
        out_logits = model(features)
        player1_outs = out_logits[:4]
        player2_outs = out_logits[4:]
        action_dicts = []
        for player in [player1_outs, player2_outs]:

            # Steering values should be [-1,1]
            steer = torch.clamp(player[1], min=-1.0, max=1.0)
            acceleration = 1 if player[0] > 0.5 else 0
            brake = 1 if player[2] > 0.5 else 0
            drift = 1 if player[3] > 0.5 else 0
            action_dicts.append(dict(acceleration=acceleration, steer=steer, brake=brake, drift=drift))
        return action_dicts

    def act(self, player_state, opponent_state, soccer_state):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT
         CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             You can ignore the camera here.
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param opponent_state: same as player_state just for other team

        :param soccer_state: dict  Mostly used to obtain the puck location
                             ball:  Puck information
                               - location: float3 world location of the puck

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        return self.generate_single_action(player_state, opponent_state, soccer_state)