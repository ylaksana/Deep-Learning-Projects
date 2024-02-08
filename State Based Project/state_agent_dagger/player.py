from os import path
import numpy as np
import torch
from state_agent_dagger.utils import extract_features, extract_single_frame_features
from torch.distributions import Bernoulli, Normal
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

class Team_single:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.model = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'imitation_final_sus.jit'))

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

    def generate_single_action(self, state):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        features = extract_features(state)
        out_logits = self.model(features.to(device))
        acc_mean = out_logits[0]
        acc_var = torch.nn.functional.softplus(out_logits[1])
        steer_mean = out_logits[2]
        steer_var = torch.nn.functional.softplus(out_logits[3])
        acc = torch.normal(acc_mean, acc_var)
        steer = torch.normal(steer_mean, steer_var)
        brake = Bernoulli(logits=torch.nn.functional.sigmoid(out_logits[4]))
        drift = Bernoulli(logits=torch.nn.functional.sigmoid(out_logits[5]))
        return dict(acceleration=acc.item(), steer=steer.item(), brake=brake.sample(), drift=drift.sample())

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
        # (frame['team1_state'], frame['team2_state'], frame['soccer_state'], player_id)

        state1 = (player_state, opponent_state, soccer_state)
        state2 = ([player_state[1], player_state[0]], opponent_state, soccer_state)
        return [self.generate_single_action(state1), self.generate_single_action(state2)]

class Team:
    agent_type = 'state'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.model_player_1 = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'imitation_final.jit'))
        self.model_player_2 = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'imitation_final.jit'))

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

    def generate_single_action(self, state, player_id):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        if player_id == 0:
            model = self.model_player_1
        else:
            model = self.model_player_2

        features = extract_features(state, self.team)
        out_logits = model(features.to(device))
        acc_mean = out_logits[0]
        acc_var = torch.nn.functional.softplus(out_logits[1])
        steer_mean = out_logits[2]
        steer_var = torch.nn.functional.softplus(out_logits[3])
        acc = torch.normal(acc_mean, acc_var)
        steer = torch.normal(steer_mean, steer_var)
        brake = Bernoulli(logits=torch.nn.functional.sigmoid(out_logits[4]))
        drift = Bernoulli(logits=torch.nn.functional.sigmoid(out_logits[5]))
        return dict(acceleration=acc.item(), steer=steer.item(), brake=brake.sample(), drift=drift.sample())

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
        # (frame['team1_state'], frame['team2_state'], frame['soccer_state'], player_id)

        state1 = (player_state, opponent_state, soccer_state)
        state2 = ([player_state[1], player_state[0]], opponent_state, soccer_state)
        return [self.generate_single_action(state1, 0), self.generate_single_action(state2, 1)]