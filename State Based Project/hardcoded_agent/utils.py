from torch.utils.data import Dataset
from tournament.utils import load_recording
from torch.utils.data import Dataset, DataLoader
from state_agent.player import extract_features_simple
import torch
import pickle
import numpy as np

def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2

'''
def extract_features(state, team_id):
    """
    All kart directions and positions and angles, velocities, (16 3d Vectors)
    Direction from puck to scoreline, puck location (2 3d vector)
    All kart to puck angles (4 3d vectors)
    Direction from all karts to puck (4 3d vectors)
    Velocity of puck (1 3d vector)
    Last touch (pos encoding 0-3, 1xembed_dim)
    All ally kart to opponent cart directions (4 3d vectors)
    Direction from all karts to goal lines (4 3d vectors)
    All actions of the previous frame
    """
    team1, team2, soccer = state
    # features of ego-vehicle
    pstate = None
    kart_states = [team1[0],team1[1], team2[0], team2[1]]
    feature_vectors = []

    # All kart directions, positions, angle, velocities
    for pstate in kart_states:
        kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
        kart_position = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_direction = (kart_front-kart_position)
        kart_angle = torch.atan2(kart_direction[1], kart_direction[0])
        kart_velocity = torch.tensor(pstate['kart']['velocity'], dtype=torch.float32)[[0, 2]]
        feature_vectors.append(kart_position)
        feature_vectors.append(kart_direction)
        feature_vectors.append(kart_angle)
        feature_vectors.append(kart_velocity)

    # Direction from puck to scoreline, puck location(2 3d vector)
    soccer_state = soccer
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(
        dim=0)
    puck_to_goal_line = (goal_line_center - puck_center)
    feature_vectors.append(puck_center)
    feature_vectors.append(puck_to_goal_line)

    # All kart to puck direction (4 3d vectors)
    for pstate in kart_states:
        kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_to_puck_direction = (puck_center - kart_center)
        feature_vectors.append(kart_to_puck_direction)

    # All ally kart to opponent cart directions (4 3d vectors)
    for ally_kart in kart_states[0:2]:
        a_kart_center = torch.tensor(ally_kart['kart']['location'], dtype=torch.float32)[[0, 2]]
        for opponent_cart in kart_states[2:]:
            o_kart_center = torch.tensor(opponent_cart['kart']['location'], dtype=torch.float32)[[0, 2]]
            a_to_o_direction = (o_kart_center - a_kart_center)
            feature_vectors.append(a_to_o_direction)

    # Direction from all karts to goal lines (4 3d vectors)
    for kart in kart_states:
        kart_center = torch.tensor(kart['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_to_goal_line = (goal_line_center - kart_center)
        feature_vectors.append(kart_to_goal_line)

    feature_tensor = torch.hstack(feature_vectors)
    return feature_tensor
'''

def min_max_scale(tensor):
    min_value = torch.min(tensor, dim=0).values
    max_value = torch.max(tensor, dim=0).values
    scaled_tensor = (tensor - min_value) / (max_value - min_value)
    return scaled_tensor

def extract_single_frame_features(current_state):
    """
    All kart directions and positions and angles, velocities, (16 3d Vectors)
    Direction from puck to scoreline, puck location (2 3d vector)
    All kart to puck angles (4 3d vectors)
    Direction from all karts to puck (4 3d vectors)
    Velocity of puck (1 3d vector)
    Last touch (pos encoding 0-3, 1xembed_dim)
    All ally kart to opponent cart directions (4 3d vectors)
    Direction from all karts to goal lines (4 3d vectors)
    All actions of the previous frame
    """
    team1_cur, team2_cur, soccer_cur, player_id = current_state
    # features of ego-vehicle
    pstate = None
    kart_states = [team1_cur[0],team1_cur[1], team2_cur[0], team2_cur[1]]
    team_id = 0
    feature_vectors = []

    # All kart directions, positions, angle, velocities
    for pstate in kart_states:
        kart_front = torch.tensor(pstate['kart']['front'], dtype=torch.float32)[[0, 2]]
        kart_position = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_direction = (kart_front-kart_position)
        kart_velocity = torch.tensor(pstate['kart']['velocity'], dtype=torch.float32)[[0, 2]]
        feature_vectors.append(kart_position)
        feature_vectors.append(kart_direction)
        feature_vectors.append(kart_velocity)


    # Direction from puck to scoreline, puck location(2 3d vector) # TODO double check team id shenanigans
    soccer_state = soccer_cur
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    goal_line_center = torch.tensor(soccer_state['goal_line'][(team_id+1)%2], dtype=torch.float32)[:, [0, 2]].mean(
        dim=0)
    puck_to_goal_line = (goal_line_center - puck_center)
    feature_vectors.append(puck_center)
    feature_vectors.append(puck_to_goal_line)

    # All kart to puck direction (4 3d vectors)
    for pstate in kart_states:
        kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_to_puck_direction = (puck_center - kart_center)
        feature_vectors.append(kart_to_puck_direction)

    # All ally kart to opponent cart directions (4 3d vectors)
    for ally_kart in kart_states[0:2]:
        a_kart_center = torch.tensor(ally_kart['kart']['location'], dtype=torch.float32)[[0, 2]]
        for opponent_cart in kart_states[2:]:
            o_kart_center = torch.tensor(opponent_cart['kart']['location'], dtype=torch.float32)[[0, 2]]
            a_to_o_direction = (o_kart_center - a_kart_center)
            feature_vectors.append(a_to_o_direction)

    #Direction from all karts to goal lines (4 3d vectors)
    for kart in kart_states:
        kart_center = torch.tensor(kart['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_to_goal_line = (goal_line_center - kart_center)
        feature_vectors.append(kart_to_goal_line)

    feature_tensor = torch.hstack(feature_vectors)
    scaled_feature_tensor = min_max_scale(feature_tensor)
    return scaled_feature_tensor

class ImitationSequenceDataset(Dataset):
    def __init__(self, dataset_path, max_pkls=1, window_size=30, window_stride=5):
        from glob import glob
        from os import path

        path = glob(path.join(dataset_path, '*.pkl'))
        idx = 0

        self.training_samples = []
        for im_r in path:  # doublecheck the keys are correct
            rollout = load_recording(im_r)
            history = []
            for frame in rollout:
                current_feature = extract_single_frame_features(
                    (frame['team1_state'], frame['team2_state'], frame['soccer_state'], 0))
                feature_size = current_feature.shape[0]
                if len(history) == 0:
                    history = [(torch.zeros(1, feature_size), torch.zeros(1, 4))] * window_size * window_stride
                current_action = frame['actions'][0]
                drift = 0
                if 'drift' in current_action:
                    drift = current_action['drift'][0]
                actions_values = [current_action['acceleration'][0], current_action['steer'][0],
                                  current_action['brake'][0],
                                  drift]
                action_label = torch.tensor(actions_values, dtype=torch.float32)
                history.append((current_feature, action_label))
                input_sequence = []
                action_sequence = []
                for i in range(-window_size*window_stride, 0, window_stride):
                    input_sequence.append(history[i][0])
                    action_sequence.append(history[i][1])
                history.pop(0)
                input_tensor = torch.vstack(input_sequence)
                action_tensor = torch.vstack(action_sequence)
                self.training_samples.append((input_tensor, action_tensor))
            idx += 1
            if idx > max_pkls:
                break
    def __getitem__(self, idx):
        return self.training_samples[idx]
    def __len__(self):
        return len(self.training_samples)

class ImitationDataset(Dataset):
    def __init__(self, dataset_path, max_pkls=1):
        from glob import glob
        from os import path

        path = glob(path.join(dataset_path, '*.pkl'))
        #print(path)
        pkl_idx = 0
        self.training_samples = []
        for im_r in path:  
            rollout = load_recording(im_r)
            for frame in rollout:
                self.training_samples.append(((frame['team1_state'], frame['team2_state'], frame['soccer_state']), 0, frame['actions'][0], frame['actions'][2] ))
                #self.training_samples.append((([frame['team1_state'][1], frame['team1_state'][0]], frame['team2_state'], frame['soccer_state']), 0, frame['actions'][2]))
                self.training_samples.append(((frame['team2_state'], frame['team1_state'], frame['soccer_state']), 1, frame['actions'][1], frame['actions'][3]))
                #self.training_samples.append((([frame['team2_state'][1], frame['team2_state'][0]], frame['team1_state'], frame['soccer_state']), 1, frame['actions'][3]))
            pkl_idx += 1
            if pkl_idx > max_pkls:
                break

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        state, team_id, player1_action, player2_action = self.training_samples[idx]
        player_state, opponent_state, soccer_state = state
        # run extract features which converts to tensor as well
        #feature_vector = extract_features(state, team_id)
        feature_vector = extract_features_simple(player_state, opponent_state, soccer_state, team_id)
        # convert actions to a tensor
        action_tensors = []
        for player in [player1_action, player2_action]:
            drift = 0
            if 'drift' in player:
                drift = player['drift'][0]
            actions_values = [player['acceleration'][0] > 0.5, player['steer'][0], player['brake'][0], drift]
            action_tensors.append(torch.tensor(actions_values, dtype=torch.float32))
        total_action_tensor = torch.hstack(action_tensors)
        return feature_vector, total_action_tensor

class ImitationJurgenDataset(Dataset):
    def __init__(self, dataset_path, max_pkls=1):
        from glob import glob
        from os import path

        path = glob(path.join(dataset_path, '*.pkl'))
        #print(path)
        pkl_idx = 0
        self.training_samples = []
        for im_r in path:  
            rollout = load_recording(im_r)
            
            if pkl_idx % 2 == 0:
                print(f'TEAM1: {im_r}')
            else:
                print(f'TEAM2: {im_r}')
            
            for frame in rollout:
                # If idx is even, then Jurgen is on team1
                #if int(im_r[22:24]) % 2 == 0:
                if pkl_idx % 2 == 0:
                    self.training_samples.append(((frame['team1_state'], frame['team2_state'], frame['soccer_state']), 0, frame['actions'][0], frame['actions'][2]))
                else:
                    self.training_samples.append(((frame['team2_state'], frame['team1_state'], frame['soccer_state']), 1, frame['actions'][1], frame['actions'][3]))
            pkl_idx += 1
            if pkl_idx > max_pkls:
                break

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        state, team_id, player1_action, player2_action = self.training_samples[idx]
        player_state, opponent_state, soccer_state = state
        # run extract features which converts to tensor as well
        feature_vector = extract_features_simple(player_state, opponent_state, soccer_state, team_id)
        # convert actions to a tensor
        action_tensors = []
        for player in [player1_action, player2_action]:
            drift = 0
            if 'drift' in player:
                drift = player['drift'][0]
            actions_values = [player['acceleration'][0] > 0.5, player['steer'][0], player['brake'][0], drift]
            action_tensors.append(torch.tensor(actions_values, dtype=torch.float32))
        total_action_tensor = torch.hstack(action_tensors)
        return feature_vector, total_action_tensor

class DaggerDataset(Dataset):
    def __init__(self, dataset):
        self.training_samples = dataset

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        # Data format:
        # ((player_team_state1, player_team_state2), opponent_team_state, soccer_state, team_id, (expert_action1, expert_action2))
        player_states, opponent_state, soccer_state, team_id, expert_actions = self.training_samples[idx]
        player_state1, player_state2 = player_states
        expert_action1, expert_action2 = expert_actions
        #feature_vector1 = extract_features((player_state1, opponent_state, soccer_state), team_id)
        #feature_vector2 = extract_features((player_state1, opponent_state, soccer_state), team_id)
        feature_vector1 = extract_features_simple(player_state1, opponent_state, soccer_state, team_id)
        feature_vector2 = extract_features_simple(player_state2, opponent_state, soccer_state, team_id)
        return feature_vector1, feature_vector2, expert_action1, expert_action2

class ReinforceDataset(Dataset):
    def __init__(self, dataset):
        self.training_samples = dataset
        print(len(dataset))

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        """
        Element of training_samples:
        ((current_state_1, current_state_2, soccer_state, team_id),(current_action_1, current_action_2),reward)
        """
        state, action, reward = self.training_samples[idx]
        team1 = state[0]
        team2 = state[1]
        soccer = state[2]
        team_id = state[3]
        feature_vector = extract_features_simple(team1, team2, soccer, team_id)
        action1 = action[0]
        action2 = action[1]
        return feature_vector, action1, action2, torch.tensor(reward, dtype=torch.float32)


def load_imitation_data(dataset_path, max_pkls=1, num_workers=0, batch_size=1024, **kwargs):
    dataset = ImitationDataset(dataset_path, max_pkls=max_pkls, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_imitation_jurgen_data(dataset_path, max_pkls=1, num_workers=0, batch_size=1024, **kwargs):
    dataset = ImitationJurgenDataset(dataset_path, max_pkls=max_pkls, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_dagger_data(dataset, num_workers=0, batch_size=512, **kwargs):
    dataset = DaggerDataset(dataset, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_reinforce_data(dataset, num_workers=0, batch_size=512, **kwargs):
    dataset = ReinforceDataset(dataset, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_imitation_sequence_data(dataset_path, max_pkls=1, window_size=30, window_stride=5, num_workers=0, batch_size=1028, **kwargs):
    dataset = ImitationSequenceDataset(dataset_path, max_pkls=max_pkls, window_size=window_size, window_stride=window_stride, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
