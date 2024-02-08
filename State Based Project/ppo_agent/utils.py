import torch

def limit_period(angle):
    # turn angle into -1 to 1
    return angle - torch.floor(angle / 2 + 0.5) * 2
def extract_features_simple(player_state, opponent_state, soccer_state, team_id):
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
    # features of ego-vehicle
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

    # Direction from puck to scoreline, puck location(2 3d vector)
    puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
    goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
    puck_to_goal_line = (goal_line_center - puck_center) / torch.norm(goal_line_center - puck_center)
    feature_vectors.append(puck_center)
    feature_vectors.append(puck_to_goal_line)

    # Direction from ally karts to goal lines (4 3d vectors)
    for pstate in player_state:
        kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_to_goal_line = (goal_line_center - kart_center) / torch.norm(goal_line_center - kart_center)
        feature_vectors.append(kart_to_goal_line)

    # Ally kart to puck direction (2 3d vectors)
    for pstate in player_state:
        kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center - kart_center)
        feature_vectors.append(kart_to_puck_direction)

    # Ally kart to opponent cart directions (2 3d vectors)
    for pstate in player_state:
        a_kart_center = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
        for opponent in opponent_state:
            o_kart_center = torch.tensor(opponent['kart']['location'], dtype=torch.float32)[[0, 2]]
            a_to_o_direction = (o_kart_center - a_kart_center) / torch.norm(o_kart_center - a_kart_center)
            feature_vectors.append(a_to_o_direction)

    feature_tensor = torch.hstack(feature_vectors)
    return feature_tensor


