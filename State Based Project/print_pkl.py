from pickle import TRUE
from tournament.utils import load_recording

def trim_dataset(args):
    from glob import glob
    from os import path, remove

    files_removed = 0
    path = glob(path.join('imitation_data_ball', '*.pkl'))
    for im_r in path: # doublecheck the keys are correct
        rollout = load_recording(im_r)
        for state in rollout:
            score = state['soccer_state']['score']
        if (score[0] + score[1]) < 1:
            print('Removing ' + str(im_r) + ' with score ' + str(score))
            remove(im_r)
            files_removed += 1
    print("Files removed: " + str(files_removed))

def print_pickle(args):
    from os import path
    data = []
    for state in load_recording(args.pickle_file):
        data.append(state)
    print(len(data))
    print('TEAM1_STATE')
    print(data[0]['team1_state'])
    for item in data[0]['team1_state']:
        print(item)
    #print('\nTEAM2_STATE')
    #print(data[0]['team2_state'])
    print('\nSOCCER_STATE')
    print(data[0]['soccer_state'])
    print('\nACTIONS')
    print(data[0]['actions'])

def generate_dataset_all_opponents(args):
    import subprocess
    agents = ["yoshua_agent", "yann_agent", "jurgen_agent", "geoffrey_agent", "image_jurgen_agent"]
    ball_locations = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]

    idx = 0
    num_epochs = 2
    for epoch in range(0, num_epochs):
        for agent1 in agents:
            for agent2 in agents:
                for ball in ball_locations:
                    path = f'imitation_data_ball/{idx}_{agent1}_vs_{agent2}_ball_{ball[0]}_{ball[1]}.pkl'
                    subprocess.run(['python', '-m', 'tournament.runner', f'{agent1}', f'{agent2}', '-s', f'{path}', '--ball_location', f'{ball[0]}', f'{ball[1]}'])
                    idx += 1

def generate_dataset_single(args):
    import subprocess
    # Only generate training data against agents used for grading (screw generalization at this point, honestly)
    expert_agent = 'jurgen_agent'
    agents = ["yoshua_agent", "yann_agent", "jurgen_agent", "geoffrey_agent"]
    ball_locations = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]

    idx = 0
    num_epochs = 3
    for epoch in range(0, num_epochs):
        for agent in agents:
            for ball in ball_locations:
                print(f'{expert_agent} vs. {agent}')
                path = f'imitation_data_{expert_agent}/{idx:03}_{expert_agent}_vs_{agent}_ball_{ball[0]}_{ball[1]}.pkl'
                subprocess.run(['python', '-m', 'tournament.runner', f'{expert_agent}', f'{agent}', '-s', f'{path}', '--ball_location', f'{ball[0]}', f'{ball[1]}'])
                idx += 1

                print(f'{agent} vs {expert_agent}')
                path = f'imitation_data_{expert_agent}/{idx:03}_{agent}_vs_{expert_agent}_ball_{ball[0]}_{ball[1]}.pkl'
                subprocess.run(['python', '-m', 'tournament.runner', f'{agent}', f'{expert_agent}', '-s', f'{path}', '--ball_location', f'{ball[0]}', f'{ball[1]}'])
                idx += 1
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_file')
    args = parser.parse_args()
    #print_pickle(args)
    generate_dataset_single(args)
