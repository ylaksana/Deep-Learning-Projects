from ppo_agent.utils import extract_features_simple
from ppo_agent.model import FeedForwardNN
import torch.optim as optim
from tournament.utils import  load_recording
import torch
from torch.distributions import Bernoulli, Normal
from os import path
import time
import numpy as np
import random
import math


def exp_decay(kart_position, puck_position):
    # Assume kart_position and puck_position are PyTorch tensors
    # Calculate the Euclidean distance between the kart and the puck
    distance = torch.sqrt(torch.sum((kart_position - puck_position) ** 2))

    # Define a scale factor for the exponential decay
    scale_factor = 1.0  # Adjust this to control how quickly the reward decreases with distance

    # Calculate the reward as an exponential decay function of the distance
    reward = torch.exp(-scale_factor * distance)

    # Ensure the reward is always positive (greater than zero)
    return torch.clamp(reward, min=0.01)

def run_match(state_file, agent1, agent2, parallel=1):
    import subprocess
    from glob import glob
    from os import path

    # Runs 5 matches of agent1 vs. agent2 with different initial ball locations
    subprocess.run(['python', '-m', 'tournament.runner', f'{agent1}', f'{agent2}', '-t', '-j', f'{parallel}', '-s', f'{state_file}'])

    trajectories = []
    path = glob(path.join('ppo_data', '*.pkl'))
    for im_r in path:
        rollout = load_recording(im_r)
        states = []
        for frame in rollout:
            states.append(frame)
        if len(states) > 0:
            result = states[-1]['soccer_state']['score']
            print(f'match result: {result} number of frames {len(states)}')
            trajectories.append((result, states))

    return trajectories


class Team:
    agent_type = 'state'

    def __init__(self, policy_class=FeedForwardNN, training_mode=False, load_training_model=False, **hyperparameters):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.team = None
        self.num_players = None
        self.actor, self.critic = None, None
        self.training_mode = training_mode
        if training_mode:
            self._init_hyperparameters(hyperparameters)
            self.actor = policy_class(34, 10)
            self.critic = policy_class(34, 1)
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
            if load_training_model:
                self.actor = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'actor.jit'),
                                            map_location='cpu')
                self.critic = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'critic.jit'),
                                             map_location='cpu')
        else:
            try:
                self.actor = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'actor.jit'), map_location='cpu')
                self.critic = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), f'critic.jit'), map_location='cpu')
                print("actor critic models found and loaded")
            except Exception as e:
                print("actor critic models not found during inference mode, using newly initialized models")
                self.actor = policy_class(34, 10)
                self.critic = policy_class(34, 1)
            self.device = torch.device('cpu')
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  # ALG STEP 3
            #print(f'batch_obs: {batch_obs.shape} batch_acts: {batch_acts.shape} batch_log_probs: {batch_log_probs.shape} batch_rtgs: {batch_rtgs.shape} batch_lens: {len(batch_lens)}')
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach().to('cpu'))

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                print(f'saving models at iteration: {i_so_far}')
                self.actor = self.actor.to('cpu')
                scripted_actor = torch.jit.script(self.actor)
                scripted_actor.save(f'ppo_agent/actor_{i_so_far}.jit')
                scripted_actor.save(f'ppo_agent/actor.jit')
                self.actor.to(self.device)

                self.critic = self.critic.to('cpu')
                scripted_critic = torch.jit.script(self.critic)
                scripted_critic.save(f'ppo_agent/critic_{i_so_far}.jit')
                scripted_critic.save(f'ppo_agent/critic.jit')
                self.critic.to(self.device)

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        state_file = "ppo_data/temp_run.pkl"
        t = 0  # Keeps track of how many timesteps we've run so far this batch
        while t < self.timesteps_per_batch:
            trajectories = run_match(state_file, 'ppo_agent', 'ppo_agent')
            # Keep simulating until we've run more than or equal to specified timesteps per batch
            for (result, states) in trajectories:
                ep_rews = []  # rewards collected per episode
                num_frames = 0
                for frame in states:
                    t += 1
                    num_frames += 1
                    team_id = random.randint(0, 1) # randomly choose which perspective
                    if team_id:
                        obs = extract_features_simple(frame['team1_state'], frame['team2_state'], frame['soccer_state'],
                                                      team_id)
                        rew = self.get_reward(frame['team1_state'], frame['team2_state'], frame['soccer_state'], result, team_id)
                    else:
                        obs = extract_features_simple(frame['team2_state'], frame['team1_state'], frame['soccer_state'],
                                                      team_id)
                        rew = self.get_reward(frame['team2_state'], frame['team1_state'], frame['soccer_state'], result, team_id)
                    batch_obs.append(obs)
                    action, log_prob = self.get_action(obs, frame['actions'], team_id) # instead of get action, use the saved action in frame
                    ep_rews.append(rew)
                    batch_acts.append(action)
                    batch_log_probs.append(log_prob)
                    if t >= self.timesteps_per_batch:
                        break
                batch_lens.append(num_frames)
                batch_rews.append(ep_rews)
                if t >= self.timesteps_per_batch:
                    break
        print(f'number of trajectories processed: {len(batch_lens)}')
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.stack(batch_obs)
        batch_acts = torch.stack(batch_acts)
        batch_log_probs = torch.stack(batch_log_probs)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return (batch_obs.to(self.device), batch_acts.to(self.device), batch_log_probs.to(self.device),
                batch_rtgs.to(self.device), batch_lens)
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_reward(self, player_states, opp_states, soccer_state, result, team_id):
        reward = 0.0
        puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
        goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(
            dim=0)

        #reward += 10*result[team_id]

        #reward += exp_decay(goal_line_center, puck_center).item()

        for pstate in player_states:
            kart_position = torch.tensor(pstate['kart']['location'], dtype=torch.float32)[[0, 2]]
            reward += 1/(torch.norm(puck_center - kart_position) + 1e-6).item() # exp_decay(kart_position, puck_center).item()#

        return reward
    def get_action(self, obs, action_states, team_id):
        action_logits = self.actor(obs.to(self.device))
        player1_outs = action_logits[:5]
        player2_outs = action_logits[5:]
        action_tensors = []
        #print(f'actions: {action_states}')
        if team_id == 0:
            player1_action, player2_action = action_states[0], action_states[1]
            for player in [player1_action, player2_action]:
                drift = 0
                if 'drift' in player:
                    drift = player['drift']
                actions_values = [player['acceleration'], player['steer'], player['brake'], drift]
                action_tensors.append(torch.tensor(actions_values, dtype=torch.float32).to(self.device))
        else:
            player1_action, player2_action = action_states[1], action_states[3]
            for player in [player1_action, player2_action]:
                drift = 0
                if 'drift' in player:
                    drift = player['drift']
                actions_values = [player['acceleration'], player['steer'], player['brake'], drift]
                action_tensors.append(torch.tensor(actions_values, dtype=torch.float32).to(self.device))

        log_prob = []
        for player, action in zip([player1_outs, player2_outs], action_tensors):
            acc_dist = Bernoulli(logits=player[0])
            # Steering values should be [-1,1]?
            steer_mean = torch.clamp(player[1], min=-1.0, max=1.0)
            steer_var = torch.nn.functional.softplus(player[2]) + 0.1
            steer_dist = Normal(steer_mean, steer_var)
            brake_dist = Bernoulli(logits=player[3])
            drift_dist = Bernoulli(logits=player[4])

            acc, steer, brake, drift = action[0], action[1], action[2], action[3] #acc_dist.sample(), steer_dist.sample(), brake_dist.sample(), drift_dist.sample()

            acc_logp, steer_logp = acc_dist.log_prob(acc), steer_dist.log_prob(steer)
            brake_logp, drift_logp = brake_dist.log_prob(brake), drift_dist.log_prob(drift)
            log_prob.append(acc_logp + steer_logp + brake_logp + drift_logp)

        total_prob = log_prob[0] + log_prob[1]
        return torch.hstack(action_tensors).detach(), total_prob.detach() # 1, 8

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs.to(self.device)).squeeze()
        action_logits = self.actor(batch_obs.to(self.device)) # B, 12
        player1_outs = action_logits[:, :5]
        player2_outs = action_logits[:, 5:]
        player1_acts = batch_acts[:, :4]
        player2_acts = batch_acts[:, 4:]
        #print(f'action_logits: {action_logits.shape} V: {V.shape}')
        log_prob = []
        for player, recorded_acts in zip([player1_outs, player2_outs], [player1_acts, player2_acts]):
            acc_dist = Bernoulli(logits=player[:,0])
            steer_mean = torch.clamp(player[:,1], min=-1.0, max=1.0)
            steer_var = torch.nn.functional.softplus(player[:,2]) + 0.1
            steer_dist = Normal(steer_mean, steer_var)
            brake_dist = Bernoulli(logits=player[:,3])
            drift_dist = Bernoulli(logits=player[:,4])

            acc, steer, brake, drift = recorded_acts[:, 0], recorded_acts[:, 1], recorded_acts[:, 2], recorded_acts[:, 3]
            #print(f'brake: {brake.shape} drift: {drift.shape}')
            acc_logp, steer_logp = acc_dist.log_prob(acc), steer_dist.log_prob(steer)
            brake_logp, drift_logp = brake_dist.log_prob(brake), drift_dist.log_prob(drift)
            log_prob.append(acc_logp + steer_logp + brake_logp + drift_logp)

        total_prob = log_prob[0] + log_prob[1]
        return V, total_prob
    def _log_summary(self):

        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def _init_hyperparameters(self, hyperparameters):
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration
        self.lr = 0.001  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA


        self.save_freq = 1  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    ### Normal Agent Code ###
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

    def generate_action_dicts(self, state):
        model = self.actor

        #features = extract_features(state, self.team)
        player_state, opponent_state, soccer_state = state
        features = extract_features_simple(player_state, opponent_state, soccer_state, self.team)
        out_logits = model(features)

        player1_outs = out_logits[:5]
        player2_outs = out_logits[5:]
        action_dicts = []
        for player in [player1_outs, player2_outs]:
            acc = Bernoulli(logits=player[0])
            # Steering values should be [-1,1]?
            steer_mean = torch.clamp(player[1], min=-1.0, max=1.0)
            steer_var = torch.nn.functional.softplus(player[2]) +0.1
            steer = Normal(steer_mean, steer_var)
            #print(f'steer_mean: {steer_mean}\tsteer_var: {steer_var}')
            #print(f'\t{steer.sample()}')
            brake = Bernoulli(logits=player[3])
            drift = Bernoulli(logits=player[4])
            action_dicts.append(dict(acceleration=acc.sample().item(), steer=steer.sample().item(), brake=brake.sample().item(), drift=drift.sample().item()))
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
        # (frame['team1_state'], frame['team2_state'], frame['soccer_state'], player_id)

        #state1 = (player_state, opponent_state, soccer_state)
        #state2 = ([player_state[1], player_state[0]], opponent_state, soccer_state)
        #return [self.generate_single_action(state1, 0), self.generate_single_action(state2, 1)]
        state = (player_state, opponent_state, soccer_state)
        return self.generate_action_dicts(state)



