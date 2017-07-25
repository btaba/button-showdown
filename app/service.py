
import os
import logging
import datetime
import json
import click
import numpy as np
import joblib

from rllab.misc import tensor_utils
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.website_env import WebsiteEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Flask
from flask import Flask
from flask_cors import CORS
from flask import request
from flask import jsonify

logger = logging.getLogger(__name__)

CONFIG = {
    'num_actions': 1,
    'num_observations': 1,
    'snapshot_dir_itr': 0,
    'snapshot_dir': '',
    'agent_dict': {},
    'batch_size': 20,
    'batch': [],
    'max_bad_updates': 60,
    'num_bad_updates': 0,
    'noise': 0.5,
    'mean_rewards': [],
    'max_bad_buttons_to_update': 5,
    'file_to_load_policy': ''
}
home = os.path.expanduser('~')
SNAPSHOT_DIR = os.path.join(home, 'rllab/experiments{}')


def listify_dict(d):
    """
    Recursively convert arrays to lists in a python object
    """

    if type(d) is np.ndarray:
        return d.tolist()

    if isinstance(d, list):
        for idx, k in enumerate(d):
            d[idx] = listify_dict(k)
        return d

    if isinstance(d, dict):
        for k in d.keys():
            d[k] = listify_dict(d[k])

    return d


def arrayify_dict(d):
    """
    Assume dict is made of values that are lists, that need to be
    converted to arrays
    """
    for k in d.keys():
        d[k] = np.array(d[k])


def validate_action_dim(action):
    """
    Validate that the observations are properly formatted
    """
    if not action:
        return False

    if len(action) == CONFIG['num_actions']:
        return True
    return False


def validate_obs_dim(obs):
    """
    Validate that the observations are properly formatted
    """
    if not obs:
        return False

    if len(obs) == CONFIG['num_observations']:
        return True
    return False


def make_path(r, obs, action, agent_info):
    path = dict()
    path["rewards"] = tensor_utils.stack_tensor_list(r)
    path["observations"] = tensor_utils.stack_tensor_list([obs])
    path["actions"] = tensor_utils.stack_tensor_list([action])
    path["env_infos"] = {}
    path["agent_infos"] = tensor_utils.stack_tensor_dict_list([agent_info])
    return path


def update_policy(obs, action, agent_info, reward):
    """
    Update the policy based on one observation, action, reward pair
    """

    obs = np.array(obs)
    action = np.array(action)
    arrayify_dict(agent_info)
    r = np.array(reward)

    path = make_path(r, obs, action, agent_info)
    update_in_batch(path)


def update_in_batch(path):
    """
    update the policy in batch (asynchronously if possible)
    """
    CONFIG['batch'].append(path)

    if len(CONFIG['batch']) > CONFIG['batch_size']:
        algo = CONFIG['agent_dict']['agent']
        good_update = algo.train_from_single_sample(
            CONFIG['batch'], log_dir=CONFIG['snapshot_dir'])

        if not good_update:
            CONFIG['num_bad_updates'] += 1

            if CONFIG['num_bad_updates'] > CONFIG['max_bad_updates']:
                print('Did max bad updates, purging batch data')
                CONFIG['batch'] = []
                return

            print('Update step was bad, skipping, but saving data')
            return

        rewards = list(map(lambda x: x['rewards'], CONFIG['batch']))
        print('rewards are ', rewards)
        mean_reward = np.mean(rewards)
        CONFIG['mean_rewards'].append(mean_reward)
        print('Mean reward at itr {} is {}'.format(
            CONFIG['snapshot_dir_itr'], mean_reward))
        CONFIG['snapshot_dir_itr'] += 1

        CONFIG['batch'] = []

    print('Saving data to ', CONFIG['data_file'])
    path_copy = dict(path)
    path_copy = listify_dict(path_copy)
    with open(CONFIG['data_file'], 'a') as f:
        f.write(json.dumps(path_copy))
        f.write('\n')


def init_agent(num_obs=1, num_actions=1):
    # set the snapshot dir stuff
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    CONFIG['snapshot_dir'] = SNAPSHOT_DIR.format(datetime_str)

    os.makedirs(CONFIG['snapshot_dir'], exist_ok=True)
    print('Making new network on ', datetime_str,
          'with {} actions and {} obs'.format(num_actions, num_obs))

    # create data file
    file = 'data{}_act{}_obs{}.json'.format(
        CONFIG['snapshot_dir_itr'], num_actions, num_obs)
    CONFIG['data_file'] = os.path.join(CONFIG['snapshot_dir'], file)
    file = open(CONFIG['data_file'], 'w')
    file.close()
    print('Made data file at ', CONFIG['data_file'])
    CONFIG['snapshot_dir_itr'] += 1

    CONFIG['num_actions'] = num_actions
    CONFIG['num_observations'] = num_obs

    # Create the env and agent
    env = WebsiteEnv(num_actions=num_actions, action_bounds=[1.] * num_actions,
                     num_observations=num_obs,
                     observation_bound=1.)
    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        adaptive_std=True
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        discount=0.99,
        step_size=0.01,
        optimizer_args={"accept_violation": False, "cg_iters": 20},
        # center_adv=False,  # we will scale rewards appropriately
    )
    algo.init_opt()

    CONFIG['agent_dict']['env'] = env
    CONFIG['agent_dict']['agent'] = algo
    CONFIG['agent_dict']['policy'] = algo.policy

    # train from a single samples to load the graph
    obs = env.observation_space.sample()
    action, agent_info = algo.policy.get_action(obs)
    path = make_path(np.array([1.]), obs, action, agent_info)
    algo.train_from_single_sample([path])


def init_agent_from_file():
    filename = CONFIG['file_to_load_policy']
    print('loading model from file', filename)
    params = joblib.load(filename)
    CONFIG['agent_dict']['env'] = params['env']
    CONFIG['agent_dict']['agent'] = params['algo']
    CONFIG['agent_dict']['policy'] = params['policy']
    CONFIG['snapshot_dir_itr'] = params['itr'] + 1
    CONFIG['snapshot_dir'] = os.path.dirname(filename)
    CONFIG['num_actions'] = params['env'].action_dim
    CONFIG['num_observations'] = params['env'].observation_space.shape[0]

    # create data file
    file = 'data{}_act{}_obs{}.json'.format(
        CONFIG['snapshot_dir_itr'], CONFIG['num_actions'],
        CONFIG['num_observations'])
    CONFIG['data_file'] = os.path.join(CONFIG['snapshot_dir'], file)
    file = open(CONFIG['data_file'], 'w')
    file.close()


def mse(x, y):
    return np.mean((x - y)**2)


def add_noise_to_action_array(x, noise):
    # orig_x = np.copy(x)
    x = np.copy(x)
    noise = abs(noise)
    noise = min(1., noise)
    noise = max(1e-8, noise)

    # get the new mean button to sample from
    # for each dimension, go left or right, whichever is valid
    for idx, xx in enumerate(x):
        if xx + noise > 1:
            x[idx] = xx - noise
        elif xx - noise < -1:
            x[idx] = xx + noise
        else:
            if np.random.rand(1) > .5:
                x[idx] = xx - noise
            else:
                x[idx] = xx + noise

    # give the mean point, generate a random point close to it, with the std being `noise` / 2.
    x_prime = np.random.multivariate_normal(x, np.diag([noise / 2.] * len(x)))
    x_prime = np.clip(x_prime, -1, 1)
    # print(x_prime)

    # any -1, or 1 values should be moved back into the space by some random value
    edge_interval = noise * 0.1
    x_prime[x_prime == 1] = np.random.uniform(1 - edge_interval, 1., len(x_prime[x_prime == 1]))
    x_prime[x_prime == -1] = np.random.uniform(-1, -1 + edge_interval, len(x_prime[x_prime == -1]))

    # d = np.mean((x_prime - orig_x)**2)

    return x_prime


"""
FLASK APP
"""


def configure_app(flask_app):
    CORS(flask_app)

    if CONFIG['file_to_load_policy'] == '':
        init_agent()
    else:
        init_agent_from_file()


# Needs to be external for gunicorn
app = Flask(__name__)
configure_app(app)


@app.route('/')
def health():
    return 'ok', 200


@app.route('/get_action', methods=['POST'])
def get_action():
    req = request.get_json()
    obs = req.get('observation')
    noise = req.get('noise', CONFIG['noise'])
    n_bads = req.get('n_bads', 1)
    
    if not validate_obs_dim(obs):
        return 'invalid input obs', 400

    action, agent_info = CONFIG['agent_dict']['policy'].get_action(obs)
    agent_info = listify_dict(agent_info)
    action = np.clip(action, -1, 1)

    # format response
    good_dict = dict()
    good_dict['action'] = list(action)
    good_dict['agent_info'] = agent_info
    
    # format bad response
    bad_button_list = []
    for i in range(n_bads):
        # bad_action = action + noise * np.random.randn(len(action))
        bad_action = add_noise_to_action_array(action, noise)
        # bad_action = np.clip(bad_action, -1, 1)
        bad_dict = dict()
        bad_dict['action'] = list(bad_action)
        bad_dict['agent_info'] = agent_info
        bad_button_list.append(bad_dict)

    resp = {
        'good': good_dict,
        'bad': bad_button_list
    }
    return jsonify(resp), 200


@app.route('/get_random_obs', methods=['GET'])
def get_random_obs():
    obs = CONFIG['agent_dict']['env'].observation_space.sample()
    return jsonify({'observation': list(obs)}), 200


@app.route('/update_policy_from_game', methods=['POST'])
def update_policy_from_game():
    req = request.get_json()
    button_list = req.get('button_list')
    obs = req.get('observation')

    if not validate_obs_dim(obs):
        return 'invalid input obs', 400

    # get the button that was picked
    picked_button = None
    unpicked_buttons = []

    for b in button_list:
        if b['picked'] is True:
            picked_button = b
        else:
            unpicked_buttons.append(b)

    if picked_button is None:
        return 'no button was picked', 400

    # the picked button gets a reward of 1
    update_policy(obs, picked_button['action'], picked_button['agent_info'], [1])

    # update policy with unpicked buttons that are maximally far from the button
    # that was picked
    mses = map(
        lambda x: mse(np.array(x['action']), picked_button['action']),
        unpicked_buttons)
    unpicked_buttons = sorted(
        zip(unpicked_buttons, mses),
        key=lambda x: x[1],
        reverse=True)
    unpicked_buttons = list(map(lambda x: x[0], unpicked_buttons))

    num_unpicked = len(unpicked_buttons)
    num_to_update = min(num_unpicked, CONFIG['max_bad_buttons_to_update'])
    for u in unpicked_buttons[:num_to_update]:
        update_policy(obs, u['action'], u['agent_info'], [-1])

    return 'ok', 200


@app.route('/init_networks', methods=['POST'])
def init_networks():
    req = request.get_json()
    num_actions = req.get('num_actions')
    num_obs = req.get('num_observations')
    init_agent(num_obs, num_actions)
    return 'ok', 200


@click.command()
@click.option('--server-port', default=8080, help='server port')
def start_app(server_port):
    logger.info('STarting on ', server_port)
    app.run(port=server_port)


if __name__ == '__main__':
    start_app()
