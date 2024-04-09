from tqdm import tqdm
import datetime
import io
import random
import traceback
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def relable_episode(env, episode, render_kwargs, load_pixels=False):
    rewards = []
    pixels = []
    reward_spec = env.reward_spec()
    states = episode['physics'] if 'physics' in episode else episode['physics_state']
    has_pixels = 'pixels' in episode or 'physics_state' in episode
    need_pixels_load = load_pixels and not has_pixels

    image = None
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        if need_pixels_load:
            env.physics.render(**render_kwargs)
            image = env.physics.render(**render_kwargs)
            pixels.append(image)
        reward = env.task.get_reward(env.physics)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode['reward'] = np.array(rewards, dtype=reward_spec.dtype)
    if need_pixels_load:
        episode['pixels'] = np.stack(pixels)
    elif 'pixels' not in episode:
        episode['pixels'] = episode['observation']
    return episode

def restack(pixels, stack_size):
    # pixels = pixels.transpose(0, 3, 1, 2)
    first = np.expand_dims(pixels[0], axis=0)
    fst = np.repeat(first, stack_size, axis=0)
    pixels = np.concatenate([fst, pixels])
    obs = []
    for i in range(len(pixels) - stack_size + 1):
        obs.append(np.concatenate(list(pixels[i:i + stack_size])))

    return np.array(obs)
    


class OfflineReplayBuffer(IterableDataset):
    def __init__(self, env, replay_dir, max_size, num_workers, discount, obs_type, relable):
        self._env = env
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._loaded = False
        self._obs_type = obs_type
        self._frame_stack = 3
        self._obs_key = 'pixels' if obs_type == 'pixels' else 'observation'
        self._relable = relable

        camera_id = dict(quadruped=2).get(env.domain, 0)
        self.render_kwargs = dict(height=84, width=84, camera_id=camera_id)

    def _load(self, relable):
        print('Labeling data...')
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'))
    
        for eps_idx, eps_fn in tqdm(enumerate(eps_fns)):
            if self._size > self._max_size:
                break
            if '_' in eps_fn.stem:
                # original offlineRL dataset
                eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            else:
                # adapted dataset from murlb 
                eps_idx, eps_len =  0, int(eps_fn.stem.split('-')[-1])
                

            if eps_idx % self._num_workers != worker_id:
                continue
            episode = load_episode(eps_fn)
            if relable:
                episode = self._relable_reward(episode)

            # simulate a stack of frames
            if len(episode['pixels'].shape) == 4 and self._obs_type == 'pixels':
                episode['pixels'] = restack(episode['pixels'], self._frame_stack)
            self._episode_fns.append(eps_fn)
            
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def _sample_episode(self):
        if not self._loaded:
            self._load(self._relable)
            self._loaded = True
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _relable_reward(self, episode):
        return relable_episode(self._env, episode, self.render_kwargs, self._obs_type == 'pixels')

    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode[self._obs_key][idx - 1]
        action = episode['action'][idx]
        next_obs = episode[self._obs_key][idx]
        reward = episode['reward'][idx]
        discount = episode['discount'][idx] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = int(np.random.get_state()[1][0]) + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(env, replay_dir, max_size, batch_size, num_workers,
                       discount, obs_type, relable):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(env, replay_dir, max_size_per_worker,
                                   num_workers, discount, obs_type, relable)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
