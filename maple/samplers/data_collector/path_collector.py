from collections import deque, OrderedDict
from functools import partial

import numpy as np

from maple.core.eval_util import create_stats_ordered_dict
from maple.samplers.data_collector.base import PathCollector
from maple.samplers.rollout_functions import rollout
from robosuite.scripts.collect_human_demonstrations import collect_human_trajectory_save_as_rollout


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
            rollout_fn_kwargs=None,
            device=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn
        if rollout_fn_kwargs is None:
            rollout_fn_kwargs = {}
        self._rollout_fn_kwargs = rollout_fn_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0

        self._num_actions_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot

        self._device = device

    def collect_new_paths(
            self,
            max_path_length,  # path means the trajectory in RL
            num_steps,  # num of env.step
            discard_incomplete_paths,  # remove the last one path if its length is less than max_path_length
    ):
        paths = []
        num_steps_collected = 0
        num_actions_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            if discard_incomplete_paths and (max_path_length_this_loop < max_path_length):
                break
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                **self._rollout_fn_kwargs
            )
            num_steps_collected += path['path_length']
            num_actions_collected += path['path_length_actions']
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._num_actions_total += num_actions_collected
        self._epoch_paths.extend(paths)
        return paths

    def collect_demo_paths(
            self,
            max_path_length,
            nums_demo_path,
            use_all_demo_path,
    ):
        paths = []
        for i in range(nums_demo_path):
            path = collect_human_trajectory_save_as_rollout(self._env, self._device, max_path_length, use_all_demo_path)
            # self._num_paths_total += 1
            # self._num_steps_total += path['path_length']
            # self._epoch_paths.extend(path)
            paths.append(path)
        return paths

    def rollout_fn_demo(
            self,
            actions,
            state,
    ):
        """
        Testing demo datas by used actions above
        take fixed action into env.step to see env-rewards is same with rewards from demo
        date:2022-0815
        """
        observations_demo = []
        next_observations_demo = []
        rewards_demo = []
        terminals = []
        agent_infos = []
        env_infos = []
        addl_infos = []
        full_observations = []
        full_next_observations = []
        path_length = len(actions)
        path_length_actions = []
        reward_actions_sum = []
        skill_names = []
        max_path_length = 150
        o = self._env.reset()
        o = state[0]
        for a in actions:
            next_o, r, d, env_info = self._env.step(a, image_obs_in_info=False)
            observations_demo.append(o)
            next_observations_demo.append(next_o)
            rewards_demo.append(r)
            terminals.append(False)
            env_infos.append(env_info)
            o = next_o

        return dict(
            observations=np.array(observations_demo),
            actions=np.array(actions),
            next_observations=np.array(next_observations_demo),
            rewards=np.array(rewards_demo),
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            addl_infos=addl_infos,
            full_observations=full_observations,
            full_next_observations=full_next_observations,
            path_length=path_length,
            path_length_actions=path_length_actions,
            reward_actions_sum=reward_actions_sum,
            skill_names=skill_names,
            max_path_length=max_path_length,
        )

    def demo_in_eval(
            self,
            paths,
    ):
        self._epoch_paths.extend(paths)

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        # path_lens = [len(path['actions']) for path in self._epoch_paths]
        path_lens = [path['path_length'] for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
            ('num actions total', self._num_actions_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict


class GoalConditionedPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            desired_goal_key='desired_goal',
            goal_sampling_mode=None,
            **kwargs
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[desired_goal_key]))

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            **kwargs
    ):
        def obs_processor(obs):
            return obs[observation_key]

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
        )
        return snapshot


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            env,
            policy,
            decode_goals=False,
            **kwargs
    ):
        """Expects env is VAEWrappedEnv"""
        super().__init__(env, policy, **kwargs)
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)
