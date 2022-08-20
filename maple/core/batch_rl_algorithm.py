import abc

import gtimer as gt
import numpy as np

from maple.core.rl_algorithm import BaseRLAlgorithm
from maple.data_management.replay_buffer import ReplayBuffer
from maple.samplers.data_collector import PathCollector

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from maple.core.dict_rw import save_paths

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            demonstration_env,
            use_demo_as_init,
            use_demo_cross,
            nums_demo_path,
            use_all_demo_path,
            min_epochs_demo_train,
            num_trains_per_train_loop_demo,
            batch_size_demo,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            demonstration_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            replay_buffer_demo: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            eval_epoch_freq=1,
            expl_epoch_freq=1,
            eval_only=False,
            no_training=False,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            demonstration_env,
            exploration_data_collector,
            evaluation_data_collector,
            demonstration_data_collector,
            replay_buffer,
            replay_buffer_demo,
            eval_epoch_freq=eval_epoch_freq,
            expl_epoch_freq=expl_epoch_freq,
            eval_only=eval_only,
            no_training=no_training,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.use_demo_as_init = use_demo_as_init
        self.use_demo_cross = use_demo_cross
        self.min_epochs_demo_train = min_epochs_demo_train
        self.num_trains_per_train_loop_demo = num_trains_per_train_loop_demo
        self.batch_size_demo = batch_size_demo
        self.nums_demo_path = nums_demo_path
        self.use_all_demo_path = use_all_demo_path
        gt.reset_root()

    def _train(self):
        if self.use_demo_as_init and not self.use_demo_cross:
            demo_paths = self.demo_data_collector.collect_demo_paths(self.max_path_length, self.nums_demo_path, self.use_all_demo_path)
            self.replay_buffer.add_paths(demo_paths)
            self.demo_data_collector.end_epoch(-1)

            for epoch in range(self.min_epochs_demo_train):
                self.training_mode(True)
                for step_demo in range(self.num_trains_per_train_loop_demo):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size_demo)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)
                print('demo epoch:', epoch)

        if self.use_demo_cross and not self.use_demo_as_init:
            demo_paths = self.demo_data_collector.collect_demo_paths(self.max_path_length, self.nums_demo_path, self.use_all_demo_path)
            # revise the rewards based on actions by experts and expl.env
            validate_paths = []
            for demo_path in demo_paths:
                validate_path = self.expl_data_collector.rollout_fn_demo(demo_path['actions'], demo_path['observations'])
                rewards_demo = np.array(demo_path['rewards'])
                rewards_demo_val = np.array(validate_path['rewards'])
                index = np.array(range(0, len(rewards_demo)))
                figure(figsize=(10, 6))
                plt.title("Compare rewards under different env with same demo-actions")
                plt.xlabel("env step")
                plt.ylabel("reward")
                plt.plot(index, rewards_demo, color='b', label="rewards by experts")
                plt.plot(index, rewards_demo_val, color='r', label="rewards by env")
                plt.legend(loc="center right")
                plt.show()
                validate_paths.append(validate_path)
                # if (np.max(validate_path['rewards']) == 5):
                #     save_paths(validate_path, '/home/jinyi/文档/code/maple/data/lift/demo')
            # self.replay_buffer_demo.add_paths(demo_paths)
            self.replay_buffer_demo.add_paths(validate_paths)
            self.demo_data_collector.end_epoch(-1)


        if self.min_num_steps_before_training > 0 and not self._eval_only and not self.use_demo_cross:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=True,  # False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs + 1),
                save_itrs=True,
        ):
            for pre_epoch_func in self.pre_epoch_funcs:
                pre_epoch_func(self, epoch)

            if epoch % self._eval_epoch_freq == 0:
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
            gt.stamp('evaluation sampling')
            print('training epoch:', epoch)
            #
            # '''
            # task1 to make sure demo data is correct
            # '''
            # self.eval_data_collector.demo_in_eval(validate_paths)
            # self.expl_data_collector.demo_in_eval(validate_paths)
            # self._end_epoch(epoch)
            # continue

            if not self._eval_only:

                if self.use_demo_cross and (epoch % 1 == 0 or (epoch < 50)):
                    self.training_mode(True)
                    for step_demo in range(self.num_trains_per_train_loop_demo):
                        train_data = self.replay_buffer_demo.random_batch(
                            self.batch_size_demo)
                        self.trainer.train(train_data)
                    gt.stamp('training', unique=False)
                    self.training_mode(False)

                    if epoch % self._expl_epoch_freq == 0:
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            self.max_path_length,
                            self.num_expl_steps_per_train_loop,
                            discard_incomplete_paths=True,  # False,
                        )
                    gt.stamp('exploration sampling', unique=False)
                    self.replay_buffer.add_paths(new_expl_paths)
                    gt.stamp('data storing', unique=False)
                    self._end_epoch(epoch)
                    continue

                for _ in range(self.num_train_loops_per_epoch):
                    if epoch % self._expl_epoch_freq == 0:
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            self.max_path_length,
                            self.num_expl_steps_per_train_loop,
                            discard_incomplete_paths=True,  # False,
                        )
                    gt.stamp('exploration sampling', unique=False)
                    self.replay_buffer.add_paths(new_expl_paths)
                    gt.stamp('data storing', unique=False)

                    if not self._no_training:
                        self.training_mode(True)
                        for _ in range(self.num_trains_per_train_loop):
                            train_data = self.replay_buffer.random_batch(
                                self.batch_size)
                            self.trainer.train(train_data)
                        gt.stamp('training', unique=False)
                        self.training_mode(False)

            self._end_epoch(epoch)


