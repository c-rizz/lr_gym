
from pytorch_autoencoders.SimpleAutoencoder import SimpleAutoencoder
from stable_baselines3 import SAC
from typing import Union, Optional, Any, Dict, Tuple
import torch as th
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common import logger
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import gym
import matplotlib.pyplot as plt
import torchvision
import time
import cv2
import copy
import os
import gazebo_gym.utils.dbg.dbg_img as dbg_img

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

class AutoencodingSAC(SAC):
    def __init__(
        self,
        autoencoder : SimpleAutoencoder,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float] = 3e-4,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        n_episodes_rollout: int = -1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        autoencDbgOutFolder : str = None):

        self._autoencoder = autoencoder
        self._optimizer = th.optim.Adam(self._autoencoder.parameters(), lr = 0.001)
        self._encoding_space = gym.spaces.Box(  low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(self._autoencoder.encoding_size,),
                                                dtype=np.float32)

        super().__init__(   policy,
                            env,
                            learning_rate,
                            buffer_size,
                            learning_starts,
                            batch_size,
                            tau,
                            gamma,
                            train_freq,
                            gradient_steps,
                            n_episodes_rollout,
                            action_noise,
                            optimize_memory_usage,
                            ent_coef,
                            target_update_interval,
                            target_entropy,
                            use_sde,
                            sde_sample_freq,
                            use_sde_at_warmup,
                            tensorboard_log,
                            create_eval_env,
                            policy_kwargs,
                            verbose,
                            seed,
                            device,
                            _init_setup_model)
        #print("AutoencodingSAC.self.observation_space = "+str(self.observation_space))
        plt.ion()
        plt.show()

        self._autoenc_batch_size = 4
        self._figure_autoenc_train = plt.figure(num = 1, figsize=(16, 4)) # size in inches
        self._dbg_autoenc_figs = [self._figure_autoenc_train.add_subplot(2,self._autoenc_batch_size,i+1) for i in range(self._autoenc_batch_size*2)]
        #plt.gray()
        self._figure_autoenc_train.show()
        self._autoencDbgOutFolder = autoencDbgOutFolder
        if self._autoencDbgOutFolder is not None:
            os.makedirs(self._autoencDbgOutFolder)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                encoded_observations = self._autoencoder.encode(self._autoencoder.resizeToInputSize(replay_data.observations))
                encoded_next_observations = self._autoencoder.encode(self._autoencoder.resizeToInputSize(replay_data.next_observations))

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(encoded_observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(encoded_next_observations)
                # Compute the target Q value: min over all critics targets
                targets = th.cat(self.critic_target(encoded_next_observations, next_actions), dim=1)
                target_q, _ = th.min(targets, dim=1, keepdim=True)
                # add entropy term
                target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                q_backup = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates for each critic network
            # using action from the replay buffer
            current_q_estimates = self.critic(encoded_observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(encoded_observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
        self.policy = self.policy_class(
            self._encoding_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)


        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)


    def learn(  self,
                total_timesteps: int,
                callback: MaybeCallback = None,
                log_interval: int = 4,
                eval_env: Optional[GymEnv] = None,
                eval_freq: int = -1,
                n_eval_episodes: int = 5,
                tb_log_name: str = "run",
                eval_log_path: Optional[str] = None,
                reset_num_timesteps: bool = True) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())


        neverTrainedEncoder = True
        while self.num_timesteps < total_timesteps:

            t0_collection = time.monotonic()
            rollout = self.collect_rollouts(
                self.env,
                n_episodes=self.n_episodes_rollout,
                n_steps=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
            tf_collection = time.monotonic()
            dur_collection = tf_collection-t0_collection
            #print(f"Collected {rollout.episode_timesteps} steps in {dur_collection}s ({rollout.episode_timesteps/(tf_collection-t0_collection)}fps)")

            #Perform autoencoder training with pre-train data
            if self.num_timesteps >= self.learning_starts and neverTrainedEncoder:
                print("Starting autoencoder training")
                neverTrainedEncoder = False
                epochs = 100
                bestLoss = float("inf")
                for epoch in range(epochs):
                    #Train
                    for i in range(self.replay_buffer.size()):
                        replay_data = self.replay_buffer.sample(self._autoenc_batch_size, env=self._vec_normalize_env)
                        batchTensor = replay_data.observations
                        self._optimizer.zero_grad()
                        with th.set_grad_enabled(True):
                            #print("batchTensor.type()="+str(batchTensor.type()))
                            resizedImgsBatch = self._autoencoder.resizeToInputSize(batchTensor)
                            outputs = self._autoencoder(resizedImgsBatch)
                            loss = th.nn.MSELoss()(outputs, resizedImgsBatch)
                            loss.backward()
                            self._optimizer.step()
                    #Evaluate
                    with th.no_grad():
                        totLoss = 0.0
                        eval_batch_size = min([self.replay_buffer.size(), 64])
                        for i in range(int(self.replay_buffer.size()/eval_batch_size)):
                            replay_data = self.replay_buffer.sample(eval_batch_size, env=self._vec_normalize_env)
                            batchTensor = replay_data.observations
                            resizedImgsBatch = self._autoencoder.resizeToInputSize(batchTensor)
                            outputs = self._autoencoder(resizedImgsBatch)
                            totLoss += th.nn.MSELoss()(outputs, resizedImgsBatch)
                        avgLoss = totLoss/(self.replay_buffer.size()/eval_batch_size)
                    #Log and show output
                    print(f"Finished epoch {epoch}, loss = {avgLoss}")
                    resizedImgs = self._autoencoder.resizeToInputSize(batchTensor.to(self.device))
                    with th.no_grad():
                        encodedDecodedImgs = self._autoencoder(resizedImgs)
                    for i in range(self._autoenc_batch_size):
                        self._dbg_autoenc_figs[i].imshow(torchvision.transforms.ToPILImage(mode="RGB")(batchTensor[i]))
                        self._dbg_autoenc_figs[i+self._autoenc_batch_size].imshow(torchvision.transforms.ToPILImage(mode="RGB")(encodedDecodedImgs[i]))
                    if self._autoencDbgOutFolder is not None:
                        self._figure_autoenc_train.savefig(fname=self._autoencDbgOutFolder+"/autoencPretrain_epoch"+str(epoch))
                    #self._figure.canvas.draw_idle()
                    plt.pause(0.001)
                    if avgLoss < bestLoss:
                        bestLoss = avgLoss
                        bestWeights = copy.deepcopy(self._autoencoder.state_dict())
                self._autoencoder.load_state_dict(bestWeights)


            t0_autoencOpt = time.monotonic()
            if self.num_timesteps > self.learning_starts:
                gradient_steps = int(rollout.episode_timesteps/self._autoenc_batch_size*0.1)+1
                #print(f"Performing {gradient_steps} gradient steps on the autoencoder")
                for gradient_step in range(gradient_steps):
                    replay_data = self.replay_buffer.sample(self._autoenc_batch_size, env=self._vec_normalize_env)
                    batchTensor = replay_data.observations
                    self._optimizer.zero_grad()
                    with th.set_grad_enabled(True):
                        #print("batchTensor.type()="+str(batchTensor.type()))
                        resizedImgsBatch = self._autoencoder.resizeToInputSize(batchTensor)
                        outputs = self._autoencoder(resizedImgsBatch)
                        loss = th.nn.MSELoss()(outputs, resizedImgsBatch)
                        loss.backward()
                        self._optimizer.step()
            dur_autoencOpt = time.monotonic() - t0_autoencOpt
            #print(f"Performed autoencoder training for {dur_autoencOpt}s")

            if rollout.continue_training is False:
                break

            t0_policyTrain = time.monotonic()
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
            tf_policyTrain = time.monotonic()
            dur_policyTrain = tf_policyTrain-t0_policyTrain
            #print(f"Performed policy training for {dur_policyTrain}s")

            tot = float(dur_collection + dur_policyTrain + dur_autoencOpt)
            #print(f"collection = {dur_collection/tot*100:3.3f}% \t autoenc = {dur_autoencOpt/tot*100:3.3f}% \t dur_policyTrain = {dur_policyTrain/tot*100:3.3f}%")


        callback.on_training_end()
        return self

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation
        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        t0 = time.monotonic()
        with th.no_grad():
            resizedImg = self._autoencoder.resizeToInputSize(th.as_tensor(observation).to(self.device))
            t1 = time.monotonic()
            #print(f"resize took {t1-t0}s")
            encodedImg = self._autoencoder.encode(resizedImg)
            t2 = time.monotonic()
            #print(f"encode took {t2-t1}s")

            decodedImg = self._autoencoder.decode(encodedImg)[0]
            t3 = time.monotonic()
            # print(f"decode took {t3-t2}s")
            decodedImgPIL = torchvision.transforms.ToPILImage(mode="RGB")(decodedImg)#Using bgr to publish (opencv standard)
            decodedImgCv = cv2.cvtColor(np.array(decodedImgPIL), cv2.COLOR_RGB2BGR)
            dbg_img.helper.publishDbgImg("img_encdec", decodedImgCv)
            # self._figure_autoenc_predict.gca().imshow(decodedImgPIL)
            # self._figure_autoenc_predict.canvas.draw()
            # #plt.pause(0.001)
            t4 = time.monotonic()
            #print(f"draw took {t4-t3}s")

        ret = self.policy.predict(encodedImg.cpu(), state, mask, deterministic)
        tf = time.monotonic()
        #print(f"Predict took {tf-t0}s")
        return ret
