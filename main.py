import time
import torch
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-graphical backend
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import win_unicode_console
import argparse
from datetime import datetime
import logging
import json
win_unicode_console.enable()
from DQN import *
from reinforce import *
from actor_critic import *
from actor_critic_v2 import *
from ppo import *
from env import Runner_Env, WebDriver
from dino_agent import Dino

def setup_logger(run_dir):
    # Create log file
    log_file = os.path.join(run_dir, 'logs', 'run.log')
    
    # Configure logger
    logger = logging.getLogger('dino_dqn')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_run_dir():
    # Create runs directory if it doesn't exist
    if not os.path.exists('runs'):
        os.makedirs('runs')
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', timestamp)
    os.makedirs(run_dir)
    
    # Create subdirectories for different types of data
    os.makedirs(os.path.join(run_dir, 'models'))  # For saved models
    os.makedirs(os.path.join(run_dir, 'images'))  # For saved images
    os.makedirs(os.path.join(run_dir, 'logs'))    # For log files
    
    return run_dir

def save_args_to_json(args, run_dir):
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Save to JSON file
    param_file = os.path.join(run_dir, 'logs', 'param.json')
    with open(param_file, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    return param_file

def online_training(
        cur_train_round, 
        trainer, 
        agent, 
        env, 
        max_epoch, 
        max_step, 
        run_dir, 
        logger, 
        debug=False, 
        model_path=None, 
        min_memory_count_to_start_training=100, 
        image_wait_time=0.1
    ):
    max_duration_achieved = 0  # Track the maximum duration time achieved
    best_model_path = None  # Track the path of the best model
    best_model_screenshot_path = None  # Track the path of the best model screenshot
    if model_path is not None:
        trainer.load_model(model_path)
        trainer.epsilon_init = 1
        logger.info(f"Loaded model from {model_path}")

    logger.info(f"Starting training with {max_epoch} epochs and maximum {max_step} steps per epoch")

    for j in range(max_epoch):
        epoch_start_time = time.time()  # Record epoch start time
        env.start()
        env.wait(4)
        init_time = env.get_time()
        image_1, image_2 = env.get_image(image_wait_time)
        s_1, s_2 = env.get_states(image_1, image_2)
        r = 0
        if debug:
            plot_track_image(env, image_1, run_dir, -1)
        
        current_steps = 0
        for i in range(max_step):
            a = trainer.choose_action(s_1, s_2)
            if a == 1:
                agent.jump()
            # elif a == 2:
            #     agent.crawl()
            
            image_1, image_2 = env.get_image(image_wait_time)
            s_1_, s_2_ = env.get_states(image_1, image_2)
            done = env.is_done(image_1)

            if debug:
                plot_track_image(env, image_1, run_dir, i)
            if done:
                r = -5
            else:
                time_diff = env.get_time() - init_time
                r = max(agent.jump_duration * time_diff, 5)

            if debug:
                cur_states = torch.cat([s_1, s_2])
                next_states = torch.cat([s_1_, s_2_])
                logger.info(f"Memory saved: \nState: \t\t{cur_states} \nNext State: \t{next_states} \nAction: {a} \nReward: {r}")

            trainer.save_memory(s_1, s_2, s_1_, s_2_, a, r)
            s_1, s_2 = s_1_, s_2_
            current_steps = i + 1
            epoch_duration = time.time() - epoch_start_time

            if done or i == max_step - 1:
                logger.info(f"Epoch {j} completed! Steps: {current_steps}, Duration: {epoch_duration:.2f}s, Current epsilon: {trainer.epsilon_init:.3f}")
                break

            if debug:
                print(f"-------------------------")

        if trainer.memory_counter >= min_memory_count_to_start_training:
            logger.info(f"Learning step - Memory counter: {trainer.memory_counter}, Epsilon: {trainer.epsilon_init:.3f}")
            trainer.learning()

        # Check and save model after epoch completion
        if epoch_duration > max_duration_achieved:
            max_duration_achieved = epoch_duration
            # Remove previous best model if it exists
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            if best_model_screenshot_path is not None and os.path.exists(best_model_screenshot_path):
                os.remove(best_model_screenshot_path)
            # save best model
            best_model_path = os.path.join(run_dir, 'models', f"best_model_at_train_round_{cur_train_round}_epoch_{j}.pt")
            trainer.save_model(best_model_path)
            logger.info(f"New best model saved! Duration: {max_duration_achieved:.2f}s")
            # Save screenshot of the game state
            best_model_screenshot_path = os.path.join(run_dir, 'images', f'best_model_screenshot_at_train_round_{cur_train_round}_epoch_{j}.png')
            og_image = env.get_game_screenshot()
            cv2.imwrite(best_model_screenshot_path, og_image)
            logger.info(f"Game screenshot saved to: {best_model_screenshot_path}")

        if debug:
            logger.info("Debug mode: Press Enter to continue...")
            input()
        else:
            env.wait(1)

    logger.info(f"Training completed! Best model achieved {max_duration_achieved:.2f}s and saved at {best_model_path}")
    return best_model_path

def online_training_reinforce(
        cur_train_round, 
        trainer, 
        agent, 
        env, 
        max_epoch, 
        max_step, 
        run_dir, 
        logger, 
        debug=False, 
        model_path=None, 
        min_memory_count_to_start_training=100,
        image_wait_time=0.1
    ):
    """
    REINFORCE Actor-Critic training function with replay memory and TD learning.
    """
    max_duration_achieved = 0  # Track the maximum duration time achieved
    best_model_path = None  # Track the path of the best model
    best_model_screenshot_path = None  # Track the path of the best model screenshot
    
    if model_path is not None:
        trainer.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")

    logger.info(f"Starting REINFORCE Actor-Critic training with {max_epoch} epochs, {max_step} steps per episode")

    for j in range(max_epoch):
        episode_start_time = time.time()  # Record episode start time
        env.start()
        env.wait(4)
        init_time = env.get_time()
        image_1, image_2 = env.get_image(image_wait_time)
        s_1, s_2 = env.get_states(image_1, image_2)
        
        if debug:
            plot_track_image(env, image_1, run_dir, -1)
        
        current_steps = 0
        episode_rewards = []  # Track rewards for this episode
        
        for i in range(max_step):
            # Choose action using REINFORCE Actor-Critic
            a = trainer.choose_action(s_1, s_2)
            if a == 1:
                agent.jump()
            # elif a == 2:
            #     agent.crawl()
            
            # Get next state and reward
            image_1, image_2 = env.get_image(image_wait_time)
            s_1_, s_2_ = env.get_states(image_1, image_2)
            done = env.is_done(image_1)

            if debug:
                plot_track_image(env, image_1, run_dir, i)
            
            # Calculate reward
            if done:
                r = -1
            else:
                time_diff = env.get_time() - init_time
                r = max(agent.jump_duration * time_diff, 1)

            # Store experience for REINFORCE training
            trainer.store_experience(s_1, s_2, a, r)
            episode_rewards.append(r)

            if debug:
                cur_states = np.concatenate([s_1, s_2])
                next_states = np.concatenate([s_1_, s_2_])
                logger.info(f"Experience stored: \nState: \t\t{cur_states} \nNext State: \t{next_states} \nAction: {a} \nReward: {r}")

            s_1, s_2 = s_1_, s_2_
            current_steps = i + 1
            episode_duration = time.time() - episode_start_time

            if done or i == max_step - 1:
                total_reward = sum(episode_rewards)
                logger.info(f"Episode {j+1} completed! Steps: {current_steps}, Duration: {episode_duration:.2f}s, Total Reward: {total_reward:.2f}")
                break

            if debug:
                print(f"-------------------------")

        # Train after each episode (REINFORCE)
        if len(trainer.states) > 0:
            logger.info(f"Training with {len(trainer.states)} experiences from episode {j+1}")
            loss_info = trainer.train()
            logger.info(f"Training completed. Policy Loss: {loss_info['policy_loss']:.4f}")
            
            # Check and save model after episode training
            if episode_duration > max_duration_achieved:
                max_duration_achieved = episode_duration
                # Remove previous best model if it exists
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                if best_model_screenshot_path is not None and os.path.exists(best_model_screenshot_path):
                    os.remove(best_model_screenshot_path)
                # save best model
                best_model_path = os.path.join(run_dir, 'models', f"best_model_at_train_round_{cur_train_round}_epoch_{j}.pt")
                trainer.save_model(best_model_path)
                logger.info(f"New best model saved! Duration: {max_duration_achieved:.2f}s")
                # Save screenshot of the game state
                best_model_screenshot_path = os.path.join(run_dir, 'images', f'best_model_screenshot_at_train_round_{cur_train_round}_epoch_{j}.png')
                og_image = env.get_game_screenshot()
                cv2.imwrite(best_model_screenshot_path, og_image)
                logger.info(f"Game screenshot saved to: {best_model_screenshot_path}")

        if debug:
            logger.info("Debug mode: Press Enter to continue...")
            input()
        else:
            env.wait(1)

    logger.info(f"REINFORCE training completed! Best model achieved {max_duration_achieved:.2f}s and saved at {best_model_path}")
    return best_model_path

def online_training_actor_critic(
        cur_train_round, 
        trainer, 
        agent, 
        env, 
        max_epoch, 
        max_step, 
        run_dir, 
        logger, 
        debug=False, 
        actor_model_path=None, 
        critic_model_path=None, 
        min_memory_count_to_start_training=100,
        image_wait_time=0.1
    ):
    """
    Actor-Critic training function with replay memory and TD learning.
    """
    max_duration_achieved = 0  # Track the maximum duration time achieved
    best_actor_model_path = None  # Track the path of the best actor model
    best_critic_model_path = None  # Track the path of the best critic model
    best_model_screenshot_path = None  # Track the path of the best model screenshot
    
    if actor_model_path is not None and critic_model_path is not None:
        trainer.load_model(actor_model_path, critic_model_path)
        logger.info(f"Loaded model from {actor_model_path} and {critic_model_path}")

    logger.info(f"Starting Actor-Critic training with {max_epoch} epochs, {max_step} steps per episode")

    for j in range(max_epoch):
        episode_start_time = time.time()  # Record episode start time
        env.start()
        env.wait(4)
        init_time = env.get_time()
        image_1, image_2 = env.get_image(image_wait_time)
        s_1, s_2 = env.get_states(image_1, image_2)
        
        if debug:
            plot_track_image(env, image_1, run_dir, -1)
        
        current_steps = 0
        episode_rewards = []  # Track rewards for this episode
        
        for i in range(max_step):
            # Choose action using Actor-Critic
            a = trainer.choose_action(s_1, s_2)
            if a == 1:
                agent.jump()
            # elif a == 2:
            #     agent.crawl()
            
            # Get next state and reward
            image_1, image_2 = env.get_image(image_wait_time)
            s_1_, s_2_ = env.get_states(image_1, image_2)
            done = env.is_done(image_1)

            if debug:
                plot_track_image(env, image_1, run_dir, i)
            
            # Calculate reward
            if done:
                r = -5
            else:
                time_diff = env.get_time() - init_time
                r = max(agent.jump_duration * time_diff, 5)

            # Store experience with done flag for TD learning
            trainer.store_experience(s_1, s_2, a, r, s_1_, s_2_, done)
            episode_rewards.append(r)

            if debug:
                cur_states = np.concatenate([s_1, s_2])
                next_states = np.concatenate([s_1_, s_2_])
                logger.info(f"Experience stored: \nState: \t\t{cur_states} \nNext State: \t{next_states} \nAction: {a} \nReward: {r} \nDone: {done}")

            s_1, s_2 = s_1_, s_2_
            current_steps = i + 1
            episode_duration = time.time() - episode_start_time

            if done or i == max_step - 1:
                total_reward = sum(episode_rewards)
                logger.info(f"Episode {j+1} completed! Steps: {current_steps}, Duration: {episode_duration:.2f}s, Total Reward: {total_reward:.2f}")
                break

            if debug:
                print(f"-------------------------")

        # Train using replay memory (similar to DQN)
        if trainer.memory_counter >= min_memory_count_to_start_training:
            logger.info(f"Training with {trainer.memory_counter} experiences from replay memory")
            loss_info = trainer.train()
            logger.info(f"Training completed. Policy Loss: {loss_info['policy_loss']:.4f}, Value Loss: {loss_info['value_loss']:.4f}")
            
            # Check and save model after training
            if episode_duration > max_duration_achieved:
                max_duration_achieved = episode_duration
                # Remove previous best model if it exists
                if best_actor_model_path is not None and os.path.exists(best_actor_model_path):
                    os.remove(best_actor_model_path)
                if best_critic_model_path is not None and os.path.exists(best_critic_model_path):
                    os.remove(best_critic_model_path)
                if best_model_screenshot_path is not None and os.path.exists(best_model_screenshot_path):
                    os.remove(best_model_screenshot_path)
                # save best model
                best_actor_model_path = os.path.join(run_dir, 'models', f"best_actor_model_at_train_round_{cur_train_round}_epoch_{j}.pt")
                best_critic_model_path = os.path.join(run_dir, 'models', f"best_critic_model_at_train_round_{cur_train_round}_epoch_{j}.pt")
                trainer.save_model(best_actor_model_path, best_critic_model_path)
                logger.info(f"New best model saved! Duration: {max_duration_achieved:.2f}s")
                # Save screenshot of the game state
                best_model_screenshot_path = os.path.join(run_dir, 'images', f'best_model_screenshot_at_train_round_{cur_train_round}_epoch_{j}.png')
                og_image = env.get_game_screenshot()
                cv2.imwrite(best_model_screenshot_path, og_image)
                logger.info(f"Game screenshot saved to: {best_model_screenshot_path}")

        if debug:
            logger.info("Debug mode: Press Enter to continue...")
            input()
        else:
            env.wait(1)

    logger.info(f"Actor-Critic training completed! Best model achieved {max_duration_achieved:.2f}s and saved at {best_actor_model_path}")
    return best_actor_model_path, best_critic_model_path

def online_training_actor_critic_v2(
        cur_train_round, 
        trainer, 
        agent, 
        env, 
        max_epoch, 
        max_step, 
        run_dir, 
        logger, 
        debug=False, 
        actor_model_path=None, 
        critic_model_path=None, 
        min_trajectories_to_start_training=30,
        image_wait_time=0.1
    ):
    """
    Actor-Critic v2 training function with trajectory-based replay memory and GAE.
    """
    max_duration_achieved = 0  # Track the maximum duration time achieved
    best_actor_model_path = None  # Track the path of the best actor model
    best_critic_model_path = None  # Track the path of the best critic model
    best_model_screenshot_path = None  # Track the path of the best model screenshot
    
    if actor_model_path is not None and critic_model_path is not None:
        trainer.load_model(actor_model_path, critic_model_path)
        logger.info(f"Loaded model from {actor_model_path} and {critic_model_path}")

    logger.info(f"Starting Actor-Critic v2 training with {max_epoch} epochs, {max_step} steps per episode")
    logger.info(f"Using trajectory-based memory with GAE, min trajectories: {min_trajectories_to_start_training}")

    for j in range(max_epoch):
        episode_start_time = time.time()  # Record episode start time
        env.start()
        env.wait(4)
        init_time = env.get_time()
        image_1, image_2 = env.get_image(image_wait_time)
        s_1, s_2 = env.get_states(image_1, image_2)
        
        if debug:
            plot_track_image(env, image_1, run_dir, -1)
        
        current_steps = 0
        episode_rewards = []  # Track rewards for this episode
        
        for i in range(max_step):
            # Choose action using Actor-Critic v2
            a = trainer.choose_action(s_1, s_2)
            if a == 1:
                agent.jump()
            # elif a == 2:
            #     agent.crawl()
            
            # Get next state and reward
            image_1, image_2 = env.get_image(image_wait_time)
            s_1_, s_2_ = env.get_states(image_1, image_2)
            done = env.is_done(image_1)

            if debug:
                plot_track_image(env, image_1, run_dir, i)
            
            # Calculate reward
            if done:
                # r = -5
                r = -10
            else:
                # time_diff = env.get_time() - init_time
                # r = max(agent.jump_duration * time_diff, 5)
                r = 0

            # Add step to trajectory (automatically handles episode completion)
            current_time = env.get_time() - init_time
            trainer.add_step(s_1, s_2, a, r, done, current_time)
            episode_rewards.append(r)

            if debug:
                cur_states = np.concatenate([s_1, s_2])
                next_states = np.concatenate([s_1_, s_2_])
                logger.info(f"Step added to trajectory: \nState: \t\t{cur_states} \nNext State: \t{next_states} \nAction: {a} \nReward: {r} \nDone: {done}")

            s_1, s_2 = s_1_, s_2_
            current_steps = i + 1
            episode_duration = time.time() - episode_start_time

            if done or i == max_step - 1:
                total_reward = sum(episode_rewards)
                logger.info(f"Episode {j+1} completed! Steps: {current_steps}, Duration: {episode_duration:.2f}s, Total Reward: {total_reward:.2f}")
                break

            if debug:
                print(f"-------------------------")

        # Train using trajectory-based memory with GAE
        memory_stats = trainer.get_memory_stats()
        if memory_stats['num_trajectories'] >= min_trajectories_to_start_training:
            logger.info(f"Training with {memory_stats['num_trajectories']} trajectories, {memory_stats['total_steps']} total steps")
            loss_info = trainer.train()
            logger.info(f"Training completed. Policy Loss: {loss_info['policy_loss']:.4f}, Value Loss: {loss_info['value_loss']:.4f}")
            
            # Check and save model after training
            if episode_duration > max_duration_achieved:
                max_duration_achieved = episode_duration
                # Remove previous best model if it exists
                if best_actor_model_path is not None and os.path.exists(best_actor_model_path):
                    os.remove(best_actor_model_path)
                if best_critic_model_path is not None and os.path.exists(best_critic_model_path):
                    os.remove(best_critic_model_path)
                if best_model_screenshot_path is not None and os.path.exists(best_model_screenshot_path):
                    os.remove(best_model_screenshot_path)
                # save best model
                best_actor_model_path = os.path.join(run_dir, 'models', f"best_actor_model_v2_at_train_round_{cur_train_round}_epoch_{j}.pt")
                best_critic_model_path = os.path.join(run_dir, 'models', f"best_critic_model_v2_at_train_round_{cur_train_round}_epoch_{j}.pt")
                trainer.save_model(best_actor_model_path, best_critic_model_path)
                logger.info(f"New best model saved! Duration: {max_duration_achieved:.2f}s")
                # Save screenshot of the game state
                best_model_screenshot_path = os.path.join(run_dir, 'images', f'best_model_screenshot_v2_at_train_round_{cur_train_round}_epoch_{j}.png')
                og_image = env.get_game_screenshot()
                cv2.imwrite(best_model_screenshot_path, og_image)
                logger.info(f"Game screenshot saved to: {best_model_screenshot_path}")
        else:
            logger.info(f"Not enough trajectories for training: {memory_stats['num_trajectories']}/{min_trajectories_to_start_training}")

        if debug:
            logger.info("Debug mode: Press Enter to continue...")
            input()
        else:
            env.wait(1)

    logger.info(f"Actor-Critic v2 training completed! Best model achieved {max_duration_achieved:.2f}s and saved at {best_actor_model_path}")
    return best_actor_model_path, best_critic_model_path

def online_training_ppo(
        cur_train_round, 
        trainer, 
        agent, 
        env, 
        max_epoch, 
        max_step, 
        run_dir, 
        logger, 
        debug=False, 
        actor_model_path=None, 
        critic_model_path=None, 
        min_memory_count_to_start_training=100,
        image_wait_time=0.1
    ):
    """
    PPO training function - trains after each episode.
    """
    max_duration_achieved = 0  # Track the maximum duration time achieved
    best_actor_model_path = None  # Track the path of the best actor model
    best_critic_model_path = None  # Track the path of the best critic model
    best_model_screenshot_path = None  # Track the path of the best model screenshot
    
    if actor_model_path is not None and critic_model_path is not None:
        trainer.load_model(actor_model_path, critic_model_path)
        logger.info(f"Loaded model from {actor_model_path} and {critic_model_path}")

    logger.info(f"Starting PPO training with {max_epoch} epochs, {max_step} steps per episode")

    for j in range(max_epoch):
        episode_start_time = time.time()  # Record episode start time
        env.start()
        env.wait(4)
        init_time = env.get_time()
        image_1, image_2 = env.get_image(image_wait_time)
        s_1, s_2 = env.get_states(image_1, image_2)
        
        if debug:
            plot_track_image(env, image_1, run_dir, -1)
        
        current_steps = 0
        episode_rewards = []  # Track rewards for this episode
        
        for i in range(max_step):
            # Choose action using PPO
            a, log_prob, value, entropy = trainer.choose_action(s_1, s_2)
            if a == 1:
                agent.jump()
            # elif a == 2:
            #     agent.crawl()
            
            # Get next state and reward
            image_1, image_2 = env.get_image(image_wait_time)
            s_1_, s_2_ = env.get_states(image_1, image_2)
            done = env.is_done(image_1)

            if debug:
                plot_track_image(env, image_1, run_dir, i)
            
            # Calculate reward
            if done:
                # r = -5
                r = -10
            else:
                # time_diff = env.get_time() - init_time
                # r = max(agent.jump_duration * time_diff, 5)
                r = 0

            # Store experience for PPO training
            current_time = env.get_time() - init_time
            trainer.add_step(s_1, s_1_, a, r, value, log_prob, done, current_time)
            episode_rewards.append(r)

            if debug:
                cur_states = np.concatenate([s_1, s_2])
                next_states = np.concatenate([s_1_, s_2_])
                logger.info(f"Experience stored: \nState: \t\t{cur_states} \nNext State: \t{next_states} \nAction: {a} \nReward: {r}")

            s_1, s_2 = s_1_, s_2_
            current_steps = i + 1
            episode_duration = time.time() - episode_start_time

            if done or i == max_step - 1:
                total_reward = sum(episode_rewards)
                logger.info(f"Episode {j+1} completed! Steps: {current_steps}, Duration: {episode_duration:.2f}s, Total Reward: {total_reward:.2f}")
                break

            if debug:
                print(f"-------------------------")

        # Train after each episode
        memory_stats = trainer.get_memory_stats()
        if memory_stats['num_trajectories'] > min_memory_count_to_start_training:
            logger.info(f"Training with {memory_stats['num_trajectories']} trajectories from episode {j+1}")
            loss_info = trainer.train()
            logger.info(f"Training completed. Loss: {loss_info}")
            
            # Check and save model after episode training
            if episode_duration > max_duration_achieved:
                max_duration_achieved = episode_duration
                # Remove previous best model if it exists
                if best_actor_model_path is not None and os.path.exists(best_actor_model_path):
                    os.remove(best_actor_model_path)
                if best_critic_model_path is not None and os.path.exists(best_critic_model_path):
                    os.remove(best_critic_model_path)
                if best_model_screenshot_path is not None and os.path.exists(best_model_screenshot_path):
                    os.remove(best_model_screenshot_path)
                # save best model
                best_actor_model_path = os.path.join(run_dir, 'models', f"best_actor_model_at_train_round_{cur_train_round}_epoch_{j}.pt")
                best_critic_model_path = os.path.join(run_dir, 'models', f"best_critic_model_at_train_round_{cur_train_round}_epoch_{j}.pt")
                trainer.save_model(best_actor_model_path, best_critic_model_path)
                logger.info(f"New best model saved! Duration: {max_duration_achieved:.2f}s")
                # Save screenshot of the game state
                best_model_screenshot_path = os.path.join(run_dir, 'images', f'best_model_screenshot_at_train_round_{cur_train_round}_epoch_{j}.png')
                og_image = env.get_game_screenshot()
                cv2.imwrite(best_model_screenshot_path, og_image)
                logger.info(f"Game screenshot saved to: {best_model_screenshot_path}")
        else:
            logger.info(f"Not enough trajectories for training: {memory_stats['num_trajectories']}/{min_memory_count_to_start_training}")

        if debug:
            logger.info("Debug mode: Press Enter to continue...")
            input()
        else:
            env.wait(1)

    logger.info(f"PPO training completed! Best model achieved {max_duration_achieved:.2f}s and saved at {best_actor_model_path}")
    return best_actor_model_path, best_critic_model_path

def image_test(env, image_wait_time, save_dir='./', logger=None):
    logger.info("Starting image test")
    env.start()
    env.wait(5)
    
    # Get full screenshot
    image_1 = env.get_game_screenshot()
    env.wait(image_wait_time)
    image_2 = env.get_game_screenshot()

    game_screenshot_path = os.path.join(save_dir, 'images', 'game_screenshot_1.png')
    cv2.imwrite(game_screenshot_path, image_1)
    logger.info(f"Original game screenshot image saved to: {game_screenshot_path}")
    
    binary_image_1 = env.get_binary_image(image_1)
    processed_image_path = os.path.join(save_dir, 'images', 'processed_image_1.png')
    cv2.imwrite(processed_image_path, binary_image_1)
    logger.info(f"Processed game screenshot image saved to: {processed_image_path}")

    track_image = env.get_track(binary_image_1)       
    track_image_path = os.path.join(save_dir, 'images', 'track_image_1.png')
    cv2.imwrite(track_image_path, track_image)
    logger.info(f"track image saved to: {track_image_path}")

    ending_image = env.get_ending(binary_image_1)       
    ending_image_path = os.path.join(save_dir, 'images', 'ending_image_1.png')
    cv2.imwrite(ending_image_path, ending_image)
    logger.info(f"ending image saved to: {ending_image_path}")

    is_done = env.is_done(binary_image_1)
    logger.info(f"is_done: {is_done}")
    
    states_grid = env.split_states(track_image)
    logger.info(f"States grid shape: {states_grid.shape}")

    real_state_value = env.compute_state_value_real(states_grid)
    logger.info(f"Real state value shape:\n{real_state_value}")


    binary_state_value = env.compute_state_value_binary(states_grid) 
    logger.info(f"Binary state value shape:\n{binary_state_value}")

    between_01_state_value = env.compute_state_value_between_01(states_grid)
    logger.info(f"Between 01 state value shape:\n{between_01_state_value}")

    
    # Create a figure to display all states
    # Calculate figure size based on grid dimensions
    fig_width = states_grid.shape[1] * 2  # 2 inches per column
    fig_height = states_grid.shape[0] * 2  # 2 inches per row
    plt.figure(figsize=(fig_width, fig_height))
    
    for i in range(states_grid.shape[0]):  # state_layer rows
        for j in range(states_grid.shape[1]):  # state_size_per_layer columns
            plt.subplot(states_grid.shape[0], states_grid.shape[1], i * states_grid.shape[1] + j + 1)
            plt.imshow(states_grid[i][j], cmap='gray')
            plt.axis('off')
            # Add state values as title for each image
            real_value = f'{real_state_value[i][j]:d}'
            between_value = f'{between_01_state_value[i][j]:.2f}'
            plt.title(f'{real_value}, {binary_state_value[i][j]}, {between_value}')
    plt.tight_layout()
    states_grid_path = os.path.join(save_dir, 'images', 'states_grid_1.png')
    plt.savefig(states_grid_path)
    plt.close()
    logger.info(f"States grid image saved to: {states_grid_path}")

    logger.info('---------------------')
    
    game_screenshot_path = os.path.join(save_dir, 'images', 'game_screenshot_2.png')
    cv2.imwrite(game_screenshot_path, image_2)
    logger.info(f"Original game screenshot image saved to: {game_screenshot_path}")
    
    binary_image_2 = env.get_binary_image(image_2)
    processed_image_path = os.path.join(save_dir, 'images', 'processed_image_2.png')
    cv2.imwrite(processed_image_path, binary_image_2)
    logger.info(f"Processed game screenshot image saved to: {processed_image_path}")

    track_image = env.get_track(binary_image_2)       
    track_image_path = os.path.join(save_dir, 'images', 'track_image_2.png')
    cv2.imwrite(track_image_path, track_image)
    logger.info(f"track image saved to: {track_image_path}")

    ending_image = env.get_ending(binary_image_2)       
    ending_image_path = os.path.join(save_dir, 'images', 'ending_image_2.png')
    cv2.imwrite(ending_image_path, ending_image)
    logger.info(f"ending image saved to: {ending_image_path}")

    is_done = env.is_done(binary_image_2)
    logger.info(f"is_done: {is_done}")
    
    states_grid = env.split_states(track_image)
    logger.info(f"States grid shape: {states_grid.shape}")

    real_state_value = env.compute_state_value_real(states_grid)
    logger.info(f"Real state value shape:\n{real_state_value}")


    binary_state_value = env.compute_state_value_binary(states_grid) 
    logger.info(f"Binary state value shape:\n{binary_state_value}")

    between_01_state_value = env.compute_state_value_between_01(states_grid)
    logger.info(f"Between 01 state value shape:\n{between_01_state_value}")
    
    # Create a figure to display all states
    # Calculate figure size based on grid dimensions
    fig_width = states_grid.shape[1] * 2  # 2 inches per column
    fig_height = states_grid.shape[0] * 2  # 2 inches per row
    plt.figure(figsize=(fig_width, fig_height))
    
    for i in range(states_grid.shape[0]):  # state_layer rows
        for j in range(states_grid.shape[1]):  # state_size_per_layer columns
            plt.subplot(states_grid.shape[0], states_grid.shape[1], i * states_grid.shape[1] + j + 1)
            plt.imshow(states_grid[i][j], cmap='gray')
            plt.axis('off')
            # Add state values as title for each image
            real_value = f'{real_state_value[i][j]:d}'
            between_value = f'{between_01_state_value[i][j]:.2f}'
            plt.title(f'{real_value}, {binary_state_value[i][j]}, {between_value}')
    plt.tight_layout()
    states_grid_path = os.path.join(save_dir, 'images', 'states_grid_2.png')
    plt.savefig(states_grid_path)
    plt.close()
    logger.info(f"States grid image saved to: {states_grid_path}")

    s_1, s_2 = env.get_states(binary_image_1, binary_image_2)
    print(s_1.shape, s_2.shape)
    states = np.vstack([s_1, s_2])
    logger.info(f"States grid shape: {states.shape}")
    logger.info(f"States grid:\n{states}")

def plot_track_image(env, binary_image, save_dir='./', idx=0):
    track_image = env.get_track(binary_image)       
    track_image_path = os.path.join(save_dir, 'images', f'track_image_{idx}.png')
    cv2.imwrite(track_image_path, track_image)

def model_test(save_dir, trainer, agent, env, max_epoch, model_path, actor_model_path, critic_model_path, model_type, save_tag='test', image_wait_time=0.1, logger=None):
    logger.info(f"Starting {model_type} model test with model from {model_path}")
    
    # Load model based on type
    if model_type == 'actor_critic' or model_type == 'actor_critic_v2' or model_type == 'PPO':
        # For Actor-Critic variants and PPO, look for actor and critic models
        trainer.load_model(actor_model_path, critic_model_path)
    else:
        # For DQN and REINFORCE, load single model
        trainer.load_model(model_path)
    
    # Check if model has CUDA parameters
    if hasattr(trainer, 'eval_net'):
        print(f"DQN model CUDA: {next(trainer.eval_net.parameters()).is_cuda}")
    elif hasattr(trainer, 'actor_net'):
        print(f"Actor-Critic/PPO model CUDA: {next(trainer.actor_net.parameters()).is_cuda}")
    elif hasattr(trainer, 'policy_net'):
        print(f"Policy model CUDA: {next(trainer.policy_net.parameters()).is_cuda}")
    
    # Set exploration to 0 for testing (deterministic behavior)
    if hasattr(trainer, 'epsilon_init'):
        trainer.epsilon_init = 0  # No exploration during testing
    
    for j in range(max_epoch):
        logger.info(f"Epoch {j} start.")
        epoch_start_time = time.time()  # Record epoch start time
        env.start()
        env.wait(3.5)
        image_1, image_2 = env.get_image(image_wait_time)
        s_1, s_2 = env.get_states(image_1, image_2)

        current_steps = 1
        
        while True:
            # Choose action based on model type
            if model_type == 'PPO':
                # PPO returns action, log_prob, value, entropy
                a, log_prob, value, entropy = trainer.choose_action(s_1, s_2)
            elif model_type == 'actor_critic_v2' or model_type == 'actor_critic':
                # Actor-Critic v2 returns action
                a = trainer.choose_action_deterministic(s_1, s_2)
            else:
                # DQN, REINFORCE, Actor-Critic variants return just action
                a = trainer.choose_action(s_1, s_2)
            
            if a == 1:
                agent.jump()
            # elif a == 2:
            #     agent.crawl()
            image_1, image_2 = env.get_image(image_wait_time)
            s_1_, s_2_ = env.get_states(image_1, image_2)
            done = env.is_done(image_1)
            current_steps += 1
            if done: 
                # save game ending image
                og_image = env.get_game_screenshot()
                game_screenshot_path = os.path.join(save_dir, 'images', f'{save_tag}_test_round_{j}_ending_image.png')
                cv2.imwrite(game_screenshot_path, og_image)
                logger.info(f"Game ending image saved to: {game_screenshot_path}")

                current_steps += 1
                epoch_duration = time.time() - epoch_start_time
                logger.info(f"Epoch {j} completed! Steps: {current_steps}, Duration: {epoch_duration:.2f}s")
                
                # input("Press Enter to continue...")
                time.sleep(2)
                break
            s_1, s_2 = s_1_, s_2_
        
        env.wait(1)
        logger.info(f"Test epoch {j} completed")
        logger.info('------------------')

    
    logger.info("Model testing completed!")

def main():
    parser = argparse.ArgumentParser(description='Chrome Dino Game DQN Training/Testing')
    
    # Mode and basic training parameters
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'image'],
                      help='Mode: train for training, test for testing model, image for image test')
    parser.add_argument('--model_type', type=str, default='DQN', choices=['DQN', 'reinforce', 'actor_critic', 'actor_critic_v2', 'PPO'],
                        help='Model type: DQN for DQN model, reinforce for REINFORCE model, actor_critic for Actor-Critic model, PPO for PPO model')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode for detailed logging')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of epochs for training/testing')
    parser.add_argument('--steps', type=int, default=70,
                      help='Number of steps per epoch for training')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to the model for testing')
    parser.add_argument('--actor_model_path', type=str, default=None,
                      help='Path to the actor model for testing')
    parser.add_argument('--critic_model_path', type=str, default=None,
                      help='Path to the critic model for testing')
    
    # Continue training parameters
    parser.add_argument('--continue_train_epochs', type=int, default=5,
                      help='Number of epochs for continue training')
    parser.add_argument('--continue_train_steps', type=int, default=70,
                      help='Number of steps per epoch for continue training')
    parser.add_argument('--continue_train_min_memory_count_to_start_training', type=int, default=100,
                      help='Minimum memory size required before starting continue training')
    parser.add_argument('--continue_train_memory_size', type=int, default=1500,
                      help='Memory size for continue training')
    
    # test parameters
    parser.add_argument('--test_epochs', type=int, default=5,
                      help='Number of epochs for testing')
    
    # Learning parameters
    parser.add_argument('--lr', type=float, default=0.01,
                      help='Learning rate')
    parser.add_argument('--value_lr', type=float, default=0.01,
                      help='Value network learning rate (for REINFORCE Actor-Critic)')
    parser.add_argument('--max_steps_per_batch', type=int, default=15000,
                      help='Maximum steps per batch for training')
    parser.add_argument('--epsilon', type=float, default=0.99,
                      help='Initial epsilon value for exploration')
    parser.add_argument('--gamma', type=float, default=0.95,
                      help='Discount factor')
    parser.add_argument('--lambda_gae', type=float, default=0.95,
                      help='GAE parameter')
    parser.add_argument('--jump_duration', type=float, default=0.355,
                      help='Duration for jump action in seconds')
    parser.add_argument('--train_rounds', type=int, default=1,
                      help='Number of training rounds')
    
    # DQN model parameters
    parser.add_argument('--epsilon_increase', type=float, default=0.005,
                      help='Epsilon increase rate')
    parser.add_argument('--net_replace_memory_gap', type=int, default=100,
                      help='Memory counter increment threshold to trigger target network update')
    parser.add_argument('--memory_size', type=int, default=1500,
                      help='Size of replay memory')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--min_memory_count_to_start_training', type=int, default=100,
                      help='Minimum memory size required before starting learning')
    parser.add_argument('--state_grid_rows', type=int, default=2,
                      help='numer of states rows')
    parser.add_argument('--state_grid_cols', type=int, default=10,
                      help='number of states columns')
    parser.add_argument('--state_binary_threshold', type=float, default=0.2,
                      help='Threshold for binary state value')
    parser.add_argument('--is_done_threshold', type=float, default=0.6,
                      help='Threshold for is_done')
    parser.add_argument('--game_url', type=str, default='http://localhost:8000/',
                      help='URL of the game')
    parser.add_argument('--number_of_actions', type=int, default=3,
                      help='Number of possible actions')
    parser.add_argument('--image_wait_time', type=float, default=0.1,
                      help='Time to wait for image to be updated')
    
    # PPO parameters
    parser.add_argument('--ppo_eps_clip', type=float, default=0.2,
                      help='PPO clipping parameter')
    parser.add_argument('--ppo_value_coef', type=float, default=0.5,
                      help='PPO value function coefficient')
    parser.add_argument('--ppo_entropy_coef', type=float, default=0.01,
                      help='PPO entropy coefficient')
    parser.add_argument('--ppo_max_grad_norm', type=float, default=0.5,
                      help='PPO maximum gradient norm for clipping')
    
    # Window parameters
    parser.add_argument('--window_width', type=int, default=800,
                      help='Width of the Chrome window')
    parser.add_argument('--window_height', type=int, default=600,
                      help='Height of the Chrome window')
    parser.add_argument('--window_position', type=str, default='center',
                      choices=['center', 'top-left', 'top-right', 'bottom-left', 'bottom-right'],
                      help='Position of the Chrome window')
    parser.add_argument('--frameless', action='store_true',
                      help='Run Chrome in frameless mode')
    
    # Ending bounding box parameters
    parser.add_argument('--ending_i', type=int, default=0,
                      help='Row index (i) of the top-left corner of the ending bounding box')
    parser.add_argument('--ending_j', type=int, default=0,
                      help='Column index (j) of the top-left corner of the ending bounding box')
    parser.add_argument('--ending_height', type=int, default=-1,
                      help='Height of the ending bounding box (-1 for full height)')
    parser.add_argument('--ending_width', type=int, default=-1,
                      help='Width of the ending bounding box (-1 for full width)')
    
    # Track bounding box parameters
    parser.add_argument('--track_i', type=int, default=0,
                      help='Row index (i) of the top-left corner of the track bounding box')
    parser.add_argument('--track_j', type=int, default=0,
                      help='Column index (j) of the top-left corner of the track bounding box')
    parser.add_argument('--track_height', type=int, default=-1,
                      help='Height of the track bounding box (-1 for full height)')
    parser.add_argument('--track_width', type=int, default=-1,
                      help='Width of the track bounding box (-1 for full width)')
    
    args = parser.parse_args()
    
    # Create run directory
    run_dir = create_run_dir()
    logger = setup_logger(run_dir)
    logger.info(f"Created run directory: {run_dir}")
    
    # Save arguments to JSON file
    param_file = save_args_to_json(args, run_dir)
    logger.info(f"Parameters saved to: {param_file}")
    
    logger.info(f"Running in {args.mode} mode with parameters: {vars(args)}")
    
    # Initialize environment
    web_driver = WebDriver(
        ChromeDriverManager().install(),
        args.game_url, 
        "t", 
        window_width=args.window_width,
        window_height=args.window_height,
        window_position=args.window_position,
        frameless=args.frameless
    )
    ending_bounding_box = (args.ending_i, args.ending_j, args.ending_height, args.ending_width)
    track_bounding_box = (args.track_i, args.track_j, args.track_height, args.track_width)
    environment = Runner_Env(
        web_driver, 
        args.state_grid_rows * args.state_grid_cols, 
        ending_bounding_box, 
        track_bounding_box, 
        args.state_grid_rows, 
        args.state_grid_cols,
        args.state_binary_threshold,
        args.is_done_threshold
    )
    runner = Dino(web_driver, args.jump_duration)
    
    # Initialize DQN model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.model_type == 'DQN':
        eval_net = DQNModel_v3(args.state_grid_rows * args.state_grid_cols, args.number_of_actions).to(device)
        target_net = DQNModel_v3(args.state_grid_rows * args.state_grid_cols, args.number_of_actions).to(device)
        
        trainer = DQN_Trainer(
            eval_net=eval_net,
            target_net=target_net,
            lr=args.lr,
            epsilon_max=args.epsilon,
            epsilon_increase=args.epsilon_increase,
            gamma=args.gamma,
            number_of_states=args.state_grid_rows * args.state_grid_cols,
            number_of_actions=args.number_of_actions,
            memory_size=args.memory_size,
            min_memory_count_to_start_training=args.min_memory_count_to_start_training,
            batch_size=args.batch_size,
            net_replace_memory_gap=args.net_replace_memory_gap
        )
    elif args.model_type == 'reinforce':
        policy_net = REINFORCE(args.state_grid_rows * args.state_grid_cols, args.number_of_actions).to(device)
        
        trainer = REINFORCETrainer(
            policy_net=policy_net,
            lr=args.lr,
            gamma=args.gamma
        )
    elif args.model_type == 'actor_critic':
        actor_net = ActorNetwork_v2(args.state_grid_rows * args.state_grid_cols, args.number_of_actions).to(device)
        critic_net = CriticNetwork_v2(args.state_grid_rows * args.state_grid_cols).to(device)
        
        trainer = ActorCriticTrainer(
            actor_net=actor_net,
            critic_net=critic_net,
            lr=args.lr,
            critic_lr=args.value_lr,
            gamma=args.gamma,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            min_memory_count_to_start_training=args.min_memory_count_to_start_training
        )
    elif args.model_type == 'actor_critic_v2':
        actor_net = ActorNetworkV2(args.state_grid_rows * args.state_grid_cols, args.number_of_actions).to(device)
        critic_net = CriticNetworkV2(args.state_grid_rows * args.state_grid_cols).to(device)
        
        trainer = ActorCriticTrainerV2(
            actor_net=actor_net,
            critic_net=critic_net,
            lr=args.lr,
            critic_lr=args.value_lr,
            gamma=args.gamma,
            lambda_gae=args.lambda_gae,  # GAE parameter
            max_trajectories=args.memory_size,  # Maximum number of trajectories to store
            batch_size=args.batch_size,
            min_trajectories_to_start_training=args.min_memory_count_to_start_training,  # Minimum trajectories before training starts
            max_steps_per_batch=args.max_steps_per_batch
        )
    elif args.model_type == 'PPO':
        actor_net = PPOActor(args.state_grid_rows * args.state_grid_cols, args.number_of_actions).to(device)
        critic_net = PPOCritic(args.state_grid_rows * args.state_grid_cols).to(device)
        
        trainer = PPOTrainer(
            actor_net=actor_net,
            critic_net=critic_net,
            lr=args.lr,
            critic_lr=args.value_lr,
            gamma=args.gamma,
            lambda_gae=args.lambda_gae,
            eps_clip=args.ppo_eps_clip,
            value_coef=args.ppo_value_coef,
            entropy_coef=args.ppo_entropy_coef,
            ppo_epochs=args.epochs,
            batch_size=args.batch_size,
            max_grad_norm=args.ppo_max_grad_norm,
            min_trajectories_to_start_training=args.min_memory_count_to_start_training,
            max_steps_per_batch=args.max_steps_per_batch
        )
    
    if args.mode == 'train':
        for cur_train_round in range(args.train_rounds):
            logger.info(f"Training round {cur_train_round} start.")
            if cur_train_round == 0:
                cur_epoch = args.epochs
                cur_steps = args.steps
                cur_min_memory_count_to_start_training = args.min_memory_count_to_start_training
            else:
                cur_epoch = args.continue_train_epochs
                cur_steps = args.continue_train_steps
                cur_min_memory_count_to_start_training = args.continue_train_min_memory_count_to_start_training
            if args.model_type == 'DQN':
                best_model_path = None
                best_model_path = online_training(
                    cur_train_round,
                    trainer,  
                    runner, 
                    environment, 
                    cur_epoch, 
                    cur_steps, 
                    run_dir, 
                    logger,
                    args.debug,
                    best_model_path,
                    cur_min_memory_count_to_start_training,
                    args.image_wait_time
                )
            elif args.model_type == 'reinforce':
                best_model_path = None
                best_model_path = online_training_reinforce(
                    cur_train_round,
                    trainer,  
                    runner, 
                    environment, 
                    cur_epoch, 
                    cur_steps, 
                    run_dir, 
                    logger,
                    args.debug,
                    best_model_path,
                    cur_min_memory_count_to_start_training,
                    args.image_wait_time
                )
            elif args.model_type == 'actor_critic':
                best_actor_model_path, best_critic_model_path = None, None
                best_actor_model_path, best_critic_model_path = online_training_actor_critic(
                    cur_train_round,
                    trainer,  
                    runner, 
                    environment, 
                    cur_epoch, 
                    cur_steps, 
                    run_dir, 
                    logger,
                    args.debug,
                    best_actor_model_path,
                    best_critic_model_path,
                    cur_min_memory_count_to_start_training,
                    args.image_wait_time
                )
            elif args.model_type == 'actor_critic_v2':
                best_actor_model_path, best_critic_model_path = None, None
                best_actor_model_path, best_critic_model_path = online_training_actor_critic_v2(
                    cur_train_round,
                    trainer,  
                    runner, 
                    environment, 
                    cur_epoch, 
                    cur_steps, 
                    run_dir, 
                    logger,
                    args.debug,
                    best_actor_model_path,
                    best_critic_model_path,
                    args.min_memory_count_to_start_training, 
                    args.image_wait_time
                )
            elif args.model_type == 'PPO':
                best_actor_model_path, best_critic_model_path = None, None
                best_actor_model_path, best_critic_model_path = online_training_ppo(
                    cur_train_round,
                    trainer,  
                    runner, 
                    environment, 
                    cur_epoch, 
                    cur_steps, 
                    run_dir, 
                    logger,
                    args.debug,
                    best_actor_model_path,
                    best_critic_model_path,
                    args.min_memory_count_to_start_training,
                    args.image_wait_time
                )
    elif args.mode == 'test':
        if args.model_path is None and args.actor_model_path is None and args.critic_model_path is None:
            logger.error("model_path or actor_model_path/critic_model_paths is required for test mode")
            return
        model_test(
            run_dir,
            trainer, 
            runner, 
            environment, 
            args.test_epochs, 
            args.model_path,     # for DQN, REINFORCE, Actor-Critic variants
            args.actor_model_path,
            args.critic_model_path,
            args.model_type,
            'test',
            args.image_wait_time,
            logger
        )
    elif args.mode == 'image':
        image_test(environment, args.image_wait_time, run_dir, logger)

    environment.env_destroy()

if __name__ == "__main__":
    main()