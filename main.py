import time
import torch
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys
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
from DQN import DQN_Trainer, DQNModel_v1
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

def online_training(trainer, agent, env, max_epoch, max_step, run_dir, logger, debug=False, model_path=None, min_memory_count_to_start_training=100):
    max_duration_achieved = 0  # Track the maximum duration time achieved
    best_model_path = None  # Track the path of the best model
    continue_train = False
    if model_path is not None:
        continue_train = True
        trainer.load_model(model_path)
        trainer.epsilon_init = 1
        logger.info(f"Loaded model from {model_path}")

    if continue_train:
        logger.info(f"Starting continue training with {max_epoch} epochs and maximum {max_step} steps per epoch")
    else:
        logger.info(f"Starting training with {max_epoch} epochs and maximum {max_step} steps per epoch")

    save_path = os.path.join(run_dir, 'models')
    for j in range(max_epoch):
        epoch_start_time = time.time()  # Record epoch start time
        env.start()
        env.wait(4)
        init_time = env.get_time()
        image = env.get_image()
        s = env.get_states(image)
        r = 0
        if debug:
            plot_track_image(env, image, run_dir, -1)
        
        current_steps = 0
        for i in range(max_step):
            a = trainer.choose_action(s)
            if a == 1:
                agent.jump()
            elif a == 2:
                agent.crawl()
            
            image = env.get_image()
            done = env.is_done(image)

            if debug:
                plot_track_image(env, image, run_dir, i)
            if done:
                s_ = s
                r = -5
            else:
                time_diff = env.get_time() - init_time
                s_ = env.get_states(image)
                if a == 0:
                    r = max((max(agent.jump_duration, agent.crawl_duration)) * time_diff, 5)
                elif a == 1:
                    r = max(agent.jump_duration * time_diff, 5)
                else:
                    r = max(agent.crawl_duration * time_diff, 5)

            if debug:
                logger.info(f"Memory saved: \nState: \t\t{s} \nNext State: \t{s_} \nAction: {a} \nReward: {r}")

            trainer.save_memory(s, s_, a, r)
            s = s_
            current_steps = i + 1
            epoch_duration = time.time() - epoch_start_time

            if done or i == max_step - 1:
                logger.info(f"Epoch {j} completed! Steps: {current_steps}, Duration: {epoch_duration:.2f}s, Current epsilon: {trainer.epsilon_init}")
                break

            if debug:
                print(f"-------------------------")

        if trainer.memory_counter >= min_memory_count_to_start_training:
            logger.info(f"Learning step - Memory counter: {trainer.memory_counter}, Epsilon: {trainer.epsilon_init}")
            trainer.learning()

        # Check and save model after epoch completion
        if epoch_duration > max_duration_achieved:
            max_duration_achieved = epoch_duration
            # Remove previous best model if it exists
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            # Save new best model
            if continue_train:
                best_model_path = os.path.join(save_path, f"best_model_at_epoch_{j}.pt")
            else:
                best_model_path = os.path.join(save_path, f"best_continue_train_model_at_epoch_{j}.pt")
            trainer.save_model(best_model_path)
            logger.info(f"New best model saved! Duration: {max_duration_achieved:.2f}s")

        if debug:
            logger.info("Debug mode: Press Enter to continue...")
            input()
        else:
            env.wait(1)

    logger.info(f"Training completed! Best model achieved {max_duration_achieved:.2f}s and saved at {best_model_path}")
    return best_model_path

def image_test(env, save_dir='./', logger=None):
    logger.info("Starting image test")
    env.start()
    env.wait(6.5)
    
    # Get full screenshot
    og_image = env.get_game_screenshot()
    game_screenshot_path = os.path.join(save_dir, 'images', 'game_screenshot.png')
    cv2.imwrite(game_screenshot_path, og_image)
    logger.info(f"Original game screenshot image saved to: {game_screenshot_path}")
    
    binary_image = env.get_binary_image(og_image)
    processed_image_path = os.path.join(save_dir, 'images', 'processed_image.png')
    cv2.imwrite(processed_image_path, binary_image)
    logger.info(f"Processed game screenshot image saved to: {processed_image_path}")

    track_image = env.get_track(binary_image)       
    track_image_path = os.path.join(save_dir, 'images', 'track_image.png')
    cv2.imwrite(track_image_path, track_image)
    logger.info(f"track image saved to: {track_image_path}")

    ending_image = env.get_ending(binary_image)       
    ending_image_path = os.path.join(save_dir, 'images', 'ending_image.png')
    cv2.imwrite(ending_image_path, ending_image)
    logger.info(f"ending image saved to: {ending_image_path}")

    is_done = env.is_done(binary_image)
    logger.info(f"is_done: {is_done}")
    
    states_grid = env.split_states(track_image)
    logger.info(f"States grid shape: {states_grid.shape}")

    real_state_value = env.compute_state_value_real(states_grid)
    logger.info(f"Real state value shape:\n{real_state_value}")


    binary_state_value = env.compute_state_value_binary(states_grid) 
    logger.info(f"Binary state value shape:\n{binary_state_value}")

    between_01_state_value = env.compute_state_value_between_01(states_grid)
    logger.info(f"Between 01 state value shape:\n{between_01_state_value}")

    s = env.get_states(binary_image)
    logger.info(f"States grid shape: {s.shape}")  
    logger.info(f"States grid:\n{s}")
    
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
    states_grid_path = os.path.join(save_dir, 'images', 'states_grid.png')
    plt.savefig(states_grid_path)
    plt.close()
    logger.info(f"States grid image saved to: {states_grid_path}")

def plot_track_image(env, binary_image, save_dir='./', idx=0):
    track_image = env.get_track(binary_image)       
    track_image_path = os.path.join(save_dir, 'images', f'track_image_{idx}.png')
    cv2.imwrite(track_image_path, track_image)

def model_test(save_dir, model, agent, env, max_epoch, model_path, save_tag='train', logger=None):
    logger.info(f"Starting model test with model from {model_path}")
    model.load_model(model_path)
    print(next(model.eval_net.parameters()).is_cuda)
    model.epsilon_init = 1
    
    for j in range(max_epoch):
        logger.info(f"Epoch {j} start.")
        epoch_start_time = time.time()  # Record epoch start time
        env.start()
        env.wait(3.5)
        image = env.get_image()
        s = env.get_states(image)
        current_steps = 1
        
        while True:
            a = model.choose_action(s)
            if a == 1:
                agent.jump()
            elif a == 2:
                agent.crawl()
            image = env.get_image()
            s_ = env.get_states(image)
            done = env.is_done(image)
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
            s = s_
        
        env.wait(1)
        logger.info(f"Test epoch {j} completed")
        logger.info('------------------')

    
    logger.info("Model testing completed!")

def main():
    parser = argparse.ArgumentParser(description='Chrome Dino Game DQN Training/Testing')
    
    # Mode and basic training parameters
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'image'],
                      help='Mode: train for training, test for testing model, image for image test')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode for detailed logging')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of epochs for training/testing')
    parser.add_argument('--steps', type=int, default=70,
                      help='Number of steps per epoch for training')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to the model for testing')
    
    # Continue training parameters
    parser.add_argument('--continue_train_epochs', type=int, default=5,
                      help='Number of epochs for continue training')
    parser.add_argument('--continue_train_steps', type=int, default=70,
                      help='Number of steps per epoch for continue training')
    parser.add_argument('--continue_train_min_memory_count_to_start_training', type=int, default=100,
                      help='Minimum memory size required before starting continue training')
    
    # test parameters
    parser.add_argument('--test_epochs', type=int, default=5,
                      help='Number of epochs for testing')
    
    # Learning parameters
    parser.add_argument('--lr', type=float, default=0.01,
                      help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.99,
                      help='Initial epsilon value for exploration')
    parser.add_argument('--gamma', type=float, default=0.95,
                      help='Discount factor')
    parser.add_argument('--jump_duration', type=float, default=0.355,
                      help='Duration for jump action in seconds')
    parser.add_argument('--crawl_duration', type=float, default=0.5,
                      help='Duration for crawl action in seconds')
    
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
    parser.add_argument('--chrome_driver_path', type=str, default="/home/newton/chrome_driver/chromedriver-linux64/chromedriver",
                      help='Path to Chrome driver')
    parser.add_argument('--game_url', type=str, default='http://localhost:8000/',
                      help='URL of the game')
    parser.add_argument('--number_of_actions', type=int, default=3,
                      help='Number of possible actions')
    
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
        args.chrome_driver_path, 
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
    runner = Dino(web_driver, args.jump_duration, args.crawl_duration)
    
    # Initialize DQN model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    eval_net = DQNModel_v1(args.state_grid_rows * args.state_grid_cols, args.number_of_actions).to(device)
    target_net = DQNModel_v1(args.state_grid_rows * args.state_grid_cols, args.number_of_actions).to(device)
    
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
    
    if args.mode == 'train':
        best_model_path = online_training(
            trainer,  
            runner, 
            environment, 
            args.epochs, 
            args.steps, 
            run_dir, 
            logger,
            args.debug,
            args.model_path,
            args.min_memory_count_to_start_training,
        )
        logger.info("\n")
        logger.info("--------------------------------")
        logger.info("\n")
        model_test(
            run_dir,
            trainer, 
            runner, 
            environment, 
            args.test_epochs, 
            best_model_path, 
            'train',
            logger
        )
        logger.info("\n")
        logger.info("--------------------------------")
        logger.info("\n")
        trainer.reset_trainer(args.epsilon, args.continue_train_min_memory_count_to_start_training)
        best_continue_train_model_path = online_training(
            trainer,  
            runner, 
            environment, 
            args.continue_train_epochs, 
            args.continue_train_steps, 
            run_dir, 
            logger,
            args.debug,
            best_model_path,
            args.continue_train_min_memory_count_to_start_training,
        )
        logger.info("\n")
        logger.info("--------------------------------")
        logger.info("\n")
        model_test(
            run_dir,
            trainer, 
            runner, 
            environment, 
            args.test_epochs, 
            best_continue_train_model_path, 
            'continue_train',
            logger
        )
    elif args.mode == 'test':
        if args.model_path is None:
            logger.error("Model path is required for test mode")
            return
        model_test(
            run_dir,
            trainer, 
            runner, 
            environment, 
            args.epochs, 
            args.model_path, 
            logger
        )
    elif args.mode == 'image':
        image_test(environment, run_dir, logger)

    environment.env_destroy()

if __name__ == "__main__":
    main()