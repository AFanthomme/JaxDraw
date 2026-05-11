import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from src.custom_types import *
import numpy as np
from typing import Dict, Optional, cast
import logging


def save_sequence_as_gif(savepath: Path, images: CanvasSequence, env_states: EnvStateSequence, agent_actions: ActionSequence, agent_rewards: RewardSequence, teacher_actions: Optional[ActionSequence]=None, teacher_rewards: Optional[RewardSequence]=None, title: str=''):  
    sort_modes = env_states.sort_mode
    decreasings = env_states.decreasing
    positions = env_states.position

    incr_str = "decreasing" if decreasings[0] else 'increasing' 
    if sort_modes[0] == 0:
        ordering_type = "projection"
    elif sort_modes[0] == 1:
        ordering_type = "proximity"
    elif sort_modes[0] == 2:
        ordering_type = "length"
    elif sort_modes[0] == 3:
        incr_str = "clockwise" if decreasings[0] else 'counter-clockwise' 
        ordering_type = "internal rotation"
    elif sort_modes[0] == 4:
        incr_str = "clockwise" if decreasings[0] else 'counter-clockwise' 
        ordering_type = "angle in the frame"
    else:
        ordering_type = str(sort_modes[0])
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.suptitle(title)
    fig.subplots_adjust(right=0.6)

    if np.any(images > 1.):
        images = .99 * images / images.max()

    # images = images.transpose(0, 2, 3, 1)
    images = images.transpose(0, 3, 2, 1)
    img_display = ax.imshow(images[0], animated=True, extent=(0, 1, 0, 1), origin='lower')
    rule_for_legend, = ax.plot([], [], color=(0,0,0), alpha=0, label= f"\nIn {incr_str} order of {ordering_type}", animated=True)
    agent_line, = ax.plot([], [], color='yellow', ls=':', label=f'Agent: p=     ', linewidth=2, animated=True)
    agent_reward_for_legend, = ax.plot([], [], color='yellow', label=f'Agent reward:      ', linewidth=2, animated=True)
    start_dots = ax.scatter([], [], color='yellow', s=60, zorder=5, animated=True, label='Start pos')
    end_dots = ax.scatter([], [], color='gray', s=60, zorder=5, animated=True, label='Next pos')
    if teacher_actions is not None:
        teacher_line, = ax.plot([], [], color='pink', label=f'Teacher: p=     ', linewidth=2, animated=True)
        teacher_reward_for_legend, = ax.plot([], [], color='pink', label=f'Teacher reward:      ', linewidth=2, animated=True)
    leg = ax.legend(loc='upper left', facecolor=None, edgecolor='gray', bbox_to_anchor=(1.1, 0.8),)

    def update_frame(t):
        incr_str = "decreasing" if decreasings[t] else 'increasing' 
        if sort_modes[t] == 0:
            ordering_type = "projection"
        elif sort_modes[t] == 1:
            ordering_type = "proximity"
        elif sort_modes[t] == 2:
            ordering_type = "length"
        elif sort_modes[t] == 3:
            incr_str = "clockwise" if decreasings[0] else 'counter-clockwise' 
            ordering_type = "internal rotation"
        elif sort_modes[t] == 4:
            incr_str = "clockwise" if decreasings[0] else 'counter-clockwise' 
            ordering_type = "angle in the frame"
        else:
            ordering_type = str(sort_modes[t])

        fig.suptitle(title+ f"\nIn {incr_str} order of\n {ordering_type}")
        img_display.set_data(images[t])
        x, y = positions[t]
        dxa, dya, pa =  agent_actions[t]
        nx, ny = positions[min(t+1, len(positions))]

        agent_line.set_data([x, x+dxa], [y, y+dya])
        leg.get_texts()[0].set_text(f"In {incr_str} order of {ordering_type}")
        leg.get_texts()[1].set_text(f'Agent draw: p={.5+pa:.2f}')
        leg.get_texts()[2].set_text(f'Agent reward: {agent_rewards[t]:.2f}')

        start_dots.set_offsets([x, y])
        end_dots.set_offsets([nx, ny])

        if teacher_actions is not None:
            assert teacher_rewards is not None
            dxt, dyt, pt = teacher_actions[t]
            teacher_line.set_data([x, x+dxt], [y, y+dyt])
            leg.get_texts()[-2].set_text(f'Teacher draw: p={.5+pt:.2f}')
            leg.get_texts()[-1].set_text(f'Teacher reward: {teacher_rewards[t]:.2f}')

        if teacher_actions is not None:
            return rule_for_legend, img_display, agent_line, teacher_line, agent_reward_for_legend, teacher_reward_for_legend, start_dots, end_dots, leg 
        else:
            return rule_for_legend, img_display, agent_line, agent_reward_for_legend, start_dots, end_dots, leg 
    
    anim = FuncAnimation(fig, update_frame, frames=images.shape[0], blit=True)
    anim.save(savepath, writer=PillowWriter(fps=1))
    plt.close()


def plot_cumulated_rewards(data: Dict, colors: Dict, env_params: EnvParams, savepath: Path, step=''):
    fig, ax = plt.subplots()
    fig.suptitle('Cumulative reward across time')
    fig.subplots_adjust(right=0.75)
    ax.set_xlabel('Time in trial')
    ax.set_ylabel('Cumulated reward')
    for i in range(1, env_params.num_target_strokes+1):
        plt.axhline(i, ls=':', c="gray", alpha=.3)

    cluster_width = 0.3 
    n_conditions = len(list(data.keys()))

    # Calculate the x-axis offsets for each condition
    offsets = np.linspace(-cluster_width/2, cluster_width/2, n_conditions)
    timesteps = np.arange(env_params.max_num_strokes)
    
    for i, (pol_name, policy_arrays) in enumerate(data.items()):
        mean = policy_arrays['mean'] 
        y_min = policy_arrays['min'] 
        y_max = policy_arrays['max'] 
        std = policy_arrays['std'] 
        logging.critical(f"Policy {pol_name} has mean final cumulated reward {mean[-1]:.2f} with std {std[-1]:.2f}, min {y_min[-1]:.2f} and max {y_max[-1]:.2f}")
        # Don't show std going lower than the min (useful if only upwards variance like in oracle)
        ymin = np.max(np.stack([y_min, mean-std], -1), -1)
        ymax = np.min(np.stack([y_max, mean+std], -1), -1)
        alpha = .7 if pol_name == 'network' else .5
        yerr = np.clip([mean-ymin, ymax-mean], 1e-6, None)
        logging.critical(yerr)
        ax.errorbar(timesteps+offsets[i], mean, yerr=yerr, label=pol_name, color=colors[pol_name], ls=':', capsize=4, capthick=1, alpha=alpha)

    ax.legend(loc='upper left', facecolor=None, edgecolor='gray', bbox_to_anchor=(1.01, 0.8),)
    fig.savefig(savepath / f'cumulated_rewards{step}.png', dpi=600)
    fig.savefig(savepath / f'cumulated_rewards{step}.pdf')