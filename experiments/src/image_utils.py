import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from src.custom_types import *
import numpy as np
from typing import Dict


def save_gif(savepath: Path, images: CanvasSequence, positions: CoordinateSequence, agent_actions: ActionSequence,
              agent_rewards: RewardSequence, teacher_actions: Optional[ActionSequence]=None, teacher_rewards: Optional[RewardSequence]=None, title: str=''):  
    T, H, W, _ = images.shape 

    fig, ax = plt.subplots()
    ax.axis('off')
    fig.suptitle(title)
    fig.subplots_adjust(right=0.6)

    if np.any(images > 1.):
        images = .99 * images / images.max()

    img_display = ax.imshow(images[0], animated=True, extent=(0, 1, 0, 1), origin='lower')
    agent_line, = ax.plot([], [], color='yellow', ls=':', label=f'Agent: p=     ', linewidth=2, animated=True)
    agent_reward_for_legend, = ax.plot([], [], color='yellow', label=f'Agent reward:      ', linewidth=2, animated=True)
    if teacher_actions is not None:
        teacher_line, = ax.plot([], [], color='pink', label=f'Teacher: p=     ', linewidth=2, animated=True)
        teacher_reward_for_legend, = ax.plot([], [], color='pink', label=f'Teacher reward:      ', linewidth=2, animated=True)

    start_dots = ax.scatter([], [], color='yellow', s=60, zorder=5, animated=True, label='Start pos')
    end_dots = ax.scatter([], [], color='gray', s=60, zorder=5, animated=True, label='Next pos')
    leg = ax.legend(loc='upper left', facecolor=None, edgecolor='gray', bbox_to_anchor=(1.1, 0.8),)

    def update_frame(t):
        img_display.set_data(images[t])
        x, y = positions[t]
        dxa, dya, pa =  agent_actions[t]
        nx, ny = positions[min(t+1, len(positions))]

        agent_line.set_data([x, x+dxa], [y, y+dya])
        leg.get_texts()[0].set_text(f'Agent draw: p={.5+pa:.2f}')
        leg.get_texts()[1].set_text(f'Agent reward: {agent_rewards[t]:.2f}')
        if teacher_actions is not None:
            assert teacher_rewards is not None
            dxt, dyt, pt = teacher_actions[t]
            teacher_line.set_data([x, x+dxt], [y, y+dyt])
            leg.get_texts()[2].set_text(f'Teacher draw: p={.5+pt:.2f}')
            leg.get_texts()[3].set_text(f'Teacher reward: {teacher_rewards[t]:.2f}')
        start_dots.set_offsets([x, y])
        end_dots.set_offsets([nx, ny])

        if teacher_actions is not None:
            return img_display, agent_line, teacher_line, agent_reward_for_legend, teacher_reward_for_legend, start_dots, end_dots, leg 
        else:
            return img_display, agent_line, agent_reward_for_legend, start_dots, end_dots, leg 
    
    anim = FuncAnimation(fig, update_frame, frames=T, blit=True)
    anim.save(savepath, writer=PillowWriter(fps=1))
    plt.close()

def plot_cumulated_rewards(data: Dict, colors: Dict, env_params: EnvParams, savepath: Path):
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
        # Don't show std going lower than the min (useful if only upwards variance like in oracle)
        y_low = np.max(np.stack([y_min, mean-std], -1), -1)
        y_high = np.min(np.stack([y_max, mean+std], -1), -1)
        alpha = .7 if pol_name == 'network' else .5
        err_low = mean-y_low
        err_high = y_high - mean
        ax.errorbar(timesteps+offsets[i], mean, yerr=[err_low, err_high], label=pol_name, color=colors[pol_name], ls=':', capsize=4, capthick=1, alpha=alpha)
    ax.legend(loc='upper left', facecolor=None, edgecolor='gray', bbox_to_anchor=(1.01, 0.8),)
    fig.savefig(savepath / 'cumulated_rewards.png', dpi=600)
    fig.savefig(savepath / 'cumulated_rewards.pdf')