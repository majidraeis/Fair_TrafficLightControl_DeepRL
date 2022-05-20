import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


# plt.style.use('ggplot')
GAMMA_range = [2, 0]
lstyle = ['-', '--', ':']
label_list = [r"DFC$_2$", r"DFC$_0$"]
# color_list = ['#66a0a3', '#543a96', '#cc6040', '#edc067']
color_list = ['b', 'r', 'g']
seed_range = [10, 55, 110]
scenarios = {
    1: {'ew': 'Poisson', 'we': 'Poisson',
        'ns': 'Poisson', 'sn': 'Poisson'},
    2: {'ew': 'Poisson', 'we': 'Poisson',
        'ns': 'Markov', 'sn': 'Markov'},
    3: {'ew': 'Poisson', 'we': 'Poisson',
        'ns': 'TimeVarying', 'sn': 'TimeVarying'}
}
ep_length = 400
window_size = 5#2
step_sample = 10#5
FontSize = 14
sampled = np.arange(0, ep_length, step_sample)
episodes = np.arange(ep_length)
episodes = episodes[sampled]
episodes = episodes[window_size-1:]


def mov_average(in_series, window_size):
  pd_series = pd.Series(in_series)
  windows = pd_series.rolling(window_size)
  moving_averages = windows.mean()
  moving_averages_list = moving_averages.tolist()
  without_nans = moving_averages_list[window_size - 1:]
  return np.array(without_nans)

arrival_distributions = scenarios[2]
FIG_SIZE = (5, 3)

fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
fig3, ax3 = plt.subplots(figsize=FIG_SIZE)
fig4, ax4 = plt.subplots(figsize=FIG_SIZE)
for i, GAMMA in enumerate(GAMMA_range):
    scenario = "RL_%3.2f_%s_%s" \
               % (GAMMA, arrival_distributions['we'], arrival_distributions['ns'])
    FIG_DIRECTORY = '/Users/majidraeis/OneDrive/SUMO/figs/' + scenario
    FILE_DIRECTORY = '/Users/majidraeis/OneDrive/SUMO/files/' + scenario
    waiting_time_max_tot = np.zeros((1, ep_length))
    waiting_time_quantile_tot = np.zeros((1, ep_length))
    waiting_time_mean_tot = np.zeros((1, ep_length))
    reward_tot = np.zeros((1, ep_length))
    for seed in seed_range:
        name = '/waiting_time_max_%d.npy' % seed
        waiting_time_max = np.reshape(np.load(FILE_DIRECTORY + name), (1, -1))
        print(waiting_time_max[0, -1])
        waiting_time_max_tot = np.append(waiting_time_max_tot, waiting_time_max, axis=0)
        name = '/waiting_time_quantile_%d.npy' % seed
        waiting_time_quantile = np.reshape(np.load(FILE_DIRECTORY + name), (1, -1))
        waiting_time_quantile_tot = np.append(waiting_time_quantile_tot, waiting_time_quantile, axis=0)
        name = '/waiting_time_mean_%d.npy' % seed
        waiting_time_mean = np.reshape(np.load(FILE_DIRECTORY + name), (1, -1))
        waiting_time_mean_tot = np.append(waiting_time_mean_tot, waiting_time_mean, axis=0)
        # name = '/reward_%d.npy' % seed
        # reward = np.reshape(np.load(FILE_DIRECTORY + name), (1, -1))
        # reward_tot = np.append(reward_tot, reward, axis=0)
        # print(reward)


    waiting_time_max_mean = np.mean(waiting_time_max_tot[1:], axis=0)
    waiting_time_max_sem = stats.sem(waiting_time_max_tot[1:], axis=0)

    waiting_time_max_mean = waiting_time_max_mean[sampled]
    waiting_time_max_sem = waiting_time_max_sem[sampled]

    waiting_time_max_mean = mov_average(waiting_time_max_mean, window_size)
    waiting_time_max_sem = mov_average(waiting_time_max_sem, window_size)



    ax1.plot(episodes, waiting_time_max_mean, alpha=0.8, color=color_list[i], label=label_list[i], linewidth=2.0)
    ax1.fill_between(episodes, waiting_time_max_mean - waiting_time_max_sem,
                    waiting_time_max_mean + waiting_time_max_sem, color=color_list[i], alpha=0.2)
    # ax.set_ylim([0,1])
    ax1.set_ylabel("Maximum waiting time", fontsize=FontSize)
    ax1.set_xlabel("Episodes", fontsize=FontSize)
    ax1.tick_params(axis='both', which='major', labelsize=FontSize)
    # ax.set_title('$\lambda=%3.2f, T=%d$'%(lam, step_len))


    # ==================================================================

    waiting_time_mean_mean = np.mean(waiting_time_mean_tot[1:], axis=0)
    waiting_time_mean_sem = stats.sem(waiting_time_mean_tot[1:], axis=0)

    waiting_time_mean_mean = waiting_time_mean_mean[sampled]
    waiting_time_mean_sem = waiting_time_mean_sem[sampled]

    waiting_time_mean_mean = mov_average(waiting_time_mean_mean, window_size)
    waiting_time_mean_sem = mov_average(waiting_time_mean_sem, window_size)


    ax2.plot(episodes, waiting_time_mean_mean, alpha=0.8, color=color_list[i], label=label_list[i], linewidth=2.0)
    ax2.fill_between(episodes, waiting_time_mean_mean - waiting_time_mean_sem,
                    waiting_time_mean_mean + waiting_time_mean_sem, color=color_list[i], alpha=0.2)
    # ax.set_ylim([0,1])
    ax2.set_ylabel("Average waiting time", fontsize=FontSize)
    ax2.set_xlabel("Episodes", fontsize=FontSize)
    ax2.tick_params(axis='both', which='major', labelsize=FontSize)
    # ax.set_title('$\lambda=%3.2f, T=%d$'%(lam, step_len))


    # ==================================================================

    waiting_time_quantile_mean = np.mean(waiting_time_quantile_tot[1:], axis=0)
    waiting_time_quantile_sem = stats.sem(waiting_time_quantile_tot[1:], axis=0)

    waiting_time_quantile_mean = waiting_time_quantile_mean[sampled]
    waiting_time_quantile_sem = waiting_time_quantile_sem[sampled]

    waiting_time_quantile_mean = mov_average(waiting_time_quantile_mean, window_size)
    waiting_time_quantile_sem = mov_average(waiting_time_quantile_sem, window_size)


    ax3.plot(episodes, waiting_time_quantile_mean, alpha=0.8, color=color_list[i], label=label_list[i], linewidth=2.0)
    ax3.fill_between(episodes, waiting_time_quantile_mean - waiting_time_quantile_sem,
                    waiting_time_quantile_mean + waiting_time_quantile_sem, color=color_list[i], alpha=0.2)
    # ax.set_ylim([0,1])
    ax3.set_ylabel("0.95-Quantile", fontsize=FontSize)
    ax3.set_xlabel("Episodes", fontsize=FontSize)
    ax3.tick_params(axis='both', which='major', labelsize=FontSize)
    # ax.set_title('$\lambda=%3.2f, T=%d$'%(lam, step_len))
    # ==================================================================

    # reward_tot_mean = np.mean(reward_tot[1:], axis=0)
    # reward_tot_sem = stats.sem(reward_tot[1:], axis=0)
    #
    # reward_tot_mean = reward_tot_mean[sampled]
    # reward_tot_sem = reward_tot_sem[sampled]
    #
    # reward_tot_mean = mov_average(reward_tot_mean, window_size)
    # reward_tot_sem = mov_average(reward_tot_sem, window_size)
    #
    #
    # ax4.plot(episodes, reward_tot_mean, alpha=0.8, color=color_list[i], label=label_list[i], linewidth=2.0)
    # ax4.fill_between(episodes, reward_tot_mean - reward_tot_sem,
    #                 reward_tot_mean + reward_tot_sem, color=color_list[i], alpha=0.2)
    # ax4.set_ylabel("Cumalative reward", fontsize=FontSize)
    # ax4.set_xlabel("Episodes", fontsize=FontSize)
    # ax4.tick_params(axis='both', which='major', labelsize=FontSize)
ax1.legend(fontsize=FontSize)
ax2.legend(fontsize=FontSize)
ax3.legend(fontsize=FontSize)
# ax4.legend(fontsize=FontSize)
fig1.savefig(FIG_DIRECTORY + '/max_wait_training.pdf', bbox_inches='tight')
fig2.savefig(FIG_DIRECTORY + '/mean_wait_training.pdf', bbox_inches='tight')
fig3.savefig(FIG_DIRECTORY + '/quantile_wait_training.pdf', bbox_inches='tight')
# fig4.savefig(FIG_DIRECTORY + '/reward_training.pdf', bbox_inches='tight')