import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import xml.etree.ElementTree as ET
from statsmodels.distributions.empirical_distribution import ECDF
# from TL_train_fairness import FIG_DIRECTORY
from TL_train import LAMBDA_FAIR
scenario = "Comaprison"
FIG_DIRECTORY = '/Users/majidraeis/OneDrive/SUMO/figs/' + scenario
FILE_DIRECTORY_BASE = '/Users/majidraeis/OneDrive/SUMO/files/'
if not os.path.exists(FIG_DIRECTORY):
    os.makedirs(FIG_DIRECTORY)
# plt.style.use('ggplot')
GAMMA_range = [2]
lstyle = ['-', '--', ':']
# color_list = ['#66a0a3', '#543a96', '#cc6040', '#edc067']
color_list = ['b', 'r',  'c', 'orange', 'k']


def run_RL(arrival_distributions):
    for i, GAMMA in enumerate(GAMMA_range):
        # scenario = "Single_intersection_indep_GAMMA_%3.2f" % GAMMA
        scenario = "RL_GAMMA_%3.2f_%s_%s" \
                   % (GAMMA, arrival_distributions['we'], arrival_distributions['ns'])
        FILE_DIRECTORY = FILE_DIRECTORY_BASE + scenario

        waiting_times_ns = np.load(FILE_DIRECTORY+"/waiting_times_ns.npy")
        waiting_times_we = np.load(FILE_DIRECTORY + "/waiting_times_we.npy")
        waiting_times_ns = np.array(waiting_times_ns)
        waiting_times_we = np.array(waiting_times_we)
        waiting_times_tot = np.append(waiting_times_ns, waiting_times_we)
        positive_waiting_times = waiting_times_tot[waiting_times_tot > 0.0]
        ax1.hist(positive_waiting_times, 80, range=(0, 50), alpha=0.7,
                 label=("Gamma=%3.2f" % GAMMA), color=color_list[i])

        # ax2.hist(waiting_times_ns, 1000, density=True, histtype='step', cumulative=True, label='Empirical')
        # ax2.hist(waiting_times_we, 1000, density=True, histtype='step', cumulative=True, label='Empirical')

        ecdf_ns = ECDF(waiting_times_ns)
        ecdf_we = ECDF(waiting_times_we)
        ecdf_tot = ECDF(waiting_times_tot)
        name_ns = r"NS, $\alpha$=%d" % GAMMA
        name_we = r"WE, $\alpha$=%d" % GAMMA
        name = r"All directions, $\alpha$=%3.2f" % GAMMA
        temp = r"DBF ($\alpha$=%d)" % GAMMA
        x_range = np.arange(0, ecdf_ns.x[-1]+1)
        ax2.plot(x_range, ecdf_ns(x_range), label=temp, color=color_list[i], lw=2.0)
        x_range = np.arange(0, ecdf_we.x[-1]+1)
        ax3.plot(x_range, ecdf_we(x_range), label=temp, color=color_list[i], lw=2.0)
        ax2.plot(ecdf_ns.x[-1], ecdf_ns.y[-1], marker="s", color=color_list[i])
        ax3.plot(ecdf_we.x[-1], ecdf_we.y[-1], marker="d", color=color_list[i])
        # ax3.plot(ecdf_tot.x, ecdf_tot.y, label=name)
        print("Ave waiting time in scenario:("+name+") = %3.2f" % np.mean(waiting_times_tot))
        print("Ave waiting time in scenario:(" + name_ns + ") = %3.2f" % np.mean(waiting_times_ns))
        print("Ave waiting time in scenario:(" + name_we + ") = %3.2f" % np.mean(waiting_times_we))
    # ax1.hist(positive_waiting_times, 80, range=(0, 50), alpha=0.7,
    #          label=("Gamma=%3.2f" % GAMMA))
    # return positive_waiting_times


def run_RL_Weighted(arrival_distributions):
    scenario = "RL_WEIGHTED_%3.2f_%s_%s" \
               % (LAMBDA_FAIR, arrival_distributions['we'], arrival_distributions['ns'])
    # scenario = "Intersection_LAMBDA_FAIR_%3.2f" % LAMBDA_FAIR
    FILE_DIRECTORY = FILE_DIRECTORY_BASE + scenario
    waiting_times_ns = np.load(FILE_DIRECTORY + "/waiting_times_ns.npy")
    waiting_times_we = np.load(FILE_DIRECTORY + "/waiting_times_we.npy")
    waiting_times_ns = np.array(waiting_times_ns)
    waiting_times_we = np.array(waiting_times_we)
    waiting_times_tot = np.append(waiting_times_ns, waiting_times_we)
    positive_waiting_times = waiting_times_tot[waiting_times_tot > 0.0]
    # ax1.hist(positive_waiting_times, 80, range=(0, 50), alpha=0.7, label="Weighted Fair", color=color_list[2])

    ecdf_ns = ECDF(waiting_times_ns)
    ecdf_we = ECDF(waiting_times_we)
    ecdf_tot = ECDF(waiting_times_tot)
    name_ns = "NS, Weighted Fair"
    name_we = "WE, Weighted Fair"
    name = "All directions, Weighted Fair"
    x_range = np.arange(0, ecdf_ns.x[-1] + 1)
    ax2.plot(x_range, ecdf_ns(x_range), label="TBF", color=color_list[4], lw=2.0)
    x_range = np.arange(0, ecdf_we.x[-1] + 1)
    ax3.plot(x_range, ecdf_we(x_range), label="TBF", color=color_list[4], lw=2.0)
    ax2.plot(ecdf_ns.x[-1], ecdf_ns.y[-1], marker="s", color=color_list[4])
    ax3.plot(ecdf_we.x[-1], ecdf_we.y[-1], marker="d", color=color_list[4])
    # ax3.plot(ecdf_tot.x, ecdf_tot.y, label=name)
    print("Ave waiting time in scenario:(" + name + ") = %3.2f" % np.mean(waiting_times_tot))
    print("Ave waiting time in scenario:(" + name_ns + ") = %3.2f" % np.mean(waiting_times_ns))
    print("Ave waiting time in scenario:(" + name_we + ") = %3.2f" % np.mean(waiting_times_we))
    return positive_waiting_times


def run_FixedTime(arrival_distributions):
    scenario = "FixedTime_%s_%s" \
               % (arrival_distributions['we'], arrival_distributions['ns'])
    FILE_DIRECTORY = FILE_DIRECTORY_BASE + scenario

    waiting_times_ns = np.load(FILE_DIRECTORY+"/waiting_times_ns.npy")
    waiting_times_we = np.load(FILE_DIRECTORY + "/waiting_times_we.npy")
    waiting_times_ns = np.array(waiting_times_ns)
    waiting_times_we = np.array(waiting_times_we)
    waiting_times_tot = np.append(waiting_times_ns, waiting_times_we)
    positive_waiting_times = waiting_times_tot[waiting_times_tot > 0.0]
    ax1.hist(positive_waiting_times, 80, range=(0, 50), alpha=0.7, label="FT")

    ecdf_ns = ECDF(waiting_times_ns)
    ecdf_we = ECDF(waiting_times_we)
    ecdf_tot = ECDF(waiting_times_tot)
    name_ns = "NS, FT"
    name_we = "WE, FT"
    name = "All directions, FT"
    ax2.plot(ecdf_ns.x, ecdf_ns.y, label=name_ns)
    ax3.plot(ecdf_we.x, ecdf_we.y, label=name_we)

    # ax3.plot(ecdf_tot.x, ecdf_tot.y, label=name)
    print("Ave waiting time in scenario:("+name+") = %3.2f" % np.mean(waiting_times_tot))
    print("Ave waiting time in scenario:(" + name_ns + ") = %3.2f" % np.mean(waiting_times_ns))
    print("Ave waiting time in scenario:(" + name_we + ") = %3.2f" % np.mean(waiting_times_we))


def run_MaxPressure(arrival_distributions):
    scenario = "MaxPressure_%s_%s" \
               % (arrival_distributions['we'], arrival_distributions['ns'])
    FILE_DIRECTORY = FILE_DIRECTORY_BASE + scenario

    waiting_times_ns = np.load(FILE_DIRECTORY+"/waiting_times_ns.npy")
    waiting_times_we = np.load(FILE_DIRECTORY + "/waiting_times_we.npy")
    waiting_times_ns = np.array(waiting_times_ns)
    waiting_times_we = np.array(waiting_times_we)
    waiting_times_tot = np.append(waiting_times_ns, waiting_times_we)
    positive_waiting_times = waiting_times_tot[waiting_times_tot > 0.0]
    ax1.hist(positive_waiting_times, 80, range=(0, 50), alpha=0.7, label="MP", color=color_list[2])

    ecdf_ns = ECDF(waiting_times_ns)
    ecdf_we = ECDF(waiting_times_we)
    ecdf_tot = ECDF(waiting_times_tot)
    name_ns = "NS, MP"
    name_we = "WE, MP"
    name = "All directions, MP"
    x_range = np.arange(0, ecdf_ns.x[-1]+1)
    ax2.plot(x_range, ecdf_ns(x_range), label=scenario, color=color_list[2], lw=2.0)
    x_range = np.arange(0, ecdf_we.x[-1]+1)
    ax3.plot(x_range, ecdf_we(x_range), label=scenario, color=color_list[2], lw=2.0)
    ax2.plot(ecdf_ns.x[-1], ecdf_ns.y[-1], marker="s", color=color_list[2])
    ax3.plot(ecdf_we.x[-1], ecdf_we.y[-1], marker="d", color=color_list[2])
    # ax3.plot(ecdf_tot.x, ecdf_tot.y, label=name)
    print("Ave waiting time in scenario:("+name+") = %3.2f" % np.mean(waiting_times_tot))
    print("Ave waiting time in scenario:(" + name_ns + ") = %3.2f" % np.mean(waiting_times_ns))
    print("Ave waiting time in scenario:(" + name_we + ") = %3.2f" % np.mean(waiting_times_we))



def run_SOTL(arrival_distributions):
    scenario = "SOTL_%s_%s" \
               % (arrival_distributions['we'], arrival_distributions['ns'])
    FILE_DIRECTORY = FILE_DIRECTORY_BASE + scenario

    waiting_times_ns = np.load(FILE_DIRECTORY+"/waiting_times_ns.npy")
    waiting_times_we = np.load(FILE_DIRECTORY + "/waiting_times_we.npy")
    waiting_times_ns = np.array(waiting_times_ns)
    waiting_times_we = np.array(waiting_times_we)
    waiting_times_tot = np.append(waiting_times_ns, waiting_times_we)
    positive_waiting_times = waiting_times_tot[waiting_times_tot > 0.0]
    ax1.hist(positive_waiting_times, 80, range=(0, 50), alpha=0.7, label="SOTL", color=color_list[3])

    ecdf_ns = ECDF(waiting_times_ns)
    ecdf_we = ECDF(waiting_times_we)
    ecdf_tot = ECDF(waiting_times_tot)
    name_ns = "NS, SOTL"
    name_we = "WE, SOTL"
    name = "All directions, SOTL"
    x_range = np.arange(0, ecdf_ns.x[-1]+1)
    ax2.plot(x_range, ecdf_ns(x_range), label=scenario, color=color_list[3], lw=2.0)
    x_range = np.arange(0, ecdf_we.x[-1]+1)
    ax3.plot(x_range, ecdf_we(x_range), label=scenario, color=color_list[3], lw=2.0)
    ax2.plot(ecdf_ns.x[-1], ecdf_ns.y[-1], marker="s", color=color_list[3])
    ax3.plot(ecdf_we.x[-1], ecdf_we.y[-1], marker="d", color=color_list[3])
    # ax3.plot(ecdf_tot.x, ecdf_tot.y, label=name)
    print("Ave waiting time in scenario:("+name+") = %3.2f" % np.mean(waiting_times_tot))
    print("Ave waiting time in scenario:(" + name_ns + ") = %3.2f" % np.mean(waiting_times_ns))
    print("Ave waiting time in scenario:(" + name_we + ") = %3.2f" % np.mean(waiting_times_we))

if __name__ == "__main__":

    scenarios = {
                 1: {'ew': 'Poisson', 'we': 'Poisson',
                     'ns': 'Poisson', 'sn': 'Poisson'},
                 2: {'ew': 'Markov', 'we': 'Markov',
                     'ns': 'Markov', 'sn': 'Markov'},
                 3: {'ew': 'Poisson', 'we': 'Poisson',
                     'ns': 'Markov', 'sn': 'Markov'},
                 4: {'ew': 'TimeVarying', 'we': 'TimeVarying',
                     'ns': 'TimeVarying', 'sn': 'TimeVarying'}
                 }

    fig1 = plt.figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure(figsize=(4, 2.5))
    ax2 = fig2.add_subplot(111)
    #
    fig3 = plt.figure(figsize=(4, 2.5))
    ax3 = fig3.add_subplot(111)

    arrival_distributions = scenarios[3]
    run_RL(arrival_distributions)
    run_RL_Weighted(arrival_distributions)
    # run_FixedTime()
    run_MaxPressure(arrival_distributions)
    run_SOTL(arrival_distributions)
    FONT_SIZE = 14
    ax1.set_xlabel('waiting time', fontsize=FONT_SIZE)
    ax1.set_ylabel('Number', fontsize=FONT_SIZE)
    ax1.tick_params(axis="x", labelsize=FONT_SIZE)
    ax1.tick_params(axis="y", labelsize=FONT_SIZE)
    ax1.legend(fontsize=12)
    fig1.savefig(FIG_DIRECTORY + '/hist.pdf', bbox_inches='tight')

    ax2.set_xlabel('Waiting time', fontsize=FONT_SIZE)
    ax2.set_ylabel('CDF', fontsize=FONT_SIZE)
    ax2.tick_params(axis="x", labelsize=FONT_SIZE)
    ax2.tick_params(axis="y", labelsize=FONT_SIZE)
    ax2.legend(fontsize=12)
    ax2.set_xlim(0, 100)
    fig2.savefig(FIG_DIRECTORY + '/cdfs_ns.pdf', bbox_inches='tight')

    ax3.set_xlabel('Waiting time', fontsize=FONT_SIZE)
    ax3.set_ylabel('CDF', fontsize=FONT_SIZE)
    ax3.tick_params(axis="x", labelsize=FONT_SIZE)
    ax3.tick_params(axis="y", labelsize=FONT_SIZE)
    ax3.legend(fontsize=12)
    ax3.set_xlim(0, 40)
    fig3.savefig(FIG_DIRECTORY + '/cdfs_we.pdf', bbox_inches='tight')