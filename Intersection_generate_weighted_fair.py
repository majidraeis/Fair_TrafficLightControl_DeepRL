# from __future__ import absolute_import
# from __future__ import print_function
#
# import os
# import sys
# import random
# import gym
# import copy
# from gym import spaces
# import numpy as np
# we need to import python modules from the $SUMO_HOME/tools directory
from Intersection_generate import *

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


# import traci

PHASE_NS = 0
PHASE_WE = 1
QL_MAX = 40
# D_TH = 20
DIR_IN_PHASE = [['u', 'd'], ['l', 'r']]
DIR = ['u', 'r', 'd', 'l']
SLOT = 5
MIN_TIME = 10
LAMBDA_FAIR = 0.001
# Q_TH = 15
# EPS = 0.1
LOOP_NUM_EDGE = 3
X_NUM_LANE = 3
Y_NUM_LANE = 2
STATE_SIZE = 6
# STATE_SIZE = 29
# NS_PRIORITY_OVER_WE = 1
NS_WEIGHT = 1.0
WE_WEIGHT = 2.5
NS_WEIGHT /= LAMBDA_FAIR
WE_WEIGHT /= LAMBDA_FAIR

# def generate_node(N_tl):
#     # N_tl = 5  # number of traffic lights
#     x_dist = 250.0
#     y_dist = 200.0
#     with open("data/tandem.nod.xml", "w") as nodes:
#
#         print("""<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
#         xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
# """, file=nodes)
#         for i in range(1, N_tl+1):
#             print('     <node id="%i" x="%3.2f" y="0"  type="traffic_light"/>' % (
#                 i, i*x_dist), file=nodes)
#             print('     <node id="%iu" x="%3.2f" y="%3.2f"  type="priority"/>' % (
#                 i, i*x_dist, y_dist), file=nodes)
#             print('     <node id="%id" x="%3.2f" y="%3.2f"  type="priority"/>' % (
#                 i, i*x_dist, -y_dist), file=nodes)
#         print('     <node id="1l" x="0" y="0"  type="priority"/>', file=nodes)
#         print('     <node id="%ir" x="%3.2f" y="0"  type="priority"/>' % (
#             i, (i+1) * x_dist), file=nodes)
#
#         print("</nodes>", file=nodes)
#
#
# def generate_edge(N_tl):
#     # N_tl = 5  # number of traffic lights
#     max_speed_we = 13.89 #(m/s) 50km/h
#     max_speed_ew = 13.89 #(m/s) 50km/h
#     max_speed_ns = 8.33 #(m/s) 30km/h
#     max_speed_sn = 8.33 #(m/s) 30km/h
#     N_lanes_we = 3
#     N_lanes_ew = 3
#     N_lanes_ns = 2
#     N_lanes_sn = 2
#     high_priority = 78
#     low_priority = 46
#
#     with open("data/tandem.edg.xml", "w") as edg:
#
#         print("""<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
#          xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
# """, file=edg)
#         for i in range(1, N_tl+1):
#             if i < N_tl:
#                 # WE
#                 print('     <edge id="%i_%i" from="%i" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#                     i, i+1, i, i+1, high_priority, N_lanes_we, max_speed_we), file=edg)
#                 # EW
#                 print('     <edge id="%i_%i" from="%i" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#                     i+1, i, i+1, i, high_priority, N_lanes_ew, max_speed_ew), file=edg)
#             # up_NS
#             print('     <edge id="%iu_%i" from="%iu" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#                 i, i, i, i, low_priority, N_lanes_ns, max_speed_ns), file=edg)
#             # up_SN
#             print('     <edge id="%i_%iu" from="%i" to="%iu" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#                 i, i, i, i, low_priority, N_lanes_sn, max_speed_sn), file=edg)
#             # down_NS
#             print('     <edge id="%i_%id" from="%i" to="%id" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#                 i, i, i, i, low_priority, N_lanes_ns, max_speed_ns), file=edg)
#             # down_SN
#             print('     <edge id="%id_%i" from="%id" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#                 i, i, i, i, low_priority, N_lanes_sn, max_speed_sn), file=edg)
#
#         # 1_l to/from 1
#         print('     <edge id="1l_1" from="1l" to="1" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#             high_priority, N_lanes_we, max_speed_we), file=edg)
#         print('     <edge id="1_1l" from="1" to="1l" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#             high_priority, N_lanes_ew, max_speed_ew), file=edg)
#         # N_tl_r to/from N_tl
#         print('     <edge id="%i_%ir" from="%i" to="%ir" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#             i, i, i, i, high_priority, N_lanes_we, max_speed_we), file=edg)
#         print('     <edge id="%ir_%i" from="%ir" to="%i" priority="%i" numLanes="%i" speed="%3.2f" />' % (
#             i, i, i, i, high_priority, N_lanes_ew, max_speed_ew), file=edg)
#
#         print("</edges>", file=edg)
#
#
# def generate_con(N_tl):
#         # N_tl = 5  # number of traffic lights
#         with open("data/tandem.con.xml", "w") as con:
#             print("""<connections xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
#              xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/connections_file.xsd">""", file=con)
#             for i in range(1, N_tl+1):
#                 if i < N_tl-1:
#                     # WE
#                     print('     <connection from="%i_%i" to="%i_%i"/>' % (
#                             i, i+1, i+1, i+2), file=con)
#                     #WE (Right turn)
#                     # print('     <connection from="<FROM_EDGE_ID>" to="<T0_EDGE_ID>" fromLane="<INT_1>"'
#                     #       ' toLane="<INT_2>"/>' % (i, i + 1, i + 1, i + 2), file=con)
#
#                     # EW
#                     print('     <connection from="%i_%i" to="%i_%i"/>' % (
#                             i+2, i+1, i+1, i), file=con)
#                 # NS
#                 print('     <connection from="%iu_%i" to="%i_%id"/>' % (
#                     i, i, i, i), file=con)
#                 # SN
#                 print('     <connection from="%id_%i" to="%i_%iu"/>' % (
#                     i, i, i, i), file=con)
#             # WE
#             print('     <connection from="1l_1" to="1_2"/>', file=con)
#             print('     <connection from="%i_%i" to="%i_%ir"/>' % (N_tl-1, N_tl, N_tl, N_tl), file=con)
#             # EW
#             print('     <connection from="2_1" to="1_1l"/>', file=con)
#             print('     <connection from="%ir_%i" to="%i_%i"/>' % (N_tl, N_tl, N_tl, N_tl-1), file=con)
#
#             print("</connections>", file=con)
#
# # def inter_arr_gen():
# #     lambda_a = self.n_servers * self.rho
# #     return np.random.exponential(1 / lambda_a)
# #
# # def service_gen():
# #     lambda_s = 1.0
# #     return np.random.exponential(1 / lambda_s)
#
#
# def generate_routefile(N_tl, N_time):
#     random.seed(22)  # make tests reproducible
#     # demand per second from different directions
#     # p_we_on_off = 0.3
#     # p_we_off_on = 0.05
#     # p_ew_on_off = 0.3
#     # p_ew_off_on = 0.05
#     p_ns_on_off = 0.28
#     p_ns_off_on = 0.02
#     p_sn_on_off = 0.28
#     p_sn_off_on = 0.02
#
#     p_we = 1/5.0
#     p_ew = 1/5.0
#     # p_ns = 1/10.0
#     # p_sn = 1/10.0
#
#     with open("data/tandem.rou.xml", "w") as routes:
#         print("""<routes>
#         <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
#         guiShape="passenger"/>
#         <vType id="bus" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" \
#         guiShape="bus"/>""", file=routes)
#
#         # WE
#         route_we = "1l_1 "
#         # EW
#         route_ew = "%ir_%i " % (N_tl, N_tl)
#         for j in range(1, N_tl + 1):
#             if j < N_tl:
#                 # WE
#                 route_we += "%i_%i " % (j, j+1)
#                 # EW
#                 route_ew += "%i_%i " % (N_tl-j+1, N_tl-j)
#             # NS
#             print('    <route edges = "%iu_%i %i_%id" color = "yellow" id = "route_ns%i" />' % (j, j, j, j, j),
#                   file=routes)
#             # SN
#             print('    <route edges = "%id_%i %i_%iu" color = "yellow" id = "route_sn%i" />' % (j, j, j, j, j),
#                   file=routes)
#         # WE
#         route_we += "%i_%ir " % (N_tl, N_tl)
#         print('    <route edges = "%s" color = "red" id = "route_we" />' % route_we,
#               file=routes)
#         # EW
#         route_ew += "1_1l "
#         print('    <route edges = "%s" color = "red" id = "route_ew" />' % route_ew,
#               file=routes)
#
#         vehNr = 0
#         s_we = 0
#         s_ew = 0
#         s_ns = 0
#         s_sn = 0
#         for i in range(N_time):
#             # WE
#             if random.uniform(0, 1) < p_we:
#                 print('    <vehicle id="we_%i" type="car" route="route_we" depart="%i" departLane="free" '
#                       'departSpeed="speedLimit" />' % (vehNr, i), file=routes)
#                 vehNr += 1
#             # EW
#             if random.uniform(0, 1) < p_ew:
#                 print('    <vehicle id="ew_%i" type="car" route="route_ew" depart="%i" departLane="free" '
#                       'departSpeed="speedLimit" />' % (vehNr, i), file=routes)
#                 vehNr += 1
#
#             # if s_we:
#             #     print('    <vehicle id="we_%i" type="car" route="route_we" depart="%i" departLane="free" '
#             #           'departSpeed="speedLimit" />' % (vehNr, i), file=routes)
#             #     vehNr += 1
#             #     s_we = 0 if random.uniform(0, 1) < p_we_on_off else 1
#             # else:
#             #     s_we = 1 if random.uniform(0, 1) < p_we_off_on else 0
#             #
#             # # EW
#             # if s_ew:
#             #     print('    <vehicle id="ew_%i" type="car" route="route_ew" depart="%i" departLane="free" '
#             #           'departSpeed="speedLimit" />' % (vehNr, i), file=routes)
#             #     vehNr += 1
#             #     s_ew = 0 if random.uniform(0, 1) < p_ew_on_off else 1
#             # else:
#             #     s_ew = 1 if random.uniform(0, 1) < p_ew_off_on else 0
#
#             for j in range(1, N_tl+1):
#                 # NS
#                 if s_ns:
#                     print('    <vehicle id="ns%i_%i" type="car" route="route_ns%i" depart="%i" departLane="free" />' % (
#                         j, vehNr, j, i), file=routes)
#                     vehNr += 1
#                     s_ns = 0 if random.uniform(0, 1) < p_ns_on_off else 1
#                 else:
#                     s_ns = 1 if random.uniform(0, 1) < p_ns_off_on else 0
#                 # if random.uniform(0, 1) < p_ns:
#                 #     print('    <vehicle id="ns%i_%i" type="car" route="route_ns%i" depart="%i" departLane="free" />' % (
#                 #         j, vehNr, j, i), file=routes)
#                 #     vehNr += 1
#                 # SN
#                 if s_sn:
#                     print('    <vehicle id="sn%i_%i" type="car" route="route_sn%i" depart="%i" departLane="free" />' % (
#                         j, vehNr, j, i), file=routes)
#                     vehNr += 1
#                     s_sn = 0 if random.uniform(0, 1) < p_sn_on_off else 1
#                 else:
#                     s_sn = 1 if random.uniform(0, 1) < p_sn_off_on else 0
#                 # if random.uniform(0, 1) < p_sn:
#                 #     print('    <vehicle id="sn%i_%i" type="car" route="route_sn%i" depart="%i" departLane="free" />' % (
#                 #         j, vehNr, j, i), file=routes)
#                 #     vehNr += 1
#         print("</routes>", file=routes)
#
# # def get_options():
# #     optParser = optparse.OptionParser()
# #     optParser.add_option("--nogui", action="store_true",
# #                          default=False, help="run the commandline version of sumo")
# #     options, args = optParser.parse_args()
# #     return options
#
#
# def close_traci():
#     traci.close()
#     sys.stdout.flush()


class TandemEnv(gym.Env):

    def __init__(self, gui, N_tl, N_time):
        #         self.__version__ = "0.1.0"
        # General variables defining the environment
        self.N_tl = N_tl
        self.gui = gui
        self.action_space = spaces.Discrete(2)
        # self.B_max = 8
        self.lam = 1
        self.time = 0
        self.phase = 0
        self.fair_dev = 0
        self.ids = [[], [], [], []]
        self.old_ids = [[], [], [], []]
        self.flag_phase_change = 0
        # self.observation_space = spaces.MultiBinary(n=self.B_max * self.N_tl * 4)
        # state_min = np.zeros(28)
        # state_max = np.append(100 * np.ones(4), 3 * np.ones(12))
        # state_max = np.append(state_max, 15 * np.ones(12))
        state_min = np.zeros(STATE_SIZE-1)
        state_min = np.append(state_min, -100)
        state_max = np.append(100 * np.ones(4), [1, 100])
        # state_max = np.append(state_max, 3 * np.ones(12))
        # state_max = np.append(state_max, 15 * np.ones(12))
        self.observation_space = spaces.Box(state_min, state_max)
        self.MAX_STEPS = N_time
        generate_node(self.N_tl)
        generate_edge(self.N_tl)
        # generate_con(self.N_tl)
        os.system("netconvert data/tandem.netccfg")
        generate_routefile(self.N_tl, self.MAX_STEPS)
        # this script has been called from the command line. It will start sumo as a
        # server, then connect and run
        self.qls = np.zeros(4)
        self.ql_prev_slot = np.zeros(4)
        self.wait_times = {0: [], 1: [], 2: [], 3: []}
        self.sum_wait = np.zeros((4, 2))
        self.loop_counts = np.zeros((4, LOOP_NUM_EDGE))
        self.loop_speeds = np.zeros((4, LOOP_NUM_EDGE))
        self.tot_ns_throughput = 0
        self.tot_we_throughput = 0

    # def loop_count_cal(self, edge):
    #
    #     loop_count_per_edge = np.zeros((1, LOOP_NUM_EDGE))
    #     direction = edge[1]
    #     num_line = {"u": Y_NUM_LANE, "r": X_NUM_LANE, "d": Y_NUM_LANE, "l": X_NUM_LANE}
    #     for loop_loc in range(LOOP_NUM_EDGE):
    #         for line in range(num_line[direction]):
    #             loop_id = "%s_%i_%i" % (direction, loop_loc, line)
    #             loop_count_per_edge[0, loop_loc] += traci.inductionloop.getLastStepVehicleNumber(loop_id)
    #     return loop_count_per_edge
    #
    # def loop_speed_cal(self, edge):
    #
    #     loop_speed_per_edge = np.zeros((1, LOOP_NUM_EDGE))
    #     direction = edge[1]
    #     num_line = {"u": Y_NUM_LANE, "r": X_NUM_LANE, "d": Y_NUM_LANE, "l": X_NUM_LANE}
    #     for loop_loc in range(LOOP_NUM_EDGE):
    #         num_cars = 0
    #         for line in range(num_line[direction]):
    #             loop_id = "%s_%i_%i" % (direction, loop_loc, line)
    #             loop_speed_per_edge[0, loop_loc] += traci.inductionloop.getLastStepVehicleNumber(loop_id) *\
    #                                                 max(0, traci.inductionloop.getLastStepMeanSpeed(loop_id))
    #             num_cars += traci.inductionloop.getLastStepVehicleNumber(loop_id)
    #         loop_speed_per_edge[0, loop_loc] = loop_speed_per_edge[0, loop_loc]/num_cars if num_cars else 0
    #     return loop_speed_per_edge

    def loop_count_cal(self, edge):

        loop_count_per_edge = 0
        direction = edge[1]
        num_lines = {"u": Y_NUM_LANE, "r": X_NUM_LANE, "d": Y_NUM_LANE, "l": X_NUM_LANE}
        for line in range(num_lines[direction]):
            loop_id = "%s_%i" % (direction, line)
            loop_count_per_edge += traci.lanearea.getLastStepVehicleNumber(loop_id)
        return loop_count_per_edge

    def step(self, action):

        self._take_action(action)
        self.time = traci.simulation.getTime()
        veh_num = []
        for i, j in enumerate(DIR):
            edge = "1%s_1" % j
            # ql_old = 0 if self.flag_phase_change else self.qls[i]
            self.qls[i] = traci.edge.getLastStepHaltingNumber(edge)
            veh_num.append(self.loop_count_cal(edge))
            # delta_ql = self.qls[i] - ql_old
            # self.loop_counts[i] = self.loop_count_cal(edge)
            # self.loop_speeds[i] = self.loop_speed_cal(edge)
            # self._calculate_delays_per_edge(action, delta_ql, i)
            # if j in DIR_IN_PHASE[action]:
            #     self.sum_wait[i] = np.zeros(2)
            # else:
            #     new_sum_wait = 2 * self.sum_wait[i, 0] - self.sum_wait[i, 1] + delta_ql
            #     self.sum_wait[i, 1] = self.sum_wait[i, 0]
            #     self.sum_wait[i, 0] = new_sum_wait
        state = np.append(veh_num, self.phase)
        state = np.append(state, self.fair_dev)
        state = np.reshape(state, (1, -1))
        reward = self._get_reward(action)
        self.old_ids = copy.copy(self.ids)
        done = True
        if traci.simulation.getMinExpectedNumber() > 0 and np.sum(self.qls < QL_MAX) > 3:
            done = False
        return state, reward, done, {}

    # def _calculate_delays_per_edge(self):
    #     # directions that have to wait in phase
    #     wait_phase_mapping = {0: ['u', 'd'], 1: ['l', 'r', 'u', 'd'], 2: ['l', 'r'], 3: ['l', 'r', 'u', 'd']}
    #     for i, j in enumerate(DIR):
    #         edge = "1%s_1" % j
    #         phase = traci.trafficlight.getPhase('1')
    #         # ql_old = 0 if self.flag_phase_change else self.qls[i]
    #         ql = traci.edge.getLastStepHaltingNumber(edge)
    #         delta_ql = ql - self.ql_prev_slot[i]
    #         if DIR[i] in wait_phase_mapping[phase]:
    #             if delta_ql > 0:
    #                 self.wait_times[i] = np.append(self.wait_times[i], np.zeros(int(delta_ql)))
    #             if ql > 0:
    #                 if self.wait_times[i] == []:
    #                     self.wait_times[i] = np.zeros(int(ql))
    #                 self.wait_times[i] += 1
    #         else:
    #             self.wait_times[i] = []
    #         # print("Phase=", phase, "Direction=", DIR[i], "wait_time=", self.wait_times[i], "QL_old", self.ql_prev_slot[i],
    #         #       "QL", ql, "QL increase", delta_ql)
    #         self.ql_prev_slot[i] = ql

    def _calculate_delays_per_edge(self, action, delta_ql, edge_index):
        if DIR[edge_index] in DIR_IN_PHASE[action]:
            self.wait_times[edge_index] = []
        else:
            if delta_ql > 0:
                self.wait_times[edge_index] = np.append(self.wait_times[edge_index],
                                                        np.zeros(int(delta_ql)))
            if self.qls[edge_index] > 0:
                self.wait_times[edge_index] += 1
                # print("direction=", DIR[edge_index], "wait_times=", self.wait_times[edge_index],
                #       "QL", self.qls[edge_index],"QL increase", delta_ql)
    # def _take_action(self, action):
    #     phase = traci.trafficlight.getPhase('1')
    #     time_inc = SLOT
    #     if action == 1:
    #         phase = (phase + 1) % 4
    #         time_inc = MIN_TIME
    #     traci.trafficlight.setPhase('1', phase)
    #     # print("phase=", phase)
    #     for _ in range(time_inc):
    #         traci.simulationStep()

    def _take_action(self, action):
        time_inc = SLOT
        self.flag_phase_change = action != self.phase
        self.phase = action
        if action == PHASE_NS:
            tl_phase = 2
            if self.flag_phase_change:
                tl_phase = 1
                time_inc = MIN_TIME
        else:
            tl_phase = 0
            if self.flag_phase_change:
                tl_phase = 3
                time_inc = MIN_TIME

        traci.trafficlight.setPhase('1', tl_phase)
        # reward = 0
        for _ in range(time_inc):
            traci.simulationStep()
            # for j in DIR:
            #     edge = "1%s_1" % j
            #     print("edge=", edge, ", count=", self.loop_count_cal(edge))
            # print("-----------")
        for i, j in enumerate(DIR):
            edge = "1%s_1" % j
            self.ids[i] = traci.edge.getLastStepVehicleIDs(edge)

            # self._calculate_delays_per_edge()
            # reward += self._get_subreward()
        # return reward

    # def _get_reward(self, action):
    #     reward = -np.sum(self.qls)
    #     return reward

    # def _get_reward(self, action):
    #     weighted_delay_sum = NS_PRIORITY_OVER_WE * np.sum(self.sum_wait[[0, 2], 0]) +\
    #                          np.sum(self.sum_wait[[1, 3], 0])
    #     return - weighted_delay_sum
    def _cal_throughput(self, ids, old_ids):
        passed_veh_ids = [[], [], [], []]
        passed_veh_nums = []
        for dir_index in range(len(ids)):
            passed_veh_nums.append(0)
            for i in old_ids[dir_index]:
                if i not in ids[dir_index]:
                    passed_veh_ids[dir_index].append(i)
                    passed_veh_nums[dir_index] += 1
        return passed_veh_nums


    # def _get_reward(self, action):
    #     cost = 0
    #     for edge in range(len(DIR)):
    #         for d in self.wait_times[edge]:
    #             cost += 1 + GAMMA * (2 * d - 1)
    #     return - cost

    def _get_reward(self, action):
        throughputs = np.array(self._cal_throughput(self.ids, self.old_ids))
        ns_throughputs = sum(throughputs[[0, 2]])
        we_throughputs = sum(throughputs[[1, 3]])
        backlog = []
        for j in DIR:
            edge = "1%s_1" % j
            backlog.append(self.loop_count_cal(edge) > 0)

        backlog = np.array(backlog)
        backlog_flag = np.sum(backlog[[0, 2]]) > 0 and np.sum(backlog[[1, 3]]) > 0
        # print(backlog_flag)
        if backlog_flag:
            self.tot_ns_throughput += ns_throughputs
            self.tot_we_throughput += we_throughputs
            self.fair_dev += ns_throughputs/NS_WEIGHT - we_throughputs/WE_WEIGHT
        reward = - np.sum(self.qls) - abs(self.fair_dev) * backlog_flag
        # print(reward)
        return reward


    # def _get_subreward(self):
    #     cost = 0
    #     for edge in range(len(DIR)):
    #         for d in self.wait_times[edge]:
    #             # cost += 1 + GAMMA * (2 * d - 1)
    #             cost += 2 * d - 1
    #     return - cost

    def reset(self):
        # options.nogui = options
        if not self.gui:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        traci.start([sumoBinary, "-c", "data/tandem.sumocfg",
                     "--tripinfo-output", "tripinfo.xml", "--duration-log.disable", "true",
                     "--no-step-log", "true", "--time-to-teleport", "-2"])
        self.qls = np.zeros(4)
        self.ql_prev_slot = np.zeros(4)
        self.wait_times = {0: [], 1: [], 2: [], 3: []}
        self.ids = [[], [], [], []]
        self.old_ids = [[], [], [], []]
        self.phase = 0
        self.flag_phase_change = 0
        self.time = 0
        self.fair_dev = 0
        self.sum_wait = np.zeros((4, 2))
        self.tot_ns_throughput = 0
        self.tot_we_throughput = 0
        state = np.zeros((1, STATE_SIZE))
        return state


