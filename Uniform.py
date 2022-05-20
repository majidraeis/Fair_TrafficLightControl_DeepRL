from Intersection_generate import *


class General_Env(gym.Env):

    def __init__(self, N_tl, N_time):
        #         self.__version__ = "0.1.0"
        # General variables defining the environment
        self.N_tl = N_tl
        self.action_space = spaces.Discrete(2)
        # self.B_max = 8
        self.lam = 1
        self.time = 0
        self.phase = 0
        self.ids = [[], [], [], []]
        self.old_ids = [[], [], [], []]
        self.flag_phase_change = 0
        state_min = np.zeros(STATE_SIZE)
        state_max = np.append(100 * np.ones(8), 1)
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
        self.priority = [NS_P, WE_P, NS_P, WE_P]
        self.sum_wait = np.zeros((4, 2))
        self.loop_counts = np.zeros((4, LOOP_NUM_EDGE))
        self.loop_speeds = np.zeros((4, LOOP_NUM_EDGE))
        self.tot_ns_throughput = 0
        self.tot_we_throughput = 0
        self.fair_dev = 0


    def step(self, action):

        self._take_action(action)
        self.time = traci.simulation.getTime()
        for i, j in enumerate(DIR):
            edge = "1%s_1" % j
            ql_old = 0 if self.flag_phase_change else self.qls[i]
            self.qls[i] = traci.edge.getLastStepHaltingNumber(edge)
            delta_ql = self.qls[i] - ql_old
            # self.loop_counts[i] = self.loop_count_cal(edge)
            # self.loop_speeds[i] = self.loop_speed_cal(edge)
            self._calculate_delays_per_edge(action, delta_ql, i)
            if j in DIR_IN_PHASE[action]:
                self.sum_wait[i] = np.zeros(2)
            else:
                new_sum_wait = 2 * self.sum_wait[i, 0] - self.sum_wait[i, 1] + delta_ql
                self.sum_wait[i, 1] = self.sum_wait[i, 0]
                self.sum_wait[i, 0] = new_sum_wait

        self.old_ids = copy.copy(self.ids)
        done = True
        if traci.simulation.getMinExpectedNumber() > 0 and np.sum(self.qls < QL_MAX) > 3:
            done = False
        # print("-------------")
        # print("states=", state, "reward=", reward)
        return state, reward, done, {}


    def _calculate_delays_per_edge(self, action, delta_ql, edge_index):
        if DIR[edge_index] in DIR_IN_PHASE[action]:
            self.wait_times[edge_index] = []
        else:
            if delta_ql > 0:
                self.wait_times[edge_index] = np.append(self.wait_times[edge_index],
                                                        np.zeros(int(delta_ql)))
            if self.qls[edge_index] > 0:
                self.wait_times[edge_index] += 1


    def _take_action(self, action):
        self.flag_phase_change = action != self.phase
        self.phase = action
        if action == PHASE_NS:
            tl_phase = 2
            if self.flag_phase_change:
                tl_phase = 1
        else:
            tl_phase = 0
            if self.flag_phase_change:
                tl_phase = 3

        traci.trafficlight.setPhase('1', tl_phase)
        # reward = 0

        traci.simulationStep()
        for i, j in enumerate(DIR):
            edge = "1%s_1" % j
            self.ids[i] = traci.edge.getLastStepVehicleIDs(edge)

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
            self.fair_dev += ns_throughputs - we_throughputs


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

    def loop_count_cal(self, edge):

        loop_count_per_edge = 0
        direction = edge[1]
        num_lines = {"u": Y_NUM_LANE, "r": X_NUM_LANE, "d": Y_NUM_LANE, "l": X_NUM_LANE}
        for line in range(num_lines[direction]):
            loop_id = "%s_%i" % (direction, line)
            loop_count_per_edge += traci.lanearea.getLastStepVehicleNumber(loop_id)
        return loop_count_per_edge

    def _get_reward(self, action):
        cost = 0
        for edge in range(len(DIR)):
            for d in self.wait_times[edge]:
                cost += 1 + GAMMA * (2 * d - 1)
                # cost += 1 + self.priority[edge] * (2 * d - 1)
        return - cost


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
        self.sum_wait = np.zeros((4, 2))
        self.tot_ns_throughput = 0
        self.tot_we_throughput = 0
        self.fair_dev = 0
        state = np.zeros((1, STATE_SIZE))
        return state

