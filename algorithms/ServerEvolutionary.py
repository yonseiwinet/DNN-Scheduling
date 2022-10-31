import numpy as np
import math
import random
import multiprocessing as mp
import time
from copy import deepcopy


class PSOGA:
    def __init__(self, dataset, num_particles, w_max=1.0, w_min=0.5, c1_s=0.5, c1_e=0.5, c2_s=0.5, c2_e=0.5):
        self.num_particles = num_particles
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = dataset.num_partitions

        self.coarsened_graph = deepcopy(dataset.coarsened_graph)
        self.graph = [np.unique(cg) for cg in self.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
        self.piece_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])

        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.w_max = w_max            # constant inertia max weight (how much to weigh the previous velocity)
        self.w_min = w_min            # constant inertia min weight (how much to weigh the previous velocity)
        self.c1_s = c1_s                  # cognative constant (start)
        self.c1_e = c1_e                  # cognative constant (end)
        self.c2_s = c2_s                  # social constant (start)
        self.c2_e = c2_e                  # social constant (end)

    def init(self):
        self.coarsened_graph = deepcopy(self.dataset.coarsened_graph)
        self.graph = [np.unique(cg) for cg in self.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
    
    def get_uncoarsened_x(self, x):
        result = []
        start = end = 0
        for svc in self.dataset.svc_set.services:
            uncoarsened_x = np.zeros_like(self.coarsened_graph[svc.id])
            start = end
            end += len(self.graph[svc.id])
            for i, x_i in enumerate(x[start:end]):
                uncoarsened_x[np.where(self.coarsened_graph[svc.id]==self.graph[svc.id][i])] = x_i
            result.append(uncoarsened_x)
        return np.concatenate(result, axis=None)

    def update_w(self, x_t, g_t):
        ps = 0
        for i in range(self.num_particles):
            ps += np.sum(np.equal(x_t[i], g_t), axis=None)
        ps /= self.num_particles * self.num_timeslots * self.num_pieces
        w = self.w_max - (self.w_max - self.w_min) * np.exp(ps / (ps - 1.01))
        return w

    def generate_random_solutions(self):
        server = np.random.choice(self.server_lst, size=(self.num_particles, self.num_timeslots, self.num_pieces))
        server[0,:,:] = self.server_lst[-1]
        return server

    def run_algo(self, loop, verbose=True, local_search=False, early_exit_loop=5):
        start = time.time()
        max_reward = -np.inf
        not_changed_loop = 0
        eval_lst = []

        x_t = self.generate_random_solutions()
        x_t, p_t, g_t, p_t_eval_lst = self.selection(x_t, local_search=local_search, local_prob=0.5)

        for i in range(loop):
            v_t = self.mutation(x_t, g_t)
            c_t = self.crossover(v_t, p_t, crossover_rate=(self.c1_e-self.c1_s) * (i / loop) + self.c1_s)
            x_t = self.crossover(c_t, g_t, crossover_rate=(self.c2_e-self.c2_s) * (i / loop) + self.c2_s, is_global=True)
            x_t, p_t, g_t, p_t_eval_lst = self.selection(x_t, p_t, g_t, p_t_eval_lst=p_t_eval_lst, local_search=local_search, local_prob=0.5)
            eval_lst.append(np.max(np.sum(p_t_eval_lst, axis=1)))

            if max_reward < np.max(np.sum(p_t_eval_lst, axis=1)):
                max_reward = np.max(np.sum(p_t_eval_lst, axis=1))
                not_changed_loop = 0
            elif not_changed_loop > early_exit_loop:
                uncoarsened_x = np.array([self.get_uncoarsened_x(g_t[t]) for t in range(self.num_timeslots)])
                self.system_manager.set_env(deployed_server=uncoarsened_x[0])
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                r = self.system_manager.get_reward()
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                # print(g_t[0])
                print("t:", np.max(t), t)
                print("e:", sum(e), e, "\033[0m")
                # print("r:", r, "\033[0m")
                return (uncoarsened_x, eval_lst, time.time() - start)
            else:
                not_changed_loop += 1

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                if local_search:
                    print("---------- Memetic PSO-GA #{} loop ----------".format(i))
                else:
                    print("---------- PSO-GA #{} loop ----------".format(i))
                self.system_manager.init_env()
                total_time = []
                total_energy = []
                total_reward = []
                for t in range(self.num_timeslots):
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(g_t[t]))
                    #print("#timeslot {} x: {}".format(t, g_t[t]))
                    #print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(self.system_manager.total_time_dp())
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    # self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(g_t[t]), timeslot=t)
                print("mean t: {:.5f}".format(np.max(total_time, axis=None)), np.max(np.array(total_time), axis=0))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                # print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))

        uncoarsened_x = np.array([self.get_uncoarsened_x(g_t[t]) for t in range(self.num_timeslots)])
        self.system_manager.set_env(deployed_server=uncoarsened_x[0])
        t = self.system_manager.total_time_dp()
        e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
        r = self.system_manager.get_reward()
        print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
        # print(g_t[0])
        print("t:", np.max(t), t)
        print("e:", sum(e), e, "\033[0m")
        # print("r:", r, "\033[0m")
        return (uncoarsened_x, eval_lst, time.time() - start)

    def selection(self, x_t, p_t=None, g_t=None, p_t_eval_lst=None, local_search=False, local_prob=0.5):
        if p_t is None and g_t is None:
            x_t, p_t_eval_lst = self.evaluation(x_t, local_search, local_prob)
            p_t_eval_sum = np.sum(p_t_eval_lst, axis=1)
            p_t = np.copy(x_t)
            g_t = np.copy(x_t[np.argmax(p_t_eval_sum),:,:])
        else:
            x_t, new_eval_lst = self.evaluation(x_t, local_search, local_prob)
            new_eval_sum = np.sum(new_eval_lst, axis=1)
            p_t_eval_sum = np.sum(p_t_eval_lst, axis=1)
            indices = np.where(new_eval_sum > p_t_eval_sum)
            p_t[indices,:,:] = x_t[indices,:,:]
            p_t_eval_lst[indices,:] = new_eval_lst[indices,:]
            p_t_eval_sum[indices] = new_eval_sum[indices]
            g_t = np.copy(p_t[np.argmax(p_t_eval_sum),:,:])
        return x_t, p_t, g_t, p_t_eval_lst

    @staticmethod
    def crossover_multiprocessing(inputs):
        np.random.seed(random.randint(0,2147483647))
        a_t, b_t, num_timeslots, num_pieces = inputs
        for t in range(num_timeslots):
            cross_point = np.random.choice(num_pieces, size=2, replace=False)
            if cross_point[0] > cross_point[1]:
                cross_point[0], cross_point[1] = cross_point[1], cross_point[0]

            # crossover x: deployed_server
            a_t[t,cross_point[0]:cross_point[1]+1] = b_t[t,cross_point[0]:cross_point[1]+1]

            # # crossover x: deployed_server
            # for cp in cross_point:
            #     a_t[t,cp] = b_t[t,cp]
        return a_t

    def crossover(self, a_t, b_t, crossover_rate, is_global=False):
        new_a_t = np.copy(a_t)
        crossover_idx = np.random.rand(self.num_particles)
        crossover_idx = crossover_idx < crossover_rate
        crossover_idx = np.where(crossover_idx)[0]
        if len(crossover_idx) == 0:
            crossover_idx = np.array([np.random.randint(low=0, high=self.num_particles)])
        
        if is_global:
            temp = [self.crossover_multiprocessing((new_a_t[i], b_t, self.num_timeslots, self.num_pieces)) for i in crossover_idx]
        else:
            temp = [self.crossover_multiprocessing((new_a_t[i], b_t[i], self.num_timeslots, self.num_pieces)) for i in crossover_idx]
        new_a_t[crossover_idx] = np.array(temp)
        return new_a_t
        if is_global:
            working_queue = [(new_a_t[i], b_t) for i in crossover_idx]
        else:
            working_queue = [(new_a_t[i], b_t[i]) for i in crossover_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.crossover_multiprocessing, working_queue))
        new_a_t[crossover_idx] = np.array(temp)
        return new_a_t

    @staticmethod
    def mutation_multiprocessing(inputs):
        v_t, server_lst, num_timeslots, num_pieces = inputs
        np.random.seed(random.randint(0,2147483647))
        for t in range(num_timeslots):
            mutation_point = np.random.randint(low=0, high=num_pieces)

            # mutate x: deployed_server
            another_s_id = np.random.choice(server_lst)
            while v_t[t,mutation_point] == another_s_id:
                another_s_id = np.random.choice(server_lst)
            v_t[t,mutation_point] = another_s_id
        return v_t

    def mutation(self, x_t, g_t, mutation_ratio=None):
        v_t = np.copy(x_t)
        if mutation_ratio == None:
            w = self.update_w(v_t[:,:,:self.num_pieces], g_t[:,:self.num_pieces])
        else:
            w = mutation_ratio
        mutation_idx = np.random.rand(self.num_particles)
        mutation_idx = mutation_idx < w
        mutation_idx = np.where(mutation_idx)[0]
        if len(mutation_idx) == 0:
            mutation_idx = np.array([np.random.randint(low=0, high=self.num_particles)])
        
        temp = [self.mutation_multiprocessing((v_t[i], self.server_lst, self.num_timeslots, self.num_pieces)) for i in mutation_idx]
        v_t[mutation_idx] = np.array(temp)
        return v_t
        working_queue = [(v_t[i], self.server_lst, self.num_timeslots, self.num_pieces) for i in mutation_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.mutation_multiprocessing, working_queue))
        v_t[mutation_idx] = np.array(temp)
        return v_t

    # convert invalid action to valid action, deployed server action.
    def deployed_server_reparation(self, x):
        # request device constraint
        for c_id, s_id in enumerate(x):
            if s_id not in self.server_lst:
                x[c_id] = self.piece_device_map[c_id]
        self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
        while False in self.system_manager.constraint_chk():
            # 각 서버에 대해서,
            for s_id in range(self.num_servers-1):
                deployed_container_lst = list(np.where(x == s_id)[0])
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                while self.system_manager.constraint_chk(s_id=s_id) == False:
                    c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                    x[c_id] = self.server_lst[-1] # edge server로 보냄
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
    
    def local_search(self, action):
        np.random.seed(random.randint(0,2147483647))
        # local search x: deployed_server
        self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action))
        max_reward = self.system_manager.get_reward()
        for j in np.random.choice(self.num_pieces, size=2, replace=False): # for jth layer
            # local search x: deployed_server
            if action[j] not in self.server_lst:
                server_lst = list(set(self.server_lst)-set([0]))
            else:
                server_lst = list(set(self.server_lst)-set([action[j]]))
            for s_id in server_lst:
                if s_id == 0:
                    s_id = self.piece_device_map[j]
                temp = action[j]
                action[j] = s_id
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action))
                cur_reward = self.system_manager.get_reward()
                if cur_reward > max_reward and all(self.system_manager.constraint_chk()):
                    max_reward = cur_reward
                else:
                    action[j] = temp
    
    def evaluation_multiprocessing(self, inputs):
        position, local_search, local_prob = inputs
        self.system_manager.init_env()
        reward = []
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(position[t])
            if local_search and random.random() < local_prob:
                self.local_search(position[t])
            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(position[t]))
            reward.append(self.system_manager.get_reward())
            # self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(position[t]), timeslot=t)
        return position, reward

    def evaluation(self, positions, local_search, local_prob):
        working_queue = [(position, local_search, local_prob) for position in positions]
        with mp.Pool(processes=30) as pool:
            outputs = list(pool.map(self.evaluation_multiprocessing, working_queue))
        positions = np.array([output[0] for output in outputs])
        evaluation_lst = np.array([output[1] for output in outputs])
        return positions, evaluation_lst


class Genetic(PSOGA):
    def __init__(self, dataset, num_solutions, mutation_ratio=0.3, cross_over_ratio=0.7):
        self.num_solutions = num_solutions
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = dataset.num_partitions

        self.coarsened_graph = deepcopy(dataset.coarsened_graph)
        self.graph = [np.unique(cg) for cg in self.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
        self.piece_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])

        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.mutation_ratio = mutation_ratio
        self.cross_over_ratio = cross_over_ratio

    def generate_random_solutions(self):
        server = np.random.choice(self.server_lst, size=(self.num_solutions, self.num_timeslots, self.num_pieces))
        server[0,:,:] = self.server_lst[-1]
        return server
    
    def local_search_multiprocessing(self, action):
        np.random.seed(random.randint(0,2147483647))
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            # local search x: deployed_server
            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]))
            max_reward = self.system_manager.get_reward()
            for j in np.random.choice(self.num_pieces, size=2, replace=False): # for jth layer
                # local search x: deployed_server
                if action[t,j] not in self.server_lst:
                    server_lst = list(set(self.server_lst)-set([0]))
                else:
                    server_lst = list(set(self.server_lst)-set([action[t,j]]))
                for s_id in server_lst:
                    if s_id == 0:
                        s_id = self.piece_device_map[j]
                    temp = action[t,j]
                    action[t,j] = s_id
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]))
                    cur_reward = self.system_manager.get_reward()
                    if cur_reward > max_reward and all(self.system_manager.constraint_chk()):
                        max_reward = cur_reward
                    else:
                        action[t,j] = temp

            # self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(action[t]), timeslot=t)
        return action

    # we have to find local optimum from current chromosomes.
    def local_search(self, action, local_prob=0.5):
        local_idx = np.random.rand(action.shape[0])
        local_idx = local_idx < local_prob
        local_idx = np.where(local_idx)[0]
        if len(local_idx) == 0:
            local_idx = np.array([np.random.randint(low=0, high=action.shape[0])])
        np.random.shuffle(local_idx)

        working_queue = [action[i] for i in local_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.local_search_multiprocessing, working_queue))
        action[local_idx] = np.array(temp)
        return action

    def run_algo(self, loop, verbose=True, local_search=False, early_exit_loop=5):
        start = time.time()
        max_reward = -np.inf
        not_changed_loop = 0
        eval_lst = []

        p_t = self.generate_random_solutions()
        p_known = np.copy(p_t)
        if local_search:
            p_t = self.local_search(p_t, local_prob=0.5)

        for i in range(loop):
            q_t = self.selection(p_t, p_known)
            q_t = self.mutation(q_t, self.mutation_ratio)
            q_t = self.crossover(q_t, self.cross_over_ratio)
            if local_search:
                q_t = self.local_search(q_t, local_prob=0.5)
            p_known = np.copy(q_t)
            p_t, v = self.fitness_selection(p_t, q_t)
            eval_lst.append(v)

            if max_reward < v:
                max_reward = v
                not_changed_loop = 0
            elif not_changed_loop > early_exit_loop:
                uncoarsened_x = np.array([self.get_uncoarsened_x(p_t[0,t]) for t in range(self.num_timeslots)])
                self.system_manager.set_env(deployed_server=uncoarsened_x[0])
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                r = self.system_manager.get_reward()
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                # print(p_t[0,0])
                print("t:", np.max(t), t)
                print("e:", sum(e), e, "\033[0m")
                # print("r:", r, "\033[0m")
                return (uncoarsened_x, eval_lst, time.time() - start)
            else:
                not_changed_loop += 1

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                if local_search:
                    print("---------- Memetic Genetic #{} loop ----------".format(i))
                else:
                    print("---------- Genetic #{} loop ----------".format(i))
                self.system_manager.init_env()
                total_time = []
                total_energy = []
                total_reward = []
                for t in range(self.num_timeslots):
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(p_t[0,t]))
                    #print("#timeslot {} x: {}".format(t, p_t[0,t]))
                    #print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(self.system_manager.total_time_dp())
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    # self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(p_t[0,t]), timeslot=t)
                print("mean t: {:.5f}".format(np.max(total_time, axis=None)), np.max(np.array(total_time), axis=0))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                # print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))

        uncoarsened_x = np.array([self.get_uncoarsened_x(p_t[0,t]) for t in range(self.num_timeslots)])
        self.system_manager.set_env(deployed_server=uncoarsened_x[0])
        t = self.system_manager.total_time_dp()
        e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
        r = self.system_manager.get_reward()
        print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
        # print(p_t[0,0])
        print("t:", np.max(t), t)
        print("e:", sum(e), e, "\033[0m")
        # print("r:", r, "\033[0m")
        return (uncoarsened_x, eval_lst, time.time() - start)

    @staticmethod
    def union(*args):
        if len(args) == 0:
            union = np.array([])
            max_len = 0
        elif len(args) == 1:
            union = args[0]
            max_len = len(args[0])
        else:
            union = np.append(args[0], args[1], axis=0)
            max_len = max(len(args[0]), len(args[1]))

        for i in range(2, len(args)):
            if max_len < len(args[i]):
                max_len = len(args[i])
            union = np.append(union, args[i], axis=0)
        return union, max_len

    def selection(self, *args):
        union, max_len = self.union(*args)
        order = np.arange(union.shape[0])
        np.random.shuffle(order)
        return union[order[:max_len], :]

    def fitness_selection(self, *args):
        union, max_len = self.union(*args)
        ev_lst = np.sum(self.evaluation(union), axis=1)
        ev_lst = list(map(lambda x: (x[0], x[1]), enumerate(ev_lst)))
        ev_lst.sort(key=lambda x: x[1], reverse=True)
        sorted_idx = list(map(lambda x: x[0], ev_lst))
        union = union[sorted_idx[:max_len], :]
        return union, ev_lst[0][1]

    @staticmethod
    def crossover_multiprocessing(inputs):
        np.random.seed(random.randint(0,2147483647))
        a_t, b_t, num_timeslots, num_pieces = inputs
        for t in range(num_timeslots):
            cross_point = np.random.choice(num_pieces, size=2, replace=False)
            if cross_point[0] > cross_point[1]:
                cross_point[0], cross_point[1] = cross_point[1], cross_point[0]

            # crossover x: deployed_server
            temp = np.copy(a_t[t,cross_point[0]:cross_point[1]+1])
            a_t[t,cross_point[0]:cross_point[1]+1] = b_t[t,cross_point[0]:cross_point[1]+1]
            b_t[t,cross_point[0]:cross_point[1]+1] = temp

            # # crossover x: deployed_server
            # for cp in cross_point:
            #     temp = a_t[t,cp]
            #     a_t[t,cp] = b_t[t,cp]
            #     b_t[t,cp] = temp
        return np.concatenate([a_t.reshape(1,num_timeslots,-1), b_t.reshape(1,num_timeslots,-1)], axis=0)

    def crossover(self, action, crossover_rate):
        crossover_idx = np.random.rand(self.num_solutions)
        crossover_idx = crossover_idx < crossover_rate
        crossover_idx = np.where(crossover_idx)[0]
        if len(crossover_idx) < 2:
            crossover_idx = np.array([np.random.randint(low=0, high=self.num_solutions, size=2)])
        np.random.shuffle(crossover_idx)
        if len(crossover_idx) - math.floor(crossover_idx.size / 2) * 2:
            crossover_idx = crossover_idx[:-1]
        
        temp = [self.crossover_multiprocessing((action[crossover_idx[i * 2]], action[crossover_idx[i * 2 + 1]], self.num_timeslots, self.num_pieces)) for i in range(math.floor(crossover_idx.size / 2))]
        action[crossover_idx] = np.concatenate(temp, axis=0)
        return action
        working_queue = [(action[crossover_idx[i * 2]], action[crossover_idx[i * 2 + 1]], self.num_timeslots, self.num_pieces) for i in range(math.floor(crossover_idx.size / 2))]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.crossover_multiprocessing, working_queue))
        action[crossover_idx] = np.concatenate(temp, axis=0)
        return action

    @staticmethod
    def mutation_multiprocessing(inputs):
        action, server_lst, num_timeslots, num_pieces = inputs
        np.random.seed(random.randint(0,2147483647))
        for t in range(num_timeslots):
            mutation_point = np.random.randint(low=0, high=num_pieces)

            # mutate x: deployed_server
            another_s_id = np.random.choice(server_lst)
            while action[t,mutation_point] == another_s_id:
                another_s_id = np.random.choice(server_lst)
            action[t,mutation_point] = another_s_id
        return action.reshape(1,num_timeslots,-1)

    def mutation(self, action, mutation_ratio):
        mutation_idx = np.random.rand(self.num_solutions)
        mutation_idx = mutation_idx < mutation_ratio
        mutation_idx = np.where(mutation_idx)[0]
        if len(mutation_idx) == 0:
            mutation_idx = np.array([np.random.randint(low=0, high=self.num_solutions)])
        
        temp = [self.mutation_multiprocessing((action[i], self.server_lst, self.num_timeslots, self.num_pieces)) for i in mutation_idx]
        action[mutation_idx] = np.concatenate(temp, axis=0)
        return action
        working_queue = [(action[i], self.server_lst, self.num_timeslots, self.num_pieces) for i in mutation_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.mutation_multiprocessing, working_queue))
        action[mutation_idx] = np.concatenate(temp, axis=0)
        return action