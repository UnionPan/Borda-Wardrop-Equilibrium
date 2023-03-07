from os import link
from graph import TrafficNetwork, Graph
import numpy as np


class TrafficFlowModel:
    ''' TRAFFIC FLOW ASSIGN MODEL
        Inside the Frank-Wolfe algorithm is given, one can use
        the method `solve` to compute the numerical solution of
        User Equilibrium problem.
    '''
    def __init__(self, graph= None, origins= [], destinations= [], 
    demands= [], link_free_time= None, link_capacity= None, theta=None, d=None):

        self.__network = TrafficNetwork(graph= graph, O= origins, D= destinations)

        # Initialization of parameters
        self.__link_free_time = np.array(link_free_time)
        self.__link_capacity = np.array(link_capacity)
        self.__demand = np.array(demands)

        # Alpha and beta (used in performance function)
        self._alpha = 0.15
        self._beta = 4 

        # Convergent criterion
<<<<<<< HEAD
        self._conv_accuracy = 3e-3
=======
        self._conv_accuracy = 1e-8
>>>>>>> 981e6a6 (debug)

        # Boolean varible: If true print the detail while iterations
        self.__detail = False
        

        # Boolean varible: If true the model is solved properly with respect to the WE and SO
        self.__solved = False
        self.__so_solved = False

        # Some variables for contemporarily storing the
        # computation results for both Wardrop equilibrium and system optimum
        self.__final_link_flow = None
        self.__final_link_time = None 
        self.so_final_link_flow = None
        self.so_final_link_time = None
        self.__iterations_times = None
        self.__so_iterations_times = None

        #stealthy modification parameter from the attacker
        self.theta = theta
        self.d = d


        #self.so_max_iter = 30
        

    def __insert_links_in_order(self, links):
        ''' Insert the links as the expected order into the
            data structure `TrafficFlowModel.__network`
        '''
        first_vertice = [link[0] for link in links]
        for vertex in first_vertice:
            self.__network.add_vertex(vertex)
        for link in links:
            self.__network.add_edge(link)
        
    def solve(self, initial_flow=None, Learning=True):
        ''' Solve the traffic flow assignment model (user equilibrium)
            by Frank-Wolfe algorithm, all the necessary data must be 
            properly input into the model in advance. 

            (Implicitly) Return
            ------
            self.__solved = True
        '''
        if self.__detail:
            print(self.__dash_line())
            print("FLOW LATENCY ATTACKED TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - DETAIL OF ITERATIONS")
            print(self.__dash_line())
            print(self.__dash_line())
            print("Initialization")
            print(self.__dash_line())
        
        # Step 0: based on the x0, generate the x1
        empty_flow = np.zeros(self.__network.num_of_links())
        link_flow = self.__all_or_nothing_assign(empty_flow)
        flag = False
        if initial_flow is not None:
            flag = True
            link_flow =  self.__path_flow_to_link_flow(initial_flow)

        flows = []
        potentials = []
        counter = 0
        while True:
            
            if self.__detail:
                print(self.__dash_line())
                print("Iteration %s" % counter)
                print(self.__dash_line())
                print("Current link flow:\n%s" % link_flow)

            # Step 1 & Step 2: Use the link flow matrix -x to generate the time, then generate the auxiliary link flow matrix -y
            
            auxiliary_link_flow = self.__all_or_nothing_assign(link_flow)
            
            potential = self.__object_function(link_flow)
            potentials.append(potential)
            flows.append(link_flow)
            # Step 3: Linear Search
            if Learning == True:
                opt_theta = self.__golden_section(link_flow, auxiliary_link_flow)
            else:
                opt_theta = 0.8 #* (counter + 1) ** (-0.75)
            #opt_theta = 0.1 * (counter+1) ** (-1.2)
            # Step 4: Using optimal theta to update the link flow matrix
            new_link_flow = (1 - opt_theta) * link_flow + opt_theta * auxiliary_link_flow
            #print(np.sum(new_link_flow))
            # Print the detail if necessary
            if self.__detail:
                print("Optimal theta: %.8f" % opt_theta)
                print("Auxiliary link flow:\n%s" % auxiliary_link_flow)

            # Step 5: Check the Convergence, if FALSE, then return to Step 1
            #if self.__is_convergent(link_flow, new_link_flow) and counter >= 30:
            if (counter >= 29 and flag == False) or (counter >=70 and flag == True):
                if self.__detail:
                    print(self.__dash_line())
                self.__solved = True
                self.__final_link_flow = new_link_flow
                self.__final_link_time = self.__link_flow_to_link_time_actual(self.__final_link_flow)
                self.__iterations_times = counter
                break
            else:
                link_flow = new_link_flow
                counter += 1
        
        return potentials, flows

    def exp_weight(self, initial_flow, iter):
        ''' Solve the traffic flow assignment model (user equilibrium)
            by EXP algorithm, all the necessary data must be 
            properly input into the model in advance. 

            (Implicitly) Return
            ------
            self.__solved = True
        '''
        if self.__detail:
            print(self.__dash_line())
            print("FLOW LATENCY ATTACKED TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nEXP3 ALGORITHM - DETAIL OF ITERATIONS")
            print(self.__dash_line())
            print(self.__dash_line())
            print("Initialization")
            print(self.__dash_line())

        path_flow = initial_flow
        link_flow = self.__path_flow_to_link_flow(path_flow)
        
        #print("the initial link_flow{}".format(link_flow))

        potentials = []

        for t in range(iter):

            eta = 1e-5 * (t+1) ** (-1)

            if self.__detail:
                print(self.__dash_line())
                print("Iteration %s" % t)
                print(self.__dash_line())
                print("Current link flow:\n%s" % link_flow)
                print("Current path flow:\n%s" % path_flow)
            
            
            potential = self.__object_function(link_flow)
            potentials.append(potential)
           
            link_time = self.__link_flow_to_link_time_actual(link_flow)
            print("the link time is {} ". format(link_time))
            latency_feedback = self.__link_time_to_path_time(link_time)
            print("the latency is {} ". format(latency_feedback))
            new_path_flow = self.soft_max(path_flow, latency_feedback, eta)
            
            link_flow = self.__path_flow_to_link_flow(new_path_flow)
            print("the link flow is {} ".format(link_flow))

            path_flow = new_path_flow

            if self.__detail: 
                print("Learning Rate {}".format(eta))
                print("Current Beckmann Potential: {}".format(potential))

        
        return potentials 

    def soft_max(self, path_flow, latency_feedback, eta):
        print("latency_feedback is")
        print(latency_feedback)
        new_path_flow = np.zeros(self.__network.num_of_paths())
        unnormalized =  path_flow * np.exp( - eta * latency_feedback)
        
        print(unnormalized)
        print('after')
        

        for OD_pair_index in  range(self.__network.num_of_OD_pairs()):
            indice_grouped = []
            for path_index in range(self.__network.num_of_paths()):
                if self.__network.paths_category()[path_index] == OD_pair_index:
                    indice_grouped.append(path_index) 
            #print(unnormalized[indice_grouped])
            #print(np.sum(unnormalized[indice_grouped]))
            new_path_flow[indice_grouped] = self.__demand[OD_pair_index] * unnormalized[indice_grouped] / np.sum(unnormalized[indice_grouped])
        print(new_path_flow)
        return new_path_flow



    def _formatted_solution(self):
        ''' According to the link flow we obtained in `solve`,
            generate a tuple which contains four elements:
            `link flow`, `link travel time`, `path travel time` and
            `link vehicle capacity ratio`. This function is exposed 
            to users in case they need to do some extensions based 
            on the computation result.
        '''
        if self.__solved:
            link_flow = self.__final_link_flow
            link_time = self.__final_link_time
            path_time = self.__link_time_to_path_time(link_time)
            link_vc = link_flow / self.__link_capacity
            
            return link_flow, link_time, path_time, link_vc
        else:
            return None

    def so_formatted_solution(self):
        '''
           According to the link flow obtained from so_solve,
           generate a tuple which contains four elements:
           'so_link_flow', 'so_link_time', 'path 
        '''

        if self.__so_solved:
            so_link_flow = self.so_final_link_flow
            so_link_time = self.so_final_link_time
            so_path_time = self.__link_time_to_path_time(so_link_time)
            so_link_vc = so_link_flow / self.__link_capacity
            return so_link_flow, so_link_time, so_path_time, so_link_vc    
        else:
            return None

    def price_of_anarchy(self):
        if self.__so_solved and self.__solved:
            poa = self.__final_link_flow.dot(self.__final_link_time) / self.so_final_link_flow.dot(self.so_final_link_time)
            return poa
        else:
            return None
    
    def __link_flow_to_link_time_actual(self, link_flow):
        n_links = self.__network.num_of_links()
        link_time = np.zeros(n_links)
        for i in range(n_links):
            link_time[i] = self.__link_time_performance(link_flow[i], self.__link_free_time[i], self.__link_capacity[i])
        return link_time

    def report(self):
        ''' Generate the report of the result in console,
            this function can be invoked only after the
            model is solved.
        '''
        if self.__solved and self.__so_solved:
            # Print the input of the model
            print(self)
            
            # Print the report
            
            # Do the computation
            link_flow, link_time, path_time, link_vc = self._formatted_solution()

            print(self.__dash_line())
            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - REPORT OF SOLUTION")
            print(self.__dash_line())
            print(self.__dash_line())
            print("TIMES OF ITERATION : %d" % self.__iterations_times)
            print(self.__dash_line())
            print(self.__dash_line())
            print("PERFORMANCE OF LINKS")
            print(self.__dash_line())
            for i in range(self.__network.num_of_links()):
                print("%2d : link= %12s, flow= %8.2f, time= %8.3f, v/c= %.3f" % (i, self.__network.edges()[i], link_flow[i], link_time[i], link_vc[i]))
            print(self.__dash_line())
            print("PERFORMANCE OF PATHS (GROUP BY ORIGIN-DESTINATION PAIR)")
            print(self.__dash_line())
            counter = 0
            for i in range(self.__network.num_of_paths()):
                if counter < self.__network.paths_category()[i]:
                    counter = counter + 1
                    print(self.__dash_line())
                print("%2d : group= %2d, time= %8.3f, path= %s" % (i, self.__network.paths_category()[i], path_time[i], self.__network.paths()[i]))
            print(self.__dash_line())

            so_link_flow, so_link_time, so_path_time, so_link_vc = self.so_formatted_solution()
            print(self.__dash_line())
            print("TRAFFIC FLOW ASSIGN MODEL (SYSTEM OPTIMUM) \nFRANK-WOLFE ALGORITHM - REPORT OF SOLUTION")
            print(self.__dash_line())
            print(self.__dash_line())
            print("TIMES OF ITERATION : %d" % self.__so_iterations_times)
            print(self.__dash_line())
            print(self.__dash_line())
            print("PERFORMANCE OF LINKS")
            print(self.__dash_line())
            for i in range(self.__network.num_of_links()):
                print("%2d : link= %12s, flow= %8.2f, time= %8.3f, v/c= %.3f" % (i, self.__network.edges()[i], so_link_flow[i], so_link_time[i], so_link_vc[i]))
            print(self.__dash_line())
            print(self.__dash_line())
            print("PERFORMANCE OF PATHS (GROUP BY ORIGIN-DESTINATION PAIR)")
            print(self.__dash_line())
            counter = 0
            for i in range(self.__network.num_of_paths()):
                if counter < self.__network.paths_category()[i]:
                    counter = counter + 1
                    print(self.__dash_line())
                print("%2d : group= %2d, time= %8.3f, path= %s" % (i, self.__network.paths_category()[i], so_path_time[i], self.__network.paths()[i]))
            print(self.__dash_line())

            poa = self.price_of_anarchy()
            print(self.__dash_line())
            print("THE PRICE OF ANARCHY")
            print(self.__dash_line())
            print("PoA = {}".format(poa))
        else:
            raise ValueError("The report could be generated only after the model is solved!")

    def __all_or_nothing_assign(self, link_flow):
        ''' Perform the all-or-nothing assignment of
            Frank-Wolfe algorithm in the User Equilibrium
            Traffic Assignment Model.
            This assignment aims to assign all the traffic
            flow, within given origin and destination, into
            the least time consuming path

            Input: link flow -> Output: new link flow
            The input is an array.
        '''
        # LINK FLOW -> LINK TIME
        link_time = self.__link_flow_to_link_time_actual(link_flow)
        # LINK TIME -> PATH TIME
        path_time = self.__link_time_to_path_time(link_time)

        # PATH TIME -> PATH FLOW
        # Find the minimal traveling time within group 
        # (splited by origin - destination pairs) and
        # assign all the flow to that path
        path_flow = np.zeros(self.__network.num_of_paths())
        for OD_pair_index in range(self.__network.num_of_OD_pairs()):
            indice_grouped = []
            for path_index in range(self.__network.num_of_paths()):
                if self.__network.paths_category()[path_index] == OD_pair_index:
                    indice_grouped.append(path_index)
            sub_path_time = [path_time[ind] for ind in indice_grouped]
            min_in_group = min(sub_path_time)
            ind_min = sub_path_time.index(min_in_group)
            target_path_ind = indice_grouped[ind_min]
<<<<<<< HEAD
            path_flow[target_path_ind] = self.__demand[OD_pair_index]
            #path_flow[target_path_ind] = sum(self.__demand) * self.d[OD_pair_index]
            path_flow[target_path_ind] = self.d.dot(self.__demand)[OD_pair_index] 
=======
            path_flow[target_path_ind] = self.__demand[OD_pair_index] 
            #path_flow[target_path_ind] = sum(self.__demand) * self.d[OD_pair_index]
>>>>>>> 981e6a6 (debug)
        if self.__detail:
            print("Link time:\n%s" % link_time)
            print("Path flow:\n%s" % path_flow)
            print("Path time:\n%s" % path_time)
        
        # PATH FLOW -> LINK FLOW
        new_link_flow = self.__path_flow_to_link_flow(path_flow)

        return new_link_flow
        
    def __link_flow_to_link_time(self, link_flow):
        ''' Based on current link flow, use link 
            time performance function to compute the link 
            traveling time.
            The input is an array.
        '''
        n_links = self.__network.num_of_links()
        link_flow = self.theta.dot(link_flow)
        link_time = np.zeros(n_links)
        for i in range(n_links):
            link_time[i] = self.__link_time_performance(link_flow[i], self.__link_free_time[i], self.__link_capacity[i])
        return link_time

    def __link_time_to_path_time(self, link_time):
        ''' Based on current link traveling time,
            use link-path incidence matrix to compute 
            the path traveling time.
            The input is an array.
        '''
        path_time = link_time.dot(self.__network.LP_matrix())
        return path_time
    
    def __path_flow_to_link_flow(self, path_flow):
        ''' Based on current path flow, use link-path incidence 
            matrix to compute the traffic flow on each link.
            The input is an array.
        '''
        link_flow = self.__network.LP_matrix().dot(path_flow)
        return link_flow

    def _get_path_free_time(self):
        ''' Only used in the final evaluation, not the recursive structure
        '''
        path_free_time = self.__link_free_time.dot(self.__network.LP_matrix())
        return path_free_time

    def __link_time_performance(self, link_flow, t0, capacity):
        ''' Performance function, which indicates the relationship
            between flows (traffic volume) and travel time on 
            the same link. According to the suggestion from Federal
            Highway Administration (FHWA) of America, we could use
            the following function:
                t = t0 * (1 + alpha * (flow / capacity))^beta
        '''
        value = t0 * (1 + self._alpha * ((link_flow/capacity)**self._beta)) + np.random.normal(0, t0/3)
        return value

    def __link_time_performance_integrated(self, link_flow, t0, capacity):
        ''' The integrated (with repsect to link flow) form of
            aforementioned performance function.
        '''
        val1 = t0 * link_flow
        # Some optimization should be implemented for avoiding overflow
        val2 = (self._alpha * t0 * link_flow / (self._beta + 1)) * (link_flow / capacity)**self._beta
        value = val1 + val2
        return value

    def __object_function(self, mixed_flow):
        ''' Objective function in the linear search step 
            of the optimization model of user equilibrium 
            traffic assignment problem, the only variable
            is mixed_flow in this case.
        '''
        val = 0
        for i in range(self.__network.num_of_links()):
            val += self.__link_time_performance_integrated(link_flow=mixed_flow[i], t0=self.__link_free_time[i], capacity=self.__link_capacity[i])
        return val

    def __so_object_function(self, mixed_flow):
        ''' Objective function in the linear search step 
            of the optimization model of system optimum convex optimization
             problem, the only variable is mixed_flow in this case.
        '''
        val = 0
        for i in range(self.__network.num_of_links()):
            val += mixed_flow[i] * self.__link_time_performance(link_flow=mixed_flow[i], t0=self.__link_free_time[i], capacity=self.__link_capacity[i])
        return val

    def __golden_section(self, link_flow, auxiliary_link_flow, accuracy= 1e-8):
        ''' The golden-section search is a technique for 
            finding the extremum of a strictly unimodal 
            function by successively narrowing the range
            of values inside which the extremum is known 
            to exist. The accuracy is suggested to be set
            as 1e-8. For more details please refer to:
            https://en.wikipedia.org/wiki/Golden-section_search
        '''
        # Initial params, notice that in our case the
        # optimal theta must be in the interval [0, 1]
        LB = 0
        UB = 1
        goldenPoint = 0.618
        leftX = LB + (1 - goldenPoint) * (UB - LB)
        rightX = LB + goldenPoint * (UB - LB)
        while True:
            val_left = self.__object_function((1 - leftX) * link_flow + leftX * auxiliary_link_flow)
            val_right = self.__object_function((1 - rightX) * link_flow + rightX * auxiliary_link_flow)
            if val_left <= val_right:
                UB = rightX
            else:
                LB = leftX
            if abs(LB - UB) < accuracy:
                opt_theta = (rightX + leftX) / 2.0
                return opt_theta
            else:
                if val_left <= val_right:
                    rightX = leftX
                    leftX = LB + (1 - goldenPoint) * (UB - LB)
                else:
                    leftX = rightX
                    rightX = LB + goldenPoint*(UB - LB)

    def __so_all_or_nothing(self, link_flow):
        ''' Perform the all-or-nothing assignment of
            Frank-Wolfe algorithm with system optimum objective, equivalent to adding toll price to every road,

            Input: link flow -> Output: new link flow
            The input is an array.
        '''
        # LINK FLOW -> LINK TIME
        link_time = self.__so_grad(link_flow)
        # LINK TIME -> PATH TIME
        path_time = self.__link_time_to_path_time(link_time)

        # PATH TIME -> PATH FLOW
        # Find the minimal traveling time within group 
        # (splited by origin - destination pairs) and
        # assign all the flow to that path
        path_flow = np.zeros(self.__network.num_of_paths())
        for OD_pair_index in range(self.__network.num_of_OD_pairs()):
            indice_grouped = []
            for path_index in range(self.__network.num_of_paths()):
                if self.__network.paths_category()[path_index] == OD_pair_index:
                    indice_grouped.append(path_index)
            sub_path_time = [path_time[ind] for ind in indice_grouped]
            min_in_group = min(sub_path_time)
            ind_min = sub_path_time.index(min_in_group)
            target_path_ind = indice_grouped[ind_min]
            path_flow[target_path_ind] = self.__demand[OD_pair_index]
            #path_flow[target_path_ind] = sum(self.__demand) * self.d[OD_pair_index]
        if self.__detail:
            print("Link time:\n%s" % link_time)
            print("Path flow:\n%s" % path_flow)
            print("Path time:\n%s" % path_time)
        
        # PATH FLOW -> LINK FLOW
        new_link_flow = self.__path_flow_to_link_flow(path_flow)

        return new_link_flow

    def __so_golden_section(self, link_flow, auxiliary_link_flow, accuracy= 1e-8):
        LB = 0
        UB = 1
        goldenPoint = 0.618
        leftX = LB + (1 - goldenPoint) * (UB - LB)
        rightX = LB + goldenPoint * (UB - LB)
        while True:
            val_left = self.__so_object_function((1 - leftX) * link_flow + leftX * auxiliary_link_flow)
            val_right = self.__so_object_function((1 - rightX) * link_flow + rightX * auxiliary_link_flow)
            if val_left <= val_right:
                UB = rightX
            else:
                LB = leftX
            if abs(LB - UB) < accuracy:
                opt_theta = (rightX + leftX) / 2.0
                return opt_theta
            else:
                if val_left <= val_right:
                    rightX = leftX
                    leftX = LB + (1 - goldenPoint) * (UB - LB)
                else:
                    leftX = rightX
                    rightX = LB + goldenPoint*(UB - LB)

    def __is_convergent(self, flow1, flow2):
        ''' Regard those two link flows lists as the point
            in Euclidean space R^n, then judge the convergence
            under given accuracy criterion.
            Here the formula
                ERR = || x_{k+1} - x_{k} || / || x_{k} ||
            is recommended.
        '''
        err = np.linalg.norm(flow1 - flow2) / np.linalg.norm(flow1)
        if self.__detail:
            print("ERR: %.8f" % err)
        if err < self._conv_accuracy:
            return True
        else:
            return False

    def unif_assign(self):
        #print(self.__network.num_of_paths())
        assigned_flow = np.zeros(self.__network.num_of_paths()) 
        for OD_pair_index in range(self.__network.num_of_OD_pairs()):
            indice_grouped = []
            for path_index in range(self.__network.num_of_paths()):
                if self.__network.paths_category()[path_index] == OD_pair_index:
                    indice_grouped.append(path_index)
            
            assigned_flow[indice_grouped] = self.__demand[OD_pair_index]/len(indice_grouped)
            #print(len(indice_grouped))
            #print(assigned_flow)
            link_flow = self.__path_flow_to_link_flow(assigned_flow)
        return assigned_flow, link_flow

    
    def disp_detail(self):
        ''' Display all the numerical details of each variable
            during the iteritions.
        '''
        self.__detail = True

    def set_disp_precision(self, precision):
        ''' Set the precision of display, which influences only
            the digit of numerical component in arrays.
        '''
        np.set_printoptions(precision= precision)


    def __dash_line(self):
        ''' Return a string which consistently 
            contains '-' with fixed length
        '''
        return "-" * 80
    
    def __so_grad(self, link_flow):
        n_links = self.__network.num_of_links()
        grad = np.zeros(n_links)
        for i in range(len(link_flow)):
            val1 = self.__link_time_performance(link_flow[i], self.__link_free_time[i], self.__link_capacity[i]) 
            val2 = link_flow[i] * self.__link_free_time[i] * (self._alpha * ((link_flow[i]/self.__link_capacity[i])**(self._beta - 1))) / self.__link_capacity[i]
            
            grad[i] = val1 + val2 
        return grad
    
    def so_solve(self):
        ''' Solve the traffic flow assignment model (system optimum)
            by Frank-Wolfe algorithm, all the necessary data must be 
            properly input into the model in advance. 

            (Implicitly) Return
            ------
            self.__solved = True
        '''
        if self.__detail:
            print(self.__dash_line())
            print("FLOW LATENCY ATTACKED TRAFFIC FLOW ASSIGN MODEL (System Optimum) \nFRANK-WOLFE ALGORITHM - DETAIL OF ITERATIONS")
            print(self.__dash_line())
            print(self.__dash_line())
            print("Initialization")
            print(self.__dash_line())
        
        # Step 0: based on the x0, generate the x1
        empty_flow = np.zeros(self.__network.num_of_links())
        link_flow = self.__so_all_or_nothing(empty_flow)

        counter = 0
        while True:
            
            if self.__detail:
                print(self.__dash_line())
                print("Iteration %s" % counter)
                print(self.__dash_line())
                print("Current link flow:\n%s" % link_flow)

            # Step 1 & Step 2: Use the link flow matrix -x to generate the time, then generate the auxiliary link flow matrix -y
            
            auxiliary_link_flow = self.__so_all_or_nothing(link_flow)

            # Step 3: Linear Search
            opt_theta = self.__so_golden_section(link_flow, auxiliary_link_flow)
            
            # Step 4: Using optimal theta to update the link flow matrix
            new_link_flow = (1 - opt_theta) * link_flow + opt_theta * auxiliary_link_flow

            # Print the detail if necessary
            if self.__detail:
                print("Optimal theta: %.8f" % opt_theta)
                print("Auxiliary link flow:\n%s" % auxiliary_link_flow)

            # Step 5: Check the Convergence, if FALSE, then return to Step 1
            if self.__is_convergent(link_flow, new_link_flow):
                if self.__detail:
                    print(self.__dash_line())
                self.__so_solved = True
                self.so_final_link_flow = new_link_flow
                self.so_final_link_time = self.__link_flow_to_link_time_actual(self.so_final_link_flow)
                self.__so_iterations_times = counter
                break
            else:
                link_flow = new_link_flow
                counter += 1

    

    def __str__(self):
        string = ""
        string += self.__dash_line()
        string += "\n"
        string += "TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - PARAMS OF MODEL"
        string += "\n"
        string += self.__dash_line()
        string += "\n"
        string += self.__dash_line()
        string += "\n"
        string += "LINK Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.__network.num_of_links()):
            string += "%2d : link= %s, free time= %.2f, capacity= %s \n" % (i, self.__network.edges()[i], self.__link_free_time[i], self.__link_capacity[i])
        string += self.__dash_line()
        string += "\n"
        string += "OD Pairs Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.__network.num_of_OD_pairs()):
            string += "%2d : OD pair= %s, demand= %d \n" % (i, self.__network.OD_pairs()[i], self.__demand[i])
        string += self.__dash_line()
        string += "\n"
        string += "Path Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.__network.num_of_paths()):
            string += "%2d : Conjugated OD pair= %s, Path= %s \n" % (i, self.__network.paths_category()[i], self.__network.paths()[i])
        string += self.__dash_line()
        string += "\n"
        string += f"Link-Path Incidence Matrix (Rank: {self.__network.LP_matrix_rank()}):\n"
        string += self.__dash_line()
        string += "\n"
        string += str(self.__network.LP_matrix())
        return string