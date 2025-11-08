
import numpy as np
import collections
import copy
import math
####Offloading class
class offloading:
    def __init__(self, delay, K, M, T, mu, reliability, deadline):
        self.T = T
        self.M = M
        self.delay = delay
        self.reliability = reliability
        self.fix = False
        self.mu = mu
        self.K = K       
        self.deadline = deadline
        self.t = 0
        self.pulls = np.zeros((self.M, self.K),  dtype=np.int32)
        self.successes = np.zeros((self.M, self.K))
        self.mu_hat = np.zeros((self.M, self.K))
        self.phi_hat = np.zeros((self.M, self.M, self.K))       
        self.F_m = np.zeros((self.M, self.M))       
        self.count_failure = np.zeros(self.M)        
        self.phi_hat_success = np.zeros((self.M, self.M, self.K))
        self.count_sample = np.zeros((self.M, self.M, self.K),dtype=np.int32)
        self.pulls_with_g = np.zeros((self.M, self.M, self.K), dtype=np.int32)
        self.offload_delay = np.zeros((self.T, self.M, self.K))
        self.count_users = np.zeros((self.T, self.M, self.K))
        self.fixed_arm = np.ones(self.M,dtype = int)
        self.arm_fixed_flag = False           
        self.kl_success = np.zeros((self.M, self.M, self.K))       
        self.bound = np.zeros((self.M, self.M, self.K))  
        self.max_num = np.zeros((self.M, self.K))       
        self.G_figure = np.zeros((self.T, self.M, self.K))       
        self.reliability_figure = np.zeros((self.T, self.M))
        self.reliability_theory = np.zeros((self.T, self.M))
        self.lower = 0
        self.count_feasibility = np.zeros(self.T)
        self.bound = np.zeros((self.M, self.M, self.K))            
        self.kl_success = np.zeros((self.M, self.M, self.K))
        
    def KL(self, p, q):
        if p == 0:
            return math.log(1 / (1 - q))
        if p == 1:
            if q == 0:
                return 1e6
            return math.log(1 / q)
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    
    def kl_prime(self, p, q):
        return -p / q + (1 - p) / (1 - q)
    
    def getUCBKL(self, c):       
        for player in range(self.M):
            for arm in range(self.K):
                for G in range(self.M):           
                    if self.pulls[player][arm] == 0:
                        self.kl_success[G, player, arm] = 1
                    else:
                        self.bound[G, player, arm] = (math.log(self.t) + c * math.log(math.log(self.t))) / (self.pulls[player][arm])                 
                        if self.phi_hat_success[G, player, arm] == 1:
                            self.kl_success[G, player, arm] = 1
                        else:
                            q = (1+self.phi_hat_success[G, player, arm])/2
                            while self.KL(self.phi_hat_success[G, player, arm], q) < self.bound[G, player, arm]: 
                                q = (q+1)/2
                            compute_kl_s = self.KL(self.phi_hat_success[G, player, arm], q)
                            while np.abs(compute_kl_s -  self.bound[G, player, arm]) > 1e-7:
                                compute_kl_s = self.KL(self.phi_hat_success[G, player, arm], q)
                                compute_kl_prime = self.kl_prime(self.phi_hat_success[G, player, arm], q)
                                q -= (compute_kl_s-self.bound[G, player, arm])/compute_kl_prime
                            self.kl_success[G, player, arm] = q
        return self.kl_success

    def draw(self, arms, opt_reward):
            counts = collections.Counter(arms)
            rewards = np.zeros((self.M, self.K))
            delay_t = np.zeros((self.M, self.K))
            regret_t = opt_reward
            count = np.ones(self.M,dtype = int)*1e6
            for player in range(self.M):
                count[player] = counts[arms[player]]
                delay_t[player][arms[player]] = counts[arms[player]]*np.random.exponential(scale = self.delay[player][arms[player]])
                rewards[player][arms[player]] = self.mu[player][arms[player]]
                regret_t -= rewards[player][arms[player]]
            return count, rewards, delay_t, regret_t

    def choose_arm(self, c = 3):
        difference = np.zeros(self.M)        
        arm_final = np.zeros((self.M, self.M),dtype = int)
        self.getUCBKL(c)
        total_reward = 0
        for g in range(self.M):
            reward = 0
            arm_fixed_final = np.ones(self.M,dtype = int)
            reward_co = np.zeros((self.M, self.K))
            for player in range(self.M):
                for arms in range(self.K):
                    if self.kl_success[g,player,arms] >= self.reliability[player]:
                        reward_co[player,arms] = 1
                    else:
                        reward_co[player,arms] = -1e6
            difference = (self.mu[:,1] - self.mu[:,0] ) * reward_co[:,0]
            iteration = g+1
            while iteration >0:
                player_check = np.argmax(difference)
                reward += difference[player_check]
                difference[player_check] = -1e10
                arm_fixed_final[player_check] = 0
                iteration -= 1
            arm_final[g] = arm_fixed_final
            if reward > total_reward:
                total_reward = reward
                self.fixed_arm = copy.deepcopy(arm_final[g])
        return self.fixed_arm

            
    def optimal_allocation(self):
        final_mu = 0
        arm_final = np.zeros((self.M, self.M),dtype = int)
        total_reward = 0
        for g in range(self.M):
            difference = np.zeros(self.M)
            reward = 0
            arm_fixed_final = np.ones(self.M,dtype = int)
            reward_co = np.zeros((self.M, self.K))
            for player in range(self.M):
                for arms in range(self.K):
                    if 1 - np.exp(-self.deadline/(self.delay[player][arms]*(g+1))) >= self.reliability[player]:
                        reward_co[player,arms] = 1
                    else:
                        reward_co[player,arms] = -1e6
            difference = (self.mu[:,1]-self.mu[:,0] ) * reward_co[:,0]
            iteration = g+1
            while iteration >0:
                player_check = np.argmax(difference)
                reward += difference[player_check]
                difference[player_check] = -1e10
                arm_fixed_final[player_check] = 0
                iteration -= 1
            arm_final[g] = arm_fixed_final
            if reward > total_reward:
                total_reward = reward
                final_arm = copy.deepcopy(arm_final[g])
        for users in range(self.M):
            final_mu += (self.mu[users][final_arm[users]])
        print(final_arm)
        return final_mu
        
        
    def compute_prob(self, arm_t):
        for player in range(self.M):
            for G in range(self.M):
                if self.offload_delay[self.t,player, arm_t[player]]* (G+1) > self.deadline:
                    self.count_sample[G,player,arm_t[player]] += 1
                self.phi_hat[G,player,arm_t[player]] = self.count_sample[G,player,arm_t[player]] /(self.pulls[player, arm_t[player]])
                self.phi_hat_success[G,player,arm_t[player]] = 1 - self.phi_hat[G,player,arm_t[player]]
                if G+1 == self.count_users[self.t,player, arm_t[player]]:
                    self.reliability_figure[self.t, player] = self.phi_hat_success[G,player,arm_t[player]]
                    self.reliability_theory[self.t, player] = 1 - np.exp(-self.deadline/(self.delay[player][arm_t[player]]*(G+1)))
        return self.phi_hat, self.phi_hat_success
             
                
    def run(self,T):
        prob = np.zeros(self.M)
        regret_total = []
        opt_reward = self.optimal_allocation()
        while self.t < T: 
            boolean = [False]*self.M
            if self.t < self.K:
                arms_t = np.ones(self.M,dtype = int)*self.t
                counts_t, rewards_t, delay_t, regret_t = self.draw(arms_t, opt_reward)
                regret_total.append(regret_t)
                self.pulls[np.arange(self.M), arms_t] += 1
                self.successes[np.arange(self.M), arms_t] += rewards_t[np.arange(self.M), arms_t]
                self.offload_delay[self.t][np.arange(self.M), arms_t] = delay_t[np.arange(self.M), arms_t] /counts_t[np.arange(self.M)]
                self.count_users[self.t][np.arange(self.M), arms_t] = counts_t[np.arange(self.M)]
                self.mu_hat = self.successes / (1e-7+self.pulls)
                self.mu_hat[self.pulls == 0] = 0 
                self.compute_prob(arms_t)
                for player in range(self.M):
                    boolean[player] = [1 - np.exp(-self.deadline/(self.delay[player][arms_t[player]]*counts_t[player])) >= self.reliability[player]]
                    prob[player] =  1 - np.exp(-self.deadline/(self.delay[player][arms_t[player]]*counts_t[player]))
                if [False] in boolean:
                    self.count_feasibility[self.t] += 1
            else:
                arms_t = self.choose_arm()
                counts_t, rewards_t, delay_t, regret_t = self.draw(arms_t,opt_reward)
                regret_total.append(regret_t)
                self.pulls[np.arange(self.M), arms_t] += 1
                self.successes[np.arange(self.M), arms_t] += rewards_t[np.arange(self.M), arms_t]
                self.offload_delay[self.t][np.arange(self.M), arms_t] = delay_t[np.arange(self.M), arms_t] /counts_t[np.arange(self.M)]
                self.count_users[self.t][np.arange(self.M), arms_t] = counts_t[np.arange(self.M)]
                self.mu_hat = self.successes / (1e-7+self.pulls)
                self.compute_prob(arms_t)
                for player in range(self.M):
                    boolean[player] = [1 - np.exp(-self.deadline/(self.delay[player][arms_t[player]]*counts_t[player])) >= self.reliability[player]]
                    prob[player] =  1 - np.exp(-self.deadline/(self.delay[player][arms_t[player]]*counts_t[player]))
                if [False] in boolean:
                    self.count_feasibility[self.t] += 1
            self.t +=1
        for player in range(self.M):
            boolean[player] = [1 - np.exp(-self.deadline/(self.delay[player][arms_t[player]]*counts_t[player])) >= self.reliability[player]]
            prob[player] =  1 - np.exp(-self.deadline/(self.delay[player][arms_t[player]]*counts_t[player]))
        return boolean, regret_total, prob, self.reliability_figure, self.G_figure, self.count_feasibility, self.reliability_theory
     
    
            
