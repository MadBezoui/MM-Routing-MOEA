import random
from typing import List, Optional

import networkx as nx
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling


class PathSampling(Sampling):
    def __init__(self, G: nx.MultiDiGraph, origins: List[str], destinations: List[str]):
        super().__init__()
        self.G = G
        self.origins = origins
        self.destinations = destinations

    def _do(self, problem, n_samples, **kwargs):
        # Generate paths for each variable.
        # X will have shape (n_samples, n_var). Since n_var=1, (n_samples, 1).
        X = np.empty((n_samples, problem.n_var), dtype=object)
        
        for i in range(n_samples):
            # For each profile, there is an origin and destination stored in problem.profile
            # But the problem is evaluated for a single profile. So origin and dest are fixed per problem.
            origin = problem.profile.get("origin_node")
            dest = problem.profile.get("dest_node")
            
            if origin is None or dest is None:
                # Fallback if not assigned
                origin = self.origins[0]
                dest = self.destinations[0]

            # Randomize weights to get different paths
            def weight_func(u, v, d):
                base_time = d.get('travel_time_sec', 60)
                # Add random noise to find diverse paths
                return base_time * random.uniform(0.5, 2.0)

            try:
                # Find shortest path with randomized weights
                path = nx.shortest_path(self.G, source=origin, target=dest, weight=weight_func)
            except nx.NetworkXNoPath:
                # Fallback to direct connection or empty path
                path = [origin, dest]

            X[i, 0] = path
            
        return X


class PathCrossover(Crossover):
    def __init__(self, prob=0.9):
        # Defines 2 parents, 2 offspring
        super().__init__(2, 2, prob=prob)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.empty((2, n_matings, n_var), dtype=object)
        
        for i in range(n_matings):
            p1 = X[0, i, 0]
            p2 = X[1, i, 0]
            
            if random.random() < self.prob and p1 and p2:
                # Find common nodes
                common_nodes = list(set(p1) & set(p2))
                # Remove origin and destination from crossover points to ensure meaningful swap
                if p1[0] in common_nodes: common_nodes.remove(p1[0])
                if p1[-1] in common_nodes: common_nodes.remove(p1[-1])
                
                if common_nodes:
                    crossover_point = random.choice(common_nodes)
                    idx1 = p1.index(crossover_point)
                    idx2 = p2.index(crossover_point)
                    
                    # Swap tails
                    o1 = p1[:idx1] + p2[idx2:]
                    o2 = p2[:idx2] + p1[idx1:]
                else:
                    o1, o2 = p1, p2
            else:
                o1, o2 = p1, p2
                
            Y[0, i, 0] = o1
            Y[1, i, 0] = o2
            
        return Y


class PathMutation(Mutation):
    def __init__(self, G: nx.MultiDiGraph, prob=0.1):
        super().__init__()
        self.G = G
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        Y = np.empty_like(X, dtype=object)
        
        for i in range(X.shape[0]):
            path = X[i, 0]
            if random.random() < self.prob and len(path) > 2:
                # Pick two random nodes in the path to reroute between
                idx1, idx2 = sorted(random.sample(range(len(path)), 2))
                node1, node2 = path[idx1], path[idx2]
                
                def weight_func(u, v, d):
                    return d.get('travel_time_sec', 60) * random.uniform(0.5, 2.0)
                    
                try:
                    # Reroute subsegment
                    subpath = nx.shortest_path(self.G, source=node1, target=node2, weight=weight_func)
                    new_path = path[:idx1] + subpath + path[idx2+1:]
                    Y[i, 0] = new_path
                except nx.NetworkXNoPath:
                    Y[i, 0] = path
            else:
                Y[i, 0] = path
                
        return Y
