from typing import List, Optional
import networkx as nx
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.duplicate import ElementwiseDuplicateElimination

class PathDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.X[0] == b.X[0]

def remove_cycles(path):
    if not path or len(path) < 2:
        return path
    # Remove cycles by taking the first occurrence of a node and skipping to its last occurrence
    valid = []
    i = 0
    while i < len(path):
        node = path[i]
        valid.append(node)
        # Find the last occurrence of this node in the path
        last_idx = len(path) - 1 - path[::-1].index(node)
        i = last_idx + 1
    return valid


class PathSampling(Sampling):
    def __init__(self, G: nx.MultiDiGraph, origins: List[str], destinations: List[str], seed: int = 42):
        super().__init__()
        self.G = G
        self.origins = origins
        self.destinations = destinations
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples, **kwargs):
        X = np.empty((n_samples, problem.n_var), dtype=object)
        
        for i in range(n_samples):
            origin = problem.profile.get("origin_node")
            dest = problem.profile.get("dest_node")
            
            if origin is None or dest is None:
                origin = self.origins[0]
                dest = self.destinations[0]

            def weight_func(u, v, d):
                # In a MultiDiGraph, d is a dictionary of dictionaries keyed by edge key
                min_time = min(attrs.get('travel_time_sec', 60) for attrs in d.values())
                return min_time * self.rng.uniform(0.5, 2.0)

            path = None
            MAX_SAMPLING_ATTEMPTS = 20
            
            for attempt in range(MAX_SAMPLING_ATTEMPTS):
                try:
                    path = nx.shortest_path(self.G, source=origin, target=dest, weight=weight_func)
                    if path and len(path) >= 2:
                        break
                except nx.NetworkXNoPath:
                    pass
                    
            if not path or len(path) < 2:
                # Fallback to absolute shortest path
                try:
                    path = nx.shortest_path(self.G, source=origin, target=dest)
                except nx.NetworkXNoPath:
                    raise RuntimeError(f"No valid path found for OD {origin}->{dest}")

            X[i, 0] = remove_cycles(path)
            
        return X


class PathCrossover(Crossover):
    def __init__(self, prob=0.9, seed: int = 42, **kwargs):
        super().__init__(2, 2, prob=prob, **kwargs)
        self.crossover_prob = float(prob)
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.empty((2, n_matings, n_var), dtype=object)
        
        for i in range(n_matings):
            p1 = X[0, i, 0]
            p2 = X[1, i, 0]
            
            if self.rng.random() < self.crossover_prob and p1 and p2:
                common_nodes = list(set(p1) & set(p2))
                if p1[0] in common_nodes: common_nodes.remove(p1[0])
                if p1[-1] in common_nodes: common_nodes.remove(p1[-1])
                
                if common_nodes:
                    crossover_point = self.rng.choice(common_nodes)
                    idx1 = p1.index(crossover_point)
                    idx2 = p2.index(crossover_point)
                    
                    o1 = p1[:idx1] + p2[idx2:]
                    o2 = p2[:idx2] + p1[idx1:]
                else:
                    o1, o2 = p1, p2
            else:
                o1, o2 = p1, p2
                
            Y[0, i, 0] = remove_cycles(o1)
            Y[1, i, 0] = remove_cycles(o2)
            
        return Y


class PathMutation(Mutation):
    def __init__(self, G, prob=0.1, seed: int = 42, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.G = G
        self.mutation_prob = float(prob)
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        Y = np.empty((len(X), 1), dtype=object)
        for i in range(len(X)):
            path = X[i, 0]
            if self.rng.random() < self.mutation_prob and path and len(path) > 3:
                indices = self.rng.choice(range(len(path)), 2, replace=False)
                idx1, idx2 = sorted(indices)
                node1, node2 = path[idx1], path[idx2]
                
                def weight_func(u, v, d):
                    min_time = min(attrs.get('travel_time_sec', 60) for attrs in d.values())
                    return min_time * self.rng.uniform(0.5, 2.0)
                    
                try:
                    subpath = nx.shortest_path(self.G, source=node1, target=node2, weight=weight_func)
                    new_path = path[:idx1] + subpath + path[idx2+1:]
                    Y[i, 0] = remove_cycles(new_path)
                except nx.NetworkXNoPath:
                    Y[i, 0] = path
            else:
                Y[i, 0] = path
                
        return Y
