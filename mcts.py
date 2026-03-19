import math
import numpy as np
import torch

class MCTSNode:
    def __init__(self, state, parent=None, move=None, prob=0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.prob = prob 
        self.children = {} 
        self.visits = 0      
        self.value_sum = 0.0 
        self.untried_moves = state.get_legal_moves()

    def is_leaf(self):
        return len(self.children) == 0

    def q_value(self):
        if self.visits == 0: return 0.0
        return self.value_sum / self.visits

class NeuralMCTS:
    def __init__(self, model, device, c_puct=1.0, alpha=0.3, beta=0.25):
        self.model = model
        self.device = device
        self.c_puct = c_puct 
        self.alpha = alpha   
        self.beta = beta     

    def get_action_prob(self, root_state, simulations=400, temp=1.0, add_noise=True):
        """
        Runs MCTS and returns the policy vector (pi).
        :param temp: Temperature (1.0 = explore, ~0 = competitive/greedy)
        """
        root = MCTSNode(root_state.clone())
        
        # 1. Prediction & Noise
        with torch.no_grad():
            state_tensor = torch.FloatTensor(root_state.get_state_vector()).to(self.device).unsqueeze(0)
            pi, v = self.model(state_tensor)
            probs = torch.exp(pi).cpu().numpy()[0]
        
        legal_moves = root.untried_moves
        legal_probs = np.array([probs[m] for m in legal_moves])
        if legal_probs.sum() > 0: legal_probs /= legal_probs.sum()
        else: legal_probs = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Add Dirichlet Noise to root (Exploration)
        if add_noise:
            noise = np.random.dirichlet([self.alpha] * len(legal_moves))
            legal_probs = (1 - self.beta) * legal_probs + self.beta * noise

        for i, move in enumerate(legal_moves):
            root.children[move] = MCTSNode(root.state.clone(), parent=root, move=move, prob=legal_probs[i])
            root.children[move].state.do_move(move)

        # 2. Simulations
        for _ in range(simulations):
            node = root
            while not node.is_leaf() and node.state.winner is None:
                node = self.select_child(node)

            value = 0.0
            if node.state.winner:
                # The game is over. The person whose turn it currently is has already lost.
                value = -1.0
            else:
                state_vec = node.state.get_state_vector()
                state_tensor = torch.FloatTensor(state_vec).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    pi, v = self.model(state_tensor)
                    value = v.item() 
                    probs = torch.exp(pi).cpu().numpy()[0]

                legal = node.state.get_legal_moves()
                if legal:
                    l_probs = np.array([probs[m] for m in legal])
                    if l_probs.sum() > 0: l_probs /= l_probs.sum()
                    else: l_probs = np.ones(len(legal)) / len(legal)
                    for i, m in enumerate(legal):
                        new_state = node.state.clone()
                        new_state.do_move(m)
                        node.children[m] = MCTSNode(new_state, parent=node, move=m, prob=l_probs[i])
            
            # Backprop
            leaf_player = node.state.turn

            while node is not None:
                node.visits += 1
                if node.state.turn == leaf_player:
                    node.value_sum += value
                else:
                    node.value_sum -= value
                node = node.parent

        # 3. Calculate Policy Vector (pi) from visits
        counts = [0] * 5 
        for move, child in root.children.items():
            counts[move] = child.visits

        if temp == 0:
            # FIX: Random tie-breaker for identical visit counts to prevent deterministic looping
            max_count = max(counts)
            best_moves = [i for i, count in enumerate(counts) if count == max_count]
            best = np.random.choice(best_moves)
            
            probs = [0] * 5
            probs[best] = 1.0
            return probs
        
        counts = [x ** (1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def select_child(self, node):
        best_score = -float('inf')
        best_child = None
        N_parent = max(1, node.visits)
        sqrt_N = math.sqrt(N_parent)

        for move, child in node.children.items():
            q_val = child.q_value()
            u_val = self.c_puct * child.prob * (sqrt_N / (1 + child.visits))
            score = -q_val + u_val
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    
    def search(self, root_state, simulations=400, add_noise=False):
        probs = self.get_action_prob(root_state, simulations, temp=0, add_noise=add_noise)
        return np.argmax(probs)