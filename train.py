import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from game import SquadroBoard
from model import SquadroNet
from mcts import NeuralMCTS

# --- HYPERPARAMETERS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITERATIONS = 50           
SELF_PLAY_EPISODES = 100  
EPOCHS = 5                
BATCH_SIZE = 256          
EVAL_GAMES = 40           
WIN_THRESHOLD = 0.55      

# --- FILE PATHS ---
MODEL_PATH = "squadro_best.pth"
OPTIMIZER_PATH = "squadro_optimizer.pth" 
BUFFER_PATH = "squadro_buffer.pkl"
CHECKPOINT_DIR = "checkpoints/"

# Create checkpoints directory if it doesn't exist
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

class AlphaZeroTrainer:
    def __init__(self):
        self.writer = SummaryWriter("runs/alphazero_pipeline")
        
        self.nnet = SquadroNet().to(DEVICE)
        self.pnet = SquadroNet().to(DEVICE)
        
        if os.path.exists(MODEL_PATH):
            print("Loading existing best model...")
            self.nnet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.pnet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        else:
            print("No existing model found. Initializing new synchronized models...")
            self.pnet.load_state_dict(self.nnet.state_dict())
        
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)
        
        # Load the optimizer state to retain Adam's momentum
        if os.path.exists(OPTIMIZER_PATH):
            print("Loading existing optimizer state...")
            self.optimizer.load_state_dict(torch.load(OPTIMIZER_PATH, map_location=DEVICE))
            self.best_optimizer_state = self.optimizer.state_dict()

        self.mcts = NeuralMCTS(self.nnet, DEVICE)
        self.train_examples_history = deque(maxlen=100000) 

        # Load existing buffer if resuming a paused run
        if os.path.exists(BUFFER_PATH):
            print(f"Loading existing training buffer from {BUFFER_PATH}...")
            with open(BUFFER_PATH, "rb") as f:
                self.train_examples_history = pickle.load(f)
            print(f"Loaded {len(self.train_examples_history)} examples.")

    def execute_parallel_episodes(self, target_episodes=100, concurrent_games=64):
        self.nnet.eval()
        train_examples = []
        
        active_boards = [SquadroBoard() for _ in range(concurrent_games)]
        active_roots = [None] * concurrent_games 
        histories = [[] for _ in range(concurrent_games)]
        step_counts = [0] * concurrent_games
        episodes_completed = 0
        
        active_slots = list(range(concurrent_games))
        print(f"  Starting {concurrent_games} concurrent games...")
        
        MAX_MOVES = 300 
        
        while active_slots:
            current_boards = [active_boards[i] for i in active_slots]
            current_temps = [1.0 if step_counts[i] < 15 else 0.0 for i in active_slots]
            current_roots = [active_roots[i] for i in active_slots] 
            
            batch_pi, updated_roots = self.mcts.get_action_prob_batched(
                current_boards, roots=current_roots, simulations=200, temps=current_temps, add_noise=True
            )
            
            slots_to_retire = []
            
            for idx, slot in enumerate(active_slots):
                board = active_boards[slot]
                root = updated_roots[idx]
                pi = batch_pi[idx]
                sym = board.get_state_vector()
                
                # Store history
                histories[slot].append([sym, pi, board.turn])
                
                # Execute the chosen move
                action = np.random.choice(len(pi), p=pi)
                board.do_move(action)
                step_counts[slot] += 1
                
                # Step the root forward to the chosen child to retain the search tree
                if action in root.children:
                    active_roots[slot] = root.children[action]
                    active_roots[slot].parent = None # Detach parent to free memory
                else:
                    active_roots[slot] = None
                
                is_draw = step_counts[slot] >= MAX_MOVES
                
                if board.winner is not None or is_draw:
                    for hist_state, hist_pi, hist_player in histories[slot]:
                        if is_draw:
                            reward = 0.0
                        else:
                            reward = 1.0 if hist_player == board.winner else -1.0
                        train_examples.append((hist_state, hist_pi, reward))
                    
                    episodes_completed += 1
                    if episodes_completed % 10 == 0:
                        print(f"    Finished {episodes_completed}/{target_episodes} games...")
                    
                    if episodes_completed + len(active_slots) <= target_episodes:
                        active_boards[slot] = SquadroBoard()
                        active_roots[slot] = None # Reset root for new game
                        histories[slot] = []
                        step_counts[slot] = 0
                    else:
                        slots_to_retire.append(slot)
                        
            for slot in slots_to_retire:
                active_slots.remove(slot)
                
        return train_examples

    def learn(self):
        self.nnet.train()
        if len(self.train_examples_history) < BATCH_SIZE: 
            return 0
            
        avg_loss = 0
        batches_per_epoch = min(len(self.train_examples_history) // BATCH_SIZE, 100) 
        if batches_per_epoch == 0: batches_per_epoch = 1
        
        # FIX 1: Cast deque to list for O(1) random sampling speed
        sampling_pool = list(self.train_examples_history)
        
        for _ in range(EPOCHS):
            for _ in range(batches_per_epoch):
                batch = random.sample(sampling_pool, BATCH_SIZE)
                
                boards, pis, vs = list(zip(*batch))
                boards = torch.FloatTensor(np.array(boards)).to(DEVICE)
                target_pis = torch.FloatTensor(np.array(pis)).to(DEVICE)
                target_vs = torch.FloatTensor(np.array(vs)).to(DEVICE)
                
                out_pi, out_v = self.nnet(boards)
                
                loss_v = F.mse_loss(out_v.view(-1), target_vs)
                loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                
                total_loss = loss_v + loss_pi
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                avg_loss += total_loss.item()
                
        return avg_loss / (EPOCHS * batches_per_epoch)

    def evaluate(self):
        self.nnet.eval()
        self.pnet.eval()
        nnet_mcts = NeuralMCTS(self.nnet, DEVICE)
        pnet_mcts = NeuralMCTS(self.pnet, DEVICE)
        
        wins, draws = 0, 0
        MAX_MOVES = 300 
        
        print(f"Evaluating: Playing {EVAL_GAMES} games...")
        for i in range(EVAL_GAMES):
            board = SquadroBoard()
            p1_is_nnet = (i % 2 == 0) 
            moves_taken = 0
            
            # FIX 2: Initialize root trackers for the evaluation games
            nnet_root = None
            pnet_root = None
            
            while board.winner is None and moves_taken < MAX_MOVES:
                # Use batched MCTS with a batch size of 1 to utilize the root-passing logic
                if (board.turn == 1 and p1_is_nnet) or (board.turn == 2 and not p1_is_nnet):
                    probs, updated_roots = nnet_mcts.get_action_prob_batched(
                        [board], roots=[nnet_root], simulations=200, temps=[0.0], add_noise=False
                    )
                    move = np.argmax(probs[0])
                    nnet_root = updated_roots[0]
                else:
                    probs, updated_roots = pnet_mcts.get_action_prob_batched(
                        [board], roots=[pnet_root], simulations=200, temps=[0.0], add_noise=False
                    )
                    move = np.argmax(probs[0])
                    pnet_root = updated_roots[0]
                    
                board.do_move(move)
                moves_taken += 1
                
                # Step both MCTS trees forward to the chosen move to retain memory
                if nnet_root and move in nnet_root.children:
                    nnet_root = nnet_root.children[move]
                    nnet_root.parent = None # Free memory
                else:
                    nnet_root = None
                    
                if pnet_root and move in pnet_root.children:
                    pnet_root = pnet_root.children[move]
                    pnet_root.parent = None # Free memory
                else:
                    pnet_root = None
            
            if board.winner is None: draws += 1
            elif p1_is_nnet:
                if board.winner == 1: wins += 1
            else:
                if board.winner == 2: wins += 1
                
        decisive_games = EVAL_GAMES - draws
        win_rate = wins / decisive_games if decisive_games > 0 else 0.0
        
        if draws > 0:
            print(f"  Note: {draws} games ended in a draw (move limit reached).")
            
        return win_rate

    def run_pipeline(self):
        print(f"Starting AlphaZero Pipeline on {DEVICE}")
        
        for i in range(1, ITERATIONS + 1):
            print(f"\n--- Iteration {i}/{ITERATIONS} ---")
            
            print("Step 1: Self-Play (Gathering Data)...")
            new_examples = self.execute_parallel_episodes(target_episodes=SELF_PLAY_EPISODES, concurrent_games=64)
            self.train_examples_history.extend(new_examples)
            print(f"  Buffer size: {len(self.train_examples_history)} examples")

            with open(BUFFER_PATH, "wb") as f:
                pickle.dump(self.train_examples_history, f)
            print("  Training buffer saved to disk.")

            print("Step 2: Retraining Neural Network...")
            loss = self.learn()
            self.writer.add_scalar("Loss", loss, i)
            print(f"  Loss: {loss:.4f}")

            print("Step 3: Evaluation (Arena)...")
            win_rate = self.evaluate()
            self.writer.add_scalar("WinRate_vs_Old", win_rate, i)
            print(f"  Win Rate vs Old Model: {win_rate*100:.1f}%")

            if win_rate >= WIN_THRESHOLD:
                print("  NEW MODEL ACCEPTED! Saving...")
                torch.save(self.nnet.state_dict(), MODEL_PATH)
                torch.save(self.optimizer.state_dict(), OPTIMIZER_PATH)
                self.pnet.load_state_dict(self.nnet.state_dict()) 
                self.best_optimizer_state = self.optimizer.state_dict()
            else:
                print("  REJECTED. Reverting to previous best.")
                self.nnet.load_state_dict(self.pnet.state_dict())
                
                if hasattr(self, 'best_optimizer_state'):
                    self.optimizer.load_state_dict(self.best_optimizer_state)
                else:
                    self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)

            if i % 5 == 0:
                checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{i}.pth")
                print(f"  Saving historical checkpoint to {checkpoint_file}...")
                torch.save({
                    'iteration': i,
                    'model_state_dict': self.nnet.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_file)

if __name__ == "__main__":
    trainer = AlphaZeroTrainer()
    trainer.run_pipeline()