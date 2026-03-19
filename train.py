import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from game import SquadroBoard
from model import SquadroNet
from mcts import NeuralMCTS

# --- HYPERPARAMETERS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITERATIONS = 10           
SELF_PLAY_EPISODES = 50   
EPOCHS = 10               
BATCH_SIZE = 256           
EVAL_GAMES = 40           
WIN_THRESHOLD = 0.55      
MODEL_PATH = "squadro_best.pth"

class AlphaZeroTrainer:
    def __init__(self):
        self.writer = SummaryWriter("runs/alphazero_pipeline")
        
        self.nnet = SquadroNet().to(DEVICE)
        self.pnet = SquadroNet().to(DEVICE)
        
        if os.path.exists(MODEL_PATH):
            print("Loading existing best model...")
            self.nnet.load_state_dict(torch.load(MODEL_PATH))
            self.pnet.load_state_dict(torch.load(MODEL_PATH))
        
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)
        self.mcts = NeuralMCTS(self.nnet, DEVICE)
        self.train_examples_history = deque(maxlen=10000) 

    def execute_episode(self):
        self.nnet.eval()
        train_examples = []
        board = SquadroBoard()
        step_count = 0
        
        while True:
            step_count += 1
            temp = 1.0 if step_count < 15 else 0.0
            pi = self.mcts.get_action_prob(board, simulations=200, temp=temp, add_noise=True)
            
            sym = board.get_state_vector()
            train_examples.append([sym, pi, board.turn])
            
            action = np.random.choice(len(pi), p=pi)
            board.do_move(action)
            
            if board.winner is not None:
                return_data = []
                for hist_state, hist_pi, hist_player in train_examples:
                    reward = 1 if hist_player == board.winner else -1
                    return_data.append((hist_state, hist_pi, reward))
                return return_data

    def learn(self):
        self.nnet.train()
        if len(self.train_examples_history) < BATCH_SIZE: 
            return 0
            
        avg_loss = 0
        
        # FIX: Cap the maximum number of batches per epoch to prevent runaway loops
        batches_per_epoch = min(len(self.train_examples_history) // BATCH_SIZE, 100) 
        if batches_per_epoch == 0:
            batches_per_epoch = 1
        
        for _ in range(EPOCHS):
            for _ in range(batches_per_epoch):
                batch = random.sample(self.train_examples_history, BATCH_SIZE)
                
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
        
        wins = 0
        
        print(f"Evaluating: Playing {EVAL_GAMES} games...")
        for i in range(EVAL_GAMES):
            board = SquadroBoard()
            p1_is_nnet = (i % 2 == 0) 
            
            while board.winner is None:
                if (board.turn == 1 and p1_is_nnet) or (board.turn == 2 and not p1_is_nnet):
                    move = nnet_mcts.search(board, simulations=40, add_noise=False)
                else:
                    move = pnet_mcts.search(board, simulations=40, add_noise=False)
                board.do_move(move)
            
            if p1_is_nnet:
                if board.winner == 1: wins += 1
            else:
                if board.winner == 2: wins += 1
                
        win_rate = wins / EVAL_GAMES
        return win_rate

    def run_pipeline(self):
        print(f"Starting AlphaZero Pipeline on {DEVICE}")
        
        for i in range(1, ITERATIONS + 1):
            print(f"\n--- Iteration {i}/{ITERATIONS} ---")
            
            print("Step 1: Self-Play (Gathering Data)...")
            new_examples = []
            for _ in range(SELF_PLAY_EPISODES):
                new_examples += self.execute_episode()
            
            self.train_examples_history.extend(new_examples)
            print(f"  Buffer size: {len(self.train_examples_history)} examples")

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
                self.pnet.load_state_dict(self.nnet.state_dict()) 
            else:
                print("  REJECTED. Reverting to previous best.")
                self.nnet.load_state_dict(self.pnet.state_dict())
                # FIX: Reset the optimizer to clear bad momentum from the rejected model's gradient steps
                self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)

if __name__ == "__main__":
    trainer = AlphaZeroTrainer()
    trainer.run_pipeline()