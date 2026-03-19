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
ITERATIONS = 50           # Total training loops
SELF_PLAY_EPISODES = 20   # Games per iteration to gather data
EPOCHS = 10               # Training epochs per iteration
BATCH_SIZE = 64           # Mini-batch size (AlphaZero uses 2048)
EVAL_GAMES = 20           # Games to play Arena (New vs Old) (AlphaZero uses 400)
WIN_THRESHOLD = 0.55      # New model must win 55% to be accepted
MODEL_PATH = "squadro_best.pth"

class AlphaZeroTrainer:
    def __init__(self):
        self.writer = SummaryWriter("runs/alphazero_pipeline")
        
        # Current Network (The one being trained)
        self.nnet = SquadroNet().to(DEVICE)
        
        # Best Network (The champion to beat)
        self.pnet = SquadroNet().to(DEVICE)
        
        if os.path.exists(MODEL_PATH):
            print("Loading existing best model...")
            self.nnet.load_state_dict(torch.load(MODEL_PATH))
            self.pnet.load_state_dict(torch.load(MODEL_PATH))
        
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)
        self.mcts = NeuralMCTS(self.nnet, DEVICE)
        
        # Replay Buffer: Stores (state, policy, value)
        # "Stores the last 500k games" - we use 10,000 for laptop scale
        self.train_examples_history = deque(maxlen=10000) 

    def execute_episode(self):
        """
        Stage 1: Self-Play
        Plays one game of AI vs AI and returns the training examples.
        """
        self.nnet.eval()
        train_examples = []
        board = SquadroBoard()
        step_count = 0
        
        while True:
            step_count += 1
            # In early game, explore more (temp=1). Later, be competitive (temp=0).
            temp = 1.0 if step_count < 15 else 0.0
            
            # Get Policy Vector from MCTS
            pi = self.mcts.get_action_prob(board, simulations=50, temp=temp)
            
            # Store (State, Policy, CurrentPlayer)
            # We don't know the winner yet, so we store the current player ID to fix later
            sym = board.get_state_vector()
            train_examples.append([sym, pi, board.turn])
            
            # Execute Move
            # Sample action based on pi
            action = np.random.choice(len(pi), p=pi)
            board.do_move(action)
            
            if board.winner is not None:
                # Game Over. Backfill the rewards.
                return_data = []
                for hist_state, hist_pi, hist_player in train_examples:
                    # If the winner is the player who moved in that state: +1
                    # Else: -1
                    reward = 1 if hist_player == board.winner else -1
                    return_data.append((hist_state, hist_pi, reward))
                return return_data

    def learn(self):
        """
        Stage 2: Retrain Network
        Samples mini-batches and updates weights over multiple epochs.
        """
        self.nnet.train()
        if len(self.train_examples_history) < BATCH_SIZE: 
            return 0
            
        avg_loss = 0
        # Calculate how many batches make up a full sweep of your data
        batches_per_epoch = len(self.train_examples_history) // BATCH_SIZE
        
        for _ in range(EPOCHS):
            for _ in range(batches_per_epoch):
                # Sample batch
                batch = random.sample(self.train_examples_history, BATCH_SIZE)
                
                # Unpack
                boards, pis, vs = list(zip(*batch))
                boards = torch.FloatTensor(np.array(boards)).to(DEVICE)
                target_pis = torch.FloatTensor(np.array(pis)).to(DEVICE)
                target_vs = torch.FloatTensor(np.array(vs)).to(DEVICE)
                
                # Predict
                out_pi, out_v = self.nnet(boards)
                
                # Loss Calculation
                # Value: Mean Squared Error
                loss_v = F.mse_loss(out_v.view(-1), target_vs)
                # Policy: Cross Entropy (Negative Log Likelihood)
                loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                
                total_loss = loss_v + loss_pi
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                avg_loss += total_loss.item()
                
        # Return the average loss across all processed batches
        return avg_loss / (EPOCHS * batches_per_epoch)

    def evaluate(self):
        """
        Stage 3: Evaluate Network (Arena)
        Play New Model (nnet) vs Old Model (pnet).
        Returns fraction of games won by New Model.
        """
        self.nnet.eval()
        nnet_mcts = NeuralMCTS(self.nnet, DEVICE)
        pnet_mcts = NeuralMCTS(self.pnet, DEVICE)
        
        wins = 0
        draws = 0
        
        print(f"Evaluating: Playing {EVAL_GAMES} games...")
        for i in range(EVAL_GAMES):
            board = SquadroBoard()
            # Alternate who starts
            p1_is_nnet = (i % 2 == 0) 
            
            while board.winner is None:
                if (board.turn == 1 and p1_is_nnet) or (board.turn == 2 and not p1_is_nnet):
                    # New Net Moves
                    move = nnet_mcts.search(board, simulations=40) # Fast check
                else:
                    # Old Net Moves
                    move = pnet_mcts.search(board, simulations=40)
                board.do_move(move)
            
            # Check who won
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
            
            # 1. SELF-PLAY
            print("Step 1: Self-Play (Gathering Data)...")
            new_examples = []
            for _ in range(SELF_PLAY_EPISODES):
                new_examples += self.execute_episode()
            
            # Add to sliding window buffer
            self.train_examples_history.extend(new_examples)
            print(f"  Buffer size: {len(self.train_examples_history)} examples")

            # 2. RETRAIN
            print("Step 2: Retraining Neural Network...")
            loss = self.learn()
            self.writer.add_scalar("Loss", loss, i)
            print(f"  Loss: {loss:.4f}")

            # 3. EVALUATE
            print("Step 3: Evaluation (Arena)...")
            # Save current as temp to compare
            win_rate = self.evaluate()
            self.writer.add_scalar("WinRate_vs_Old", win_rate, i)
            print(f"  Win Rate vs Old Model: {win_rate*100:.1f}%")

            # Acceptance Logic
            if win_rate >= WIN_THRESHOLD:
                print("  NEW MODEL ACCEPTED! Saving...")
                torch.save(self.nnet.state_dict(), MODEL_PATH)
                self.pnet.load_state_dict(self.nnet.state_dict()) # Update opponent
            else:
                print("  REJECTED. Reverting to previous best.")
                self.nnet.load_state_dict(self.pnet.state_dict())

if __name__ == "__main__":
    trainer = AlphaZeroTrainer()
    trainer.run_pipeline()