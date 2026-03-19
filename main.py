import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
from game import SquadroBoard
from model import SquadroNet
from mcts import NeuralMCTS
from interface import print_board

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "squadro_net.pth"
# for the more advanced neural net:
MODEL_PATH = "squadro_best.pth"

def load_ai():
    """Helper to load the model safely."""
    model = SquadroNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("AI Brain Loaded.")
        except:
            print("Error loading model. Starting with random weights.")
    else:
        print("No trained model found. AI will play randomly (untrained).")
        print("Run 'train.py' to make it smart!")
    
    model.eval()
    return NeuralMCTS(model, DEVICE)

def human_vs_human():
    board = SquadroBoard()
    
    while board.winner is None:
        print_board(board)
        
        print(f"\nPlayer {board.turn}'s turn.")
        if board.turn == 1:
            print("Color: Yellow/Horizontal (►)")
        else:
            print("Color: Red/Vertical (▼)")

        moves = board.get_legal_moves()
        print(f"Valid pieces to move: {[m+1 for m in moves]}")
        
        while True:
            try:
                choice = int(input("Select piece to move: "))
                if (choice - 1) in moves:
                    board.do_move(choice - 1)
                    break
                print("Invalid move.")
            except ValueError:
                print("Please enter a number.")

    print_board(board)
    print(f"\nGame Over! Player {board.winner} Wins!")

def human_vs_ai():
    mcts = load_ai()
    board = SquadroBoard()
    
    while board.winner is None:
        print_board(board)

        if board.turn == 1:
            # Human Turn
            moves = board.get_legal_moves()
            print(f"\nYour moves: {[m+1 for m in moves]}")
            while True:
                try:
                    c = int(input("Select piece to move: "))
                    if (c-1) in moves:
                        board.do_move(c-1)
                        break
                    print("Invalid move.")
                except ValueError: 
                    print("Please enter a number.")
        else:
            # AI Turn
            print("\nAI (P2) Thinking...")
            move = mcts.search(board, simulations=800)
            
            print(f"AI chooses piece {move+1}")
            time.sleep(1.0) 
            board.do_move(move)
            
    print_board(board)
    print(f"\nGame Over! Player {board.winner} Wins!")

def ai_vs_ai():
    mcts = load_ai()
    board = SquadroBoard()
    
    print("Starting AI vs AI match...")
    time.sleep(1)
    
    while board.winner is None:
        print_board(board)
        
        player_name = "P1 (Horizontal)" if board.turn == 1 else "P2 (Vertical)"
        print(f"\n{player_name} AI Thinking...")
        
        # Both sides use the same MCTS brain to find the best move for the current player
        move = mcts.search(board, simulations=800)
        
        print(f"{player_name} chooses piece {move+1}")
        
        # Add a delay so the human can watch
        time.sleep(1.5) 
        board.do_move(move)
            
    print_board(board)
    print(f"\nGame Over! Player {board.winner} Wins!")

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*30)
    print("      S Q U A D R O")
    print("="*30)
    print("1. Play vs AI (Neural Net)")
    print("2. Play vs Human (Local)")
    print("3. Watch AI vs AI")
    print("-" * 30)
    
    choice = input("Select Mode: ")
    
    if choice == '1':
        human_vs_ai()
    elif choice == '2':
        human_vs_human()
    elif choice == '3':
        ai_vs_ai()
    else:
        print("Invalid selection.")