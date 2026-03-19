import numpy as np

class SquadroBoard:
    def __init__(self):
        self.P1_SPEEDS = [1, 3, 2, 3, 1]
        self.P2_SPEEDS = [3, 1, 2, 1, 3]
        # 0=Start, 1-5=Board, 6=Turnaround
        self.p1_pos = [0] * 5
        self.p2_pos = [0] * 5
        # 1=Out, -1=Back
        self.p1_dir = [1] * 5
        self.p2_dir = [1] * 5
        # 0=Not Fin, 1=Finished
        self.p1_fin = [0] * 5
        self.p2_fin = [0] * 5
        self.turn = 1 # 1 or 2
        self.winner = None

    def clone(self):
        n = SquadroBoard()
        n.p1_pos, n.p2_pos = self.p1_pos[:], self.p2_pos[:]
        n.p1_dir, n.p2_dir = self.p1_dir[:], self.p2_dir[:]
        n.p1_fin, n.p2_fin = self.p1_fin[:], self.p2_fin[:]
        n.turn, n.winner = self.turn, self.winner
        return n
        
    def get_state_vector(self):
        """
        Converts board to a (5, 5, 5) spatial tensor for the ResNet.
        Shape: [Channels, Height, Width]
        """
        # We create 5 planes of 5x5
        # Plane 0: P1 Positions (Row-wise fill)
        # Plane 1: P2 Positions (Col-wise fill)
        # Plane 2: P1 Directions
        # Plane 3: P2 Directions
        # Plane 4: Player Turn (All 1 if P1, All -1 if P2)
        state = np.zeros((5, 5, 5), dtype=np.float32)
        
        for r in range(5):
            if 1 <= self.p1_pos[r] <= 5:
                state[0, r, self.p1_pos[r] - 1] = 1.0
            
            if self.p1_fin[r] == 1:
                state[2, r, :] = 0.0  # Unique marker for finished pieces
            else:
                state[2, r, :] = self.p1_dir[r]

        for c in range(5):
            if 1 <= self.p2_pos[c] <= 5:
                state[1, self.p2_pos[c] - 1, c] = 1.0
                
            if self.p2_fin[c] == 1:
                state[3, :, c] = 0.0  # Unique marker for finished pieces
            else:
                state[3, :, c] = self.p2_dir[c]

        # Turn Plane
        turn_val = 1.0 if self.turn == 1 else -1.0
        state[4, :, :] = turn_val

        return state

    def get_legal_moves(self):
        if self.winner: return []
        fin_list = self.p1_fin if self.turn == 1 else self.p2_fin
        return [i for i, f in enumerate(fin_list) if f == 0]

    def do_move(self, piece_idx):
        if self.turn == 1:
            my_pos, my_dir, my_fin = self.p1_pos, self.p1_dir, self.p1_fin
            opp_pos, opp_dir = self.p2_pos, self.p2_dir
            base = self.P1_SPEEDS[piece_idx]
            my_track = piece_idx
        else:
            my_pos, my_dir, my_fin = self.p2_pos, self.p2_dir, self.p2_fin
            opp_pos, opp_dir = self.p1_pos, self.p1_dir
            base = self.P2_SPEEDS[piece_idx]
            my_track = piece_idx

        speed = base if my_dir[piece_idx] == 1 else (4 - base)
        curr = my_pos[piece_idx]
        d = my_dir[piece_idx]

        for _ in range(speed):
            nxt = curr + d
            
            # Bound the next step to the board limits
            if nxt < 0: nxt = 0
            if nxt > 6: nxt = 6
            
            # Check if the next step is inside the grid (1-5) and occupied by an opponent
            b_idx = nxt - 1
            if 0 <= b_idx <= 4 and opp_pos[b_idx] == (my_track + 1):
                # CHAIN JUMP LOGIC
                # Keep jumping as long as the landing spot is occupied by another opponent piece
                land = nxt
                while 0 <= (land - 1) <= 4 and opp_pos[land - 1] == (my_track + 1):
                    opp_idx = land - 1
                    
                    # Reset the jumped opponent piece to its respective start/turnaround
                    opp_pos[opp_idx] = 0 if opp_dir[opp_idx] == 1 else 6
                    
                    # Advance the landing position by our direction
                    land += d
                
                # Ensure the final landing spot doesn't go out of bounds
                if land > 6: land = 6
                if land < 0: land = 0
                
                my_pos[piece_idx] = land
                
                # In Squadro, a jump immediately ends the movement for that piece
                break 
                
            else:
                # Normal move (no collision)
                curr = nxt
                my_pos[piece_idx] = curr
            
            # Stop if we reached a dock (0 or 6)
            if my_pos[piece_idx] in [0, 6]: 
                break

        # Turnaround / Finish Logic
        if my_pos[piece_idx] == 6 and my_dir[piece_idx] == 1:
            my_dir[piece_idx] = -1
        elif my_pos[piece_idx] == 0 and my_dir[piece_idx] == -1:
            my_fin[piece_idx] = 1

        # Win Check
        if sum(self.p1_fin) >= 4: self.winner = 1
        elif sum(self.p2_fin) >= 4: self.winner = 2
        
        # Switch turns
        self.turn = 3 - self.turn