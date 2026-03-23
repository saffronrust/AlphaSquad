import logging
import os
import asyncio
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler
from dotenv import load_dotenv
from cachetools import TTLCache

# Import your existing game modules
from game import SquadroBoard
from model import SquadroNet
from mcts import NeuralMCTS

# --- CONFIGURATION ---
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEVICE = torch.device("cpu") # CPU is usually sufficient for bot inference
# MODEL_PATH = "squadro_net.pth"
# for the more advanced neural net:
MODEL_PATH = "squadro_best.pth"

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Global dictionary replaced with TTLCache
# maxsize: Max concurrent games stored in memory
# ttl: Time-to-live in seconds (86400 seconds = 24 hours)
games = TTLCache(maxsize=10000, ttl=86400)

# Load AI once at startup
print("Loading AI Model...")
model = SquadroNet().to(DEVICE)
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("AI Loaded Successfully.")
    except:
        print("Model load failed. Using random weights.")
model.eval()
ai_player = NeuralMCTS(model, DEVICE)

def render_board_text(board):
    """
    Converts the board state into a text-based representation suitable for Telegram.
    Uses directional emojis to accurately reflect piece movesets.
    """
    text = "*SQUADRO BOT*\n\n"
    
    # Header (P2 Speeds)
    text += "           "
    for i, s in enumerate(board.P2_SPEEDS):
        eff = s if board.p2_dir[i] == 1 else (4 - s)
        arrow = "v" if board.p2_dir[i] == 1 else "^"
        text += f"{eff}{arrow} "
    text += "\n"

    # Top Docks (P2 Home)
    text += "           "
    for i in range(5):
        if board.p2_pos[i] == 0:
            sym = "✅" if board.p2_fin[i] else "🔽"
            text += f"{sym}"
        else:
            text += "⬜"
    text += "\n"

    # Grid
    for r in range(5):
        # Left Header (P1 Speed)
        eff = board.P1_SPEEDS[r] if board.p1_dir[r] == 1 else (4 - board.P1_SPEEDS[r])
        arrow = ">" if board.p1_dir[r] == 1 else "<"
        text += f"{eff}{arrow} "
        
        # P1 Home Dock
        if board.p1_pos[r] == 0:
            sym = "✅" if board.p1_fin[r] else "▶️"
            text += f"{sym}|"
        else:
            text += "⬜|"

        # Board Cells
        for c in range(5):
            cell = "⚫" # Empty dot
            
            # Show exact direction for P1
            if board.p1_pos[r] == c + 1:
                cell = "▶️" if board.p1_dir[r] == 1 else "◀️"
            
            # Show exact direction for P2
            if board.p2_pos[c] == r + 1:
                cell = "🔽" if board.p2_dir[c] == 1 else "🔼"
                
            text += f"{cell}"
        
        # P1 Turnaround
        if board.p1_pos[r] == 6: 
            text += "|◀️\n"
        else: 
            text += "|⬜\n"

    # Bottom Docks (P2 Turnaround)
    text += "           "
    for i in range(5):
        if board.p2_pos[i] == 6: 
            text += "🔼 "
        else: 
            text += "⬜"
    
    text += "\n\n"
    if board.winner:
        text += f"*Player {board.winner} Wins!*"
    else:
        turn_text = "▶️ Your Turn (P1)" if board.turn == 1 else "🔽 AI Thinking..."
        text += f"Status: {turn_text}"
        
    return text

def get_keyboard(board):
    """
    Creates the clickable buttons for the valid moves.
    """
    if board.winner or board.turn == 2:
        return None # No buttons if game over or AI turn

    keyboard = []
    moves = board.get_legal_moves()
    row = []
    for m in moves:
        # Callback data format: "move_INDEX"
        row.append(InlineKeyboardButton(f"Move {m+1}", callback_data=f"move_{m}"))
        if len(row) >= 3: # 3 buttons per row
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("🔄 New Game", callback_data="new_game")])
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /start command."""
    chat_id = update.effective_chat.id
    games[chat_id] = SquadroBoard()
    
    board = games[chat_id]
    await update.message.reply_text(
        render_board_text(board),
        reply_markup=get_keyboard(board),
        parse_mode='Markdown'
    )

async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for button clicks."""
    query = update.callback_query
    await query.answer() # Acknowledge click
    
    chat_id = update.effective_chat.id
    data = query.data
    
    if data == "new_game":
        games[chat_id] = SquadroBoard()
        board = games[chat_id]
        await query.edit_message_text(
            render_board_text(board),
            reply_markup=get_keyboard(board),
            parse_mode='Markdown'
        )
        return

    # Handle Move Logic
    # If the game was dropped from the cache due to inactivity, initialize a new one
    if chat_id not in games:
        games[chat_id] = SquadroBoard()
    
    board = games[chat_id]
    
    # 1. Human Move
    if data.startswith("move_"):
        move_idx = int(data.split("_")[1])
        if move_idx in board.get_legal_moves():
            board.do_move(move_idx)
            
            # Update UI immediately to show move
            await query.edit_message_text(
                render_board_text(board),
                reply_markup=None, # Remove buttons while AI thinks
                parse_mode='Markdown'
            )
            
            # 2. Check Win / AI Turn
            if not board.winner and board.turn == 2:
                # AI Turn
                # Offload the heavy MCTS search to a separate thread so the bot remains responsive
                loop = asyncio.get_running_loop()
                best_move = await loop.run_in_executor(None, ai_player.search, board, 100)
                
                board.do_move(best_move)
                
                # Update UI again after AI move
                await query.edit_message_text(
                    render_board_text(board),
                    reply_markup=get_keyboard(board),
                    parse_mode='Markdown'
                )
            else:
                # If human won or game over
                await query.edit_message_text(
                    render_board_text(board),
                    reply_markup=get_keyboard(board), # Allow New Game
                    parse_mode='Markdown'
                )

if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_click))

    print("Bot is polling...")
    app.run_polling()