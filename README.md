# SquadroAI
An AI for the board game Squadro using Monte Carlo Tree Search and reinforcement learning.

## Setup
Run `pip install -r requirements.txt` to install the necessary dependencies.

## Training the AI
Run `python train.py`.

## Logging the Training
After running `python train.py`, create a separate terminal and run `tensorboard --logdir=runs`, and open up `http://localhost:6006/` in your browser, you'll be able to see the losses and win rate as training progresses.

## How to Play
Run `python main.py`, and select what game mode you would like.
1. Human vs AI
2. Human vs Human
3. AI vs AI

## Telegram Bot Functionality
You can also convert this into a Telegram bot. Here's how to do it.
1. Open Telegram and search for @BotFather.
2. Send the message /newbot.
3. Follow the instructions and name your bot.
4. Copy the HTTP API Token provided by BotFather.
5. Create a `.env` file containing your token:
```
TELEGRAM_BOT_TOKEN='YOUR_TOKEN_HERE'
```
6. Run `python telegram_bot.py`.
7. Open your bot in Telegram and type `/start`.

Have fun!