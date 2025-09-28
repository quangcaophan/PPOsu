# Osu! RAM-Reading Bot (That Runs Smooth AF!)

Alright, so here's the deal: this is an AI I coded to teach itself how to shred osu! mania. At its core, it's a pretty smart AI called PPO. But the real magic trick is that instead of slowly "staring" at the screen, it hacks straight into the game's brain (we're talkin' RAM) to grab the combo, score, and accuracy. The speed is just insane, way more efficient than the old way, and lets you train this beast without a monster PC!

## 🌟 The Cool Stuff
- Big Brain AI: It uses PPO from stable-baselines3. In normal-people-speak, this AI is smart because it learns steadily and doesn't just randomly change its strategy, so it doesn't "forget" what it's already mastered. It learns properly, not like some dumb bot that just spams keys.

- Blazing Fast: Reads straight from RAM, no questions asked! We're ditching the laggy, resource-hogging OCR. Game info gets updated almost instantly (sub-1ms). This means the training FPS goes through the roof! Higher FPS means the bot learns way faster and gets more practice in the same amount of time.

- Super Optimized: This whole project was built with one goal in mind: RUN FAST and LIGHT. I wanted it to work even on potatoes, so everything is optimized to cut down on system load.

- Live Bot Showcase: A window pops up so you can watch what the AI is "seeing" in real-time. You'll see the key presses light up and the combo/score numbers dance around. It's pretty fun watching it go from a total noob spamming keys to a legit pro.

- GPU Powered: Of course! All the heavy number-crunching for the neural network gets tossed to your graphics card, leaving the CPU free to handle other stuff.

- Pro-Level Training: It auto-saves the model constantly. So, if the power goes out or your PC crashes mid-training, you don't lose all your progress. It also has full TensorBoard logs so you can track how it's doing, which is pretty neat.

## 💥 The Performance Saga: The Epic Pivot from OCR to RAM Reading
Okay, let me tell you about the performance drama. This was a whole journey.

### The Big Problem at First
- Resource Wars: At first, I was using OCR to read the numbers on the screen. Turns out, the AI (PPO) and the OCR were like two hungry hippos fighting over the same GPU resources. Imagine two huge dudes trying to eat the same tiny cake—nobody wins. My computer was crying for help.

- Nosediving FPS: The result was that the OCR was slow as a snail, taking about ~300ms just for one recognition. That's a third of a second! This dragged the entire program down to a pathetic ~3 FPS. You can't train anything at that speed; it's not even a game anymore, it's a slideshow. How is a bot supposed to learn timing when the frames it sees are choppy and delayed by half a beat? Impossible!

- A performance report showing the OCR process hogging all the processing time. It was the weak link, for sure.

### The Game-Changer: Reading RAM!
So, I got fed up and decided to start a revolution: I completely ditched OCR for reading in-game stats. Now, I use Pymem to read directly from the game's RAM. It's like plugging a straw directly into the game's brain to suck out the data, no middleman needed.

And the result was a massive explosion! By getting rid of that dead weight, the performance bottleneck was completely solved. Now, the program's speed is only limited by screen capture (which is super fast) and the AI's calculations (also super fast on a GPU). The training is smooth, efficient, and actually works!

## 🔧 How It Works
It's pretty simple to understand, it's got a few main steps:

1. "Seeing" the Game: The bot takes a screenshot of the note highway. Then, it feeds this image into a CNN (think of it as the robot's eyeball). The CNN learns to tell which pixels are notes and which are just the background.

2. Making a Decision: Based on what it "sees," the AI calculates the best possible key combination to press at that exact moment.

3. Getting the Results (The Secret Sauce!): Right after pressing the keys, instead of waiting for OCR, it instantly peeks into the game's memory to pull out the combo, score, and accuracy (in under 1ms). It's like it has a telepathic link with the game.

4. Reward/Punish and Learn: Thanks to the instant feedback, it knows immediately if it did a good job. A good hit earns a reward, a miss gets a penalty. This "action -> result -> reward/punishment" loop happens incredibly fast, helping the AI develop a much better sense of rhythm and timing.

## 📋 Stuff You Need to Install
- stable-baselines3[extra]==2.1.0
- torch>=2.0.0
- gymnasium>=0.28.0
- opencv-python>=4.8.0
- mss>=9.0.1
- numpy>=1.24.0
- pydirectinput>=1.0.4
- pymem>=1.13.0
- tensorboard>=2.1.0

## 🚀 Let's Get This Party Started
### Installation
```git clone [https://github.com/your-username/OsuMania-PPO-Memory-Reader.git](https://github.com/your-username/OsuMania-PPO-Memory-Reader.git)
cd OsuMania-PPO-Memory-Reader
pip install -r requirements.txt
```
### Find Memory Pointers (The Most Important Step!)
- Okay, this part is a bit of a pain, but you only have to do it once, so hang in there! It's like a treasure hunt, and finding it feels awesome.
  - Download and open Cheat Engine.
  - Open osu! and attach Cheat Engine to the osu!.exe process.
  - Use Cheat Engine's scanning features to find the values for Score, Combo, and Accuracy. A good tip is to play a bit to change the numbers, then use "Next Scan" to filter the results.
  - Once you find the address, you have to find its "pointer path." This is crucial so the address doesn't change every time you restart the game. Just Google "cheat engine find pointer path," and you'll find tons of tutorials.
  - Open up memory_reader.py and paste the pointer paths you found. Done!

### Calibrate Your Play Area
Even though it reads RAM, the bot still needs to see the play area to know which notes are falling, right?
- Run this script:

```
python setup_calibration.py
```

- It'll guide you to draw a box around the Play Area and the results screen. EZ!

### Let's Start Training!
Once everything is set up, just run this command to start training:
```
python train_optimized.py --config config/mania_4k_config.json
```
A window will pop up showing the bot play. At first, it's gonna look like a drunk guy spamming keys—don't worry, that's just how it learns! Let it run for a while, and you'll see it get better and better. The model will be saved automatically in the models/ folder.

### 📂 File Structure
```
├── environments/          # The custom game environment
├── config/                # Config files for different key modes
├── models/                # Saved AI models
├── templates/             # Result screen image templates
├── memory_reader.py       # The big boss: reads game RAM
├── setup_calibration.py   # Tool for screen calibration
├── train_optimized.py     # The main training script
├── play_agent.py          # Script to watch the bot play
└── README.md              # This file right here
```

### 📈 Cool Ideas for the Future
- Auto Pointer Finder: Write a fancy script that can auto-scan the game's memory to find the pointers. That would make the setup way more user-friendly.

- More Key Modes: The framework is pretty flexible. It's 4K now, but adding 7K, 8K, or whatever would just require a new config file and some minor tweaks.

- Next-Level Rewards: If we could find the memory address for the "Hit Error" (timing deviation), that would be insane! We could teach the bot to aim for MARVELOUS-level accuracy instead of just PERFECT. Then the bot would truly be a god.