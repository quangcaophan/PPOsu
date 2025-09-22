# OsuMania PPO Agent

This project presents a sophisticated AI agent that learns to play the rhythm game *osu\! mania* using Reinforcement Learning (RL). The agent is built upon Proximal Policy Optimization (PPO) and leverages Optical Character Recognition (OCR) for a more robust and intelligent reward system.

## 🌟 Features

  * **Advanced RL Agent**: Utilizes Proximal Policy Optimization (PPO) from `stable-baselines3` for efficient and stable learning.
  * **OCR-Powered Rewards**: The agent's rewards are not just based on in-game events but are calculated by reading the combo, score, and accuracy directly from the screen using EasyOCR. This allows for a more nuanced and human-like learning process.
  * **Automated Setup and Calibration**: A comprehensive `setup_calibration.py` script allows for easy, interactive, and precise calibration of the play area, combo, score, and accuracy regions on the screen.
  * **Real-time Visualization**: The `game_env.py` provides a real-time visualization of the agent's view, actions, and key metrics like reward, combo, and OCR-read accuracy, offering valuable insights into the learning process.
  * **GPU Acceleration**: The training script supports both NVIDIA (CUDA) and AMD (DirectML) GPUs for accelerated training.
  * **Detailed Monitoring and Callbacks**: The training process is enhanced with custom callbacks for detailed logging, performance tracking, and model checkpointing.

## 🔧 How It Works

The agent operates by capturing screen frames of the *osu\! mania* gameplay. These frames are processed and fed into a Convolutional Neural Network (CNN) as the agent's "eyes". The PPO algorithm then determines the optimal action (which keys to press) based on this visual input.

The reward system, a crucial component of RL, is what makes this project unique. Instead of relying on simplistic reward mechanisms, the agent uses EasyOCR to read the game's UI elements in real-time. This allows for a more sophisticated reward function that considers:

  * **Combo Increases/Breaks**: The agent is rewarded for increasing its combo and penalized for breaking it.
  * **Score Differentials**: The agent is rewarded for increases in the score.
  * **Accuracy**: The agent's reward is multiplied by its accuracy, incentivizing precise timing.

This OCR-based approach leads to more intelligent and human-like gameplay from the trained agent.

## 📋 Requirements

The project's dependencies are listed in the `requirements.txt` file and include:

  * `stable-baselines3[extra]==2.1.0`
  * `torch>=2.0.0`
  * `gymnasium>=0.28.0`
  * `opencv-python>=4.8.0`
  * `mss>=9.0.1`
  * `numpy>=1.24.0`
  * `easyocr>=1.7.0`
  * `pydirectinput>=1.0.4`
  * `tensorboard>=2.13.0`

For AMD GPU support, `torch-directml` is also required.

## 🚀 Getting Started

### 1\. Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/OsuMania-PPO-EasyOCR.git
cd OsuMania-PPO-EasyOCR
pip install -r requirements.txt
```

### 2\. Calibration

Before you can train the agent, you need to calibrate it to your screen and *osu\! mania* skin. Run the `setup_calibration.py` script:

```bash
python setup_calibration.py
```

This will launch an interactive tool that will guide you through selecting the following areas on your screen:

1.  **Play Area**: The area where the notes fall.
2.  **Combo Area**: The area where the combo count is displayed.
3.  **Score Area**: The area where the score is displayed.
4.  **Accuracy Area**: The area where the accuracy percentage is displayed.

The tool will save these coordinates to an `osu_config.json` file.

### 3\. Training

Once the calibration is complete, you can start training the agent by running the `train.py` script:

```bash
python train.py
```

The script will load the configuration from `osu_config.json` and begin the training process. You will see a visualization window showing the agent's view and performance metrics. Models will be saved periodically in the `models/` directory.

### 4\. Playing

After training, you can watch your agent play *osu\! mania* by running the `play_agent.py` script and providing the path to your trained model:

```bash
python play_agent.py models/best_osu_model.zip
```

## 📂 Project Structure

```
├── game_env.py            # The custom osu! mania environment for the RL agent.
├── osu_config.json        # Configuration file for screen coordinates.
├── play_agent.py          # Script to run a trained agent.
├── README.md              # This README file.
├── requirements.txt       # Project dependencies.
├── setup_calibration.py   # Interactive tool for calibrating screen areas.
└── train.py               # The main script for training the agent.
```

## 📈 Future Improvements

  * **Hyperparameter Tuning**: Further optimization of the PPO hyperparameters could lead to faster and more effective learning.
  * **Multi-key Support**: The current agent is designed for 4K mania, but it could be extended to support other key modes.
  * **Advanced OCR**: More robust OCR techniques could improve the accuracy of reading the game's UI, especially with different skins.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 🙏 Acknowledgements

This project would not be possible without the following open-source libraries and their contributors:

  * [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
  * [PyTorch](https://pytorch.org/)
  * [OpenCV](https://opencv.org/)
  * [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  * [Gymnasium](https://gymnasium.farama.org/)



## What next? 
- Buy a better gpu to train the agent
- Build Game Setup Interface for easy to setup
- Build Game Interface to control the agent and watch the game, metric,...