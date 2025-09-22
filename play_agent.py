"""
Script để chạy agent đã được train để chơi osu! mania
"""
from stable_baselines3 import PPO
from game_env import OsuManiaEnv

def play_agent(model_path="models/best_osu_model.zip"):
    print("=== Osu! Mania AI Player ===")
    
    env = OsuManiaEnv(show_window=True)
    
    try:
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path, env=env)
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at '{model_path}'")
        print("Please train the model first.")
        env.close()
        return

    print("\nInstructions:")
    print("1. Open osu! and select a song.")
    print("2. When ready, switch back to this terminal and press Enter.")
    print("3. To stop the agent, focus the visualization window and press 'q'.")
    input("Press Enter to start the agent...")

    episodes = 0
    while True:
        episodes += 1
        print(f"\n--- Starting Episode {episodes} ---")
        
        obs, info = env.reset()
        done, truncated = False, False
        total_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        print(f"--- Episode {episodes} Finished ---")
        print(f"Total reward: {total_reward:.2f}")

        # Sau khi một ván kết thúc (do hết bài, hoặc do bạn nhấn 'q')
        # Hỏi người dùng có muốn chơi tiếp không.
        
        # Nếu người dùng đã nhấn 'q' trong game, biến done sẽ là True
        # và chúng ta sẽ thoát khỏi vòng lặp lớn.
        if done:
            print("Quit signal received from environment. Exiting.")
            break

        continue_playing = input("Play another song? (Y/n): ").strip().lower()
        if continue_playing == 'n':
            break
            
    env.close()
    print("Goodbye!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        play_agent(sys.argv[1])
    else:
        play_agent(model_path="models/best_model/best_model.zip")