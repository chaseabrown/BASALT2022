from argparse import ArgumentParser
from itertools import count
import pickle

import gym
import minerl
from colabgymrender.recorder import Recorder
from pyvirtualdisplay import Display


from openai_vpt.agent import MineRLAgent, ENV_KWARGS

def main(model, weights, env):
    envName = str(env)
    env = gym.make(env)
    display = Display(visible=0, size=(720, 480))
    display.start()
    env = Recorder(env, "/content/drive/MyDrive/BASALT2022/basalt-2022-behavioural-cloning-baseline-main/video/" + envName + "/", fps=30)

    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    print("---Launching MineRL enviroment (be patient)---")
    obs = env.reset()

    counter = 0
    while True:
        minerl_action = agent.get_action(obs)
        # ESC is not part of the predictions model.
        # For baselines, we just set it to zero.
        # We leave proper execution as an exercise for the participants :)
        minerl_action["ESC"] = 0
        obs, reward, done, info = env.step(minerl_action)
        env.render()
        if done:
            counter += 1
            if counter < 10:
                print("---Restarting---")
                obs = env.reset()
            else:
                print("---Done---")
                break
    env.stop()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)

    args = parser.parse_args()

    main(args.model, args.weights, args.env)