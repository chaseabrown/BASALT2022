#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:46:00 2022

@author: chasebrown
"""

import gym
import logging
import coloredlogs
from PIL import Image
import datetime
import os
from random import randint
from detecto import core, utils, visualize
import sys       
sys.path.append('../models')
from BASALTAgent import Agent
from DQN import DQN

coloredlogs.install(logging.DEBUG)

startUpFile = """
from typing import List, Optional, Sequence

import gym

from minerl.env import _fake, _singleagent
from minerl.herobraine import wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers, mc

from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec


MAKE_HOUSE_VILLAGE_INVENTORY = [
    {inventory}
]

class BasaltTimeoutWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0

    def reset(self):
        self.timeout = self.env.task.max_episode_steps
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        return observation, reward, done, info


class DoneOnESCWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.episode_over = False

    def reset(self):
        self.episode_over = False
        return self.env.reset()

    def step(self, action):
        if self.episode_over:
            raise RuntimeError("Expected `reset` after episode terminated, not `step`.")
        observation, reward, done, info = self.env.step(action)
        done = done or bool(action["ESC"])
        self.episode_over = done
        return observation, reward, done, info


def _basalt_gym_entrypoint(
        env_spec: "BasaltBaseEnvSpec",
        fake: bool = False,
) -> _singleagent._SingleAgentEnv:
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)

    env = BasaltTimeoutWrapper(env)
    env = DoneOnESCWrapper(env)
    return env


BASALT_GYM_ENTRY_POINT = "minerl.herobraine.env_specs.basalt_specs:_basalt_gym_entrypoint"


class BasaltBaseEnvSpec(HumanControlEnvSpec):

    LOW_RES_SIZE = 64
    HIGH_RES_SIZE = 1024

    def __init__(
            self,
            name,
            demo_server_experiment_name,
            max_episode_steps=2400,
            inventory: Sequence[dict] = (),
            preferred_spawn_biome: str = "plains"
    ):
        self.inventory = inventory  # Used by minerl.util.docs to construct Sphinx docs.
        self.preferred_spawn_biome = preferred_spawn_biome
        self.demo_server_experiment_name = demo_server_experiment_name
        super().__init__(
            name=name,
            max_episode_steps=max_episode_steps,
            # Hardcoded variables to match the pretrained models
            fov_range=[70, 70],
            resolution=[640, 360],
            gamma_range=[2, 2],
            guiscale_range=[1, 1],
            cursor_size_range=[16.0, 16.0]
        )

    def is_from_folder(self, folder: str) -> bool:
        # Implements abstractmethod.
        return folder == self.demo_server_experiment_name

    def _entry_point(self, fake: bool) -> str:
        # Don't need to inspect `fake` argument here because it is also passed to the
        # entrypoint function.
        return BASALT_GYM_ENTRY_POINT

    def create_observables(self):
        # Only POV
        obs_handler_pov = handlers.POVObservation(self.resolution)
        return [obs_handler_pov]

    def create_agent_start(self) -> List[handlers.Handler]:
        return super().create_agent_start() + [
            handlers.SimpleInventoryAgentStart(self.inventory),
            handlers.PreferredSpawnBiome(self.preferred_spawn_biome),
            handlers.DoneOnDeath()
        ]

    def create_agent_handlers(self) -> List[handlers.Handler]:
        return []

    def create_server_world_generators(self) -> List[handlers.Handler]:
        # TODO the original biome forced is not implemented yet. Use this for now.
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self) -> List[handlers.Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                (self.max_episode_steps * mc.MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[handlers.Handler]:
        return []

    def create_server_initial_conditions(self) -> List[handlers.Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=True
            )
        ]

    def get_blacklist_reason(self, npz_data: dict) -> Optional[str]:

        # TODO(shwang): Waterfall demos should also check for water_bucket use.
        #               AnimalPen demos should also check for fencepost or fence gate use.
        # TODO Clean up snowball stuff (not used anymore)
        equip = npz_data.get("observation$equipped_items$mainhand$type")
        use = npz_data.get("action$use")
        if equip is None:
            return f"Missing equip observation. Available keys: {list(npz_data.keys())}"
        if use is None:
            return f"Missing use action. Available keys: {list(npz_data.keys())}"

        assert len(equip) == len(use) + 1, (len(equip), len(use))

        for i in range(len(use)):
            if use[i] == 1 and equip[i] == "snowball":
                return None
        return "BasaltEnv never threw a snowball"

    def create_mission_handlers(self):
        # Implements abstractmethod
        return ()

    def create_monitors(self):
        # Implements abstractmethod
        return ()

    def create_rewardables(self):
        # Implements abstractmethod
        return ()

    def determine_success_from_rewards(self, rewards: list) -> bool:

        return False

    def get_docstring(self):
        return self.__class__.__doc__


MINUTE = 20 * 60


class FindCaveEnvSpec(BasaltBaseEnvSpec):

    def __init__(self):
        super().__init__(
            name="MineRLBasaltFindCave-v0",
            demo_server_experiment_name="findcaves",
            max_episode_steps=3*MINUTE,
            preferred_spawn_biome="plains",
            inventory=[],
        )


class MakeWaterfallEnvSpec(BasaltBaseEnvSpec):

    def __init__(self):
        super().__init__(
            name="MineRLBasaltMakeWaterfall-v0",
            demo_server_experiment_name="waterfall",
            max_episode_steps=5*MINUTE,
            preferred_spawn_biome="extreme_hills",
            inventory=[
                dict(type="water_bucket", quantity=1),
                dict(type="cobblestone", quantity=20),
                dict(type="stone_shovel", quantity=1),
                dict(type="stone_pickaxe", quantity=1),
            ],
        )


class PenAnimalsVillageEnvSpec(BasaltBaseEnvSpec):

    def __init__(self):
        super().__init__(
            name="MineRLBasaltCreateVillageAnimalPen-v0",
            demo_server_experiment_name="village_pen_animals",
            max_episode_steps=5*MINUTE,
            preferred_spawn_biome="plains",
            inventory=[
                dict(type="oak_fence", quantity=64),
                dict(type="oak_fence_gate", quantity=64),
                dict(type="carrot", quantity=1),
                dict(type="wheat_seeds", quantity=1),
                dict(type="wheat", quantity=1),
            ],
        )

    def create_agent_start(self) -> List[handlers.Handler]:
        return super().create_agent_start() + [
            handlers.SpawnInVillage()
        ]


class VillageMakeHouseEnvSpec(BasaltBaseEnvSpec):

    def __init__(self):
        super().__init__(
            name="MineRLBasaltBuildVillageHouse-v0",
            demo_server_experiment_name="village_make_house",
            max_episode_steps=12*MINUTE,
            preferred_spawn_biome="plains",
            inventory=MAKE_HOUSE_VILLAGE_INVENTORY,
        )

    def create_agent_start(self) -> List[handlers.Handler]:
        return super().create_agent_start() + [
            handlers.SpawnInVillage()
        ]

"""

def logRun(ac, obs, reward, done, info, startTimeSTR):
    datetimeSTR = datetime.datetime.now().strftime("D%Y-%m-%d-T%H-%M-%S-%f")
    log = open("../logs/Run Logs/" + startTimeSTR + "/" + datetimeSTR + ".txt", 'w+')
    for key in ac.keys():
        log.write(key + ": " + str(ac[key]) + "\n")
    log.write("\nReward: " + str(reward) + "\n")
    log.write("\nInfo: " + str(info) + "\n")
    log.write("\nDone: " + str(done) + "\n")
    
    log.close()
    
    img = Image.fromarray(obs['pov'], 'RGB')
    img.save("../logs/Run Logs/" + startTimeSTR + "/" + datetimeSTR + '.jpg')

def editMalmoRandom():
    template = 'dict(type="{item}", quantity={amount}),\n'
    replace = ""
    testData = []
    #Get random number between and including 0 and 40
    selected = randint(0, 40)
    for i in range(0, 41):
        if i == selected:
            replace += template.format(item="diamond", amount=1)
        else:
            replace += template.format(item="air", amount=0)
    projectFile = open("/home/chase/.local/lib/python3.10/site-packages/minerl/herobraine/env_specs/basalt_specs.py", "w")
    projectFile.write(startUpFile.replace("{inventory}", replace))
    projectFile.close()
    return selected

def resetGame(ac, env):
    ac = env.action_space.noop()
    ac["inventory"] = 0
    obs, reward, done, info = env.step(ac)
    env.render()
    ac = env.action_space.noop()
    ac["inventory"] = 1
    ac["camera"] = [-10, 0]
    obs, reward, done, info = env.step(ac)
    env.render()
    return obs


startTimeSTR = datetimeSTR = datetime.datetime.now().strftime("D%Y-%m-%d-T%H-%M-%S")

diamondLoc = editMalmoRandom()

import minerl

path = "../logs/Run Logs/" + startTimeSTR
if not os.path.exists(path):
    os.makedirs(path)
    
env = gym.make("MineRLBasaltBuildVillageHouse-v0")
obs = env.reset()

agent = Agent()

#DQN Agent Setup
#State = CursorX(float), CursorY(float), Holding(bool), startPosInv(int), endPosInv(int)
state_size = 5
#Action = Up(bool), Down(bool), Left(bool), Right(bool), Click(bool)
action_size = 5

dqn = DQN(state_size, action_size)

batch_size = 32
n_episodes = 1001
done = False
counter = 0

while not done:
    counter += 1
    ac = env.action_space.noop()
    # Spin around to see what is around us
    resetGame(ac, env)
    info = agent.act(obs)

    if diamondLoc < 5:
        self.inventory["armor"][slot]['item'] = items[slot]
        self.inventory["armor"][slot]['quantity'] = quants[slot]
    elif slot < 32:
        self.inventory["inventory"][slot-5]['item'] = items[slot]
        self.inventory["inventory"][slot-5]['quantity'] = quants[slot]
    elif slot < 41:
        self.inventory["item_bar"][slot-32]['item'] = items[slot]
        self.inventory["item_bar"][slot-32]['quantity'] = quants[slot]
    else:
        self.inventory["crafting"][slot-41]['item'] = items[slot]
        self.inventory["crafting"][slot-41]['quantity'] = quants[slot]

    holding = False
    if ac["Attack"] == 1 and info["inventory"][]
    state = [info["cursorLocation"]["x"], info["cursorLocation"]['y'], info["holding"], info["inventory_start"], info["inventory_end"]]
    state = np.reshape(state, [1, state_size])

    action = dqn.act()

    
    #
    
    if counter == 11:
        done = True
    logRun(ac, obs, reward, done, info, startTimeSTR)
"""while not simDone:
    counter += 1
    ac = env.action_space.noop()
    # Spin around to see what is around us
    ac["inventory"] = 1
    
    obs, reward, done, info = env.step(ac)
    env.render()
    
    
    info = agent.act(obs)
    
    for e in range(n_episodes): 
    #Reset Game
    ac = env.action_space.noop()
    ac["inventory"] = 0
    obs, reward, done, info = env.step(ac)
    env.render()
    ac = env.action_space.noop()
    ac["inventory"] = 1
    obs, reward, done, info = env.step(ac)
    env.render()
    
    for time in range(5000):  
        #Render Environment
        
        
        #Get Action
        action = agent.act(state)
        
        #Process Action in Environment
        next_state, reward, done, _ = env.step(action) 
        
        #Punish Failures
        reward = reward if not done else -10     
        
        #Store Datapoint
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done) 
        
        state = next_state    
        if done: 
            print("episode: {}/{}, score: {}, e: {:.2}" 
                  .format(e, n_episodes, time, agent.epsilon))
            break
        
    #When Memory filled to batch size, train model on batch
    if len(agent.memory) > batch_size:
        agent.replay(batch_size) 
        
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")




    if counter == 11:
        done = True
    logRun(ac, obs, reward, done, info, startTimeSTR)

env.close()"""



