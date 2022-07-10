#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 23:14:35 2022

@author: chasebrown
"""

from PIL import Image
import logging
import coloredlogs
import datetime
import os
import random
import sys
import pandas as pd

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

def parseInvImage(img):
    #Armor
    left = 240
    top = 105
    width = 16
    height = 16
    armor = {}
    for item in range(0,4):
        box = (left, top, left+width, top+height)
        armor.update({item: img.crop(box)})
        top += 18
    
    left = 309
    top = 159
    box = (left, top, left+width, top+height)
    armor.update({4: img.crop(box)})
    
    #Inventory
    top = 181
    inventory = {}
    
    for row in range(0, 3):
        left = 240
        for column in range(0,9):
            box = (left, top, left+width, top+height)
            inventory.update({row*9 + column: img.crop(box)})
            left += 18
        top += 18
    
    #Item Bar
    top = 239
    left = 240
    itemBar = {}
    
    for column in range(0,9):
        box = (left, top, left+width, top+height)
        itemBar.update({column: img.crop(box)})
        left += 18
        
    #Crafting
    top = 115
    crafting = {}
    
    for row in range(0, 2):
        left = 330
        for column in range(0,2):
            box = (left, top, left+width, top+height)
            crafting.update({row*2 + column: img.crop(box)})
            left += 18
        top += 18
    left = 386
    top = 125
    box = (left, top, left+width, top+height)
    crafting.update({4: img.crop(box)})
    
    return {"Armor": armor, "Inventory": inventory, "Item Bar": itemBar, "Crafting": crafting}

def logRun(obs, block, start, end, version):
    
    img = Image.fromarray(obs['pov'], 'RGB')
    img.save("../logs/Run Logs/Inventory/" + block + '-' + start + '-' + end + '-' + version + '.jpg')

def editMalmoFixed(block, start, end, startUpFile):
    template = 'dict(type="{item}", quantity={amount}),\n'
    replace = ""
    block = block.strip()
    for i in range(int(start), int(end)):
        replace += template.replace("{item}", block).replace("{amount}", str(i))
    projectFile = open("/home/chase/.local/lib/python3.10/site-packages/minerl/herobraine/env_specs/basalt_specs.py", "w")
    projectFile.write(startUpFile.replace("{inventory}", replace))
    projectFile.close()

def editMalmoRandom(startUpFile, id1, id2):
    options = []
    for file in os.listdir("../assets/datasets/Item Classifier Data/train/"):  
        options.append({"block": file.split("-")[0], "quant": file.split("-")[1].split(".")[0]})
    df = pd.DataFrame.from_dict(options)
    template = 'dict(type="{item}", quantity={amount}),\n'
    replace = ""
    testData = []
    for i in range(0, 36):
        item = df.sample()
        replace += template.replace("{item}", item.iat[0,0]).replace("{amount}", str(item.iat[0,1]))
        testData.append({"location": i, "block": item.iat[0,0], "quantity": str(item.iat[0,1])})
    df = pd.DataFrame.from_dict(testData).to_csv("../logs/Run Logs/Inventory/random" + '-' + id1 + '-' + id2 + '.csv', index=False)
    projectFile = open("/home/chase/.local/lib/python3.10/site-packages/minerl/herobraine/env_specs/basalt_specs.py", "w")
    projectFile.write(startUpFile.replace("{inventory}", replace))
    projectFile.close()


def runMalmo(block, start, end):
    import minerl    
    import gym
    

    path = "../logs/Run Logs/Inventory"
    if not os.path.exists(path):
        os.makedirs(path)
    
    
    env = gym.make("MineRLBasaltBuildVillageHouse-v0")
    obs = env.reset()


    done = False
    counter = 0
    while not done:
        counter += 1
        ac = env.action_space.noop()
        # Spin around to see what is around us
        if counter > 1:
            ac["inventory"] = 1
            if counter < 4 and counter > 1:
                ac["camera"] = [-100, -100]
            elif counter == 4:
                logRun(obs, block, start, end, "1")
            elif counter < 7:
                ac["camera"] = [100, 100]
            elif counter == 7:
                logRun(obs, block, start, end, "2")
            
        obs, reward, done, info = env.step(ac)
        env.render()
        if counter == 7:
            done = True

    env.close()
    

def main():
    needRandom = True
    if needRandom:
        rand1 = str(random.sample(range(1000, 9999), 1))
        rand2 = str(random.sample(range(1000, 9999), 1))
        editMalmoRandom(startUpFile, rand1, rand2)
        runMalmo("random", rand1, rand2)
    else:
        editMalmoFixed(sys.argv[1], sys.argv[2], sys.argv[3], startUpFile)
        runMalmo(sys.argv[1], sys.argv[2], sys.argv[3])



if __name__ == "__main__":
    main()


