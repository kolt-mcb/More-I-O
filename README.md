# MORE I/O
## What is this project?
This project is a OpenAI-powered version of SethBling's MarI/O Evolving Neural Network script for Bizhawk 1.x.

Unfortunately, Sethbling's original code relies on Bizhawk 1.x and is Windows-based. The code is also very old, dating from last edits in 2015. Trying to run the code on a newer Bizhawk instances works, but you cannot save or load your progress due to the Lua scripting runtime throwing errors.

MoreI/O aims to fill that gap using OpenAI (https://openai.com/about/) by provide a way of letting the computer do the work of playing Super Mario Bros on the NES using FCEUX emulator. Maybe one day, the AI will be able to race against the clock in speedruns in versus matches against human players.

## Requirements
You'll need at least the latest version of Python 2 and/or Python 3 that comes with your distro. Ubuntu 16.04 and similar distros (probably even Debian) should ship with 2.7.x and 3.5 in their repositories.

It doesn't matter what platform you use, as long as it's Linux and you can run FCEUX and Python. A lot of this is CPU-driven, and doesn't make use of TensorFlow. It has been successfully tested on powerhouse x86-64 servers to ARM based devices like the ODROID-XU4. It might even run on the R-Pi with poor emulated framerate.

Make sure you have the following packages installed as well. Some of the commands might be incorrect, please submit a issue ticket with the correct ones.

* fceux (Ubuntu/Debian: `apt install fceux`)
* git (Ubuntu/Debian: `apt install git`)
* python3.x (Ubuntu/Debian: `apt install python3`)
* pip3 for python 3.x (Ubuntu/Debian: `apt install python3-pip`)
* matplotlib (`pip install matplotlib`)
* pymongo (`pip install pymongo`)
* tk (Ubuntu/Debian: `apt install python-tk` or `apt install python3-tk`)
* gym (`pip install gym`)
* You might also need to install libfreetype6-dev and a few other packages if Tk throws an exception that it can't find libraries to build with.

If you wish, you can install the full OpenAI stack by going to https://gym.openai.com/docs. It is recommended you install these packages locally (don't install them as superuser or globally) because we need to edit some files.

## Installing the environment
After installing the above packages, clone this repository into a empty folder using `git clone [url of repo] MoreIO`. You can change the directory name if you wish, but keeping it as MoreIO is easier to know what it is. After git pulls the repo, move on the next step.

Do not run any python scripts yet. You should have a folder in your `$HOME/.local/lib/python3.X/site-packages/` called `gym`. Replace X with the minor version of python - for example, if you have python 3.5, you'd replace the X with 5. If you don't, you probably have a install error or forgot to install it. Otherwise, open a ticket or search google. In the directory you cloned the repo, run `git clone https://github.com/koltafrickenfer/gym-super-mario` to clone the AI environments and related data.

Now, we need to install the cloned gym environment done in the last command. Run this command, adapting the path to your needs:

`ln -s [path to the just-cloned gym-super-mario repo]/ppaquette_gym_super_mario $HOME/.local/lib/python3.X/site-packages/gym/envs/gym_super_mario`

Do not alter anything else. This sets up a symlink so gym can see the environments on disk, but now you need to register them by editing `$HOME/.local/lib/python3.X/site-packages/gym/envs/__init__.py` and putting the following at the bottom of it:

```
register(
	id='meta-SuperMarioBros-Tiles-v0',
	entry_point='gym.envs.gym_super_mario:MetaSuperMarioBrosEnv',
	kwargs={'draw_tiles': 1},
	reward_threshold=(3266 - 40),
	nondeterministic=True,
)	
```
Save the file. You're done. Let's get it started.

## Running the environment
Change to the directory where you have MoreIO cloned into and then run `python3 mario.py`. If all goes well, you should see the control panel window, with how many jobs you want to run, and the population. The defaults are set to 2 jobs and 300 population. The graph at the bottom will be generated over time, and show you the progress of the network. The more jobs, the more CPU consumption will be used as the emulator instances running at the same time. **Do not set it to absurd numbers if you have a weak CPU as that's just stupid and you're asking for a system crash.**

After changing jobs and population to your liking hit "Start Run". It will then turn to "Running" to let you know it's running the AI environment. After a bit, you should see FCEUX boot and the AI start. It is normal for the AI to do nothing for a while and then reset, you need to keep in mind that it's running a evolution simulation, and there will always be child species that "do nothing" or are duds. 

Right now, the AI will go through all 32 levels in Super Mario Bros. It won't do glitch levels like minus world or other levels that are accessible by exploiting logic flaws in the game unless the AI becomes aware of those strats. This might be slow, but give it about half a day and you'll see the AI start making progress at attempting to beat the levels. Keep in mind that this will be a slow evolution process.

## Copyright concerns
While this project does use emulated NES hardware via emulators and a copy of the Super Mario Bros NES ROM, the term "emulator" can be disputed in some countries. The version of mario that is played is cloned from an unofficial openAI and has no mention of copyright, neither I or this project are affiliated with Nintendo, openAI, or any other company for that matter. 

## Things to do, in no particular order:
1. clean up tkinter ui add in a way to save config files with mutation rates and ability to disbale rendering. basicly a bunch of boring work. 
2. write docker container that inilizes open ai. 
3. write code for inderect encoding of genes. 
