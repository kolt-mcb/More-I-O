from subprocess import check_output
import gym
from gym import wrappers
import os
import signal
import psutil
from lib import neat
from lib import networkDisplay
import numpy
import math
import time
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import threading
from multiprocessing import Queue
from queue import Empty
from tkinter import *
from tkinter import filedialog, messagebox
import pickle
from queue import Empty
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from operator import itemgetter


sentinel = object()  # tells the main tkinter window if a generattion is in progress

lock = multiprocessing.Lock()  # lock needed for env process


def poolInitializer(q):  # manages the job queue for genomes/players
    global jobs
    jobs = q


def playBest(genome,game):
    parentPipe, childPipe = multiprocessing.Pipe()
    genome.generateNetwork()
    process = multiprocessing.Process(
        target=singleGame, args=(genome, childPipe, game))
    process.start()
    display = networkDisplay.newNetworkDisplay(genome, parentPipe)
    display.checkGenomePipe()
    display.Tk.mainloop()
    process.join()

# creates multiprocessing job for pool and trains pool
def trainPool(envNum, species, runQueue, env, attemps):
    before = time.time()
    results = []
    jobs = Queue()
    s=0
    for specie in species:  # generates network for each genome and creates a job with species and genome index, env name and number of trials/attemps
        g=0
        for genome in specie.genomes:
            genome.generateNetwork()
            jobs.put((s,g, genome, attemps))
            g+=1
        s+=1
    mPool = multiprocessing.Pool(
        processes=envNum, initializer=poolInitializer, initargs=(jobs,))

    results = mPool.map(jobTrainer, [env] * envNum)
    mPool.close()
    mPool.join()

    # sets results from result list
    after = time.time()
    print("finished in ", int(after - before))

    runQueue.put(results)  # sends message to main tkinter process


# fixes observation ranges, uses hyperbolic tangent for infinite values
def obFixer(observation_space, observation):
    newObservation = []
    if observation_space.__class__ == gym.spaces.box.Box:
        for space in range(observation_space.shape[0]):
            high = observation_space.high[space]
            low = observation_space.low[space]
            if high == numpy.finfo(numpy.float32).max or high == float('Inf'):
                newObservation.append(math.tanh(observation[space]))
            else:
                dif = high - low
                percent = (observation[space] + abs(low)) / dif
                newObservation.append(((percent * 2) - 1).tolist())
    if observation_space.__class__ == gym.spaces.discrete.Discrete:
        c = 0
        for neuron in range(observation_space.n):
            if observation == neuron:
                newObservation.append(1)
            else:
                newObservation.append(0)
    return newObservation


def acFixer(action_space, action):  # fixes action ranges, uses hyperbolic tangent for infinite values
    newAction = []
    if action_space.__class__ == gym.spaces.box.Box:
        for space in range(action_space.shape[0]):
            high = action_space.high[space]
            low = action_space.low[space]
            if high != numpy.finfo(numpy.float32).max:
                dif = high - low
                percent = (action[space] + 1) / 2
                newAction.append(((percent * dif) - (dif / 2)).tolist())
            else:
                print("if your seeing me this code needs updated")
                newAction.append(math.tanh(observation[space]))
        return newAction
    if action_space.__class__ == gym.spaces.discrete.Discrete:
        c = 0
        for _, myAction in enumerate(action):
            if myAction > 0:
                newAction.append(_ + 1)
                c += 1
            if c > 1:
                newAction = [0]
                return int(newAction[0])
        if c == 0:
            newAction = [0]
        return int(newAction[0])


def jobTrainer(envName):

    env = gym.make(envName)
    # env = wrappers.Monitor(env,'tmp/'+envName,resume=True,video_callable=False) # no recoding on windows due to ffmepg	if
    if env.env.__class__ == 'gym.envs.atari.atari_env.AtariEnv':
        atari = True

    if env.action_space.__class__ == gym.spaces.discrete.Discrete:  # identifies action/observation space
        discrete = True
    else:
        discrete = False

    results = []

    while not jobs.empty():  # gets a new player index from queue
        try:
            job = jobs.get()
        except Empty:
            pass
        currentSpecies = job[0]
        currentGenome = job[1]
        genome = job[2]
        attemps = job[3]
        scores = 0
        if genome.age == 0:
            for run in range(attemps):  # runs for number of attemps
                score = 0
                done = False
                ob = env.reset()
                while not done:
                    ob = obFixer(env.observation_space, ob)
                    # evalutes brain, getting button presses
                    o = genome.evaluateNetwork(ob, discrete)
                    o = acFixer(env.action_space, o)
                    ob, reward, done, _ = env.step(o)
                    # env.render() # disabled render
                    score += reward
                    
                scores += score
            finalScore = round(scores / attemps)
            print("species:", currentSpecies, " genome:",
    			  currentGenome, " Scored:", finalScore)
        if genome.age == 0:
            results.append((finalScore, job))
        else: results.append((genome.fitness, job))
    env.close()
    return results


def singleGame(genome, genomePipe, envName):
    env = gym.make(envName)
    #env = wrappers.Monitor(env,'tmp/'+envName,resume=True)
    runs = 10

    print("playing best")
    if env.action_space.__class__ == gym.spaces.discrete.Discrete:
        discrete = True
    else:
        discrete = False
    for i in range(runs):

        ob = env.reset()
        done = False
        distance = 0
        maxDistance = 0
        staleness = 0
        score = 0
        while not done:
            ob = obFixer(env.observation_space, ob)
            o = genome.evaluateNetwork(ob, discrete)
            o = acFixer(env.action_space, o)
            ob, reward, done, _ = env.step(o)
            score += reward
            env.render()
            genomePipe.send(genome)
        print(score)
    env.reset()
    genomePipe.send("quit")
    genomePipe.close()
    env.close()


def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    gone, still_alive = psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


class gui:
    def __init__(self, master):
        self.master = master
        self.frame = Frame(self.master, height=1000, width=450)
        self.frame.grid()
        # jobs label
        self.envLabel = Label(self.master, text="Jobs: ").grid(
            row=1, column=0, sticky=W)
        self.envNum = IntVar()
        self.envNumEntry = Entry(self.master, textvariable=self.envNum)
        self.envNumEntry.insert(END, '2')
        self.envNum.set('2')
        self.envNumEntry.grid(row=1, column=0, sticky=E)
        # popluation label
        self.populationLabel = Label(self.master, text="Population")
        self.populationLabel.grid(row=2, column=0, sticky=W)
        self.population = IntVar()
        self.populationEntry = Entry(self.master, textvariable=self.population)
        self.populationEntry.insert(END, '300')
        self.population.set('300')
        self.populationEntry.grid(row=2, column=0, sticky=E)
        # file saver button
        self.fileSaverButton = Button(
            self.frame, text="save pool", command=self.saveFile)
        self.fileSaverButton.grid(row=2, column=1)
        self.fileLoaderButton = Button(
            self.frame, text="load pool", command=self.loadFile)
        self.fileLoaderButton.grid(row=2, column=2)
        # run button
        self.runButton = Button(
            self.frame, text="start run", command=self.toggleRun)
        self.runButton.grid(row=2, column=3)
        # play best button
        self.playBestButton = Button(
            self.frame, text='play best', command=self.handlePlayBest)
        self.playBestButton.grid(row=2, column=4)
        # uploadButton
        self.uploadButton = Button(
            self.frame, text="upload", command=self.handleUpload)
        self.uploadButton.grid(row=2, column=5)
        # attemps label
        self.attempsLabel = Label(self.master, text="attemps")
        self.attempsLabel.grid(row=3, column=0, sticky=W)
        self.attemps = IntVar()
        self.attempsEntry = Entry(self.master, textvariable=self.attemps)
        self.attempsEntry.insert(END, '1')
        self.attemps.set('1')
        self.attempsEntry.grid(row=3, column=0, sticky=E)
        # env label
        self.envLabel = Label(self.master, text="enviroment")
        self.envLabel.grid(row=4, column=0, sticky=W)
        self.envEntry = Entry(self.master)
        self.envEntry.insert(END, 'CartPole-v1')
        self.envEntry.grid(row=4, column=0, sticky=E)
        self.netProccess = None
        self.running = False
        self.poolInitialized = False
        self.pool = None
        self.genomeDictionary = {}
        self.plotData = {}
        self.specieID = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.stackplot([], [], baseline='wiggle')
        canvas = FigureCanvasTkAgg(self.fig, self.master)
        canvas.get_tk_widget().grid(row=5, column=0, rowspan=4, sticky="nesw")

    def updateStackPlot(self):

        for specie in self.pool.species:
            for genome in specie.genomes:
                foundSpecies = False
                for relative in genome.relatives:
                    if relative in self.genomeDictionary:
                        relativeGenome = self.pool.generations[relative[0]][relative[1]]
                        if self.pool.sameSpecies(genome,relativeGenome):
                            specieID = self.genomeDictionary[relative]
                            if len(self.plotData[specieID]) != self.pool.generation:
                                self.plotData[specieID].append(0)
                            self.plotData[specieID][self.pool.generation-1] +=1
                            foundSpecies = True
                if not foundSpecies:
                    self.genomeDictionary[genome.ID] = self.specieID
                    self.plotData[self.specieID] = [0] * self.pool.generation
                    self.plotData[self.specieID][self.pool.generation-1] += 1
                    self.specieID += 1
        for k,v in self.plotData.items():
            if len(v) != self.pool.generation:
                self.plotData[k].append(0)
        sortedPlots = sorted(self.plotData.keys())
        plotList = []
        for plot in sortedPlots:
            plotList.append(self.plotData[plot])
        self.ax.clear()
        self.ax.stackplot(
            list(range(len(plotList[0]))), *plotList, baseline='wiggle')
        canvas = FigureCanvasTkAgg(self.fig, self.master)
        canvas.get_tk_widget().grid(row=5, column=0, rowspan=5, sticky="nesw")

    def handleUpload(self):
        gym.upload('tmp/' + self.envEntry.get(),
                   api_key="")

    def handlePlayBest(self):
        playBest(self.pool.getBest(),self.envEntry.get())

    def toggleRun(self):
        env = gym.make(self.envEntry.get())
        if not self.running:
            if not self.poolInitialized:
                if env.action_space.__class__ == gym.spaces.discrete.Discrete:
                    actions = env.action_space.n - 1
                else:
                    actions = env.action_space.shape[0]
                if env.observation_space.__class__ == gym.spaces.discrete.Discrete:
                    observation = env.observation_space.n
                else:
                    observation = env.observation_space.shape[0]
                self.pool = neat.pool(
                    int(self.populationEntry.get()), observation, actions, recurrent=False,connectionCost=False)
                env.close()
                self.poolInitialized = True
            self.running = True
            self.runButton.config(text='running')
            self.master.after(250, self.checkRunPaused)
        else:
            self.running = False
            self.runButton.config(text='pausing')

    def checkRunPaused(self):
        if self.running:
            runQueue = Queue()
            self.pool.Population = self.population.get()
            self.netProcess = multiprocessing.Process(target=trainPool, args=(int(self.envNumEntry.get(
            )), self.pool.species, runQueue, self.envEntry.get(), int(self.attempsEntry.get())))
            self.netProcess.start()
            self.master.after(250, lambda: self.checkRunCompleted(
                runQueue, pausing=False))
        if not self.running:
            self.runButton.config(text='run')

    def onClosing(self):
        if messagebox.askokcancel("Quit", "do you want to Quit?"):
            self.master.destroy()
            self.master.quit()

    def checkRunCompleted(self, runQueue, pausing=True):
        try:
            msg = runQueue.get_nowait()
            if msg is not sentinel:
                self.netProcess.join()
                jobs = []
                for resultChunk in msg:
                    for result in resultChunk:
                        jobs.append(result)
                self.updateFitness(jobs)
                self.pool.nextGeneration()

                playBest(self.pool.getBest(),self.envEntry.get())
                
                print("gen ", self.pool.generation," best ", self.pool.getBest().fitness)
                self.updateStackPlot()
                
            if pausing:
                self.running = False
                self.master.after(
                    250, lambda: self.checkRunCompleted(runQueue, pausing))
                return
            else:
                self.master.after(250, self.checkRunPaused)
        except Empty:
            self.master.after(
                250, lambda: self.checkRunCompleted(runQueue, pausing))

    def updateFitness(self, jobs):
        pool = ThreadPool(4)
        pool.map(self.updateFitnessjob, jobs)

    def updateFitnessjob(self, job):
        currentSpecies = job[1][0]
        currentGenome = job[1][1]
        self.pool.species[currentSpecies].genomes[currentGenome].setFitness(
            job[0])

    def saveFile(self):
        if self.pool == None:
            return

        filename = filedialog.asksaveasfilename(defaultextension=".pool")
        if filename is None or filename == '':
            return
        file = open(filename, "wb")
        pickle.dump((self.pool.species, self.pool.best,
                     None,
                     None,
                     self.plotData,
                     self.genomeDictionary,
                     self.specieID, self.pool.generations), file)
        print("file saved",filename)

    def loadFile(self):
        filename = filedialog.askopenfilename()
        if filename is ():
            return
        f = open(filename, "rb")
        loadedPool = pickle.load(f)
        species = loadedPool[0]
        self.plotData = loadedPool[4]
        self.genomeDictionary = loadedPool[5]
        self.specieID = loadedPool[6]
        newInovation = 0
        for specie in species:
            for genome in specie.genomes:
                for gene in genome.genes:
                    if gene.innovation > newInovation:
                        newInovation = gene.innovation

        self.pool = neat.pool(sum([v for v in [len(specie.genomes) for specie in species]]),
                              species[0].genomes[0].Inputs, species[0].genomes[0].Outputs, recurrent=species[0].genomes[0].recurrent)
        self.pool.newGenome.innovation = newInovation + 1
        self.pool.species = species
        self.pool.best = loadedPool[1]
        self.pool.generation = len(self.pool.best)
        neat.pool.generations = loadedPool[7]
        self.population.set(self.pool.Population)
        self.poolInitialized = True
        f.close()
        self.ax.stackplot(
            list(range(len(self.plotData[0]))), *self.plotData, baseline='wiggle')
        canvas = FigureCanvasTkAgg(self.fig, self.master)
        canvas.get_tk_widget().grid(row=5, column=0, rowspan=5, sticky="nesw")
        print(filename, "loaded")




if __name__ == '__main__':

    root = Tk()
    root.resizable(width=False, height=False)
    app = gui(root)
    root.protocol("WM_DELETE_WINDOW", app.onClosing)
    root.mainloop()
