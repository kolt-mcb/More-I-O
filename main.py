from subprocess import check_output
import gym
from gym import wrappers
import os
import signal
import psutil
from lib import neat
from lib.networkDisplay import newNetworkDisplay
import numpy
import math
import time
import multiprocessing
import queue
from multiprocessing.pool import ThreadPool
from tkinter import *
from tkinter import filedialog, messagebox
import pickle
import matplotlib
#matplotlib.use('gg')
matplotlib.use('tkAGG')
import matplotlib.pyplot as plt
from operator import itemgetter
from ctypes import c_bool
import random

stackplotQueue = multiprocessing.Queue()
poolQueue = multiprocessing.Queue()
sharedRunning = multiprocessing.Value(c_bool,False)





def singleGame(randomQueue,displayQueue,env):
    env = gym.make(env)

    if env.env.__class__ == 'gym.envs.atari.atari_env.AtariEnv':
        atari = True
    if env.action_space.__class__ == gym.spaces.discrete.Discrete:  # identifies action/observation space
        discrete = True
    else:
        discrete = False

    while True:
        try:
            genome = randomQueue.get()
        except queue.Empty:
            time.sleep(0.5)
            pass
        done = False
        print("playing next")
        ob = env.reset()
        genome.generateNetwork()
        done = False
        while not done:
            ob = obFixer(env.observation_space, ob)
            # evalutes brain, getting button presses
            o = genome.evaluateNetwork(ob, discrete)
            o = acFixer(env.action_space, o)
            ob, reward, done, _ = env.step(o)
            if displayQueue.empty():
                    displayQueue.put(genome)
            env.render() # disabled render
            time.sleep(0.01)
            


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

def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    gone, still_alive = psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


def get_pid(name):
    return check_output(["pidof", name]).split()




class workerClass(object):
    def __init__(self,displayQueue,numJobs,env,population,attempts,input,output,sharedRunning,recurrnet=False,connectionCost=False,):
        self.lock = multiprocessing.Lock()
        self.jobs = multiprocessing.Queue()
        self.randomQueue = multiprocessing.Queue(maxsize=1)
        self.results = multiprocessing.Queue()
        self.proccesses = []
        self.numJobs = None
        self.initialized = multiprocessing.Value(c_bool,False)
        self.running = multiprocessing.Value(c_bool,False)
        self.runningNextGen = multiprocessing.Value(c_bool,False)
        self.counter = multiprocessing.Value('i',0)
        self.plotData = {}
        self.genomeDictionary = {}
        self.specieID = 0
        self.pool = neat.pool(population, input, output, recurrent=False,connectionCost=True)
        self.numJobs = numJobs
        self.env = env
        self.singleGame = None
        self.displayQueue = displayQueue
        self.attempts=attempts
        self.sharedRunning = sharedRunning


        
        
            
    def initializeProcess(self):
        
        if not self.initialized.value:
            for i in range(int(self.numJobs)):
                p = multiprocessing.Process(
                    target=self.jobTrainer,
                    args=(self.env,self.jobs,self.results,self.running,self.counter)
                    )
                self.proccesses.append(p)
                p.start()
            self.singleGame = multiprocessing.Process(target=singleGame, args=(self.randomQueue,self.displayQueue,self.env))
            self.singleGame.start()

        while True:
            if self.sharedRunning.value:
                if not self.initialized.value:
                    self.initialized.value = True
            if self.initialized.value:
                self.createJobs()
                self.sendResults()
            time.sleep(0.5)


    def createJobs(self):
        self.counter.value = 0
        s = 0
        for specie in self.pool.species:  # creates a job with species and genome pindex, env name and number of trials/attempts
            g = 0
            for genome in specie.genomes:
                self.jobs.put((s, g, genome))
                g += 1
            s += 1
        self.running.value= True
        


    def sendResults(self):
        results = []

        while self.initialized.value:
            while len(results) != self.pool.Population:
                if self.randomQueue.empty():
                    self.randomQueue.put(self.getRandomGenome())
                if not self.results.empty():
                    for result in self.results.get():
                        results.append(result)

            if len(results) == self.pool.Population:
                self.updateFitness(results)
                self.runningNextGen.value = True
                process  = multiprocessing.Process(target=self.randomQueueJob)
                process.start()
                self.pool.nextGeneration()
                self.runningNextGen.value = False
                process.join()
                stackplotQueue.put(self.generateStackPlot())
                poolQueue.put((self.pool,neat.pool.generations,self.plotData,self.genomeDictionary,self.specieID))
                print("gen ", self.pool.generation," best", self.pool.getBest().fitness)# sends message to main tkinter process
                self.initialized.value = False
            time.sleep(0.5)


    def getRandomGenome(self):
        randomSpecie = random.randint(0,len(self.pool.species)-1)
        species = self.pool.species[randomSpecie]
        randomGenome = random.randint(0,len(species.genomes)-1)
        genome = species.genomes[randomGenome]
        return genome

    def randomQueueJob(self):
        while self.runningNextGen.value:
            if self.randomQueue.empty():
                self.randomQueue.put(self.getRandomGenome())

    
    def generateStackPlot(self):
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
        return plotList





    def jobTrainer(self,envName,jobs,results,running,counter):
        job = None
        env = gym.make(envName)
        env.reset()
        resultsList = []
        resultsReady = False

        if env.action_space.__class__ == gym.spaces.discrete.Discrete:  # identifies action/observation space
            discrete = True
        else:
            discrete = False

        while True:
            if self.running.value:
                try: 
                    job = jobs.get(timeout=1)
                except queue.Empty as error:

                    if jobs.qsize() == 0: 
                        time.sleep(0.5)
                        self.lock.acquire()
                        counter.value += 1
                        self.lock.release()
                        resultsReady = True
                       
                        if counter.value == int(self.numJobs):
                            running.value = False
                        job = None
                        while running.value:
                            time.sleep(0.5)
                        pass
                    pass
                if job != None:
                    currentSpecies = job[0]
                    currentGenome = job[1]
                    genome = job[2]
                    scores = []
                    finalScore = 0
                    done = False
                    currentSpecies = job[0]
                    currentGenome = job[1]
                    genome = job[2]
                    scores = 0
                    for run in range(int(self.attempts)):  # runs for number of attempts
                        genome.generateNetwork()
                        score = 0
                        done = False
                        ob = env.reset()
                        while not done:
                            ob = obFixer(env.observation_space, ob)
                            # evalutes brain, getting button presses
                            o = genome.evaluateNetwork(ob, discrete)
                            o = acFixer(env.action_space, o)
                            ob, reward, done, _ = env.step(o)
                            #env.render() # disabled render
                            score += reward
                        scores += score
                    finalScore = round(scores / int(self.attempts))

                    resultsList.append((finalScore, job))
                    job = None
                    print("species:", currentSpecies, "genome:",currentGenome, "Scored:", finalScore)
            if resultsReady == True:
                results.put(resultsList)
                resultsList = []
                resultsReady = False
            time.sleep(0.5)



    def updateFitness(self,jobs):
        for job in jobs:
            self.updateFitnessjob(job)

    def updateFitnessjob(self,job):
        currentSpecies = job[1][0]
        currentGenome = job[1][1]
        self.pool.species[currentSpecies].genomes[currentGenome].setFitness(job[0])
            

class gui:
    def __init__(self, master):
        self.master = master
        self.frame = Frame(self.master, height=1100, width=500)
        self.frame.grid()
        # jobs label
        self.envLabel = Label(self.master, text="Jobs: ").grid(
            row=1, column=0, sticky=W)
        self.jobsEntry = Entry(self.master, textvariable=2)
        self.jobsEntry.insert(END, '2')
        self.jobsEntry.grid(row=1, column=0, sticky=E)
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
        # self.playBestButton = Button(
        #     self.frame, text='play best', command=self.handlePlayBest)
        # self.playBestButton.grid(row=2, column=4)
        # attempts label
        self.attemptsLabel = Label(self.master, text="attempts")
        self.attemptsLabel.grid(row=3, column=0, sticky=W)
        self.attempts = IntVar()
        self.attemptsEntry = Entry(self.master, textvariable=self.attempts)
        self.attemptsEntry.insert(END, '1')
        self.attempts.set('1')
        self.attemptsEntry.grid(row=3, column=0, sticky=E)
        # env label
        self.envLabel = Label(self.master, text="enviroment")
        self.envLabel.grid(row=4, column=0, sticky=W)
        self.envEntry = Entry(self.master)
        self.envEntry.insert(END, 'CartPole-v1')
        self.envEntry.grid(row=4, column=0, sticky=E)
        self.lock = multiprocessing.Lock()
        self.netProccess = None
        self.running = False
        self.poolInitialized = False
        self.firstRun = True
        self.workerClass = None
        self.pool = None
        self.plotData = {}
        self.genomeDictionary = {}
        self.specieID = 0
        self.displayQueue = multiprocessing.Queue(maxsize=1)
        self.display = None
        self.plt = plt
        self.fig, self.ax = self.plt.subplots()
        self.ax.set_xlabel('generations')
        self.ax.set_ylabel('number of genomes in a specie')
        self.plt.show(block=False)
        



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
                self.runButton.config(text='running')
                self.workerClass = workerClass(self.displayQueue,self.jobsEntry.get(),self.envEntry.get(),self.population.get(),self.attemptsEntry.get(), observation, actions,sharedRunning)
                self.display = newNetworkDisplay(self.displayQueue)
                # file saver button
                self.fileSaverButton = Button(
                self.frame, text="save pool", command=self.saveFile)
                self.fileSaverButton.grid(row=2, column=1)
            self.running = True
            self.runButton.config(text='running')
            self.checkRunPaused()
        else:
            self.running = False
            self.runButton.config(text='pausing')



    def checkRunPaused(self):
        if self.running:
            if not stackplotQueue.empty():
                plotList = stackplotQueue.get()
                self.ax.clear()
                self.ax.set_xlabel('generations')
                self.ax.set_ylabel('number of species in a specie')
                self.ax.stackplot(list(range(len(plotList[0]))), *plotList, baseline='wiggle')
                self.plt.show(block=False)
            if not poolQueue.empty():
                self.pool,self.generations,self.plotData,self.genomeDictionary,self.specieID = poolQueue.get()
            if self.firstRun:
                self.netProcess = multiprocessing.Process(target=self.workerClass.initializeProcess)
                self.poolInitialized=True
                self.netProcess.start()
                self.firstRun = False
            sharedRunning.value = True
            self.master.after(250,self.checkRunPaused)
        if not self.running:
            sharedRunning.value=False
            self.runButton.config(text='run')



    def onClosing(self):
        if messagebox.askokcancel("Quit", "do you want to Quit?"):
            for child in multiprocessing.active_children():
                kill_proc_tree(child.pid)
            self.master.destroy()
            self.master.quit()



    def saveFile(self):
        if self.pool == None:
            return
        filename = filedialog.asksaveasfilename(defaultextension=".pool")
        if filename is None or filename == '':
            return
        file = open(filename, "wb")
        pickle.dump({"species" : self.pool.species,
                    "best"     : self.pool.best,
                    "plotData" : self.plotData,
                    "specieID" : self.specieID,
                    "genomeDictionary" : self.genomeDictionary,
                    "generations" : self.generations}, file)

        print("file saved",filename)

    def loadFile(self):
        filename = filedialog.askopenfilename()
        if filename is ():
            return
        f = open(filename, "rb")
        loadedPool = pickle.load(f)
        species = loadedPool["species"]
        
        newInovation = 0
        for specie in species:
            for genome in specie.genomes:
                for gene in genome.genes:
                    if gene.innovation > newInovation:
                        newInovation = gene.innovation

        self.workerClass = workerClass(self.displayQueue,self.envNum.get(),
                                        self.env,
                                        sum([v for v in [len(specie.genomes) for specie in species]]),
                                        species[0].genomes[0].Inputs,
                                        species[0].genomes[0].Outputs)

        self.workerClass.pool.newGenome.innovation = newInovation + 1
        self.workerClass.pool.species = species
        self.workerClass.pool.best = loadedPool["best"]
        self.best = loadedPool["best"]
        self.workerClass.pool.generation = len((self.workerClass.pool.best))
        self.workerClass.plotData = loadedPool["plotData"]
        self.workerClass.genomeDictionary = loadedPool["genomeDictionary"]
        self.workerClass.specieID = loadedPool["specieID"]
        neat.pool.generations = loadedPool["generations"]
        self.population.set(self.workerClass.pool.Population)
        self.display = newNetworkDisplay(self.displayQueue)
        if not self.poolInitialized:
            # file saver button
            self.fileSaverButton = Button(
            self.frame, text="save pool", command=self.saveFile)
            self.fileSaverButton.grid(row=2, column=1)
        self.poolInitialized = True
        f.close()

        print(filename, "loaded")

    








if __name__ == '__main__':

    root = Tk()
    root.resizable(width=False, height=False)
    app = gui(root)
    root.protocol("WM_DELETE_WINDOW", app.onClosing)
    root.mainloop()
