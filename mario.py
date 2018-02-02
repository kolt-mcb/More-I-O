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
import queue
from multiprocessing.pool import ThreadPool
from tkinter import *
from tkinter import filedialog, messagebox
import pickle
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from operator import itemgetter
from ctypes import c_bool
resultQueue = multiprocessing.Queue()



def joystick(four):
    six = [0] * 6
    if four[0] > 0.5:
        six[0] = 0
        six[2] = 1
    if four[0] < -0.5:
        six[0] = 1
        six[2] = 0
    if four[0] < 0.5 and four[0] > -0.5:
        six[0] = 0
        six[2] = 0

    if four[1] >= 0.5:
        six[1] = 0
        six[3] = 1
    if four[1] <= -0.5:
        six[1] = 1
        six[3] = 0
    if four[1] < 0.5 and four[1] > -0.5:
        six[1] = 0
        six[3] = 0

    if four[2] >= 0.5:
        six[4] = 1
    if four[2] <= -0.5:
        six[4] = -1
    if four[2] < 0.5 and four[2] > -0.5:
        six[4] = 0

    if four[3] >= 0.5:
        six[5] = 1
    if four[3] <= -0.5:
        six[5] = -1
    if four[3] < 0.5 and four[3] > -0.5:
        six[5] = 0
    return six


#starts a new game with the network display.
def playBest(genome):
        parentPipe, childPipe = multiprocessing.Pipe()
        genome.generateNetwork()
        process = multiprocessing.Process(target=singleGame, args=(genome,childPipe))
        process.start()
        display = networkDisplay.newNetworkDisplay(genome, parentPipe)
        display.checkGenomePipe()
        display.Tk.mainloop()
        process.join()
        # creates multiprocessing job for pool and trains pool



def singleGame(genome, genomePipe):
    env = gym.make('meta-SuperMarioBros-Tiles-v0')
    env.reset()
    done = False
    distance = 0
    maxDistance = 0
    staleness = 0
    print("playing next")
    env.locked_levels = [False] * 32
    for LVint in range(1):
        genome.generateNetwork()
        maxDistance = 0
        staleness = 0
        oldDistance = 0
        done = False
        bonus = 0
        bonusOffset = 0

        #env.is_finished = True

        env.change_level(new_level=LVint)
        # env._write_to_pipe("changelevel#"+str(LVint))
        while not done:
            ob = env.tiles.flatten()

            o = genome.evaluateNetwork(ob.tolist(), discrete=False)
            o = joystick(o)
            genomePipe.send(genome)
            ob, reward, done, _ = env.step(o)
            if 'ignore' in _:
                done = False
                env = gym.make('meta-SuperMarioBros-Tiles-v0')
                env.reset()
                env.locked_levels = [False] * 32
                env.change_level(new_level=LVint)
            distance = env._get_info()["distance"]
            if oldDistance - distance < -100:
                bonus = maxDistance
                bonusOffset = distance
            if maxDistance - distance > 50 and distance != 0:
                maxDistance = distance
            if distance > maxDistance:
                maxDistance = distance
                staleness = 0
            if maxDistance >= distance:
                staleness += 1

            if staleness > 100 or done:
                if not done:
                    done = True
            oldDistance = distance
    env.close()
    genomePipe.send("quit")
    genomePipe.close()

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


def killFCEUX():
    for pid in get_pid("fceux"):
        pid = int(pid)
        os.kill(pid, signal.SIGKILL)



class workerClass(object):
    def __init__(self,numJobs,running,env,population,input,output,recurrnet=False,connectionCost=False):
        self.pool = self.pool = neat.pool(population, input, output, recurrent=False,connectionCost=False)
        self.lock = multiprocessing.Lock()
        self.jobs = multiprocessing.Queue()
        self.results = multiprocessing.Queue()
        self.numJobs = numJobs
        self.env = env
        self.proccesses = []
        self.initialized = False
        self.running = running
        self.counter = multiprocessing.Value('i',0)
    
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

            
    def initializeProcess(self):
        for i in range(self.numJobs):
            p = multiprocessing.Process(
                target=self.jobTrainer,
                args=([self.env])
                )
            self.proccesses.append(p)
            p.start()
        
        while self.running.value:
            if not self.initialized:
                self.initialized = True
                self.startRun()
            time.sleep(1)


    def startRun(self):
        self.counter.value = 0
        c2 = 0
        for specie in self.pool.species:
            for genome in specie.genomes:
                c2 += 1
        print(c2)
        self.createJobs()
        


    def createJobs(self):
        s = 0
        for specie in self.pool.species:  # creates a job with species and genome index, env name and number of trials/attemps
            g = 0
            for genome in specie.genomes:
                self.jobs.put((s, g, genome))
                g += 1
            s += 1


    def sendResults(self):
        self.updateFitness(msg)
        self.pool.nextGeneration()
        playBest(self.pool.getBest())
        print("gen ", self.pool.generation," best", self.pool.getBest().fitness)# sends message to main tkinter process
        self.initialized = False


    def jobTrainer(self,envName):
        job =None
        env = gym.make(envName)
        env.lock = self.lock
        env.lock.acquire()
        env.reset()
        env.lock.release()
        env.locked_levels = [False] * 32
        running = True
        c = 0
        while True:
            if self.running.value:
                print("try get job?")
                try: 
                    job = self.jobs.get_nowait()
                except queue.Empty: 
                    time.sleep(0.5)
                    self.counter.value += 1
                    print(self.counter.value)
                    if self.counter.value == self.numJobs:
                        print("sending")
                        self.sendResults()
                        self.running.value = False
                    job = None
                    while self.running.value:
                        time.sleep(0.5)
                    pass

                
                if job != None:
                    c+=1
                    currentSpecies = job[0]
                    currentGenome = job[1]
                    genome = job[2]
                    maxDistance = 0
                    distance = None
                    staleness = 0
                    scores = []
                    finalScore = 0
                    done = False
                    maxReward = 0
                    for LVint in range(1):
                        genome.generateNetwork()
                        maxDistance = 0
                        oldDistance = 0
                        bonus = 0
                        bonusOffset = 0
                        staleness = 0
                        done = False
                        env.change_level(new_level=LVint)
                        while not done:
                            ob = env.tiles.flatten()
                            o = genome.evaluateNetwork(ob.tolist(), discrete=False)
                            o = joystick(o)
                            ob, reward, done, _ = env.step(o)
                            if 'ignore' in _:
                                done = False
                                env = gym.make('meta-SuperMarioBros-Tiles-v0')
                                env.lock.acquire()
                                env.reset()
                                env.locked_levels = [False] * 32
                                env.change_level(new_level=LVint)
                                env.lock.release()
                            distance = env._get_info()["distance"]
                            if oldDistance - distance < -100:
                                bonus = maxDistance
                                bonusOffset = distance
                            if maxDistance - distance > 50 and distance != 0:
                                maxDistance = distance
                            if distance > maxDistance:
                                maxDistance = distance
                                staleness = 0
                            if maxDistance >= distance:
                                staleness += 1

                            if staleness > 80 or done:
                                scores.append(maxDistance - bonusOffset + bonus)
                                if not done:
                                    done = True
                            oldDistance = distance
                        for score in scores:
                            finalScore += score
                        finalScore = round(finalScore / 32)
                        
                    print(job)
                    self.results.put((finalScore, job))
                    job = None
                    print("species:", currentSpecies, "genome:",currentGenome, "Scored:", finalScore,c)
            time.sleep(1)
        
            

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
        self.netProccess = None
        self.running = False
        self.poolInitialized = False
        self.pool = None
        self.env = 'meta-SuperMarioBros-Tiles-v0'
        self.plotData = {}
        self.genomeDictionary = {}
        self.specieID = 0
        self.fig, self.ax = plt.subplots(figsize=(3.7, 3))
        self.ax.stackplot([], [], baseline='wiggle')
        canvas = FigureCanvasTkAgg(self.fig, self.master)
        canvas.get_tk_widget().grid(row=5, column=0, rowspan=4, sticky="nesw")
        self.sentinel = object()  # tells the main tkinter window if a generattion is in progress
        self.workerClass = None
        self.firstRun = True
        self.sharedRunning = multiprocessing.Value(c_bool,False)
        self.sharedPopulation = multiprocessing.Value('i',self.population.get())

        

    def handlePlayBest(self):
        playBest(self.pool.getBest(),self.envEntry.get())

    
    


    

    def updateStackPlot(self,plotList):
        self.ax.clear()
        self.ax.stackplot(
            list(range(len(plotList[0]))), *plotList, baseline='wiggle')
        canvas = FigureCanvasTkAgg(self.fig, self.master)
        canvas.get_tk_widget().grid(row=5, column=0, rowspan=5, sticky="nesw")


    def toggleRun(self):

        if not self.running:
            if not self.poolInitialized:
                self.runButton.config(text='running')
                self.workerClass = workerClass(self.envNum.get(),self.sharedRunning,self.env,self.population.get(), 208, 4)
            self.running = True
            self.runButton.config(text='running')
            self.master.after(250, self.checkRunPaused)
        else:
            self.running = False
            self.runButton.config(text='pausing')

    def checkRunPaused(self):
        if self.running:
            self.sharedPopulation.value = self.population.get()
            if self.firstRun:
                self.netProcess = multiprocessing.Process(target=self.workerClass.initializeProcess)
                self.netProcess.start()
                self.firstRun = False
            self.sharedRunning.value = True
            self.master.after(
                250, lambda: self.checkRunCompleted(pausing=False))
        if not self.running:
            self.runButton.config(text='run')
            self.shared

    def onClosing(self):
        if messagebox.askokcancel("Quit", "do you want to Quit?"):
            for child in multiprocessing.active_children():
                kill_proc_tree(child.pid)
            if self.running:
                killFCEUX()
            self.master.destroy()
            self.master.quit()

    def checkRunCompleted(self, pausing=True):
        try:
            msg = resultQueue.get()
        except queue.Empty:
            self.master.after(250, lambda: self.checkRunCompleted(pausing))
        if msg is not self.sentinel:
            self.updateStackPlot(self.workerClass.generateStackPlot())
        if pausing:
            self.running = False
            self.master.after(250,self.checkRunCompleted)
            return
        else:
            self.master.after(250, self.checkRunPaused)


    

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
                    "generations" : self.pool.generations}, file)

        print("file saved",filename)

    def loadFile(self):
        filename = filedialog.askopenfilename()
        if filename is ():
            return
        f = open(filename, "rb")
        loadedPool = pickle.load(f)
        species = loadedPool["species"]
        self.plotData = loadedPool["plotData"]
        self.genomeDictionary = loadedPool["genomeDictionary"]
        self.specieID = loadedPool["specieID"]
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
        self.pool.best = loadedPool["best"]
        self.pool.generation = len(self.pool.best)
        neat.pool.generations = loadedPool["generations"]
        self.population.set(self.pool.Population)
        self.poolInitialized = True
        f.close()

        print(filename, "loaded")


    def updateFitness(self,jobs):
        for job in jobs:
            self.updateFitnessjob(job)

    def updateFitnessjob(self,job):
        currentSpecies = job[1][0]
        currentGenome = job[1][1]
        self.workerClass.pool.species[currentSpecies].genomes[currentGenome].setFitness(job[0])




if __name__ == '__main__':

    root = Tk()
    root.resizable(width=False, height=False)
    app = gui(root)
    root.protocol("WM_DELETE_WINDOW", app.onClosing)
    root.mainloop()
