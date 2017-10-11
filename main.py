from subprocess import check_output
import gym
from gym import wrappers
import os,signal,psutil
from lib import neat 
from lib import networkDisplay
import numpy
import math
import time 
import multiprocessing
from multiprocessing import Queue
from queue import Empty
from tkinter import *
from tkinter import filedialog,messagebox
import pickle
from queue import Empty
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from operator import itemgetter




sentinel = object() # tells the main tkinter window if a generattion is in progress

lock = multiprocessing.Lock() # lock needed for env process

def poolInitializer(q): # manages the job queue for genomes/players
	global jobs
	jobs = q


def trainPool(envNum,species,queue,env,attemps): # creates multiprocessing job for pool and trains pool 
    before = time.time()
    results = []
    jobs = Queue()
    s = 0
    for specie in species: # generates network for each genome and creates a job with species and genome index, env name and number of trials/attemps
      g=0
      for genome in specie.genomes:
        genome.generateNetwork()
        jobs.put((s,g,genome,attemps))
        g+=1
      s+=1
    mPool = multiprocessing.Pool(processes=envNum,initializer =poolInitializer,initargs=(jobs,))
    results = mPool.map(jobTrainer,[env]*envNum)
    mPool.close()
    mPool.join()
    
    for resultChunk in results:
        for result in resultChunk:
            currentSpecies = result[1][0]
            currentGenome = result[1][1]
            species[currentSpecies].genomes[currentGenome].fitness = result[0] # sets results from result list
    after = time.time()
    print("finished in ",int(after-before))

    queue.put(species) # sends message to main tkinter process
   

def obFixer(observation_space,observation): # fixes observation ranges, uses hyperbolic tangent for infinite values
  newObservation = []
  if observation_space.__class__ == gym.spaces.box.Box:
    for space in range(observation_space.shape[0]):
      high = observation_space.high[space]
      low = observation_space.low[space]
      if high == numpy.finfo(numpy.float32).max or high == float('Inf'):
        newObservation.append(math.tanh(observation[space]))
      else:
        dif = high - low 
        percent = (observation[space]+abs(low))/dif
        newObservation.append(((percent * 2)-1).tolist())


  return newObservation

def acFixer(action_space,action): # fixes action ranges, uses hyperbolic tangent for infinite values
  newAction = []
  if action_space.__class__ == gym.spaces.box.Box:
    for space in range(action_space.shape[0]):
      high = action_space.high[space]
      low = action_space.low[space]
      if high != numpy.finfo(numpy.float32).max:
        dif = high - low 
        percent = (action[space]+1)/2
        newAction.append(((percent * dif)-(dif/2)).tolist())
      else:
        print("if your seeing me this code needs updated")
        newAction.append(math.tanh(observation[space]))
    return newAction
  if action_space.__class__ == gym.spaces.discrete.Discrete:
    c = 0
    for _,myAction in enumerate(action):
      if myAction > 0:
        newAction.append(_+1)
        c += 1
      if c > 1:
        newAction = [0]
        return int(newAction[0])
    if c == 0:
      newAction = [0]
    return int(newAction[0])





def jobTrainer(envName):

	env = gym.make(envName)
	env = wrappers.Monitor(env,'tmp/'+envName,resume=True,video_callable=False) # no recoding on windows due to ffmepg


	if env.action_space.__class__ == gym.spaces.discrete.Discrete: # identifies action/observation space
		discrete = True
	else:
		discrete = False
	
	results = []
	
	while not jobs.empty(): # gets a new player index from queue
		try: 
			job = jobs.get()
		except Empty:
			pass

		currentSpecies = job[0]
		currentGenome = job[1]
		genome = job[2]
		attemps = job[3]
		scores = 0
		for run in range(attemps): # runs for number of attemps
			score = 0
			done = False
			ob = env.reset()	
			
			while not done: 
			  ob = obFixer(env.observation_space,ob)
			  o = genome.evaluateNetwork(ob,discrete) # evalutes brain, getting button presses
			  o = acFixer(env.action_space,o)
			  ob, reward, done, _ = env.step(o)
			  #env.render() # disabled render
			  score += reward
			
			scores += score
		finalScore = round(scores/attemps)	
		print("species:",currentSpecies, " genome:",currentGenome," Scored:",finalScore)
		results.append((finalScore,job))
	env.close()
	return results

def singleGame(genome,genomePipe,envName,eval):
	env = gym.make(envName)
	env = wrappers.Monitor(env,'tmp/'+envName,resume=True)
	runs = 1
	print("playing best")
	if eval:
		#env = wrappers.Monitor(env,'tmp/'+envName,resume=True,video_callable=False)
		runs = 100
	for i in range(runs):
		if env.action_space.__class__ == gym.spaces.discrete.Discrete:
			discrete = True
		else:
			discrete = False
		ob = env.reset()
		done = False
		distance = 0
		maxDistance = 0
		staleness = 0
		score = 0 
		while not done:
			ob = obFixer(env.observation_space,ob)
			o = genome.evaluateNetwork(ob,discrete)
			o = acFixer(env.action_space,o)
			ob, reward, done, _ = env.step(o)
			score += reward
			if not eval:
				env.render()
			genomePipe.send(genome)
		print(score)
	env.reset
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
    self.frame = Frame(self.master,height=1000,width=450)
    self.frame.grid()
    #jobs label
    self.envLabel = Label(self.master,text="Jobs: ").grid(row=1,column=0,sticky=W)
    self.envNum = IntVar()
    self.envNumEntry  = Entry(self.master,textvariable=self.envNum)
    self.envNumEntry.insert(END,'2')
    self.envNum.set('2')
    self.envNumEntry.grid(row=1,column=0,sticky=E)
    #popluation label
    self.populationLabel = Label(self.master,text="Population")
    self.populationLabel.grid(row=2,column=0,sticky=W)
    self.population = IntVar()
    self.populationEntry = Entry(self.master,textvariable=self.population)
    self.populationEntry.insert(END,'300')
    self.population.set('300')
    self.populationEntry.grid(row=2,column=0,sticky=E)
    #file saver button
    self.fileSaverButton = Button(self.frame,text="save pool",command=self.saveFile)
    self.fileSaverButton.grid(row=2,column=1)
    self.fileLoaderButton = Button(self.frame,text="load pool", command=self.loadFile)
    self.fileLoaderButton.grid(row=2,column=2)
    #run button
    self.runButton = Button(self.frame,text="start run", command=self.toggleRun)
    self.runButton.grid(row=2,column=3)
    #play best button
    self.playBestButton = Button(self.frame,text='play best',command =self.playBest)
    self.playBestButton.grid(row=2,column=4)
    #uploadButton 
    self.uploadButton = Button(self.frame,text="upload",command=self.handleUpload)
    self.uploadButton.grid(row=2,column=5)
    #attemps label
    self.attempsLabel = Label(self.master,text="attemps")
    self.attempsLabel.grid(row=3,column=0,sticky=W)
    self.attemps = IntVar()
    self.attempsEntry = Entry(self.master,textvariable=self.attemps)
    self.attempsEntry.insert(END,'1')
    self.attemps.set('1')
    self.attempsEntry.grid(row=3,column=0,sticky=E)
    #env label
    self.envLabel = Label(self.master,text="enviroment")
    self.envLabel.grid(row=4,column=0,sticky=W)
    self.envEntry = Entry(self.master)
    self.envEntry.insert(END,'CartPole-v1')
    self.envEntry.grid(row=4,column=0,sticky=E)
    self.netProccess = None
    self.running= False
    self.poolInitialized = False
    self.pool = None
    self.lastPopulation = []
    self.plotDictionary = {}
    self.plotData = []
    self.genomeDictionary = {}
    self.specieID = 0
    self.fig,self.ax = plt.subplots(figsize=(10,6))
    self.ax.stackplot([],[],baseline='wiggle')
    canvas = FigureCanvasTkAgg(self.fig,self.master)
    canvas.get_tk_widget().grid(row=5,column=0,rowspan=4,sticky="nesw")
	
	
	
  def updateStackPlot(self,species):
    if self.lastPopulation == []:
        for specie in species:
            genome = specie.genomes[0] 
            self.plotDictionary[self.specieID] = len(specie.genomes)
            self.genomeDictionary[genome] = self.specieID
            self.specieID += 1
    else:
        self.plotDictionary = dict.fromkeys(self.plotDictionary,0)
        for specie in species:
            for genome in specie.genomes:
                foundSpecies = False
                for oldSpecie,specieID in self.genomeDictionary.items():
                    definingGenome = oldSpecie
                    if not foundSpecies and self.pool.sameSpecies(genome,definingGenome):
                        specieID = self.genomeDictionary[definingGenome]
                        self.plotDictionary[specieID] +=1
                        foundSpecies = True
                if not foundSpecies:
                    definingGenome = specie.genomes[0]
                    if self.genomeDictionary.get(definingGenome,None) != None:
                        specieID = self.genomeDictionary[definingGenome]
                        self.plotDictionary[specieID] +=1
                    else:
                        self.plotDictionary[self.specieID] = 1
                        self.genomeDictionary[definingGenome] = self.specieID
                        self.specieID +=1
    self.lastPopulation = species

    for genome,specieID in sorted(self.genomeDictionary.items(),key=itemgetter(1)):
        speciesLen = self.plotDictionary[specieID]
        if speciesLen == 0 :
          del self.plotDictionary[specieID]
          del self.genomeDictionary[genome]

        if len(self.plotData) <= specieID:
            if len(self.plotData) == 0:
                self.plotData.append([])
            else:
                self.plotData.append([0]*(len(self.plotData[0])-1))
            self.plotData[specieID].append(speciesLen)
        else:
            self.plotData[specieID].append(speciesLen)
    for specieArray in self.plotData:
      if len(specieArray) != self.pool.generation:
        specieArray.append(0)
    self.ax.clear()
    self.ax.stackplot(list(range(len(self.plotData[0]))),*self.plotData,baseline='wiggle')
    canvas = FigureCanvasTkAgg(self.fig,self.master)
    canvas.get_tk_widget().grid(row=5,column=0,rowspan=5,sticky="nesw")
	
	
	
  def handleUpload(self):
    gym.upload('tmp/'+self.envEntry.get(),api_key="sk_8j3LQ561SH20sk0YN3qpg")


  def toggleRun(self):
    env = gym.make(self.envEntry.get())
	
    if not self.running:
      if not self.poolInitialized:
        if env.action_space.__class__ == gym.spaces.discrete.Discrete:
           actions = env.action_space.n -1
        else:
           actions = env.action_space.shape[0]
        if env.observation_space.__class__ == gym.spaces.discrete.Discrete:
           observation = env.observation_space.n 
        else:
           observation = env.observation_space.shape[0]
        self.pool = neat.pool(int(self.populationEntry.get()),observation,actions,recurrent=False)
        env.close()
        self.poolInitialized = True
      self.running = True
      self.runButton.config(text='running')
      self.master.after(250,self.checkRunPaused)
    else:
      self.running = False
      self.runButton.config(text='pausing')


  def checkRunPaused(self):
    if self.running:
      queue = multiprocessing.Queue()
      self.pool.Population = self.population.get()
      self.netProcess = multiprocessing.Process(target=trainPool,args=(int(self.envNumEntry.get()),self.pool.species,queue,self.envEntry.get(),int(self.attempsEntry.get())))
      self.netProcess.start()
      self.master.after(250,lambda: self.checkRunCompleted(queue,pausing=False))
    if not self.running:
      self.runButton.config(text='run')
         


  def onClosing(self):
    if messagebox.askokcancel("Quit","do you want to Quit?"):
      for child in multiprocessing.active_children():
        kill_proc_tree(child.pid)
      self.master.destroy()
      self.master.quit()



  def checkRunCompleted(self,queue,pausing=True):
    try:
        msg = queue.get_nowait()
        if msg is not sentinel:
          self.pool.species = msg
          self.netProcess.join()
          print("next generation")
          self.pool.nextGeneration() # applies rewards and breeds new species
          print("gen " ,self.pool.generation," best", self.pool.getBest().fitness)
          #self.updateStackPlot(self.pool.species)
          self.playBest(eval = False)
          if pausing:
            self.running = False
            self.master.after(250,lambda: self.checkRunCompleted(queue,pausing))
            return
          else:
           self.master.after(250,self.checkRunPaused)
        self.master.after(250,lambda: self.checkRunCompleted(queue,pausing))
    except Empty:
        self.master.after(250,lambda: self.checkRunCompleted(queue,pausing))

  def saveFile(self):
    if self.pool == None:
      return

    filename = filedialog.asksaveasfilename(defaultextension=".pool")
    if filename is None or filename == '':
      return
    file = open(filename,"wb")
    pickle.dump((self.pool.species,self.pool.best,
                 self.lastPopulation,
                 self.plotDictionary,
                 self.plotData,
                 self.genomeDictionary,
                 self.specieID),file)

  def loadFile(self):
    filename = filedialog.askopenfilename()
    if filename is ():
      return
    f = open(filename,"rb")
    loadedPool = pickle.load(f)
    species  = loadedPool[0]
    self.lastPopulation = loadedPool[2]
    self.plotDictionary = loadedPool[3]
    self.plotData = loadedPool[4]
    self.genomeDictionary = loadedPool[5]
    self.specieID = loadedPool[6]
    newInovation = 0
    for specie in species:
      for genome in specie.genomes:
        for gene in genome.genes:
          if gene.innovation > newInovation:
            newInovation = gene.innovation
    
    self.pool = neat.pool(sum([v for v in [len(specie.genomes) for specie in species]]),species[0].genomes[0].Inputs,species[0].genomes[0].Outputs,recurrent=species[0].genomes[0].recurrent)
    self.pool.newGenome.innovation = newInovation +1
    self.pool.species = species
    self.pool.best = loadedPool[1]
    self.pool.generation = len(self.pool.best)
    self.population.set(self.pool.Population)
    self.poolInitialized = True
    f.close()
	
  def playBest(self,eval=True):
    parentPipe, childPipe = multiprocessing.Pipe()
    genome = self.pool.getBest()
    process = multiprocessing.Process(target = singleGame,args=(genome,childPipe,self.envEntry.get(),eval))
    process.start()
    display = networkDisplay.newNetworkDisplay(genome,parentPipe)
    display.checkGenomePipe()
    display.Tk.mainloop()
    process.join()




if __name__ == '__main__':
    
    root = Tk()
    root.resizable(width=False,height=False)
    app = gui(root)
    root.protocol("WM_DELETE_WINDOW",app.onClosing)
    root.mainloop()



           
           
           











        

