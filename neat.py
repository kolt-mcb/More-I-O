from operator import attrgetter
import sys
import time
import random
import math
from tkinter import *



CrossoverChance = 0.75
PerturbChance = 0.90
MutateConnectionsChance = 0.25
LinkMutationChance = 2.0
BiasMutationChance = 0.40
NodeMutationChance = .25
EnableMutationChance = 0.2
DisableMutationChance = 0.4
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0
StepSize = 0.1
Outputs = None
MaxNodes = 10000
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
GRIDSIZE=80
GRID_WIDTH = SCREEN_WIDTH // GRIDSIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRIDSIZE
Inputs = None
StaleSpecies = 15
pool = None

width = 720
height = 240



class networkDisplay(Frame):
    def __init__(self):
        Frame.__init__(self)
        #Set up the main window frame as a grid
        self.grid()     
        #Set up main frame for game as a grid
        frame1 = Frame(self, width = width, height = height)
        frame1.grid()
        #Add a canvas to frame1 as self.canvas member 
        self.canvas = Canvas(frame1, width = width, height = height,bg ="white")
        
nd= networkDisplay()             
class Cell():
    def __init__(self):
        self.x= None
        self.y = None
        self.value = None
        
        
class newPool(): #holds all species data
    def __init__(self):
        self.species = []
        self.generation = 0
        self.currentSpecies = 0
        self.currentGenome = 0
        self.maxFitness = 0
        self.innovation = Outputs
        self.Population = None

class newSpecies():
    def __init__(self):
        self.topFitness = 0
        self.staleness = 0
        self.genomes = []
        self.averageFitness = 0
   
class newGenome():
    def __init__(self):
        self.genes = []
        self.fitness = 0
        self.neurons = {}
        self.maxneuron = Inputs
        self.mutationRates = {}
        self.globalRank = 0
        self.buttonPresses = {}
        self.mutationRates["connections"] =  MutateConnectionsChance
        self.mutationRates["link"] =  LinkMutationChance
        self.mutationRates["bias"] = BiasMutationChance
        self.mutationRates["node"] = NodeMutationChance
        self.mutationRates["enable"] = EnableMutationChance
        self.mutationRates["disable"] = DisableMutationChance
        self.mutationRates["step"] = StepSize
        newGenome.__eq__ = lambda self, other: self.fitness == other.fitness
        newGenome.__ne__ = lambda self, other: self.fitness != other.fitness
        newGenome.__lt__ = lambda self, other: self.fitness < other.fitness
        newGenome.__le__ = lambda self, other: self.fitness <= other.fitness
        newGenome.__gt__ = lambda self, other: self.fitness > other.fitness
        newGenome.__ge__ = lambda self, other: self.fitness >= other.fitness

class newGene():
    def __init__(self):
        self.into = 0
        self.out = 0
        self.weight = 0.0
        self.innovation = 0
        self.enabled = True

class newNeuron():
    def __init__(self):
        self.incoming = []
        self.value = 0.0
   
def newInnovation():
    
    pool.innovation += 1
    return pool.innovation

def addToSpecies(child):
    foundSpecies = False
    generateNetwork(child)
    for specie in pool.species:
        if not foundSpecies and sameSpecies(child,specie.genomes[0]):
            specie.genomes.append(child)
            foundSpecies = True
    if not foundSpecies:
        childSpecies = newSpecies()
        childSpecies.genomes.append(child)
        pool.species.append(childSpecies)
  
def basicGenome():
    genome = newGenome()
      
    genome.maxneuron = Inputs
    mutate(genome)
    return genome 
  
def initializePool(Population,_Input,_Outputs):
    
    setInputOutput(_Input,_Outputs)
    global pool
    pool = newPool()
    pool.Population = Population
    for x in range(pool.Population):
        basic = basicGenome()
        addToSpecies(basic)        
                          
def mutate(genome):
    
    for mutation,rate in genome.mutationRates.items():
        if random.randint(1,2) == 1:
            genome.mutationRates[mutation] = 0.95*rate
        else:
            genome.mutationRates[mutation] = 1.05263*rate
    if random.random() < genome.mutationRates["connections"]:
        pointMutate(genome)
    p = genome.mutationRates["link"]
    while p > 0:
        if random.random() < p:
            linkMutate(genome, False)
            p -= 1
    p = genome.mutationRates["bias"]
    while p > 0:
        if random.random() < p:
            linkMutate(genome,True)
            p -= 1
    p = genome.mutationRates["node"] 
    while p > 0:
        if random.random() < p:
            nodeMutate(genome)
            p -= 1
    p = genome.mutationRates["enable"]
    while p > 0:
        if random.random() < p:
            enableDisableMutate(genome, True)
            p -= 1
    p = genome.mutationRates["disable"]
    while p > 0:
        if random.random() < p:
            enableDisableMutate(genome, False)
            p -= 1
               
def pointMutate(genome):
    step = genome.mutationRates["step"]
    for gene in genome.genes:
        if random.random() < PerturbChance:
            gene.weight = gene.weight + random.random()*step*2-step
        else:
            gene.weight = random.random()*4-2

def linkMutate(genome,forceBias):
    neuron1 = randomNeuron(genome.genes,False)
    neuron2 = randomNeuron(genome.genes,True)
    newLink = newGene()
    if neuron1 <= Inputs and neuron2 <= Inputs:
        return
    if neuron2 <= Inputs:
        temp = neuron1
        neuron1 = neuron2
        neuron2 = temp
        
    newLink.into = neuron1
    newLink.out = neuron2
    
    if forceBias:
        newLink.into = Inputs
    if containsLink(genome.genes,newLink):
        return 
    
    newLink.innovation = newInnovation()
    newLink.weight = (random.random()*4-2)
    genome.genes.append(newLink)
        
def containsLink(genes,link):
    
    for gene in genes:
        if gene.into == link.into and gene.out == link.out:
            return True
            
def nodeMutate(genome):

    if len(genome.genes) == 0:
        return
    genome.maxneuron += 1
    
    for k,_ in enumerate(genome.genes):
        r = random.randint(0,k)
        gene = genome.genes[r]
    if not gene.enabled:
        return
    
    gene.enabled = False
    
    gene1 = copyGene(gene)
    gene1.out = genome.maxneuron
    gene1.weight = 1.0
    gene1.innovation = newInnovation()
    gene1.enabled = True
    genome.genes.append(gene1)
    
    gene2 = copyGene(gene)
    gene2.out = genome.maxneuron
    gene2.innovation = newInnovation()
    gene2.enabled = True
    genome.genes.append(gene2)
    
def enableDisableMutate(genome, enable):
    candidates = []
    
    for g in genome.genes:
        if g.enabled == (not enable):
            candidates.append(g)
    if len(candidates) == 0:
        return
    r = random.randint(0,len(candidates)-1)
    gene = candidates[r]
    gene.enabled = (not gene.enabled)
   
def copyGene(gene):
    gene2 = newGene()
    gene2.into = gene.into
    gene2.out = gene.out
    gene2.weight = gene.weight
    gene2.enabled = gene.enabled
    gene2.innovation = gene.innovation
    return gene2
    
def copyGenome(genome):
    genome2 = newGenome()
    for gene in genome.genes:
        genome2.genes.append(copyGene(gene))
    genome2.maxneuron = genome.maxneuron
    genome2.mutationRates["connections"] = genome.mutationRates["connections"]
    genome2.mutationRates["link"] = genome.mutationRates["link"]
    genome2.mutationRates["bias"] = genome.mutationRates["bias"]
    genome2.mutationRates["node"] = genome.mutationRates["node"]
    genome2.mutationRates["enable"] = genome.mutationRates["enable"]
    genome2.mutationRates["disable"] = genome.mutationRates["disable"]
    
    return genome2
       
def randomNeuron(genes,nonInput):
    neurons = {}
    if not nonInput:
        for i in range(1,Inputs):
            neurons[i] = True
    for o in range(1,Outputs+1):
        neurons[MaxNodes+o] = True
      
 
    for gene in genes:
        if (not nonInput) or gene.into > Inputs:
            neurons[gene.into]  = True
        if (not nonInput) or gene.out > Inputs:
            neurons[gene.into] = True
    count = 0
    for k,v in neurons.items():
        count += 1
    n = random.randint(1,count)
    for k,v in neurons.items():
        n = n-1
        if n==0:
            return k

    
    return(neurons[n])

def generateNetwork(genome):
    neurons = {}
    
    for i in range(1,Inputs+1):
        neurons[i] = newNeuron()
    for o in range(1,Outputs+1):
        neurons[MaxNodes+o] = newNeuron()
    
    genome.genes = sorted(genome.genes,key=attrgetter('out'))
    count = 0
    for gene in genome.genes:
        if gene.enabled:
            if neurons.get(gene.out, 'N/A'):
                neurons[gene.out] = newNeuron()
            neuron = neurons[gene.out]
            
            neuron.incoming.append(gene)
            if neurons.get(gene.into, 'N/A'):
                neurons[gene.into] = newNeuron()
            count += 1
    genome.neurons = neurons
      
def evaluateNetwork(neurons,inputs):
    inputs.append(1)
    if len(inputs) != Inputs:
        print("incorrect neural network")
    for i in range(1,Inputs):
        neurons[i].value = inputs[i-1] #start at +1
    
    for neuron in neurons:
        total = 0
        for incoming in neurons[neuron].incoming:
            other = neurons[incoming.into]
            total += incoming.weight * other.value
        if len(neurons[neuron].incoming) > 0:
            neurons[neuron].value = sigmoid(total)
    
    
    
    for o in range(Outputs):
        if neurons[MaxNodes+o+1].value > 0:
            outputs = 1
        else:
            outputs = 0
    return outputs


             
def sigmoid(x):
    return 2/(1+math.exp(-4.9*x))-1

def evaluateCurrent(inputs):

    species = pool.species[pool.currentSpecies]
    genome = species.genomes[pool.currentGenome]
    output = evaluateNetwork(genome.neurons,inputs) 
    updateCanvas(genome)
     
    return output
        
def evaluateAll(inputs): 
    
    for input in inputs:
        currentNetwork = pool.species[pool.currentSpecies].genomes[pool.currentGenome]
        generateNetwork(currentNetwork)
        currentNetwork.buttonPresses = evaluateNetworkRaw(currentNetwork.neurons,input)
        nextGenomeCompetition()
    
    return pool

def setFitness(score):
    pool.species[pool.currentSpecies].genomes[pool.currentGenome].fitness = score

def setAllFitness(gamePool):
    currentSpecie = 0
    for specie in gamePool.species:
        currentGenome = 0
        for genome in specie.genomes:
            pool.species[currentSpecie].genomes[currentGenome].fitness = genome.creature.score
            currentGenome += 1
        currentSpecie += 1
    
def nextGenome():
    pool.currentGenome = pool.currentGenome + 1
    if pool.currentGenome +1 > len(pool.species[pool.currentSpecies].genomes):
        pool.currentGenome = 0
        pool.currentSpecies = pool.currentSpecies+1
        if pool.currentSpecies >= len(pool.species):
            pool.currentSpecies = 0
            pool.currentGenome = 0
            newGeneration()
               
def nextGenomeCompetition():
    pool.currentGenome = pool.currentGenome + 1
    if pool.currentGenome +1 > len(pool.species[pool.currentSpecies].genomes):
        pool.currentGenome = 0
        pool.currentSpecies = pool.currentSpecies+1
        if pool.currentSpecies >= len(pool.species):
            pool.currentSpecies = 0
            pool.currentGenome = 0            
            
def initializeRun():
    species = pool.species[pool.currentSpecies]
    genome = species.genomes[pool.currentGenome]
    generateNetwork(genome)   

def newGeneration():
    cullSpecies(False)  
    rankGlobally()
    removeStaleSpecies()
    rankGlobally()
    for specie in pool.species:
        calculateAverageFitness(specie)
        
    removeWeakSpecies()
    sum = totalAverageFitness()
    children = []
    for specie in range(len(pool.species)):
        breed = math.floor(pool.species[specie].averageFitness / sum * pool.Population) -1
        for i in range(breed):
            children.append(breedChildren(pool.species[specie]))
    cullSpecies(True)
    while (len(children) + len(pool.species) < pool.Population):
        test = breedChildren(pool.species[random.randint(0,len(pool.species)-1)])
        children.append(test)
    for child in children:
        addToSpecies(child)
    pool.generation = pool.generation + 1
            
def cullSpecies(cutToOne):
    for specie in pool.species:
        specie.genomes = sorted(specie.genomes,key=attrgetter('fitness'),reverse=True)
        remaining = math.ceil(len(specie.genomes)/2)
        if cutToOne:
            remaining = 1
        total = len(specie.genomes)
        while len(specie.genomes) > remaining:
            specie.genomes.remove(specie.genomes[total-1])
            total += -1


def removeStaleSpecies():
    survived = []
    for specie in pool.species:
        specie.genomes = sorted(specie.genomes, key=attrgetter('fitness'),reverse=True)
        for genome in specie.genomes:
            if genome.fitness > specie.topFitness:
                specie.topFitness = genome.fitness
                specie.staleness = 0
            else:
                specie.staleness = specie.staleness + 1
        if specie.staleness < StaleSpecies or specie.topFitness >= pool.maxFitness:
            survived.append(specie)
    pool.species = survived

def removeWeakSpecies():
    survived = []
    sum = totalAverageFitness()
    for specie in pool.species:
        breed = math.floor(specie.averageFitness / sum * pool.Population)
        if breed >= 1:
            survived.append(specie)
    
    pool.species = survived

def totalAverageFitness():
    total = 0 
    for specie in pool.species:
        total = total + specie.averageFitness
    return total

def averageFitness():
    total = 0
    count =+ 1
    for specie in pool.species:
        for genome in specie.genomes:
            count += 1
            total += genome.fitness
    total = total/count
    return total

def calculateAverageFitness(Specie):
    total = 0
    for genome in Specie.genomes:
        total = total + genome.globalRank
    Specie.averageFitness = total / len(Specie.genomes)

def breedChildren(species):
    if random.random() < CrossoverChance:
        rand1 = random.randint(0,len(species.genomes)-1)
        rand2 = random.randint(0,len(species.genomes)-1)
        g1 = species.genomes[rand1]
        g2 = species.genomes[rand2]
        child = crossover(g1,g2)
    else:
        g =  species.genomes[random.randint(0,len(species.genomes)-1)]
        child = copyGenome(g)
    mutate(child)
    return child

def crossover(g1,g2):
    if g2.fitness > g1.fitness:
        tempg = g1
        g1 = g2
        g2 = tempg
    innovations2 = {}
    child = newGenome()
    for gene2 in g2.genes:
        innovations2[gene2.innovation] = gene2
    for gene1 in g1.genes:
        gene2 = innovations2.get(gene1.innovation)
        if gene2 != None and  random.randint(1,2) == 1 and gene2.enabled:
            child.genes.append(copyGene(gene2))
        else:
            child.genes.append(copyGene(gene1))
    child.maxneuron = max(g1.maxneuron,g2.maxneuron)
    for mutation,rate in g1.mutationRates.items():
        child.mutationRates[mutation] = rate
    return child
        
def sameSpecies(genome1,genome2):
    dd = DeltaDisjoint*disjoint(genome1.genes,genome2.genes)
    dw = DeltaWeights*weights(genome1.genes,genome2.genes)
    return dd + dw < DeltaThreshold

def disjoint(genes1,genes2):
    i1 = {}
    for gene in genes1:
        i1[gene.innovation] = True
    i2 = {}
    for gene in genes2:
        i2[gene.innovation] = True
    
    disjointGenes = 0
    for gene in genes1:
        if  not i2.get(gene.innovation,False):
            disjointGenes = disjointGenes+1
    for gene in genes2:
        if not i1.get(gene.innovation,False):
            disjointGenes = disjointGenes+1
            
    n = max(len(genes1),len(genes2))
    return disjointGenes / n
        
def weights(genes1,genes2):
    i2 = {}
    for gene in genes2:
        i2[gene.innovation] = gene
    sum = 0
    coincident = 0
    for gene in genes1:
        if i2.get(gene.innovation) != None:
            gene2 = i2.get(gene.innovation)
            sum = sum + abs(gene.weight - gene2.weight)
            coincident = coincident + 1
    if sum == 0 and coincident == 0:
        return 0
    return sum / coincident

def rankGlobally():
    sIndex = []
    s = 0
    for specie in pool.species:
        g = 0
        for genome in specie.genomes:
           sIndex.append((s,g,genome))
           g += 1
        s += 1
    
    sIndex.sort(key=lambda tup: tup[2])
    c = 0
    for rank in sIndex:
        pool.species[rank[0]].genomes[rank[1]].globalRank = c
        c += 1
     
    
def getNeurons():
	return pool.species[pool.currentSpecies].genomes[pool.currentGenome].neurons
	
def getGenome():
	return pool.species[pool.currentSpecies].genomes[pool.currentGenome]
	
def getSpecies():
	return pool.species

def getPool():
    return pool

def setInputOutput(_Inputs,_Outputs):
    global Inputs
    global Outputs
    Inputs = len(range(_Inputs+1))
    Outputs = _Outputs
    
def updateCanvas(genome):
    nd.canvas.delete(ALL)
    cells = {}
    i = 1
    GridSize = 15
    
    x=2        
    square = 1
    while x <= Inputs:
        x= 2**square
        square +=1
    k = int(math.sqrt(x))

    for dx in range(k):
        dx = dx-k//2
        for dy in range(k):
            dy = dy-k//2
            r = (height/2)+(dx*(height/k))+(height/k)/2
            z = (height/2)+(dy*(height/k)) + (height/k)/2
            if i < Inputs:                
                cell = Cell()
                cell.x = r
                cell.y =  z
                cell.value = genome.neurons[i].value
                cells[i] = cell
            i += 1
    biasCell = Cell()
    biasCell.x = r
    biasCell.y = z
    biasCell.value = genome.neurons[Inputs].value
    cells[Inputs] = biasCell
    
    for o in range(1,Outputs+1):
        cell = Cell()
        cell.x = width*.9
        cell.y = (height*.2)*o
        cell.value = genome.neurons[MaxNodes+o].value
        cells[MaxNodes+o]=cell
        

    for n,neuron in genome.neurons.items():
        cell = Cell()
        if n>Inputs and n<=MaxNodes:
            cell.x = 20*n
            cell.y = 10 * n
            cell.value = neuron.value
            cells[n] = cell
    x = 0
    y = 10
    for mutation,Rate in genome.mutationRates.items():
        Text = mutation," ",Rate
        nd.canvas.create_text((x,y),text=Text)
        x += .2*width
        
    for k,cell in cells.items():
        value = math.floor((cell.value+1)/2*256)
        if value > 255:
            value = 255
        if value < 0:
            value = 0
        color = '#%02x%02x%02x' % (0, 0,value)
        nd.canvas.create_rectangle(cell.x-GridSize,cell.y-GridSize,cell.x+GridSize,cell.y+GridSize,fill=color)

    for o in range(1,Outputs+1):
        cell = cells[MaxNodes+o]
        nd.canvas.create_text((cell.x+20,cell.y),text=str(o))
    

    for gene in genome.genes:
        if gene.enabled:
            c1 = cells[gene.into]
            c2 = cells[gene.out]
            value = int(math.fabs(gene.weight)*255)
            if value > 255:
                value = 255
            color = '#%02x%02x%02x' % (0, 0,value)
            nd.canvas.create_line(c1.x+1, c1.y, c2.x-3, c2.y,fill=color,width=5)
    nd.canvas.focus_set()
    nd.canvas.pack()
    nd.canvas.update()
    
    
def getBest():
    _max = 0
    for specie in pool.species:
        for genome in specie.genomes:
            if genome.fitness > _max:
               _max = genome.fitness
    return _max

