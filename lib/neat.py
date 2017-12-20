from operator import attrgetter
from operator import itemgetter
import sys
import time
import random
import math
from pymongo import MongoClient





class pool: #holds all species data, crossspecies settings and the current gene innovation
	generations = []
	client = None
	timeStamp = time.time()
	def __init__(self,population,Inputs,Outputs,recurrent=False,database=None,timeStamp=None,connectionCost=False):
		self.species = []
		self.generation = 0
		self.currentSpecies = 0
		self.currentGenome = 0
		self.maxFitness = 0
		self.Population = population
		self.best = []
		self.StaleSpecies = 15
		self.Inputs = Inputs
		self.Outputs = Outputs
        #  sets the class variable to the current number of inputs
		self.newGenome.innovation = Inputs 
		self.recurrent = recurrent
		self.connectionCost = connectionCost
		self.databaseName = database
        
        # use saved time stamp
		if timeStamp != None:
			pool.timeStamp = timeStamp
		#children for initial population
		children = []
		if database != None and timeStamp == None:
			pool.client = MongoClient(self.databaseName, 27017)
			db = pool.client["runs"]
			collection = db["Runs"]
			collection.insert_one({
				"time stamp" : pool.timeStamp,
				"inputs" : self.Inputs,
				"outputs" : self.Outputs,
				"population" : self.Population
								  })
		#potpulate with random nets
		for x in range(self.Population):#
			newGenome = self.newGenome(Inputs,Outputs,recurrent)
			newGenome.mutate()
			children.append(newGenome)
    
		#adds to population
		self.addToPool(children)

        #generate networks
		for specie in self.species:
			for genome in specie.genomes:
				genome.generateNetwork()
				
				
				


	#updates a mongo database when the genome is added to the population
	def updateMongoGenerations(self,doc):
		db = pool.client["runs"]
		collection = db["Generations"]
		collection.insert_one(doc)
	
    # the main genome doc with genes and lots of things. 
	def updateMongoGenome(self,genome,specie):
		doc = {
		"time stamp" : pool.timeStamp,
		"genes" : self.getGenesBSON(genome.genes),
		"relatives" : list(genome.relatives),
		"weightMaps" : self.getGeneWeightsBSON(genome.genes),
		"fitness" : genome.fitness,
		"age" : genome.age,
		"maxNeuron" : genome.maxneuron,
		"mutationRates" : genome.mutationRates,
		"globalRank" : genome.globalRank,
		"maxNodes" : genome.maxNodes,
		"inputs" : genome.Inputs,
		"Outputs" : genome.Outputs,
		"recurrent" : genome.recurrent,
		"parents" : self.getParentsBSON(genome.parents),
		"generation" : genome.ID[0],
		"genome" : genome.ID[1],
		"specie" : specie
		}
		db = pool.client["runs"]
		collection = db["Genomes"]
		collection.insert_one(doc)
	
    # turns parent tuple into bson
	def getParentsBSON(self,parents):
		parentsBSON = {}
		parentsBSON["parent1"] = parents[0]
		parentsBSON["parent2"] = parents[1]
		return parentsBSON

	# turns ID tuple into bson
	def getIDBSON(self,ID):
		IDBSON = {}
		IDBSON["generation"] = ID[0]
		IDBSON["genome"] = ID[1]
		return IDBSON
		
    # turns gene array into bson
	def getGenesBSON(self,genes):
		genesArray = []
		for gene in genes:
			genesArray.append(gene.innovation)
		return genesArray
	
    # turns gene weights into bson
	def getGeneWeightsBSON(self,genes):
		genesWeightMapArray = []
		for gene in genes:
			genesWeightMapArray.append({
			str(gene.innovation) : gene.weight
			})
		return genesWeightMapArray
	
    # adds a list of children to the pool
	def addToPool(self,children):
		pool.generations.append([])
		for child in children:
			if child.relatives == set():
				child.relatives = self.getRelatives(child)
			child.mates = set()
			foundSpecies = False
			foundedSpecie = None
			closestMatch = math.inf
			updatedMates = []
			if child.relatives != None:
				for specie in range(len(self.species)):
					for genome in range(len(self.species[specie].genomes)):
						_genome = self.species[specie].genomes[genome]
						if not child.relatives.isdisjoint(_genome.relatives):
							match = self.sameSpecies(child,_genome,rating=True)
							if match < math.inf:
								if match < closestMatch:
									closestMatch = match
									foundedSpecie = specie
									foundSpecies = True
								updatedMates.append((specie,genome))
								child.mates.add(_genome.ID)
			idSet = False
			if child.ID != None:
				idSet = True
				child.ID = (self.generation,len(self.generations[self.generation]))
			for mate in updatedMates:
				specie = mate[0]
				genome = mate[1]
				self.species[specie].genomes[genome].mates.add(child.ID)
			pool.generations[self.generation].append(child)
			if foundSpecies:
				self.species[foundedSpecie].genomes.append(child)
				s = foundedSpecie
			else:
				specie = self.newSpecies(self.Inputs,self.Outputs,self.recurrent)
				child.defining = True
				specie.genomes.append(child)
				s = len(self.species)
				self.species.append(specie)
				foundSpecies = True
			if idSet:
				print("child specie:",s," genome:",len(self.species[s].genomes)-1,"was born")
			if pool.client != None:
				doc = self.getIDBSON(child.ID)
				doc["game"] = pool.timeStamp
				doc["generation"] = self.generation
				doc = {**doc,**self.getParentsBSON(child.parents)}
				self.updateMongoGenerations(doc)
				if idSet:
					self.updateMongoGenome(child,s)


				
				
	def getRelatives(self,child,parent=None):
		relatives = set()
		if parent == None:
			genomeToCheck = child
		else:
			genomeToCheck = parent
			
		for parentGenomeTup in genomeToCheck.parents:
			
			if parentGenomeTup != None:
				generation = parentGenomeTup[0]
				genome = parentGenomeTup[1]
				parentGenome = self.generations[generation][genome]
				if self.sameSpecies(child,parentGenome):
					if parentGenome.ID != None:
						relatives.add(parentGenome.ID)
						if not parentGenome.defining:
							parentRelatives = self.getRelatives(child,parentGenome)
							if parentRelatives != None:
								relatives.update(parentRelatives)

		return relatives
				  
	def sameSpecies(self,genome1,genome2,rating=False):
		Threshold1 = genome1.mutationRates["DeltaThreshold"]
		DeltaDisjoint1 = genome1.mutationRates["DeltaDisjoint"]
		DeltaWeights1 = genome1.mutationRates["DeltaWeights"]

		Threshold2 = genome2.mutationRates["DeltaThreshold"]
		DeltaDisjoint2 = genome2.mutationRates["DeltaDisjoint"]
		DeltaWeights2 = genome2.mutationRates["DeltaWeights"]
		

		DeltaDisjoint = (DeltaDisjoint1 + DeltaDisjoint2)/2
		DeltaWeights = (DeltaWeights1 + DeltaWeights2)/2
		Threshold = (Threshold1 + Threshold2)/2
		
		
		dd = DeltaDisjoint*self.disjoint(genome1.genes,genome2.genes) #checks for genes
		dw = DeltaWeights*self.weights(genome1.genes,genome2.genes) # checks values in genes
		
		if rating:
			if	dd + dw < Threshold:
				return  Threshold - dd+dw
			else:
				return math.inf
		return dd + dw< Threshold

	#generates a network for current species
	def initializeRun(self): 
		species = self.species[self.currentSpecies]
		genome = species.genomes[self.currentGenome]
		generateNetwork(genome) 
	

				
	def evaluateCurrent(self,inputs,discrete=False): # runs current species network 
		species = self.species[self.currentSpecies]
		genome = species.genomes[self.currentGenome]
		output = genome.evaluateNetwork(inputs,discrete)
		return output
				   
	#cuts poor preforming genomes and performs crossover of remaining genomes.
	def nextGeneration(self):
		self.generation += 1
		self.cullSpecies(False)  
		self.rankGlobally() 
		self.removeStaleSpecies()
		# reranks after removeing stales species and  stores best player for later play
		self.rankGlobally(addBest=True)
		for specie in self.species:
			#calculateAverageFitness of a specie
			specie.calculateAverageFitness()
			
		self.removeWeakSpecies()
		_sum = self.totalAverageFitness()
		c = 0
		for specie in self.species:
			for genome in specie.genomes:
				c += 1
		#defines new children list
		children = []
		while (len(children)+c < self.Population):
			for specie in self.species:
				 # if a species average fitness is over the pool averagefitness it can breed
				breed = math.floor(specie.averageFitness / _sum * self.Population)-1
				for i in range(breed):
						if len(children)+c < self.Population:
							children.append(specie.breedChildren())

		# leave only the top member of each species.
		self.cullSpecies(True) 
		c = 0
		for specie in self.species:
			for genome in specie.genomes:
				c += 1
				
		
		while (len(children)+c < self.Population):
			parent = random.choice(self.species)
			child = parent.breedChildren()
			children.append(child)

		for specie in self.species:
			for genome in specie.genomes:
				children.append(genome)
		self.species = []
		# adds all children to there species in the pool
		self.addToPool(children)

	def cullSpecies(self,cutToOne): #sorts genomes by fitness and removes half of them or cuts to one
		for specie in self.species:
			specie.genomes = sorted(specie.genomes,key=attrgetter('fitness'),reverse=True)
			genomes = []
			remaining = math.ceil(len(specie.genomes)/2)
			if cutToOne:
				remaining = 2
			while len(specie.genomes) > remaining:
				specie.genomes.pop()
	
	# removes species that have not gotten a high score past a threshold		
	def removeStaleSpecies(self): 
		survived = []
		for specie in self.species:
			specie.genomes = sorted(specie.genomes, key=attrgetter('fitness'),reverse=True)
			for genome in specie.genomes:
				if genome.fitness > specie.topFitness:
					specie.topFitness = genome.fitness
					specie.staleness = 0
				else:
					specie.staleness = specie.staleness + 1
			if specie.staleness < self.StaleSpecies or specie.topFitness >= self.maxFitness:
				survived.append(specie)
		self.species = survived
	
	# removes poor performing species
	def removeWeakSpecies(self): 
		survived = []
		_sum = self.totalAverageFitness()
		for specie in self.species:
			breed = specie.averageFitness / _sum * self.Population
			if breed >= 1:
				survived.append(specie)
		self.species = survived

	#total of all averages of pool
	def totalAverageFitness(self):
		total = 0 
		for specie in self.species:
			total = total + specie.averageFitness
		return total

	# average fitness of entire pool
	def averageFitness(self): 
		total = 0
		count = 0
		for specie in self.species:
			for genome in specie.genomes:
				count += 1
				total += genome.fitness
		total = total/count
		return total
		
	# sets globalRank value for all genomes.
	def rankGlobally(self,addBest=False): 
		sIndex = []
		s = 0
		c = 0
		for specie in self.species:
			g = 0
			for genome in specie.genomes:
				if self.connectionCost:
					geneEnabledCount = 0 
					for gene in genome.genes:
						if gene.enabled:
							geneEnabledCount += 1
					geneEnabledCount = 0 - geneEnabledCount
					sIndex.append((s,g,genome.fitness,genome.mutationRates["ConectionCostRate"]*geneEnabledCount))
				else:
					sIndex.append((s,g,genome.fitness))
				c += 1
				g += 1
			s += 1
		if self.connectionCost:
			sIndex.sort(key=lambda tup: (tup[2],tup[3]))
		else:
			sIndex.sort(key=lambda tup: (tup[2]))
		c = 1
		for rank in sIndex:
			self.species[rank[0]].genomes[rank[1]].globalRank = c
			print(c,self.species[rank[0]].genomes[rank[1]].fitness)
			if c == len(sIndex):
				topGenome = self.species[rank[0]].genomes[rank[1]]
				if addBest:
					self.best.append(topGenome)
				self.maxFitness = topGenome.fitness
			c += 1

	# returns best genome for current generation
	def getBest(self): 
		return self.best[len(self.best)-1]

	# measures the amount of shared in a genes in a genomes genes.
	def disjoint(self,genes1,genes2): 
		i1 = []
		i2 = []
		for gene in genes1:
			i1.append(gene.innovation)
		for gene in genes2:
			i2.append(gene.innovation)
		n = max(len(genes1),len(genes2))
		disjointGenes = n - len(list(set(i1).intersection(i2)))

		if n == 0:
			return 0
		return disjointGenes / n


		
	# mesures the difference of weights in a shared gene due to mutation 
	def weights(self,genes1,genes2): 
		i2 = {}
		for gene in genes2:
			i2[gene.innovation] = gene
		_sum = 0
		coincident = 0
		for gene in genes1:
			if i2.get(gene.innovation) != None:
				gene2 = i2.get(gene.innovation)
				_sum = _sum + abs(gene.weight - gene2.weight)
				coincident = coincident + 1
		if _sum == 0 or coincident == 0:
			return 0
		return _sum / coincident




	class newGenome:
		def __init__(self,Inputs,Outputs,recurrent):
			self.genes = []
			self.fitness = 0
			self.neurons = {}
			self.maxneuron = Inputs
			self.mutationRates = {}
			self.globalRank = 0
			self.maxNodes = 10000
			self.mutationRates["connections"] =  0.7
			self.mutationRates["link"] =  .7
			self.mutationRates["bias"] = 0.1
			self.mutationRates["node"] = 0.7
			self.mutationRates["enable"] = 0.05
			self.mutationRates["disable"] = 0.1
			self.mutationRates["step"] = 0.1
			self.mutationRates["DeltaThreshold"] = 1
			self.mutationRates["DeltaDisjoint"] = 1
			self.mutationRates["DeltaWeights"] = .4
			self.mutationRates["ConectionCostRate"] = 1
			self.perturbChance = .9
			self.age = 0
			self.parents = ()
			self.relatives = set()
			self.mates = set()
			self.defining = False
			self.Inputs = Inputs
			self.Outputs = Outputs
			self.recurrent = recurrent
			self.ID = (0,0)
			self.parents = (None,None)
			# initializes first run for reccurrent networks
			if self.recurrent: 
				self.lastEval = Outputs*[0]
				self.maxneuron = self.Inputs+self.Outputs

		

		def newInnovation(self):
			#sets genome class variable, this is shared between all genomes
			pool.newGenome.innovation += 1
			return pool.newGenome.innovation

		# runs all mutation types at rate set, while probabilty is over the rate set.
		def mutate(self): 
			for mutation,rate in self.mutationRates.items():
				if random.randint(1,2) == 1:
					self.mutationRates[mutation] = 0.95*rate
				else:
					self.mutationRates[mutation] = 1.05*rate
			
			p = self.mutationRates["connections"]
			while p > 0:
				if random.random() < p:
					self.pointMutate()
				p = p -1	
			p = self.mutationRates["link"] 
			while p > 0:
				if random.random() < p:
					self.linkMutate(False)
				p = p -1
			p = self.mutationRates["bias"] 
			while p > 0:
				if random.random() < p:
					self.linkMutate(True)
				p = p -1
			p = self.mutationRates["node"] 
			while p > 0:
				if random.random() < p:
					self.nodeMutate()
				p = p -1
			p = self.mutationRates["enable"]
			while p > 0:

				if random.random() < p:
					self.enableDisableMutate(True)
				p = p -1
			p = self.mutationRates["disable"]
			while p > 0:
				if random.random() < p:
					self.enableDisableMutate(False)
				p = p -1

					
		#mutates the weight of a gene
		def pointMutate(self): 
			step = self.mutationRates["step"]
			if len(self.genes) > 1:
				r = random.randint(0,len(self.genes)-1)
				gene = self.genes[r]
				if random.random() < self.perturbChance:
					gene.weight = gene.weight + random.random()*step*2-step
				else:
					gene.weight = 1-random.random()*2
	
		#adds a link to the network, potentialy forced to bias neuron
		def linkMutate(self,forceBias): 
			neuron1 = self.randomNeuron(False)
			neuron2 = self.randomNeuron(True)
			newLink = newGene()
			if self.recurrent:
				Inputs = self.Inputs+self.Outputs
			else:
				Inputs = self.Inputs
			if neuron1 <=Inputs and neuron2 <= Inputs:	 
				return 
			if neuron1 == neuron2:
				return
			if neuron2 <= Inputs:
				temp = neuron1
				neuron1 = neuron2
				neuron2 = temp

			newLink.into = neuron1

			newLink.out = neuron2 
			if forceBias:
				newLink.into = Inputs
			if self.containsLink(newLink):
				return	 
			newLink.innovation = self.newInnovation()
			newLink.weight = (1-random.random()*2)
			self.genes.append(newLink)
			
		# checks if link already exists between neurons
		def containsLink(self,link): 
			for gene in self.genes:
				if gene.into == link.into and gene.out == link.out:
					return True
		
		# addds a node 
		def nodeMutate(self):
			if len(self.genes) == 0:
				return
			r = random.randint(0,len(self.genes)-1)
			gene = self.genes[r]
			# index of next neuron to point at as seen 3 lines below, gene1.out = self.maxneuron
			self.maxneuron += 1 
			gene.enabled = False
			gene1 = gene.copyGene()
			gene1.out = self.maxneuron
			gene1.weight = 1.0
			gene1.innovation = self.newInnovation()
			gene1.enabled = True
			self.genes.append(gene1)
			gene2 = gene.copyGene()
			gene2.into = self.maxneuron
			gene2.innovation = self.newInnovation()
			gene2.enabled = True
			self.genes.append(gene2)
		
		# enables or disables a gene
		def enableDisableMutate(self, enable):
			candidates = []
			for g in self.genes:
				if g.enabled == (not enable):
					candidates.append(g)
			if len(candidates) == 0:
				return
			r = random.randint(0,len(candidates)-1)
			gene = candidates[r]
			gene.enabled = (not gene.enabled)

		# copies a genome perfectly. 
		def copyGenome(self): 
			genome2 = pool.newGenome(self.Inputs,self.Outputs,self.recurrent)
			for gene in self.genes:
				genome2.genes.append(gene.copyGene())
			genome2.Inputs = self.Inputs
			genome2.Outputs = self.Outputs
			genome2.maxneuron = self.maxneuron
			genome2.mutationRates = self.mutationRates
			genome2.parents = (self.ID,None)
			return genome2
		
		# generates a network based on genes.
		def generateNetwork(self): 
			neurons = {}
			self.lastEval = self.Outputs * [0]
			#input neurons
			if self.recurrent:
				for i in range(self.Inputs+self.Outputs+1): 
					neurons[i] = newNeuron()
			else:
				for i in range(self.Inputs+1):
					neurons[i] = newNeuron()

		
			#Output neurons
			for o in range(1,self.Outputs+1):
				neurons[self.maxNodes+o] = newNeuron()

		
			self.genes = sorted(self.genes,key=attrgetter('out'))
			for gene in self.genes:
				if gene.enabled:
					if not(gene.out in neurons):
						# checks all out links for missing neurons
						neurons[gene.out] = newNeuron() 

					neuron = neurons[gene.out]
					neuron.incoming.append(gene)
					if not(gene.into in neurons):
						
						neurons[gene.into] = newNeuron()
			self.neurons = neurons

		# selects a random neuron with choice of input or not.
		def randomNeuron(self,nonInput): 
			neurons = {}
			if self.recurrent:
				inputs = self.Inputs + self.Outputs
			else:
				inputs = self.Inputs
			if not nonInput:
				
				for i in range(inputs):
					neurons[i] = True
			for o in range(1,self.Outputs+1):
				neurons[self.maxNodes+o] = True
			for gene in self.genes: 
				if (not nonInput) or gene.into > inputs:
					neurons[gene.into]  = True
				if (not nonInput) or gene.out > inputs:
					neurons[gene.into] = True
			n = random.randint(1,len(neurons))
			for k,v in neurons.items():
				n = n-1
				if n==0:
					return k
			return 0
		
		# runs inputs through neural network, this is what runs the brain. 
		def evaluateNetwork(self,inputs,discrete=False): 
			if self.recurrent:
				inputs = inputs + self.lastEval
			inputs = inputs + [1]

			if self.recurrent:
				if len(inputs) != self.Inputs+self.Outputs+1:
					print(self.Inputs,self.Outputs+1,len(inputs))
					print("incorrect neural network")
			else:
				if len(inputs) != self.Inputs+1:
					print(self.Inputs,len(inputs))
					print("incorrect neural network")
			for i in range(len(inputs)):
				self.neurons[i].value = inputs[i] #start at +1
			


			for n,neuron in self.neurons.items():
				total = 0
				for incoming in neuron.incoming:
					other = self.neurons[incoming.into]
					total += incoming.weight * other.value

				if len(neuron.incoming) > 0:
					neuron.value = self.sigmoid(total)
			outputs = []
			# discrete means 0 or 1 eg on or off other wise send unmodified output layer values
			if discrete: 
				for o in range(1,self.Outputs+1):
					if self.neurons[self.maxNodes+o].value > 0:
						outputs.append(1)
					else:
						outputs.append(0)
			else:
				for o in range(1,self.Outputs+1):	
					outputs.append(self.neurons[self.maxNodes+o].value)
			if self.recurrent:
				self.lastEval=[]
				for o in range(1,self.Outputs+1):	
					self.lastEval.append(self.neurons[self.maxNodes+o].value)
			return outputs
				
		
		def setFitness(self,fitness):
			if self.age > 0:
				self.fitness = (self.fitness + fitness) / 2
			else:
				self.fitness = fitness
			if pool.client != None:
				db = pool.client["runs"]
				genomeCollection = db["Genomes"]
				genomeCollection.update({
										"genome": self.ID[1],
										"generation": self.ID[0],
										"time stamp": pool.timeStamp
										},
										{
										"$set": {
											"fitness": fitness
											}
										})
		def sigmoid(self,x):
			try:
				value =  2/(1+math.exp(-5*x))-1
			except OverflowError:
				if x > 1:
					value = 1
				if x < -1:
					value = -1
			return value
			

	 
	class newSpecies():
		def __init__(self,Inputs,Outputs,recurrent):
			self.Inputs = Inputs
			self.Outputs = Outputs
			self.topFitness = 0
			self.staleness = 0
			self.genomes = []
			self.averageFitness = 0
			self.recurrent = recurrent
			
		def calculateAverageFitness(self): 
			total = 0
			for genome in self.genomes:
				total = total + genome.globalRank
			self.averageFitness = total / len(self.genomes)

		 # breeds children of a species
		def breedChildren(self):
			genome1 = random.choice(self.genomes)
			if random.random() < .75 and len(genome1.mates)>0:
				mate = random.sample(genome1.mates,1)
				generation = mate[0][0]
				genome = mate[0][1]
				genome2 = pool.generations[generation][genome]
				child = self.crossover(genome1,genome2)
			else:
				child = genome1.copyGenome()
			child.mutate()
			return child
		
		# mixes genes of 2 species
		def crossover(self,g1,g2): 
			if g2.fitness > g1.fitness:
				tempg = g1
				g1 = g2
				g2 = tempg
			innovations2 = {}
			child = pool.newGenome(self.Inputs,self.Outputs,self.recurrent)
			child.mutate()
			for gene2 in g2.genes:
				innovations2[gene2.innovation] = gene2
			for gene1 in g1.genes:
				gene2 = innovations2.get(gene1.innovation)
				if gene2 != None and  random.randint(1,2) == 1 and gene2.enabled:
					child.genes.append(gene2.copyGene())
				else:
					child.genes.append(gene1.copyGene())
			child.maxneuron = max(g1.maxneuron,g2.maxneuron)
			for mutation,rate in g1.mutationRates.items():
				g2Rate= g2.mutationRates[mutation]
				child.mutationRates[mutation] = (rate+g2Rate)/2
			child.parents = (g1.ID,g2.ID)   
			return child

class newGene:
	def __init__(self):
		self.into = 0
		self.out = 0
		self.weight = 0.0
		self.innovation = 0
		self.enabled = True
	
	# copies gene perfectly
	def copyGene(self): 
		gene2 = newGene()
		gene2.into = self.into
		gene2.out = self.out
		gene2.weight = self.weight
		gene2.enabled = self.enabled
		gene2.innovation = self.innovation
		return gene2

class newNeuron():
	def __init__(self):
		self.incoming = []
		self.value = 0.0
		







