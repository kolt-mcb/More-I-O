from operator import attrgetter
import sys
import time
import random
import math





class pool: #holds all species data, crossspecies settings and the current gene innovation
	def __init__(self,population,Inputs,Outputs,recurrent=False):
		self.generations = []
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
		self.newGenome.innovation = Inputs #  sets the class variable to the current number of inputs
		self.recurrent = recurrent
		self.database = None
		self.client = None
		for x in range(self.Population):# potpulate with random nets
			newGenome = self.newGenome(Inputs,Outputs,recurrent)
			while newGenome.genes == []:
				newGenome.mutate()
			self.addToPool(newGenome)
		for specie in self.species:
			for genome in specie.genomes:
				genome.generateNetwork()		
 

			
	def updateMongo(self):
		self.client = MongoClient(self.database, 27017)
		db = self.client["pool"]
		collection = db["gen"+str(self.generation+1)]
		collection.insert_many(self.getSpeciesBSON(self.species))

		
	def getSpeciesBSON(self,species):
		specieArray = []
		c = 0
		for specie in species:
			specieBSON = {
				"specie" : c,
				"inputs"  : specie.Inputs,
				"outputs" : specie.Outputs,
				"topFitness" : specie.topFitness,
				"staleness" : specie.staleness, 
				"genomes" : self.getGenomeBSON(specie.genomes),
				"averageFitness" : specie.averageFitness,
				"CrossoverChance" : specie.CrossoverChance,
				"recurrent" : specie.recurrent
				}
			c += 1
			specieArray.append(specieBSON)
		return specieArray
			
	def getGenomeBSON(self,genomes):
		genomesArray = []
		c = 0
		for genome in genomes:
			genomeBSON = {
				"genome" : c,
				"genes" : self.getGenesBSON(genome.genes),
				"fitness" : genome.fitness,
				"neurons" : self.getNeuronsBSON(genome.neurons),
				"max neuron" : genome.Inputs,
				"muation rates" : {
					"connections" : genome.mutationRates["connections"],
					"link" : genome.mutationRates["link"],
					"bias" : genome.mutationRates["bias"],
					"node" : genome.mutationRates["node"],
					"enable" : genome.mutationRates["enable"],
					"disable" : genome.mutationRates["disable"],
					"step" :  genome.mutationRates["step"]
				},
				"global rank" :  genome.globalRank,
				"max nodes" : genome.maxNodes,
				"perturb chance" : genome.PerturbChance,
				"parents" : self.getGenomeParentsBSON(genome.parents)
			}
			c += 1
			genomesArray.append(genomeBSON)
		return genomesArray
	
	def getGenesBSON(self,genes):
		genesArray = []
		c = 0
		for gene in genes:
			geneBSON = {
				"into": gene.into,
				"out" : gene.out,
				"weight" : gene.weight,
				"innovation" : gene.innovation,
				"enabled" : gene.enabled
			}
			c += 1
			genesArray.append(geneBSON)
		return genesArray
	
	def getNeuronsBSON(self,neurons):
		neuronsBSON = {}
		for k,neuron in neurons.items():
			neuronsBSON[str(k)] = neuron.value 
		return neuronsBSON
		
	def getGenomeParentsBSON(self,parents):
		parentsBSON = {}
		
		if parents != None:
			parentsBSON={
				"1":parents["parent1"],
				"2":parents["parent2"]
			}
		return parentsBSON
	
	



	def addToPool(self,child): # adds a species to its family of species, if not within threshold of any existing species creates a new species.
		foundSpecies = False
		for specie in range(len(self.species)):
			if not foundSpecies and self.sameSpecies(child,self.species[specie].genomes[0]):
				child.ID ={
					"generation" : self.generation,
					"specie" : specie,
					"genome" : len(self.species[specie].genomes)+1
				}
				self.species[specie].genomes.append(child)
				foundSpecies = True
		if not foundSpecies:
			childSpecies = self.newSpecies(self.Inputs,self.Outputs,self.recurrent)
			child.ID = {
				"generation" : self.generation,
				"specie"	:len(self.species)+1,
				"genome"	 :0
			}
			childSpecies.genomes.append(child)
			self.species.append(childSpecies)
				  
	def sameSpecies(self,genome1,genome2):
		threshold = 1
		if genome1.fitness == genome2.fitness:
			threshold = genome1.mutationRates["DeltaThreshold"]
			DeltaDisjoint = genome1.mutationRates["DeltaDisjoint"]
			DeltaWeights = genome1.mutationRates["DeltaWeights"]

		if genome1.fitness > genome2.fitness:
			threshold = genome1.mutationRates["DeltaThreshold"]
			DeltaDisjoint = genome1.mutationRates["DeltaDisjoint"]
			DeltaWeights = genome1.mutationRates["DeltaWeights"]

		if genome1.fitness < genome2.fitness:
			threshold = genome2.mutationRates["DeltaThreshold"]
			DeltaDisjoint = genome2.mutationRates["DeltaDisjoint"]
			DeltaWeights = genome2.mutationRates["DeltaWeights"]

		dd = DeltaDisjoint*self.disjoint(genome1.genes,genome2.genes) #checks for genes
		dw = DeltaWeights*self.weights(genome1.genes,genome2.genes) # checks values in genes 

		if genome1.fitness > genome2.fitness:
			threshold = genome1.mutationRates["DeltaThreshold"]
		if genome1.fitness < genome2.fitness:
			threshold = genome2.mutationRates["DeltaThreshold"]
		dd = DeltaDisjoint*self.disjoint(genome1.genes,genome2.genes) #checks for genes
		dw = DeltaWeights*self.weights(genome1.genes,genome2.genes) # checks values in genes 
		return dd + dw < threshold

	def initializeRun(self): #generates a network for current species
		species = self.species[self.currentSpecies]
		genome = species.genomes[self.currentGenome]
		generateNetwork(genome) 
			
	def nextGenome(self):# cycles through genomes
		self.currentGenome = self.currentGenome + 1
		if self.currentGenome +1 > len(self.species[self.currentSpecies].genomes):
			self.currentGenome = 0
			self.currentSpecies = self.currentSpecies+1
			if self.currentSpecies >= len(self.species):
				self.currentSpecies = 0
				self.currentGenome = 0
				newGeneration()
				
	def evaluateCurrent(self,inputs,discrete=False): # runs current species network 
		species = self.species[self.currentSpecies]
		genome = species.genomes[self.currentGenome]
		output = genome.evaluateNetwork(inputs,discrete)
		return output
				   
	#cuts poor preforming genomes and performs crossover of remaining genomes.
	def nextGeneration(self):
		self.generations.append(self.species)
		population = 0
		for specie in self.generations[self.generation].species:
			for genome in specie.genomes:
				population += 1
		self.Population = population
		self.cullSpecies(False)  
		self.rankGlobally() 
		self.removeStaleSpecies()
		# reranks after removeing stales species and  stores best player for later play
		self.rankGlobally(addBest=True)
		for specie in self.generations[self.generation].species:
			#calculateAverageFitness of a specie
			specie.calculateAverageFitness() 
		
		self.removeWeakSpecies() 
		Sum = self.totalAverageFitness()

		#defines new children list
		children = []

		for k, species in enumerate(self.generations[self.generation].species):
			breed = math.floor(specie.averageFitness / Sum * self.Population)-1 # if a species average fitness is over the pool averagefitness it can breed
			for i in range(breed):
				children.append(specie.breedChildren())
		# leave only the top member of each species.
		self.cullSpecies(True) 
		self.cullOldSpecies()
		while (len(children)+len(self.species) < self.Population):
			parent = random.choice(self.species)
			child = parent.breedChildren()
			children.append(child)


		for child in children: # adds all children there species in the pool
			self.addToPool(child)
		for specie in lastGen:
			for genome in specie.genomes:
				self.addToPool(genome)
			
		self.generation = self.generation + 1
		
	def cullOldSpecies(self):
		species = self.species
		s = 0
		for specie in species:
			g = 0
			for genome in specie.genomes:
				p = genome.currentAge
				if random.random() > p:
					self.species[s].genomes.pop(g)
					if len(self.species[s].genomes) == 0:
						self.species.pop(s)
				else:
					if len(self.species[s].genomes) != 0:
						self.species[s].genomes[g].currentAge -= 1
				g += 1
			s += 1	

	def cullSpecies(self,cutToOne): #sorts genomes by fitness and removes half of them or cuts to one
		species = self.species
		for specie in species:
			specie.genomes = sorted(specie.genomes,key=attrgetter('fitness'),reverse=True)
			if not cutToOne:
				remaining = math.ceil(len(specie.genomes)/specie.calculateAverageRemainingMultiplyer())
				if remaining < 1:
					remaining = 1
		
			if cutToOne:
				remaining = specie.calculateAverageRemainingRate()
				if remaining < 1:
					remaining = 1
			print(cutToOne,remaining)			
			total = len(specie.genomes)
			while len(specie.genomes) > remaining:
				self.specie.genomes.remove(specie.genomes[total-1])
				total += -1
		
	def removeStaleSpecies(self): # removes species that have not gotten a high score past a threshold
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
		
	def removeWeakSpecies(self): # removes poor performing species
		survived = []
		sum = self.totalAverageFitness()
		for specie in self.species:
			breed = math.floor(specie.averageFitness / sum * self.Population)
			if breed >= 1:
				survived.append(specie)
	
		self.species = survived

	def totalAverageFitness(self): #total of all averages of pool
		total = 0 
		for specie in self.species:
			total = total + specie.averageFitness
		return total

	def averageFitness(self): #average fitness of entire pool
		total = 0
		count =+ 1
		for specie in self.species:
			for genome in specie.genomes:
				count += 1
				total += genome.fitness
		total = total/count
		return total
		
	def rankGlobally(self,addBest=False): # sets globalRank value for all genomes.
		sIndex = []
		s = 0
		for specie in self.species:
			g = 0
			for genome in specie.genomes:
				geneEnabledCount = 0 
				for gene in genome.genes:
					if gene.enabled:
						geneEnabledCount += 1
				geneEnabledCount = 0 -geneEnabledCount
				sIndex.append((s,g,genome.fitness,genome.mutationRates["ConectionCostRate"]*geneEnabledCount))
				g += 1
			s += 1

		sIndex.sort(key=lambda tup: (tup[2],tup[3]))
		
		c = 1
		for rank in sIndex:
			self.species[rank[0]].genomes[rank[1]].globalRank = c
			if c == len(sIndex):
				topGenome = self.species[rank[0]].genomes[rank[1]]
				if addBest:
					self.best.append(topGenome)
				self.maxFitness = topGenome.fitness
			c += 1

	def getBest(self): # returns best genome for current generation
		return self.best[len(self.best)-1]

	def disjoint(self,genes1,genes2): # mesures the amount of shared in a genes in a genomes genes.
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
		if n == 0:
			return 0
		return disjointGenes / n
		
	def weights(self,genes1,genes2): # mesures the difference of weights in a shared gene due to mutation 
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

	def getOutputs(self):
		return self.Outputs

	

	class newGenome:
		def __init__(self,Inputs,Outputs,recurrent):
			self.genes = []
			self.fitness = 0
			self.neurons = {}
			self.maxneuron = Inputs
			self.mutationRates = {}
			self.speciesRates = {}
			self.globalRank = 0
			self.maxNodes = 10000
			self.mutationRates["connections"] =  0.4
			self.mutationRates["link"] =  .4
			self.mutationRates["bias"] = 0.1
			self.mutationRates["node"] = 0.4
			self.mutationRates["enable"] = 0.05
			self.mutationRates["disable"] = 0.1
			self.mutationRates["step"] = 0.1
			self.mutationRates["speciesStep"]= 0.5
			self.mutationRates["DeltaThreshold"] = 1
			self.mutationRates["DeltaDisjoint"] = 2
			self.mutationRates["DeltaWeights"] = 0.4 
			self.mutationRates["CrossOverRate"] = .75
			self.mutationRates["PerturbChance"] = 0.5
			self.mutationRates["ConectionCostRate"] = 1
			self.speciesRates["RemainingMultiplyer"] = 2
			self.speciesRates["Remaining"] = 1 
			self.mutationRates["age"] = 5
			self.currentAge = self.mutationRates["age"]
			self.parents = []
			self.Inputs = Inputs
			self.Outputs = Outputs
			self.recurrent = recurrent
			self.ID = None
			self.parents = {
				"parent1" : None,
				"parent2" : None
				}
		
			if self.recurrent: # initializes first run for reccurrent networks
				self.lastEval = Outputs*[0]
				self.maxneuron = self.Inputs+self.Outputs
		
		

		def newInnovation(self):
			#sets genome class variable, this is shared between all genomes
			pool.newGenome.innovation += 1
			return pool.newGenome.innovation

			
		def mutate(self): # runs all mutation types at rate set, while probabilty is over the rate set.
			for mutation,rate in self.mutationRates.items():
				if random.randint(1,2) == 1:
					self.mutationRates[mutation] = 0.95*rate
				else:
					self.mutationRates[mutation] = 1.05263*rate
			for mutation,rate in self.speciesRates.items():
				speciesRate = self.mutationRates["speciesStep"]
				if random.randint(1,2) == 1:
					self.speciesRates[mutation] = rate + speciesRate
				else:
					self.speciesRates[mutation] = rate-speciesRate
			if random.random() < self.mutationRates["connections"]:

				self.pointMutate()
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

					
						   
		def pointMutate(self): #mutates the weight of a gene
			step = self.mutationRates["step"]
			for gene in self.genes:
				if random.random() < self.mutationRates["PerturbChance"]:
					gene.weight = gene.weight + random.random()*step*2-step
				else:
					gene.weight = 1-random.random()*2

		def linkMutate(self,forceBias): #adds a link to the network, potentialy forced to bias neuron
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
			
		def containsLink(self,link): # checks if link already exists between neurons
			for gene in self.genes:
				if gene.into == link.into and gene.out == link.out:
					return True
				
		def nodeMutate(self):# addds a node 
			if len(self.genes) == 0:
				return
			r = random.randint(0,len(self.genes)-1)
			gene = self.genes[r]
			if not gene.enabled:
				return
			self.maxneuron += 1 # index of next neuron to point at as seen 3 lines below, gene1.out = self.maxneuron
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
		
		def enableDisableMutate(self, enable):# enables or disables a gene
			candidates = []
			for g in self.genes:
				if g.enabled == (not enable):
					candidates.append(g)
			if len(candidates) == 0:
				return
			r = random.randint(0,len(candidates)-1)
			gene = candidates[r]
			gene.enabled = (not gene.enabled)
	   
		def copyGenome(self): # copies a genome perfectly. 
			genome2 = pool.newGenome(self.Inputs,self.Outputs,self.recurrent)
			for gene in self.genes:
				genome2.genes.append(gene.copyGene())
			genome2.Inputs = self.Inputs
			genome2.Outputs = self.Outputs
			genome2.maxneuron = self.maxneuron
			genome2.mutationRates = self.mutationRates
			genome2.speciesRates = self.speciesRates
			genome2.parents["parents1"] = self.ID
			genome2.parents["parent2"] = self.ID 
			return genome2
		
		def generateNetwork(self): # generates a network based on genes.
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

						neurons[gene.out] = newNeuron() # checks all out links for missing neurons

					neuron = neurons[gene.out]
					neuron.incoming.append(gene)
					if not(gene.into in neurons):
						
						neurons[gene.into] = newNeuron()
			self.neurons = neurons

		
		def randomNeuron(self,nonInput): # selects a random neuron with choice of input or not.
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
		
		def evaluateNetwork(self,inputs,discrete=False): # runs inputs through neural network, this is what runs the brain. 
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
			if discrete: # discrete means 0 or 1 eg on or off other wise send unmodified output layer values
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
				
		def sigmoid(self,x): # sigmoid function.
			return 2/(1+math.exp(-4.9*x))-1

	 
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

		def calculateAverageRemainingRate(self):
			total = 0			
			for genome in self.genomes:
				total += genome.speciesRates["Remaining"] 
			total = total / len(self.genomes)
			if total < 1:
				total = 1
			return total

		def calculateAverageRemainingMultiplyer(self):
			total = 0
			for genome in self.genomes:
				total += genome.speciesRates["RemainingMultiplyer"] 
			total = total / len(self.genomes)
			if total < 1:
				total = 1
			return total

		def breedChildren(self): # breeds children of a species
			if random.random() < self.getAverageCrossOverRate():
				rand1 = random.randint(0,len(self.genomes)-1)
				rand2 = random.randint(0,len(self.genomes)-1)
				g1 = self.genomes[rand1]
				g2 = self.genomes[rand2]
				child = self.crossover(g1,g2)
			else:
				g =  self.genomes[random.randint(0,len(self.genomes)-1)]
				child = g.copyGenome()
			child.mutate()
			return child
		

		def crossover(self,g1,g2): # mixes genes of 2 species
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
				child.mutationRates[mutation] = rate
			child.parents["parent1"] = g1.ID
			child.parents["parent2"] = g2.ID   
			return child


		def getAverageCrossOverRate(self):
			totalAverage = 0
			for genome in self.genomes:
				totalAverage += genome.mutationRates["CrossOverRate"]
			CrossOverRate = totalAverage/len(self.genomes)
			return CrossOverRate

		
	   
	 


	   
		
class newGene:
	def __init__(self):
		self.into = 0
		self.out = 0
		self.weight = 0.0
		self.innovation = 0
		self.enabled = True
		
	def copyGene(self): # copies gene perfectly
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
	 








