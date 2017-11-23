from operator import attrgetter
from operator import itemgetter
import sys
import time
import random
import math
from pymongo import MongoClient





class pool: #holds all species data, crossspecies settings and the current gene innovation
	generations = []
	def __init__(self,population,Inputs,Outputs,recurrent=False,database=None,gameName=None):
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
		self.databaseName = database
		self.client = None
		self.gameName = gameName
		self.timeStamp = time.time()
		children = []
		if gameName == None:
			self.gameName = str(self.Inputs)+" "+str(self.Outputs)+" "+str(self.timeStamp)
		if database != None:
			self.client = MongoClient(self.databaseName, 27017)
		for x in range(self.Population):# potpulate with random nets
			newGenome = self.newGenome(Inputs,Outputs,recurrent)
			newGenome.mutate()
			children.append(newGenome)
			
		self.addToPool(children)



		for specie in self.species:
			for genome in specie.genomes:
				genome.generateNetwork()
		if self.client != None:
			self.updateMongoGenomes(self.species)




			
	def updateMongoGenerations(self,doc):
		db = self.client["runs"]
		collection = db["Generations"]
		collection.insert_one(doc)
		
	def updateMongoGenomes(self,species):
		docs = []
		for specie in species:
			for genome in specie.genomes:
				docs.append({
				"game" : str(self.Inputs)+" "+str(self.Outputs)+" "+str(self.timeStamp),
				"genes" : self.getGenesBSON(genome.genes),
				"weightMaps" : self.getGeneWeightsBSON(genome.genes),
				"fitness" : genome.fitness,
				"maxNeuron" : genome.maxneuron,
				"mutationRates" : genome.mutationRates,
				"globalRank" : genome.globalRank,
				"maxNodes" : genome.maxNodes,
				"currentAge" : genome.mutationRates["age"],
				"inputs" : genome.Inputs,
				"Outputs" : genome.Outputs,
				"recurrent" : genome.recurrent,
				"parents" : genome.parents,
				"generation" : genome.ID["generation"],
				"genome"	 : genome.ID["position"]
				})
		db = self.client["runs"]
		collection = db["Genomes"]
		collection.insert_many(docs)
		
	
	def getGenesBSON(self,genes):
		genesArray = []
		for gene in genes:
			genesArray.append(gene.innovation)
		return genesArray
	
	def getGeneWeightsBSON(self,genes):
		genesWeightMapArray = []
		for gene in genes:
			genesWeightMapArray.append({
			str(gene.innovation) : gene.weight
			})
		return genesWeightMapArray
	

	def addToPool(self,children):
	
		self.generations.append([])
		for child in children:
			child.relatives = self.getRelatives(child)
			foundSpecies = False
			foundedSpecie = None
			maxRating = 0
			mates = []
			if child.relatives != None:
				for childRelative in child.relatives:
					for specie in range(len(self.species)):
						for genome in range(len(self.species[specie].genomes)):
							_genome = self.species[specie].genomes[genome]
							if childRelative in _genome.relatives or childRelative == _genome.ID:
								rating = self.sameSpecies(child,_genome,rating=True)
								if rating>0:
									if rating > maxRating:
										maxRating = rating
										foundedSpecie = specie
										print("family",rating,foundedSpecie)
										foundSpecies = True
									mates.append({
										"specie" : specie,
										"genome" : genome})
									child.mates.append(_genome.ID)


			child.ID = {
				"generation" : self.generation,
				"position" : len(self.generations[self.generation])
				}
			for mate in mates:
				specie = mate["specie"]
				genome = mate["genome"]
				self.species[specie].genomes[genome].mates.append(child.ID)
			self.generations[self.generation].append(child)
			if foundSpecies:
				self.species[foundedSpecie].genomes.append(child)
			if not foundSpecies:
				specie = self.newSpecies(self.Inputs,self.Outputs,self.recurrent)
				child.defining = True
				specie.genomes.append(child)
				self.species.append(specie)
				foundSpecies = True
			if self.client != None:
				doc = child.ID
				doc["game"] = self.gameName
				doc = {**doc,**child.parents}

				
				
	def getRelatives(self,child,parent=None):
		relatives = []
		if parent == None:
			genomeToCheck = child
		else:
			genomeToCheck = parent

		for parentGenomeDic in genomeToCheck.parents.values():
			
			if parentGenomeDic != None:
				generation = parentGenomeDic["generation"]
				position = parentGenomeDic["position"]
				parentGenome = self.generations[generation][position]
				if not generation == 0:
					if self.sameSpecies(child,parentGenome):
						if parentGenome.ID != None:
							relatives.append(parentGenome.ID)
							parentRelatives = self.getRelatives(child,parentGenome)
							if parentRelatives != None:
								relativesSet = (frozenset(relative.items()) for relative in relatives)
								parentRelatives = (frozenset(relative.items()) for relative in parentRelatives)
								newRelatives = set(parentRelatives).difference(relativesSet)
								newRelatives = [dict(tuple) for tuple in newRelatives]
								relatives.extend(list(newRelatives))
		if relatives != None or relatives != []:
			return relatives
				  
	def sameSpecies(self,genome1,genome2,rating=False):

		Threshold = genome1.mutationRates["DeltaThreshold"]
		DeltaDisjoint = genome1.mutationRates["DeltaDisjoint"]
		DeltaWeights = genome1.mutationRates["DeltaWeights"]

		dd = DeltaDisjoint*self.disjoint(genome1.genes,genome2.genes) #checks for genes
		dw = DeltaWeights*self.weights(genome1.genes,genome2.genes) # checks values in genes
		if rating:
			
			if	dd + dw > Threshold:
				return dd+dw - Threshold 
			else:
				return 0
		return dd + dw > Threshold

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
		for specie in self.species:
			specie.calculateAverageRemainingMultiplyer()
			specie.caclulateAverageBreedRate()
		if self.client != None:
			self.updateMongoGenomes(self.species)
		self.cullSpecies(False)  
		self.rankGlobally() 
		self.removeStaleSpecies()
		# reranks after removeing stales species and  stores best player for later play
		self.rankGlobally(addBest=True)
		for specie in self.species:
			#calculateAverageFitness of a specie
			specie.calculateAverageFitness()
			specie.calculateAverageRemainingRate()
			specie.calculateAverageCrossoverRate()
		self.removeWeakSpecies() 
		Sum = self.totalAverageFitness()
		c = 0
		for specie in self.species:
			for genome in specie.genomes:
				c += 1
		#defines new children list
		children = []
		while (len(children)+c < self.Population):
			for specie in self.species:
				breed = math.floor(specie.averageFitness / Sum * self.Population)-1 # if a species average fitness is over the pool averagefitness it can breed
				for i in range(breed):
						if len(children)+c < self.Population:
							children.append(specie.breedChildren())
		# leave only the top member of each species.
		self.cullSpecies(True) 
		self.cullOldSpecies()
		
		c = 0
		for specie in self.species:
			for genome in specie.genomes:
				c += 1
		while (len(children)+c < self.Population):
			parent = random.choice(self.species)
			child = parent.breedChildren()
			children.append(child)

		# adds all children to there species in the pool
		self.addToPool(children)



		
	def cullOldSpecies(self):
		species = []
		for specie in self.species:
			survivedGenomes = []
			for genome in specie.genomes:
				p = genome.currentAge
				if 1 < p:
					survivedGenomes.append(genome)
			if len(survivedGenomes) >= 1:
				specie.genomes = survivedGenomes
				species.append(specie)
		self.species = species


	def cullSpecies(self,cutToOne): #sorts genomes by fitness and removes half of them or cuts to one
		species = []
		
		for specie in self.species:
			specie.genomes = sorted(specie.genomes,key=attrgetter('fitness'),reverse=True)
			genomes = []
			if not cutToOne:
				remaining = math.ceil(len(specie.genomes)/specie.remainingMultiplyer)
			if cutToOne:
				remaining = specie.remainingRate
			cutoff = 0
			while (not len(genomes) > remaining) and len(specie.genomes) != len(genomes):
				genomes.append(specie.genomes[cutoff])
				cutoff +=1
			specie.genomes=genomes.copy()
			species.append(specie)
		self.species = species
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
		S= 0 
		for specie in self.species:
			breed = specie.averageFitness / sum * self.Population
			if breed >= specie.averageBreed:
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
		#self.client = MongoClient(self.databaseName, 27017)
		#db = self.client["runs"]
		#generationsCollection = db["Genomes"]
		#i = []
		#for gene in genes1:
		#	i.append(gene.innovation)
		#generation = generationsCollection.find({"game": self.gameName ,"generation":self.generation,"genes":{"$in":i}})
		#for specie in generation:
		#	print("specie",i)
		#	print(specie["genes"])
		i1 = []
		i2 = []
		for gene in genes1:
			i1.append(gene.innovation)
		for gene in genes2:
			i2.append(gene.innovation)
		disjointGenes = len(list(set(i1).intersection(i2)))
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
		if sum == 0 or coincident == 0:
			return 0
		return sum / coincident



	

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
			self.mutationRates["DeltaThreshold"] = .25
			self.mutationRates["DeltaDisjoint"] = 5
			self.mutationRates["DeltaWeights"] = 1
			self.mutationRates["crossoverRate"] = .2
			self.mutationRates["PerturbChance"] = 0.5
			self.mutationRates["ConectionCostRate"] = 1
			self.mutationRates["RemainingMultiplyer"] = 3
			self.mutationRates["Remaining"] = 2
			self.mutationRates["breed"] = 1
			self.mutationRates["age"] = 10
			self.currentAge = self.mutationRates["age"]
			self.parents = []
			self.relatives = []
			self.mates = []
			self.defining = False
			self.Inputs = Inputs
			self.Outputs = Outputs
			self.recurrent = recurrent
			self.ID = {
				"genome" : 0,
				"specie" : 0,
				"generation" : 0
			}
			self.defining = False
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
			genome2.parents["parent1"] = self.ID
			genome2.parents["parent2"] = None
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
			self.remainingRate = None
			self.remainingMultiplyer = 	None
			self.crossoverRate = None
			self.averageBreed = None
			
		def calculateAverageFitness(self): 
			total = 0
			for genome in self.genomes:
				total = total + genome.globalRank
			self.averageFitness = total / len(self.genomes)

		def calculateAverageRemainingRate(self):
			total = 0			
			for genome in self.genomes:
				total += genome.mutationRates["Remaining"] 
			total = total / len(self.genomes)
			self.remainingRate = total
		
		def caclulateAverageBreedRate(self):
			total = 0
			for genome in self.genomes:
				total += genome.mutationRates["breed"]
			total = total / len(self.genomes)
			self.averageBreed = total

		def calculateAverageRemainingMultiplyer(self):
			total = 0
			for genome in self.genomes:
				total += genome.mutationRates["RemainingMultiplyer"] 
			total = total / len(self.genomes)
			self.remainingMultiplyer = total

		def calculateAverageCrossoverRate(self):
			totalAverage = 0
			for genome in self.genomes:
				totalAverage += genome.mutationRates["crossoverRate"]
			self.crossoverRate = totalAverage/len(self.genomes)

		def breedChildren(self): # breeds children of a species
			genome1 = random.choice(self.genomes)
			if random.random() < self.crossoverRate and len(genome1.mates)>0:
				mate = random.choice(genome1.mates)
				generation = mate["generation"]
				position = mate["position"]
				genome2 = pool.generations[generation][position]
				child = self.crossover(genome1,genome2)
			else:
				child = genome1.copyGenome()
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
			print("PARNTS BREWEEDING",g1.ID,g2.ID)
			child.parents["parent1"] = g1.ID
			child.parents["parent2"] = g2.ID   
			return child

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
		







