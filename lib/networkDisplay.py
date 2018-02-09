


from tkinter import *
import math
import time
import queue

width = 1530
height = 480



class newNetworkDisplay(Toplevel):
    def __init__(self,displayQueue):
        Toplevel.__init__(self)
        #Set up the main window frame as a grid
        #self.grid() 
        self.Frame = Frame(self, width = width, height = height)
        self.Frame.grid()
        #Add a canvas to frame1 as self.canvas member 
        self.canvas = Canvas(self.Frame, width = width, height = height,bg ="white")
        self.lastGenome = None
        self.cells = {} 
        self.lines = {}
        self.drawnCells = {}
        self.drawnLines = {}
        self.initialized = False
        self.displayQueue = displayQueue
        self.checkDisplayQueue()

        
        
    def update(self,genome):
        self.canvas.delete("all")
        self.lastGenome = None
        self.cells = {} 
        self.lines = {}
        self.drawnCells = {}
        self.drawnLines = {}
        self.placeInputNeurons(genome)
        self.placeOutputNeurons(genome)
        self.placeHiddenNeurons(genome)
        self.mutationRates(genome)
        self.drawCells() 
        self.drawLines(genome)
        self.canvas.focus_set()
        self.canvas.pack()
        self.canvas.update()

    def _quit(self):
        self.Tk.destroy()
        self.Tk.quit()
        #self.master.destroy()
        #self.master.quit()

    def placeInputNeurons(self,genome):
        if genome.recurrent:
            Inputs = genome.Inputs + genome.Outputs
        else:
            Inputs = genome.Inputs
        neurons = genome.neurons
        for Input in range(Inputs+1): # add 1 for python indexing
            percent  = (Input+1)/(Inputs+1)
            cell = Cell()
            if (height//Inputs) <5:
                 cell.gridSize = 5
            else:
                 cell.gridSize = (height//(Inputs+1)) 
            cell.x = 0
            cell.y = percent*height
            cell.value=neurons[Input].value
            self.cells[Input] = cell



    def placeHiddenNeurons(self,genome):
        Outputs = genome.Outputs
        if genome.recurrent:
            Inputs = genome.Inputs + genome.Outputs
        else:
            Inputs = genome.Inputs
        maxNodes = genome.maxNodes
        neurons = genome.neurons

        for k,v in neurons.items():

            if k > Inputs and k < maxNodes: 
                    cell = Cell()
                    cell.x = width*.5
                    cell.y = height
                    cell.gridSize = 10
                    cell.value = v.value
                    self.cells[k] = cell


        for gene in genome.genes:
            if gene.enabled:
                c1 = self.cells[gene.into]
                c2 = self.cells[gene.out]
                displace = width*.2
                leftLimit = height//(Inputs)
                rightLimit = width - (height//Outputs)
                if gene.into > Inputs and gene.into <= maxNodes:
                    c1.x = 0.65*c1.x + 0.15*c2.x
                    if c1.x>= c2.x:
                        c1.x - c1.x-displace
                    if c1.x < leftLimit:
                        c1.x = leftLimit+displace
                    if c1.x > rightLimit:
                        c1.x = rightLimit-displace
                    c1.y = 0.75*c1.y + 0.30*c2.y
                if gene.out > Inputs and gene.out <= maxNodes:
                    c2.x = 0.15 * c1.x + 0.65*c2.x
                    if c1.x >= c2.x:
                        c2.x = c2.x+displace
                    if c2.x < leftLimit:
                        c2.x = leftLimit+displace
                    if c2.x > rightLimit:
                        c2.x = rightLimit-displace
                    c2.y = 0.15*c1.y + 0.60*c2.y
        return
      

    def placeOutputNeurons(self,genome):
        Outputs = genome.Outputs
        maxNodes = genome.maxNodes
        neurons = genome.neurons
        

        for Output in range(1,Outputs+1):
            percent  = Output/Outputs
            cell = Cell()
            if (height//Outputs) <5:
                cell.gridSize = 5
            else:
                cell.gridSize = (height//Outputs) 
            cell.x = width*.9
            cell.y = percent*height
            cell.value=neurons[maxNodes+Output].value
            self.cells[maxNodes+Output] = cell
        for o in range(Outputs):
            cell = self.cells[maxNodes+Output]


    def mutationRates(self,genome):  
        if genome.recurrent:
            Inputs = genome.Inputs + genome.Outputs
        else:
            Inputs = genome.Inputs
        x = (height//(Inputs+1))+100
        y = 10
        for mutation,Rate in genome.mutationRates.items():
            self.canvas.create_text((x,y),text='{} {}'.format(mutation,str(round(Rate,2))))
            x += (.05*width)+25 
        self.canvas.create_text((x,y),''.join(genome.ID))




    def pickCellColor(self,value):
        color = "#000000"
        if value > 0:
            adjustedValue = math.floor((value)/2*256)
            if adjustedValue > 255:
                adjustedValue = 255
            if adjustedValue < 0:
                adjustedValue = 0
            color = '#%02x%02x%02x' % (0, 0,adjustedValue)
        if value < 0:
            adjustedValue =  math.floor((math.fabs(value))/2*256)
            if adjustedValue > 255:
                adjustedValue = 255
            if adjustedValue < 0:
                adjustedValue = 0
            color = '#%02x%02x%02x' % (adjustedValue, 0,0)
        return color
		
    def drawCells(self):
        for k,cell in self.cells.items():
            color = self.pickCellColor(cell.value)
            self.drawnCells[cell] = self.canvas.create_rectangle(cell.x-cell.gridSize,cell.y-cell.gridSize,cell.x+cell.gridSize,cell.y+cell.gridSize,fill=color)

    def pickLineColor(self,weight,other):
        if other == 0:
            return "#000000"
        color = "#000000"
        if weight > 0:
            value = int(weight*255)
            if value > 255:
                value = 255
            color = '#%02x%02x%02x' % (0, 0,value)
        if weight < 0:
            value = int(math.fabs(weight)*255)
            if value > 255:
                value = 255
            color = '#%02x%02x%02x' % (value, 0,0)
        return color
				
    def drawLines(self,genome):
        genes = genome.genes
        for gene in genes:
            if gene.enabled:
                c1 = self.cells[gene.into]
                c2 = self.cells[gene.out]
                color = self.pickLineColor(gene.weight,genome.neurons[gene.into].value)
                self.drawnLines[gene.innovation] = self.canvas.create_line(c1.x-(c1.gridSize//2), c1.y-(c1.gridSize//2), c2.x-(c2.gridSize//2), c2.y-(c2.gridSize//2),fill=color,width=3)

    def updateCanvas(self,genome):
        if self.lastGenome != genome.ID:
            self.lastGenome= genome.ID
            self.update(genome)
        self.updateCells(genome)
        self.updateLines(genome)
        self.canvas.focus_set()
        self.canvas.pack()
        self.canvas.update()
	
    def updateCells(self,genome):
        neurons = genome.neurons
        for k,neuron in neurons.items():
            cell = self.cells[k]
            rectangle = self.drawnCells[cell]
            self.canvas.itemconfig(rectangle,fill=self.pickCellColor(neuron.value))
		
    def updateLines(self,genome):
        for gene in genome.genes:
            if gene.enabled:
                    color = self.pickLineColor(gene.weight,genome.neurons[gene.into].value)
                    self.canvas.itemconfig(self.drawnLines[gene.innovation],fill=color)
    
    def checkDisplayQueue(self):
        try:
            genome = self.displayQueue.get_nowait()
            self.updateCanvas(genome)
            self.after(250,self.checkDisplayQueue)
        except queue.Empty:
            self.after(250,self.checkDisplayQueue)





class Cell():
    def __init__(self):
        self.x= None
        self.y = None
        self.value = None
        self.gridSize = None


