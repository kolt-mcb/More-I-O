from tkinter import *
from tkinter import filedialog,messagebox
import pickle
from lib import networkDisplay


class gui:
    def __init__(self, master):
        self.master = master
        self.frame = Frame(self.master)
        self.frame.grid()
        self.button1 = Button(self.frame,text="button1",command=self.button1)
        self.button2 = Button(self.frame,text="button2",command=self.button2)
        self.button3 = Button(self.frame,text="button3",command=self.loadFile)
        self.button1.grid(row=0, column=0)
        self.button2.grid(row=0, column=1)
        self.button3.grid(row=0, column=2)
        self.best = []
        self.currentGenome = 0
        self.display = None



    def button1(self):
        if self.currentGenome >= 0:
            self.currentGenome -= 1
            self.refreshNetworkDisplay()

    
    def button2(self):
        self.currentGenome +=1
        self.refreshNetworkDisplay()

    def refreshNetworkDisplay(self):
        self.display._quit()
        genome = self.best[self.currentGenome]
        self.display = networkDisplay.newNetworkDisplay(genome)
        self.display.Tk.mainloop()

        

    def loadFile(self):
        filename = filedialog.askopenfilename()
        if filename is ():
            return
        f = open(filename,"rb")
        loadedPool = pickle.load(f)
        self.best = loadedPool[1]
        for genome in self.best:
            print(genome.fitness)
        f.close()
        genome = self.best[0]
        self.display = networkDisplay.newNetworkDisplay(genome)
        self.display.Tk.mainloop()
    
    def onClosing(self):
        if messagebox.askokcancel("Quit","do you want to Quit?"):
            if self.display != None:
                self.display._quit()
            self.master.destroy()
            self.master.quit()
            

if __name__ == '__main__':
    
    root = Tk()
    root.resizable(width=False,height=False)
    app = gui(root)
    root.protocol("WM_DELETE_WINDOW",app.onClosing)
    root.mainloop()