import gym
from lib import neat as ai
import universe



def trainNetwork(env):   
    beaten = False
    while not beaten:
        for i in range(population):
           ob = env.reset()
           ai.initializeRun()
           cum_reward = 0
           for j in range(max_steps):
              env.render()
              o = ai.evaluateCurrent(ob.tolist())
              ob, reward, done, _ = env.step(o)
              cum_reward +=reward
              if done:
                 print(ai.pool.currentSpecies,ai.pool.currentGenome, "scored: ",cum_reward)
                 ai.setFitness(cum_reward)     
                 if cum_reward > 350:
                    beaten  = True
                    return ai.pool.species[ai.pool.currentSpecies].genomes[ai.pool.currentGenome]
                 ai.nextGenome() 
                 break
        print("******* new gen best: ",ai.getBest())
        ai.newGeneration()
    
       
    

if __name__ == '__main__':
    population = 50
    env = gym.make('CartPole-v1')
    #print(env.observation_space)
    ai.initializePool(population,len(env.observation_space.high),env.action_space.n)
    max_steps = 400
    
    net = trainNetwork(env)
    
    while True:
       ob = env.reset()    
       for w in range(max_steps):
           env.render()
           o = ai.evaluateCurrent(ob.tolist())
           ob, reward, done, _ = env.step(o)
           if done:
               break











        

