# encoding: utf8

#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Particle Swarm Optimization (PSO) with Python
#   July, 2016
#   https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import random
import math
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import ceil, floor

global the_best_pso

# Semente do random
random.seed(17)

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

def cost_function(x):
    # F16
    #return x[:,0] * np.sin(np.sqrt(np.absolute(x[:,0]-(x[:,1]+9))))- (x[:,1]+9) * np.sin(np.sqrt(np.absolute(x[:,1]+0.5*x[:,0]+9)))
    # F15
    return -np.exp(0.2*np.sqrt((x[:,0] - 1)**2 + (x[:,1] - 1)**2) + (np.cos(2 * x[:,0]) + np.sin(2 * x[:,0])))
    # F12
    #return 0.5+(np.sin(np.sqrt(x[:,0]**2 + x[:,1]**2)**2) - 0.5) / (1 + 0.1 * (x[:,0]**2 + x[:,1]**2))

def my_sort(par, cost):
    #print 'Custo minimo:', np.min(cost),
    ordenar = dict()
    par_ordenado = np.zeros(par.shape)
    for i in xrange(len(cost)):
        ordenar[cost[i]] = i
    #print '| Par: ', par[ordenar[np.min(cost)]]
    cost = sorted(ordenar.keys())
    for i in xrange(len(cost)):
        par_ordenado[i] = par[ordenar[cost[i]],:]
    
    return par_ordenado, cost

def F10(x):
    return 20 + np.sum(x.T**2 - 10*np.cos(2 * math.pi * x.T)).T

def F11(x):
    return 1 + np.sum(abs(x.T)**2 / 4000).T - np.prod(np.cos(x.T)).T

def F12(x):
    return 0.5+(np.sin(np.sqrt(x[0]**2 + x[1]**2)**2) - 0.5) / (1 + 0.1 * (x[0]**2 + x[1]**2))

def F15(x):
    return -np.exp(0.2*np.sqrt((x[0] - 1)**2 + (x[1] - 1)**2) + (np.cos(2 * x[0]) + np.sin(2 * x[0])))

def F16(x):
    return x[0] * np.sin(np.sqrt(np.absolute(x[0]-(x[1]+9))))- (x[1]+9) * np.sin(np.sqrt(np.absolute(x[1]+0.5*x[0]+9)))

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc, func):
        if (func == 10):
            self.err_i = F10(np.array(self.position_i))
        elif (func == 11):
            self.err_i = F11(np.array(self.position_i))
        elif (func == 12):
            self.err_i = F12(np.array(self.position_i))
        elif (func == 15):
            self.err_i = F15(np.array(self.position_i))
        elif (func == 16):
            self.err_i = F16(np.array(self.position_i))
        

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i
        
        return self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g, w):
        #w = 0.5      # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1        # cognative constant
        c2 = 4 - c1   # social constant
        
        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter,func):
        global num_dimensions
        global the_best_pso

        num_dimensions=len(x0)          # Dimension of the problem
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group
        the_best_pso = np.zeros((maxiter, 2))
        minc = meanc = np.zeros(maxiter)

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        plt.ion()
        while i < maxiter:
            media = 0
            menor = 99999999
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                cost = swarm[j].evaluate(costFunc, func)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

                media += cost
                menor = min(cost, menor)
            
            meanc[i] = media / num_particles
            minc[i] = menor

            # Posição da melhor
            the_best_pso[i] = np.asarray(pos_best_g)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g, (maxiter - i) / maxiter)
                swarm[j].update_position(bounds)
            i+=1

            positions = np.zeros((num_particles, 2))
            for j in xrange(num_particles):
                positions[j,:] = swarm[j].position_i

            plt.cla()
            plt.clf()
            plt.plot(positions[:,0], positions[:,1], '*', the_best_pso[i-1,0], the_best_pso[i-1,1], 'r*')
            plt.axis([bounds[0][0] - 5, bounds[0][1] + 5, bounds[1][0] - 5, bounds[1][1] + 5])
            plt.title('Position of Particules')
            plt.text(bounds[0][0] - (abs(bounds[0][0]) * 0.2), bounds[1][0] - (abs(bounds[1][1]) * 0.2), 'Iteration: ' + str(i))
            if (i == maxiter - 1):
                plt.ioff()
                plt.pause(0.1)
            else:
                plt.pause(0.1)
        
        # print final results
        print 'FINAL:'
        print pos_best_g
        print err_best_g

        X = np.arange(bounds[0][0], bounds[0][1], 1)
        Y = np.arange(bounds[1][0], bounds[1][1], 1)
        Z = np.zeros((len(X), len(Y)))

        for i in xrange(len(X)):
            for j in xrange(len(Y)):
                if (func == 10):
                    Z[i,j] = F10(np.array([X[i], Y[j]]))
                elif (func == 11):
                    Z[i,j] = F11(np.array([X[i], Y[j]]))
                elif (func == 12):
                    Z[i,j] = F12(np.array([X[i], Y[j]]))
                elif (func == 15):
                    Z[i,j] = F15(np.array([X[i], Y[j]]))
                elif (func == 16):
                    Z[i,j] = F16(np.array([X[i], Y[j]]))
        
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        if (func == 10):
            plt.title('F10')
        elif (func == 11):
            plt.title('F11')
        elif (func == 12):
            plt.title('F12')
        elif (func == 15):
            plt.title('F15')
        elif (func == 16):
            plt.title('F16')

        plt.figure(2)
        plt.subplot(211)
        plt.title('Mean Cost')
        plt.plot(meanc, 'r--')
        plt.subplot(212)
        plt.title('Minimun Cost')
        plt.plot(minc, 'g')

        # Plotando a trajetória do PSO
        plt.figure(3)
        plt_the_best, = plt.plot(the_best_pso[:,0], the_best_pso[:,1], 'b--', label="the_best")
        pos_inicial, = plt.plot(the_best_pso[0,0], the_best_pso[0,1], 'go', label="pos_inicial")
        pos_final, = plt.plot(the_best_pso[i - 1,0], the_best_pso[i - 1,1], 'ro', label="pos_final")
        plt.legend([plt_the_best, pos_inicial, pos_final], ['Trajetoria do PSO', 'Pos. Inicial', 'Pos. Final'])
        plt.axis([bounds[0][0] - 5, bounds[0][1] + 5, bounds[1][0] - 5, bounds[1][1] + 5])
        plt.title('PSO - Trajetoria do Melhor')

        plt.show()

if __name__ == "__PSO__":
    main()

#--- RUN ----------------------------------------------------------------------+

initial=[random.random(),random.random()]               # initial starting location [x1,x2...]
bounds=[(-20,20),(-20,20)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(func1,initial,bounds,num_particles=100,maxiter=50,func=15)

#--- END ----------------------------------------------------------------------+

# ALGORITMO GA
# Parte I - Configurando o GA
npar = 2
varhi = 20 # Limite superior das variaveis
varlo = -20 # Limite inferior das variaveis

# Parte II - Criterios de parada
maxit = 50 # numero maximo de iteracoes
mincost = -9999999 # custo minimo

# Parte III - Parametros do GA
popsize = 100
mutrate = 0.2 # Taxa de mutação
selection = 0.5 # A porcentagem de indivíduos que irão permancer na população
Nt = npar
keep = int(floor(selection * popsize)) # Quantidade de indivíduos que irão permanecer na população
nmut = int(ceil((popsize - 1) * Nt * mutrate)) # Quantos indíviduos sofrerão mutação
M = ceil((popsize - keep) / 2.0) # Quantidade de cruzamentos (está associado ao número de indivíduos que irão morrer)

# Parte IV - Criando a população inicial
iga = 0
#par = (varhi - varlo) * np.random.rand(popsize, npar) + varlo
#par = (10 - (5)) * np.random.rand(popsize, npar) + (5)
par = (1 - (0)) * np.random.rand(popsize, npar) + (0)
cost = cost_function(par)
par, cost = my_sort(par, cost)
minc = np.zeros(maxit + 1)
meanc = np.zeros(maxit + 1)
minc[0] = np.min(cost)
meanc[0] = np.mean(cost)

# Exibindo a situação inicial das partículas
plt.figure(0)
plt.ion()
plt.cla()
plt.clf()
the_best = par[0]
plt.plot(par[:,0], par[:,1], '.', the_best[0], the_best[1], 'r*')
plt.axis([varlo - 5, varhi + 5, varlo - 5, varhi + 5])
plt.title('Position of Particules')
#plt.text(varlo + (abs(varlo) * 0.03), varhi + (abs(varhi) * 0.03), 'Iteration: ' + str(1))
plt.pause(3)

# Inicializando o vetor com a posição do melhor (the best)
the_best_ga = np.zeros((maxit + 1, 2))
the_best_ga[0] = par[0]

# Iterando por meio das geracoes
while iga <= maxit - 1:
    iga += 1
    M = ceil((popsize - keep) / 2) # Numero de cruzamentos
    prob = np.flipud(np.linspace(1, keep, keep).T / np.sum(np.linspace(1, keep, keep))) # Pesos dos cromossomos
    odds = np.insert(np.cumsum(prob[0:int(keep)]).T, 0, 0) # Funcao de distribuicao de probabilidade
    pick1 = np.random.rand(int(M)) # Cruzamento 1
    pick2 = np.random.rand(int(M)) # Cruzamento 2
    # ma e pa contêm os indices dos cromossomos que estarão cruzando
    ma = np.zeros(pick1.shape).astype(int)
    pa = np.zeros(pick2.shape).astype(int)
    ic = 0
    while ic < M:
        for idx in xrange(1, int(keep) + 1):
            if pick1[ic] <= odds[idx] and pick1[ic] > odds[idx - 1]:
                ma[ic] = idx - 1
            if pick2[ic] <= odds[idx] and pick2[ic] > odds[idx - 1]:
                pa[ic] = idx - 1
        ic += 1

    # Executa o cruzamento usando o crossover de ponto único
    # Na página 25 do material desse link (http://omnipax.com.br/livros/2013/MHPO/mhpo-cap02.pdf)
    # contém informações sobre esse tipo de cruzamento
    ix = np.arange(0, int(keep), 2)
    xp = (np.floor(np.random.rand(int(M)) * Nt)).astype(int)
    r = np.random.rand(int(M))
    for ic in xrange(int(M)):
        xy = par[ma[ic], xp[ic]] - par[pa[ic], xp[ic]]
        par[keep + ix[ic],:] = par[ma[ic],:] # Obtém a posição do primeiro filho, que inicia nas posições dos indivíduos que estão com os maiores custos. Por exemplo, se eu devo manter 500 indivíduos, eu tenho 1000 - 500 = 500 indivíduos que devem ser descartados, logo, teremos 250 cruzamentos. Assim, iremos substituir os indivíduos da população que estão da posição 500 adiante.
        par[keep + ix[ic] + 1,:] = par[pa[ic],:] # Obtém a posição do segundo filho, que inicia nas posições dos indivíduos que estão com os maiores custos e sempre é a posição posterior a do primeiro filho (por isso, ix incrementa de dois em dois)
        par[keep + ix[ic],xp[ic]] = par[ma[ic],xp[ic]] - r[ic] * xy
        par[keep + ix[ic] + 1,xp[ic]] = par[pa[ic],xp[ic]] + r[ic] * xy
        if xp[ic] < npar - 1:
            par[keep + ix[ic],0], par[keep + ix[ic],1] = par[keep + ix[ic], 0], par[keep + ix[ic] + 1,xp[ic]+1:npar]
            par[keep + ix[ic]+1,0], par[keep + ix[ic]+1,1] = par[keep + ix[ic] + 1, 0], par[keep + ix[ic],xp[ic]+1:npar]
    
    # Mutação da população
    # Cria um vetor 499 números aleatórios (entre 0 e 999) para escolher a linha em que vai ocorrer a mutação
    mrow = np.sort(np.ceil(np.random.rand(nmut) * (popsize - 1))).astype(int)
    # Cria um vetor com números aleatórios entre (0 e 1) para escolher a linha em que vai ocorrer a mutação
    mcol = np.floor(np.random.rand(nmut) * Nt).astype(int)

    for i in xrange(nmut):
        par[mrow[i], mcol[i]] = (varhi - varlo) * random.random() + varlo

    cost = cost_function(par)
    par, cost = my_sort(par, cost)

    minc[iga] = np.min(cost)
    meanc[iga] = np.mean(cost)
    the_best_ga[iga] = par[0]

    plt.cla()
    plt.clf()
    plt.plot(par[:,0], par[:,1], '.', the_best_ga[iga, 0], the_best_ga[iga,1], 'r*')
    plt.axis([varlo - 5, varhi + 5, varlo - 5, varhi + 5])
    plt.title('Position of Particules')
    plt.text(varlo - (abs(varlo) * 0.2), varlo - (abs(varhi) * 0.2), 'Iteration: ' + str(iga))
    if (iga > maxit - 1):
        plt.ioff()
        plt.show()
    else:
        plt.pause(0.05)

    if (iga > maxit) or cost[0] < mincost:
        break

print par[0], cost_function(par)[0]

plt.figure(1)
plt.subplot(211)
plt.title('Mean Cost')
plt.plot(meanc, 'r--')
plt.subplot(212)
plt.title('Minimun Cost')
plt.plot(minc, 'g')

# Plotando a trajetória do GA
plt.figure(2)
plt_the_best, = plt.plot(the_best_ga[:,0], the_best_ga[:,1], 'b--', label="the_best")
pos_inicial, = plt.plot(the_best_ga[0,0], the_best_ga[0,1], 'go', label="pos_inicial")
pos_final, = plt.plot(the_best_ga[iga - 1,0], the_best_ga[iga - 1,1], 'ro', label="pos_final")
plt.legend([plt_the_best, pos_inicial, pos_final], ['Trajetoria do GA', 'Pos. Inicial', 'Pos. Final'])
plt.axis([varlo - 5, varhi + 5, varlo - 5, varhi + 5])
plt.title('GA - Trajetoria do Melhor')
plt.show()

# Plotando a trajetória do GA e do PSO
plt.figure(3)
plt_the_best_ga, = plt.plot(the_best_ga[:,0], the_best_ga[:,1], 'k--', label="the_best_ga")
plt_the_best_pso, = plt.plot(the_best_pso[:,0], the_best_pso[:,1], 'y--', label="the_best_pso")
pos_inicial, = plt.plot(the_best_ga[0,0], the_best_ga[0,1], 'go', label="pos_inicial_ga")
pos_final_ga, = plt.plot(the_best_ga[iga - 1,0], the_best_ga[iga - 1,1], 'b*', label="pos_final_ga")
pos_final_pso, = plt.plot(the_best_pso[iga - 1,0], the_best_pso[iga - 1,1], 'r*', label="pos_final_pso")
plt.legend([plt_the_best_ga, plt_the_best_pso, pos_inicial, pos_final_ga, pos_final_pso], ['Trajetoria do GA', 'Trajetoria do PSO', 'Pos. Inicial', 'Pos. Final (GA)', 'Pos. Final (PSO)'])
plt.axis([varlo - 5, varhi + 5, varlo - 5, varhi + 5])
plt.title('GA - Trajetoria do Melhor')
plt.show()