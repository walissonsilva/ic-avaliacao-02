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

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def F1(x):
    return x[0]**2 + x[1]**2

def F2(x):
    return (-3 * x[1]) / (x[0]**2 + x[1]**2 + 200)#np.exp((x[0] - x[1]) / 30) * np.sin(x[0] * 0.15) * np.cos(x[1] * 0.15)

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

def F17(x):
    return x[0]**2 + x[1]**2

#--- MAIN ---------------------------------------------------------------------+

"""class Particle:
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
        if (func == 1):
            self.err_i = F1(np.array(self.position_i))
        elif (func == 2):
            self.err_i = F2(np.array(self.position_i))
        elif (func == 10):
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
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

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
            plt.xlabel('x')
            plt.ylabel('y')
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
                if (func == 1):
                    Z[i,j] = F1(np.array([X[i], Y[j]]))
                elif (func == 2):
                    Z[i,j] = F2(np.array([X[i], Y[j]]))
                elif (func == 10):
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
        plt.xlabel('x')
        plt.ylabel('y')
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
# initial starting location [x1,x2...]
initial=[random.random() - 15,random.random() - 15]
bounds=[(-20,20),(-20,20)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(F1,initial,bounds,num_particles=20,maxiter=50,func=2)

#--- END ----------------------------------------------------------------------+"""

def cost_function(x, func):
    if (func == 1):
        return x[:,0]**2 + x[:,1]**2
    elif (func == 2):
        return (-3 * x[:,1]) / (x[:,0]**2 + x[:,1]**2 + 200)

def find_matlab(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

func_cost = 2
popsize = 20
npar = 2
maxit = 50
c1 = 1
c2 = 4 - c1
C = 1

varhi = 20 # Limite superior das variaveis
varlo = -20 # Limite inferior das variaveis

#par = np.random.rand(popsize, npar)
par = (-10 - (-15)) * np.random.rand(popsize, npar) + (-15)
vel = np.random.rand(popsize, npar)

cost = cost_function(par, func_cost)

minc = np.zeros(maxit + 1)
meanc = np.zeros(maxit + 1)
globalmin = np.zeros(maxit + 1)

minc[0] = np.min(cost)
meanc[0] = np.mean(cost)

globalmin[0] = minc[0]

localpar = par
localcost = cost
globalcost = min(cost)
pos = cost.tolist().index(globalcost)
globalpar = par[pos,:]

the_best_pso = np.zeros((maxit + 1, npar))
the_best_pso[0] = globalpar

# Start iterations
it = 0
plt.ion()
while it < maxit:
    w = (maxit - it) / maxit

    r1 = np.random.rand(popsize, npar)
    r2 = np.random.rand(popsize, npar)
    vel = C * (w * vel + c1 * r1 * (localpar - par) + c2 * r2 * (np.ones((popsize,1)) * globalpar - par))

    par += vel
    overlimit = par <= varhi
    underlimit = par >= varlo
    par = par * overlimit + np.logical_not(overlimit)
    par= par * underlimit

    cost = cost_function(par, func_cost)
    bettercost = cost < localcost
    localcost = localcost * np.logical_not(bettercost) + cost * bettercost
    idx = find_matlab(bettercost.tolist(), lambda x: x > 0)
    localpar[idx,:] = par[idx,:]
    
    temp = min(cost)
    t = cost.tolist().index(temp)
    
    if temp < globalcost:
        globalpar = par[t,:]
        indx = t
        globalcost = temp
    
    it += 1
    minc[it] = np.min(cost)
    globalmin[it] = globalcost
    meanc[it] = np.mean(cost)

    # Exibindo a nova posição das partículas
    plt.cla()
    plt.clf()
    the_best_pso[it] = par[t,:]
    plt.plot(par[:,0], par[:,1], '*', the_best_pso[it,0], the_best_pso[it,1], 'r*')
    plt.axis([varlo - 5, varhi + 5, varlo - 5, varhi + 5])
    plt.title('Position of Particules')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.text(varlo - (abs(varlo) * 0.2), varlo - (abs(varlo) * 0.2), 'Iteration: ' + str(it))
    if (it == maxit):
        plt.ioff()
        plt.show()
    else:
        plt.pause(0.1)


plt.figure(3)
plt_the_best, = plt.plot(the_best_pso[:,0], the_best_pso[:,1], 'b--', label="the_best")
pos_inicial, = plt.plot(the_best_pso[0,0], the_best_pso[0,1], 'go', label="pos_inicial")
pos_final, = plt.plot(the_best_pso[it - 1,0], the_best_pso[it - 1,1], 'ro', label="pos_final")
plt.legend([plt_the_best, pos_inicial, pos_final], ['Trajetoria do PSO', 'Pos. Inicial', 'Pos. Final'])
plt.axis([varlo - 5, varhi + 5, varlo - 5, varhi + 5])
plt.title('PSO - Trajetoria do Melhor')
plt.xlabel('x')
plt.ylabel('y')
plt.show()