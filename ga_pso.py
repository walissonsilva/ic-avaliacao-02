# encoding: utf8

from math import ceil, floor
import random
import matplotlib.pyplot as plt
import numpy as np

def cost_function(x):
    # F16
    #return x[:,0] * np.sin(np.sqrt(np.absolute(x[:,0]-(x[:,1]+9))))- (x[:,1]+9) * np.sin(np.sqrt(np.absolute(x[:,1]+0.5*x[:,0]+9)))
    # F15
    #return -np.exp(0.2*np.sqrt((x[:,0] - 1)**2 + (x[:,1] - 1)**2) + (np.cos(2 * x[:,0]) + np.sin(2 * x[:,0])))
    # F12
    return 0.5+(np.sin(np.sqrt(x[:,0]**2 + x[:,1]**2)**2) - 0.5) / (1 + 0.1 * (x[:,0]**2 + x[:,1]**2))

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
par = (10 - (5)) * np.random.rand(popsize, npar) + (5)
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
plt.pause(1)

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