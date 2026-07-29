import matplotlib.pyplot as plt 
import torch 
import random 
import numpy as np
from collections import deque
from AI import AI,Direcao,Ponto,TAMANHO_BLOCO
from metodo_Q import Q_linear,Treino_Q
import plot
MAX_MEMORY = 50_000
BATCH_SIZE = 500
taxa_aprendizado = 0.002


class agente:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # aleatoriedade
        self.gamma = 0.8 # taxa desconto
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Q_linear(11,256,3)
        self.trainer = Treino_Q(self.model,taxa_aprendizado,self.gamma)
    # estado (11 valores) -> arbitrario
    #[ bater_reto, bater_direita, bater_esquerda,
    #   
    # direcao esquerda, direcao direita,
    # direcao cima, direcao baixo
    # 
    # comida esquerda,comida direita,
    # comida cima, comida baixo]
    def estado(self,game):
        head = game.cobra[0]
        point_l=Ponto(head.x - TAMANHO_BLOCO, head.y)
        point_r=Ponto(head.x + TAMANHO_BLOCO, head.y)
        point_u=Ponto(head.x, head.y - TAMANHO_BLOCO)
        point_d=Ponto(head.x, head.y + TAMANHO_BLOCO)

        dir_l = game.direction == Direcao.LEFT
        dir_r = game.direction == Direcao.RIGHT
        dir_u = game.direction == Direcao.UP
        dir_d = game.direction == Direcao.DOWN

        estado = [
            # bater_reto
            (dir_u and game.colisao(point_u))or
            (dir_d and game.colisao(point_d))or
            (dir_l and game.colisao(point_l))or
            (dir_r and game.colisao(point_r)),

            # bater direita
            (dir_u and game.colisao(point_r))or
            (dir_d and game.colisao(point_l))or
            (dir_u and game.colisao(point_u))or
            (dir_d and game.colisao(point_d)),

            #bater esquerda
            (dir_u and game.colisao(point_r))or
            (dir_d and game.colisao(point_l))or
            (dir_r and game.colisao(point_u))or
            (dir_l and game.colisao(point_d)),

            # direcao movimento
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #posicao comida
            game.comida.x < game.head.x, # esquerda
            game.comida.x > game.head.x, # direita
            game.comida.y < game.head.y, # cima
            game.comida.y > game.head.y  # baixo
        ]
        return np.array(estado,dtype=int)

    def guardar(self,estado, acao, recompensa, proximo_estado,done):
        self.memory.append((estado,acao,recompensa,proximo_estado,done)) # popleft if memory exceed

    def treino_longo(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        estados, acoes, recompensas, proximos_estados, dones = zip(*mini_sample)
        self.trainer.etapa_treino(estados,acoes,recompensas,proximos_estados,dones)

    def treino_curto(self, estado, acao, recompensa, proximo_estado, done):
        self.trainer.etapa_treino(estado, acao, recompensa, proximo_estado, done)

    def jogada(self,estado):
        # movimentos random
        self.epsilon = 80 - self.n_game
        movimento_final = [0,0,0]
        if(random.randint(0,200)<self.epsilon):
            movimento = random.randint(0,2)
            movimento_final[movimento]=1
        else:
            state0 = torch.tensor(estado,dtype=torch.float)
            previsao = self.model(state0) # previsao do modelo
            movimento = torch.argmax(previsao).item()
            movimento_final[movimento]=1 
        return movimento_final

def treino():
    total_score = 0
    record = 0
    agent = agente()
    game = AI()
    eventos = 0
    while eventos < 500:
        # estado antigo
        antigo_estado = agent.estado(game)

        # acao atual
        movimento_final = agent.jogada(antigo_estado)

        # executa e pega uma nova acao
        recompensa, done, score = game.jogada(movimento_final)
        novo_estado = agent.estado(game)

        # treino curto
        agent.treino_curto(antigo_estado,movimento_final,recompensa,novo_estado,done)

        #guardar
        agent.guardar(antigo_estado,movimento_final,recompensa,novo_estado,done)

        if done:
            # treino longo,plot 
            eventos+= 1
            game.reset()
            agent.n_game += 1
            agent.treino_longo()
            agent.model.guardar()
            if(score > recompensa): 
                recompensa = score
            if(score > record):
                record = score
                agent.model.guardar()
            print('Partida:',agent.n_game,'Placar:',score,'Record:',record)

if(__name__=="__main__"):
    treino()