import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
pygame.init()
minha_fonte = pygame.font.SysFont('times new roman', 50)


class Direcao(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
 
Ponto = namedtuple('Point','x , y')

TAMANHO_BLOCO=20
VELOCIDADE = 40
preto = pygame.Color(0, 0, 0)
branco = pygame.Color(255, 255, 255)
vermelho= pygame.Color(255, 0, 0)
azul = pygame.Color(0, 255, 0)
preto = pygame.Color(0, 0, 255)

class AI:
    def __init__(self,janela_x=640,janela_y=640):
        self.w=janela_x
        self.h=janela_y
        #display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Cobrinha')
        self.clock = pygame.time.Clock()
        
        #estado inicial
        self.reset()
    def reset(self):
        self.direction = Direcao.RIGHT
        self.head = Ponto(self.w/2,self.h/2)
        self.cobra = [self.head,
                      Ponto(self.head.x-TAMANHO_BLOCO,self.head.y),
                      Ponto(self.head.x-(2*TAMANHO_BLOCO),self.head.y)]
        self.placar = 0
        self.comida = None
        self.posicao_comida()
        self.frame_iteration = 0
      

    def posicao_comida(self):
        x = random.randint(0,(self.w-TAMANHO_BLOCO)//TAMANHO_BLOCO)*TAMANHO_BLOCO
        y = random.randint(0,(self.h-TAMANHO_BLOCO)//TAMANHO_BLOCO)*TAMANHO_BLOCO
        self.comida = Ponto(x,y)
        if(self.comida in self.cobra):
            self.posicao_comida()


    def jogada(self,acao):
        self.frame_iteration+=1
        # 1. input
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()
            
        # 2. movimento
        self.movimento(acao)
        self.cobra.insert(0,self.head)

        # 3. game over
        recompensa = 0  # comer: +10 , game over: -10 , else: 0
        game_over = False 
        if(self.colisao() or self.frame_iteration > 100*len(self.cobra) ):
            game_over=True
            recompensa = -10
            return recompensa,game_over,self.placar
        # 4. nova comida ou movimento
        if(self.head == self.comida):
            self.placar+=1
            recompensa=10
            self.posicao_comida()
            
        else:
            self.cobra.pop()
        
        # 5. atualizar hud
        self.hud()
        self.clock.tick(VELOCIDADE)
        # 6. game over e placar
        
        return recompensa,game_over,self.placar

    #atualizar hud
    def hud(self):
        self.display.fill(preto)
        for pt in self.cobra:
            pygame.draw.rect(self.display,branco,pygame.Rect(pt.x,pt.y,TAMANHO_BLOCO,TAMANHO_BLOCO))
            pygame.draw.rect(self.display,azul,pygame.Rect(pt.x+4,pt.y+4,12,12))
        pygame.draw.rect(self.display,vermelho,pygame.Rect(self.comida.x,self.comida.y,TAMANHO_BLOCO,TAMANHO_BLOCO))
        text = minha_fonte.render("Placar: "+str(self.placar),True,branco)
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def movimento(self,acao):
        # Acao
        # [1,0,0] -> Reto (nao faz nada)
        # [0,1,0] -> direita
        # [0,0,1] -> esquerda

        clock_wise = [Direcao.RIGHT,Direcao.DOWN,Direcao.LEFT,Direcao.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(acao,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(acao,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # direita
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # esquerda
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if(self.direction == Direcao.RIGHT):
            x+=TAMANHO_BLOCO
        elif(self.direction == Direcao.LEFT):
            x-=TAMANHO_BLOCO
        elif(self.direction == Direcao.DOWN):
            y+=TAMANHO_BLOCO
        elif(self.direction == Direcao.UP):
            y-=TAMANHO_BLOCO
        self.head = Ponto(x,y)

    def colisao(self,pt=None):
        if(pt is None):
            pt = self.head
        #hit boundary
        if(pt.x>self.w-TAMANHO_BLOCO or pt.x<0 or pt.y>self.h - TAMANHO_BLOCO or pt.y<0):
            return True
        if(pt in self.cobra[1:]):
            return True
        return False