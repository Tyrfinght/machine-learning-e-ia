import pygame
import time
import random
from enum import Enum
from collections import namedtuple
pygame.init()


# Reset 
# Recompensa
# Acao -> Direcao
# Iteracao
# Colisao


class Direcao(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
 
Ponto = namedtuple('Point','x , y')

#algumas coisas basicas
TAMANHO_BLOCO= 20
VELOCIDADE = 10
preto = pygame.Color(0, 0, 0)
branco = pygame.Color(255, 255, 255)
vermelho = pygame.Color(255, 0, 0)
verde = pygame.Color(0, 255, 0)
azul = pygame.Color(0, 0, 255)

class Jogo_Cobrinha:
    def __init__(self,janela_x=640,janela_y=640):
        self.w=janela_x
        self.h=janela_y
        #init display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Cobrinha')
        self.clock = pygame.time.Clock()
        
        #estado inicial do jogo
        self.direction = Direcao.RIGHT
        self.head = Ponto(self.w/2,self.h/2)
        self.cobra = [self.head,
                      Ponto(self.head.x-TAMANHO_BLOCO,self.head.y),
                      Ponto(self.head.x-(2*TAMANHO_BLOCO),self.head.y)]
        self.placar = 0
        self.comida = None
        self.posicao_comida()

    def posicao_comida(self):
        x = random.randint(0,(self.w-TAMANHO_BLOCO)//TAMANHO_BLOCO)*TAMANHO_BLOCO
        y = random.randint(0,(self.h-TAMANHO_BLOCO)//TAMANHO_BLOCO)*TAMANHO_BLOCO
        self.comida = Ponto(x,y)
        if(self.comida in self.cobra):
            self.posicao_comida()


    def jogada(self):
        # 1. input
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()
            if(event.type == pygame.KEYDOWN):
                if(event.key == pygame.K_LEFT):
                    self.direction = Direcao.LEFT
                elif(event.key == pygame.K_RIGHT):
                    self.direction = Direcao.RIGHT
                elif(event.key == pygame.K_UP):
                    self.direction = Direcao.UP
                elif(event.key == pygame.K_DOWN):
                    self.direction = Direcao.DOWN
           
        
        # 2. movimento
        self.movimento(self.direction)
        self.cobra.insert(0,self.head)

        # 3. game over
        game_over = False 
        if(self.colisao()):
            game_over=True
            return game_over,self.placar
        # 4. gerar comida ou se mover
        if(self.head == self.comida):
            self.placar+=1
            self.posicao_comida()
        else:
            self.cobra.pop()
        # 5. atualizar hud
        self.hud()
        self.clock.tick(VELOCIDADE)
        # 6. game over e mostra o placar
        
        return game_over,self.placar

    def hud(self):
        self.display.fill(preto)
        for pt in self.cobra:
            pygame.draw.rect(self.display, branco,pygame.Rect(pt.x,pt.y,TAMANHO_BLOCO,TAMANHO_BLOCO))
            pygame.draw.rect(self.display, azul,pygame.Rect(pt.x+4,pt.y+4,12,12))
        pygame.draw.rect(self.display, vermelho,pygame.Rect(self.comida.x,self.comida.y,TAMANHO_BLOCO,TAMANHO_BLOCO))
        minha_fonte = pygame.font.SysFont('times new roman', 50)
        text = minha_fonte.render("Placar: "+str(self.placar),True,branco)
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def movimento(self,direction):
        x = self.head.x
        y = self.head.y
        if(direction == Direcao.RIGHT):
            x+=TAMANHO_BLOCO
        elif(direction == Direcao.LEFT):
            x-=TAMANHO_BLOCO
        elif(direction == Direcao.DOWN):
            y+=TAMANHO_BLOCO
        elif(direction == Direcao.UP):
            y-=TAMANHO_BLOCO
        self.head = Ponto(x,y)
    def colisao(self):
        #bater na borda perde
        if(self.head.x>self.w-TAMANHO_BLOCO or self.head.x<0 or self.head.y>self.h - TAMANHO_BLOCO or self.head.y<0):
            return True
        if(self.head in self.cobra[1:]):
            return True
        return False

if __name__=="__main__":
    game = Jogo_Cobrinha()

    #Game loop
    #game_over=False
    while True:
        game_over,score=game.jogada()
        if(game_over == True):
            break
    print('Placar Final',score)

    pygame.quit()