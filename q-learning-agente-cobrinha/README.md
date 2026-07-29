# Aprendizado por Reforço: Agente Q-Learning (Jogo da Cobrinha de Nokia)

## O Projeto
O Aprendizado por Reforço (*Reinforcement Learning*) é a base por trás de sistemas autônomos que aprendem a tomar decisões interagindo com o ambiente, através de recompensas e penalidades. Neste projeto, construí um agente virtual integrado a uma Rede Neural profunda para jogar o clássico jogo da "Cobrinha" (Snake) de forma 100% autônoma.

O modelo foi treinado para maximizar sua pontuação, mapeando estados matriciais (posição da comida, perigos adjacentes e direção atual) para ações ótimas através da equação de Bellman.

## Ferramentas e Métodos
* **Linguagem:** Python
* **Bibliotecas:** PyTorch (Deep Learning), Pygame (Ambiente Virtual)
* **Arquitetura:** Deep Q-Learning (DQN)

## Modelagem do Agente
O ecossistema do projeto foi arquitetado em três módulos interdependentes:
1. **O Ambiente:** Onde o jogo é renderizado e a física de colisão e pontuação ocorre.
2. **O Modelo (`metodo_Q.py` / `AI.py`):** A rede neural multicamadas que recebe o estado atual com 11 valores (perigo à frente/lados, direção atual e localização da comida) e prediz o valor Q de cada ação possível.
3. **O Agente (`agente.py`):** O orquestrador que coleta o estado, toma as decisões balanceando Exploração (movimentos randômicos iniciais) vs. *Exploitation* (decisões da rede treinada), e atualiza a memória de curto e longo prazo (Replay Memory).

## Evolução e Resultados
Durante as iterações iniciais, o agente explora o mapa colidindo frequentemente. Após as primeiras centenas de iterações, a curva de aprendizado estabiliza, demonstrando que o algoritmo compreendeu as fronteiras do ambiente e a recompensa (comida), chegando a alcançar pontuações sólidas de forma consistente e visualmente fluida.

## Código Fonte e Documentação
Os scripts modulares do treinamento da IA estão disponíveis nesta pasta, bem como o Whitepaper teórico do projeto.
👉 **[Ler o Relatório Técnico (PDF)](./Trabalho_MS571.pdf)**
