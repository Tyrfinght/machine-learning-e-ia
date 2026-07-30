# Otimização Computacional em Redes Neurais (Multiplicação em cadeia matricial)

## O Projeto
O treinamento de Redes Neurais Profundas (Deep Learning) exige um esforço computacional massivo, majoritariamente focado em multiplicações de matrizes (pesos, vieses e entradas). O que muitos pessoas ignoram é que a **ordem** em que essas matrizes são multiplicadas altera drasticamente a quantidade de operações escalares necessárias.

Neste projeto, implementei o clássico algoritmo de **Multiplicação de Cadeia de Matrizes** utilizando Programação Dinâmica para encontrar a parentização ótima. Em seguida, apliquei essa lógica matemática para otimizar o custo computacional do treinamento de uma Rede Neural no PyTorch.

## Ferramentas e Métodos
* **Linguagem:** Python
* **Bibliotecas:** PyTorch, Torchvision
* **Método Matemático:** Programação Dinâmica 
* **Dataset de Teste:** MNIST

## Modelagem e Algoritmo
O projeto foi dividido em duas fases lógicas:

1. **O Motor de Otimização:** Construção de uma classe Orientada a Objetos (`MultiplicadorDeCadeiaDeMatrizes`) que calcula a tabela de custos de multiplicação e define as partições ótimas (divisores) para qualquer conjunto de matrizes em tempo `O(n³)`.
2. **A Aplicação em IA:** Construção de uma Rede Neural Simples (`RedeNeuralSimples`) em PyTorch (camadas: entrada `28x28`, oculta `128`, saída `10`).
3. **Benchmarking:** Execução de um loop de treinamento padrão via *Stochastic Gradient Descent (SGD)* e *CrossEntropyLoss*, comparado matematicamente com o custo das dimensões das matrizes otimizadas.

## Resultados Computacionais
Para um conjunto grande de matrizes (benchmarking de estresse com mais de 50 dimensões testadas), a diferença no número de operações caiu de uma escala de **878 bilhões** para apenas **866 bilhões** de operações escalares, evidenciando o poder da programação dinâmica em problemas de escalabilidade em *Machine Learning*.

## Código 
Toda a modelagem matemática, as classes de programação dinâmica e a integração com o treinamento no PyTorch estão detalhadas no Jupyter Notebook presente nesta pasta.
