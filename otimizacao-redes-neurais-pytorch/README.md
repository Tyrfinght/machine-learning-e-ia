# Otimização Computacional em Redes Neurais: Teoria vs. Abstração (Multplicação em Cadeia Matricial)

## O Projeto
O treinamento de Redes Neurais Profundas (Deep Learning) exige um esforço computacional massivo, majoritariamente focado em multiplicações de matrizes. Matematicamente, a **ordem** (parentização) em que essas matrizes são multiplicadas altera drasticamente a quantidade de operações escalares necessárias.

Neste projeto, o objetivo foi duplo: 
1. Implementar o clássico algoritmo de **Multiplicação de Cadeia de Matrizes** utilizando Programação Dinâmica (*Dynamic Programming*) para encontrar a rota matemática de menor custo computacional.
2. Analisar como essa teoria se comporta na prática ao ser confrontada com a abstração de grafos computacionais de *frameworks* modernos de Deep Learning, como o PyTorch.

## Ferramentas e Métodos
* **Linguagem:** Python
* **Bibliotecas:** PyTorch, Torchvision
* **Método Matemático:** Programação Dinâmica (Dynamic Programming)
* **Dataset de Teste:** MNIST

## Modelagem e Algoritmo
O projeto foi estruturado para contrastar a otimização teórica com a execução em *framework*:

1. **O Motor de Otimização (Teoria):** Construção de uma classe Orientada a Objetos (`MultiplicadorDeCadeiaDeMatrizes`) que calcula a tabela de custos de multiplicação e define as partições ótimas para qualquer conjunto de matrizes em tempo `O(n³)`.
2. **A Aplicação em IA (Prática):** Construção de uma Rede Neural Simples (`RedeNeuralSimples`) em PyTorch, executando um loop de treinamento via *Stochastic Gradient Descent (SGD)* e *CrossEntropyLoss*.

## Resultados e Análise Crítica
Na fase de benchmarking teórico (com um teste de estresse de mais de 50 matrizes), o algoritmo de Programação Dinâmica provou matematicamente que a reorganização da cadeia derrubaria o custo de **878 bilhões** para apenas **866 bilhões** de operações escalares.

Contudo, ao integrar essa lógica ao PyTorch, o experimento evidenciou um comportamento arquitetural crucial: *frameworks* de alto nível abstraem a multiplicação matricial das camadas (`nn.Linear`) para rotinas otimizadas em C++/CUDA. A tentativa de forçar uma parentização matemática via Python puro sem modificar o backend de compilação em C++ não altera o grafo computacional da rede e pode gerar um *overhead* indesejado.

**Conclusão do Estudo:** O projeto comprova a validade da otimização teórica da Programação Dinâmica, mas demonstra que a engenharia de Machine Learning moderna exige alinhar o custo matemático à arquitetura do compilador (C++/CUDA), evidenciando a diferença entre o modelo "caixa-branca" (matemática pura) e a abstração "caixa-preta" dos *frameworks*.

## Código Fonte
Toda a modelagem matemática, as classes de programação dinâmica e a integração com o treinamento no PyTorch estão detalhadas no Jupyter Notebook presente nesta pasta.
