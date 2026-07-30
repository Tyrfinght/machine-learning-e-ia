# Processamento de Linguagem Natural: Tradução (Seq2Seq + Attention)

## O Projeto
A tradução automática de textos evoluiu de métodos estatísticos rígidos para redes neurais profundas capazes de compreender contextos complexos. Neste projeto, construí um sistema de Processamento de Linguagem Natural (NLP) focado na tradução e padronização de formatos: o modelo recebe datas escritas em linguagem natural (exemplo: "12th of april 2003") e as converte para o formato padronizado de máquina (`YYYY-MM-DD`).

O objetivo central foi implementar e analisar a arquitetura **Sequence-to-Sequence (Seq2Seq)** combinada com um **Mecanismo de Atenção**, que permite à rede "focar" em partes específicas da entrada durante a geração da saída, superando o problema de decaimento de memória das redes recorrentes tradicionais.

## Ferramentas e Métodos
* **Linguagem:** Python
* **Framework:** TensorFlow / Keras (Otimizador Adam, Categorical Crossentropy)
* **Arquitetura:** Bi-LSTM (Encoder), LSTM (Decoder), Mecanismo de Atenção (Bahdanau).

## Modelagem e Treinamento
A rede foi arquitetada da seguinte forma:
1. **Codificador (Encoder):** Uma camada LSTM Bidirecional processa a sequência de texto de entrada, capturando dependências do passado e do futuro.
2. **Atenção Dinâmica:** Uma ativação softmax avalia a importância (peso) de cada estado oculto do codificador para gerar o token atual.
3. **Decodificador (Decoder):** Uma rede LSTM recebe o vetor de contexto ponderado e gera a data formatada, token por token.

O treinamento envolveu o teste de hiperparâmetros com variações de épocas (1 a 100) e dimensões de unidades de ativação (32x32, 32x64 e 64x64).

## Descobertas e Análise Crítica
A análise empírica do treinamento revelou comportamentos cruciais sobre redes neurais:
* **Overfitting / Platô:** O ganho de precisão foi drástico de 1 para 20 épocas. Contudo, entre 30 e 100 épocas, o desempenho estagnou e até apresentou leves quedas, indicando o limite de aprendizado do modelo para aquele dataset.
* **Viés de Dados (Data Bias):** O modelo falhou sistematicamente ao tentar traduzir formatos específicos (como "1999 09 27" ou "01/12/1929"). A análise concluiu que essa deficiência não era falha da arquitetura, mas sim um viés do *dataset* de treino, que provavelmente carecia de variabilidade representativa desses padrões.

## Relatório 
O estudo detalhado sobre o histórico da tradução automática, a fundamentação teórica da arquitetura LSTM com Atenção e as tabelas de validação cruzada estão no relatório abaixo:
**[Ler o Relatório (PDF)](./Relatório_Projeto_2_LSTM.pdf)**
