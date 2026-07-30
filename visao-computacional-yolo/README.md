# Visão Computacional e Detecção de Objetos (YOLOv3)

## O Projeto
A detecção de objetos em imagens é um dos pilares da automação visual moderna, aplicável desde carros autônomos até diagnósticos médicos. O desafio clássico dessa área sempre foi o balanço entre **Precisão** (com modelos de múltiplas etapas, como R-CNN) e **Velocidade**.

Neste projeto, estudei e implementei o modelo **YOLO (You Only Look Once)**, que revolucionou a Visão Computacional ao tratar a detecção como um problema único de regressão, permitindo inferências em tempo real (*Single Shot Detector*).

## Ferramentas e Métodos
* **Linguagem:** Python
* **Arquitetura:** Redes Neurais Convolucionais (CNN) / YOLOv3
* **Conceitos Matemáticos Aplicados:** Non-Max Suppression (NMS), Intersection over Union (IoU), Função Softmax, Regressão de Bounding Boxes.

## Arquitetura e Modelagem
O projeto focou na compreensão profunda do "motor" do YOLO. O modelo divide a imagem de entrada em um grid (grade) `NxN`. Cada célula do grid é responsável por prever:
1. Coordenadas do centro do objeto `(x, y)` e dimensões `(w, h)`.
2. Probabilidade (Confidence Score) de existir um objeto ali.
3. Classificação do objeto detectado.

Para evitar detecções duplicadas de um mesmo objeto (múltiplas *Bounding Boxes* sobrepostas), apliquei ativamente o limiar matemático de **IoU (Intersection over Union)** aliado ao filtro **Non-Max Suppression (NMS)**.

## Desafios e Análise Crítica
O algoritmo obteve excelente taxa de acerto em cenários com objetos isolados ou classes distintas no mesmo plano. O verdadeiro teste técnico ocorreu em imagens com alta sobreposição de objetos (ex: pessoas em primeiro plano bloqueando parcialmente um sofá no segundo plano). 

A análise evidenciou que a detecção depende de um *fine-tuning* preciso dos hiperparâmetros de `Threshold` (Limiar de Confiança). A diminuição consciente desse limiar permitiu que o modelo localizasse objetos parcialmente obstruídos, provando que parâmetros puramente teóricos devem ser calibrados para a "sujeira" do mundo real.

## Relatório e Exemplos
O texto do projeto, contendo a fundamentação teórica detalhada (comparativo com a arquitetura R-CNN) e as amostras visuais de *Bounding Boxes* geradas, está disponível abaixo:
👉 **[Ler o Relatório  (PDF)](./Relatorio_Projeto_1_Florindo_versao_final.pdf)**

## Pesos e Dados
Devido às restrições de tamanho de arquivo do GitHub, os arquivos de pesos pré-treinados (`.weights`) e os datasets completos não estão incluídos neste repositório. 
* Para executar o código localmente, baixe os pesos oficiais do YOLOv3 e coloque na pasta main do projeto.
