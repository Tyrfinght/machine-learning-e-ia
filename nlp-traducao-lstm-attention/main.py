import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
from string_to_int import string_to_int
from plot_attention_map import plot_attention_map

# fiz isso para conseguir usar o cuda e o cudnn corretamente
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CUDNN_VERSION'] = '8'
os.environ['CUDA_TOOLKIT_ROOT_DIR'] = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2'

# imports do keras
Model = keras.models.Model
load_model = keras.models.load_model
LSTM = keras.layers.LSTM
RepeatVector = keras.layers.RepeatVector
Input = keras.layers.Input
Activation = keras.layers.Activation
Dot = keras.layers.Dot
Concatenate = keras.layers.Concatenate
Bidirectional = keras.layers.Bidirectional
Dense = keras.layers.Dense

# pegando o dataset e os vocabulários
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

with open("human_vocab.pkl", "rb") as f:
    human_vocab = pickle.load(f)

with open("machine_vocab.pkl", "rb") as f:
    machine_vocab = pickle.load(f)

with open("inv_machine_vocab.pkl", "rb") as f:
    inv_machine_vocab = pickle.load(f)

# constantes 
Tx = 30  # tamanho máximo do input (data humana)
Ty = 10  # tamanho máximo do output (data máquina)
n_a = 64
n_s = 64 
humanVocabSize = len(human_vocab)
machineVocabSize = len(machine_vocab)

# preprocessamento (one-hot vector)
def preprocessar(dataset, human_vocab, machine_vocab, Tx, Ty):
    encoder = np.zeros((len(dataset), Tx, humanVocabSize), dtype=np.float32)
    decoder = np.zeros((len(dataset), Ty, machineVocabSize), dtype=np.float32)
    
    for i, (human_readable, machine_readable) in enumerate(dataset):
        human_seq = string_to_int(human_readable, Tx, human_vocab)
        machine_seq = string_to_int(machine_readable, Ty, machine_vocab)
        
        encoder[i] = np.array([keras.utils.to_categorical(x, num_classes=humanVocabSize) for x in human_seq])
        decoder[i] = np.array([keras.utils.to_categorical(y, num_classes=machineVocabSize) for y in machine_seq])
        
    return encoder, decoder

encoder_input, decoder_input = preprocessar(dataset, human_vocab, machine_vocab, Tx, Ty)

# definindo o modelo
def camadaAtencao(outputEncoder, estadoPrevio):

    # repete para bater dimensões
    estadoRepetido = RepeatVector(Tx)(estadoPrevio)
    concatenado = Concatenate(axis=-1)([outputEncoder, estadoRepetido])

    # camadas densas que serão usadas na ativação
    densa = Dense(10, activation="tanh")(concatenado)
    preAtt = Dense(1, activation="relu")(densa)

    # output da densa é usada na ativação
    ativacao = Activation("softmax")(preAtt)
    contexto = Dot(axes=1)([ativacao, outputEncoder])

    return contexto

def criaModelo(Tx, Ty, n_a, n_s, humanVocabSize, machineVocabSize):
    inputEncoder = Input(shape=(Tx, humanVocabSize)) # camada de input
    s0 = Input(shape=(n_s,))
    c0 = Input(shape=(n_s,)) 
    s, c = s0, c0
    outputDecoder = []
    
    # cria uma camada bidirecional com a lstm
    outputEncoder = Bidirectional(LSTM(n_a, return_sequences=True))(inputEncoder)
    
    # pos-atencao LSTM
    for t in range(Ty):

        # cria a camada de atencao
        vetorContexto = camadaAtencao(outputEncoder, s)
        s, _, c = LSTM(n_s, return_state=True)(vetorContexto, initial_state=[s, c])

        # camada densa de output com softmax
        output = Dense(machineVocabSize, activation='softmax')(s)
        outputDecoder.append(output)
    
    model = Model(inputs=[inputEncoder, s0, c0], outputs=outputDecoder)
    return model

# instanciando e compilando os modelos
model1 = criaModelo(Tx, Ty, n_a, n_s, humanVocabSize, machineVocabSize)
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model30 = criaModelo(Tx, Ty, n_a, n_s, humanVocabSize, machineVocabSize)
model30.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model100 = criaModelo(Tx, Ty, n_a, n_s, humanVocabSize, machineVocabSize)
model100.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

s0 = np.zeros((len(encoder_input), n_s))
c0 = np.zeros((len(encoder_input), n_s))

# treinando
print("\nTreinando para 1 época.\n")
model1.fit([encoder_input, s0, c0], list(decoder_input.swapaxes(0, 1)), epochs=1, batch_size = 64)
print("\nTreinando para 30 épocas.\n")
model30.fit([encoder_input, s0, c0], list(decoder_input.swapaxes(0, 1)), epochs=30, batch_size = 64)
print("\nTreinando para 100 épocas.\n")
model100.fit([encoder_input, s0, c0], list(decoder_input.swapaxes(0, 1)), epochs=100, batch_size = 64)


# converter para foramto legível (basicamente int_to_string)
def decodePrediction(pred, inv_vocab):
    decoded = ''.join([inv_vocab[np.argmax(p)] for p in pred])
    return decoded.strip('<pad>')

testesExemplo = ["12th of april 2003", "jan 23 2004", "wednesday 26 aug 1914", "23 september 61", "30/10/76"]

for exemplo in testesExemplo:
    # encoding
    exemploInt = string_to_int(exemplo, Tx, human_vocab)
    exemploEncoded = np.array([keras.utils.to_categorical(x, num_classes=humanVocabSize) for x in exemploInt])
    exemploEncoded = exemploEncoded.reshape((1, Tx, humanVocabSize))
    s0, c0 = np.zeros((1, n_s)), np.zeros((1, n_s))
    
    # realizar predição com os modelos
    pred1 = model1.predict([exemploEncoded, s0, c0])
    pred30 = model30.predict([exemploEncoded, s0, c0])
    pred100 = model100.predict([exemploEncoded, s0, c0])
    
    # decoding
    decoded1 = decodePrediction(pred1, inv_machine_vocab)
    decoded30 = decodePrediction(pred30, inv_machine_vocab)
    decoded100 = decodePrediction(pred100, inv_machine_vocab)
    
    print(f"Entrada: {exemplo}")
    print(f"Resultado modelo 1 época: {decoded1}")
    print(f"Resultado modelo 30 épocas: {decoded30}")
    print(f"Resultado modelo 100 épocas: {decoded100}")
    print("\n")


#plotando
#for exemplo in testesExemplo:
#
#    print(f"\nPlotando modelo 1 época para: {exemplo}")
#    plot_attention_map(model1, human_vocab, inv_machine_vocab, exemplo, n_s)
#    plt.show()
#    plt.close()
#
#    print(f"\nPlotando modelo 30 épocas para: {exemplo}")
#    plot_attention_map(model30, human_vocab, inv_machine_vocab, exemplo, n_s)
#    plt.show()
#    plt.close()
#
#    print(f"\nPlotando modelo 100 épocas para: {exemplo}")
#    plot_attention_map(model100, human_vocab, inv_machine_vocab, exemplo, n_s)
#    plt.show()
#    plt.close()