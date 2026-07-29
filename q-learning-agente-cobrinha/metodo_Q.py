import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Q_linear(nn.Module): #rede neural 
    def __init__(self,dimensao_entrada,dimensao_camada_oculta,dimensao_saida):
        super().__init__()
        self.linear1 = nn.Linear(dimensao_entrada,dimensao_camada_oculta)
        self.linear2 = nn.Linear(dimensao_camada_oculta,dimensao_saida)
   
 
    def forward(self, x):  #como os dados sao passados
        x = F.relu(self.linear1(x))   #opera a primeira camada com a funcao reLU 
        x = self.linear2(x)   #opera a segunda camada com a saida da camada anterior
        return x       #retorna a saida da segunda camada

    def guardar(self, pesos='pesos'):   #funcao pra salvar os pesos 
        model_folder_path = 'K:\\Programação\\Projetos VS Code\\Projeto MS571'
        pesos = os.path.join(model_folder_path,pesos)         #cria o arquivo no qual os pesos vao ser salvos
        torch.save(self.state_dict(),pesos) 

class Treino_Q:
    def __init__(self,modelo,taxa_aprendizado,taxa_desconto):
        self.lr = taxa_aprendizado
        self.gamma = taxa_desconto
        self.model = modelo
        self.optimizer = optim.Adam(modelo.parameters(),lr = self.lr)    
        self.criterion = nn.MSELoss()        #funcao perda e a funcao do erro quadratico medio 
        for i in self.model.parameters():
            print(i.is_cuda)

    
    def etapa_treino(self, estado, acao, recompensa, proximo_estado, done):
        estado = torch.tensor(estado,dtype=torch.float)
        proximo_estado = torch.tensor(proximo_estado,dtype=torch.float)
        acao = torch.tensor(acao,dtype=torch.long)
        recompensa = torch.tensor(recompensa,dtype=torch.float)              #convertendo tudo pra tensor


        if(len(estado.shape) == 1): # conferindo se tem um parametro so, se sim, converte pra tupla
            #(1 , x)
            estado = torch.unsqueeze(estado,0)
            proximo_estado = torch.unsqueeze(proximo_estado,0)
            acao = torch.unsqueeze(acao,0) #unsqueeze add uma dimensao pra usar como tensor bidimensional
            recompensa = torch.unsqueeze(recompensa,0)
            done = (done, )

           

        # previsao do valor Q com o estado atual
        previsao = self.model(estado)
        copia = previsao.clone()
        for idx in range(len(done)):
            novo_Q = recompensa[idx]
            if not done[idx]:
                novo_Q = recompensa[idx] + self.gamma * torch.max(self.model(proximo_estado[idx]))
            copia[idx][torch.argmax(acao).item()] = novo_Q 
        # formula de Bellman
        # previsao.clone()
        # previsao[argmax(action)] = novo_Q
        self.optimizer.zero_grad()
        perda = self.criterion(copia,previsao)
        perda.backward()

        self.optimizer.step()