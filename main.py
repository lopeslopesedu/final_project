from chatbot import Chatbot
#conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
#pip3 install tqdm, sklearn, torch, nltk
#pip3 install sklearn

if __name__ == '__main__':
    chat = Chatbot()
    #chat.rede="LSTM"
    #carrega as bases gerais e especificas e prepara as informações para serem processadas
    #chat.dataset.treino = 10 --- delimita 10% das perguntas para fins de testes
    chat.dataset.usar_base_especialista = False
    chat.dataset.carregar_bases()
    #método de treino
    chat.treinar()
    #método de avaliação automatizada das perguntas e respostas retiradas para testes
    #chat.avaliar()
    #permite a interação individual com o bot
    chat.interar()


