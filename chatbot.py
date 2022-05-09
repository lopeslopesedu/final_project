import torch
import torch.nn as nn
import random
from torch import optim
from GreedySearchDecoder import GreedySearchDecoder
from common import unicodeToAscii, normalizeString, readVocs, trimRareWords
from common import indexesFromSentence, zeroPadding,binaryMatrix,inputVar,outputVar, batch2TrainData
from common import batch2TrainData
from tqdm import tqdm
from dataset import Data_set
from encoderRNN import EncoderRNN
from luongAttnDecoderRNN import LuongAttnDecoderRNN

# tokens
SOS_token = 1  # Start-of-sentence token
MAX_LENGTH = 20  # Maximum sentence length to consider

from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu

class Chatbot:
    def __init__(self):
        self.dataset = Data_set()
        #delimita o uso de placa de vídeo caso esteja disponível
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")
        self.encoder=None
        self.decoder=None
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.embedding = None
        self.searcher = None
        #parâmetros de configuração e treinamento
        self.teacher_forcing_ratio = 1.0
        self.model_name = 'cb_model'
        self.attn_model = 'dot'
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 64
        self.clip = 50.0
        self.teacher_forcing_ratio = 1.0
        self.learning_rate = 0.0001
        self.decoder_learning_ratio = 5.0
        self.n_iteration = 4000
        self.avg_pontuacao_bleu = 0
        self.pontuacao_bleu = []
        self.resultados = []


    #Métodos privados de treinamento
    def __maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss, nTotal.item()

    def __treino(self, input_variable, lengths, target_variable, mask, max_target_len,max_length=MAX_LENGTH):
        """
        Método utilizado para treinar efetivamente o modelo
        :param input_variable:
        :param lengths:
        :param target_variable:
        :param mask:
        :param max_target_len:
        :param max_length:
        :return:
        """
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        #Configuro para rodar na placa de vídeo caso disponível
        input_variable = input_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.to(self.device)

        lengths = lengths.to("cpu")
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(self.batch_size)]])
        decoder_input = decoder_input.to(self.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.__maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.__maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def __trainIters(self):
        """
        Método utilizado para gerir as interações de treinamento
        """
        training_batches = [batch2TrainData(self.dataset.voc, [random.choice(self.dataset.pairs) for _ in range(self.batch_size)])
                            for _ in range(self.n_iteration)]

        start_iteration = 1
        print_loss = 0
        resultado_loss=[]
        # Training loop
        print("->treinando: ")
        for iteration in tqdm(range(self.n_iteration)):
            training_batch = training_batches[iteration - 1]
            input_variable, lengths, target_variable, mask, max_target_len = training_batch
            loss = self.__treino(input_variable, lengths, target_variable, mask, max_target_len)
            resultado_loss.append(loss)
            print_loss += loss
        print("Parâmetro Loss Final: "+ str(resultado_loss[-1]))

    #Métodos privados de interação
    def __interacao(self, sentence):
        """
        Método que realiza a interação direta com o chatbot
        :param sentence: entrada de texto para interação
        :return: resposta do chatbot
        """
        indexes_batch = [indexesFromSentence(self.dataset.voc, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        input_batch = input_batch.to(self.device)
        lengths = lengths.to("cpu")
        tokens, scores = self.searcher(input_batch, lengths, MAX_LENGTH)
        #faz a tradução da resposta da partir das chaves geradas
        decoded_words = [self.dataset.voc.index2word[token.item()] for token in tokens]
        return decoded_words

    def __interacao_entrada(self):
        """
        Método para realizar a interação com o chatbot
        Tem a função principal de receber e formatar a entrada para o padrão do chatbot
        """
        entrada = ''
        while (True):
            try:
                entrada = input('Pergunta > : ')
                if entrada == 'q' or entrada == 'quit': break
                # Normaliza a entrada
                entrada = normalizeString(entrada)
                #interage
                output_words = self.__interacao(entrada)
                #formata a resposta
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('T.I.A:', ' '.join(output_words))

            except KeyError:
                print("Palavra não encontrada")

    def __avaliar_questoes_testes(self, base_testes):
        """
        Método utilizar para avaliar cada uma das perguntas e repostas
        que foram separadas para testes
        organiza o resultado para armazenamento
        :param searcher:
        :param base_testes:
        """
        input_sentence = ''
        tamanho_base = len(base_testes)
        self.resultados = []
        for i in tqdm(range(tamanho_base)):
            try:
                #pergunta
                input_sentence = base_testes[i][0]
                # Normaliza a pergunta
                input_sentence = normalizeString(input_sentence)
                # Avaliar a pergunta
                output_words = self.__interacao(input_sentence)
                # Formata e prepara a resposta armazenando no resultado
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                resposta = ""
                resposta = " ".join([str(item) for item in output_words])
                self.resultados.append("Question: "+ input_sentence)
                self.resultados.append('T.I.A: ' + resposta)
                self.resultados.append("Esperada: "+base_testes[i][1])

            except KeyError:
                pass
                #print("Error: Encountered unknown word.")
        #Salva o resultado no arquivo referente.
        self.dataset.salvar_resultados(self.resultados)


    #Métodos de pontuação das repsostas

    def __calcular_pontuacao_bleu(self,candidata, referencia):
        return sentence_bleu(candidata, referencia,smoothing_function=bleu_score.SmoothingFunction(epsilon=1e-12).method2)

    def __calcular_pontuacao_respostas(self):
        i = 0
        while(i<len(self.resultados)):
            pergunta = self.resultados[i]
            candidata = list(self.resultados[i+1].split())[1:]
            referencia = list(self.resultados[i+2].split())[1:]
            self.pontuacao_bleu.append(self.__calcular_pontuacao_bleu(candidata,referencia))
            self.avg_pontuacao_bleu+=self.pontuacao_bleu[-1]
            i+=3
            print(candidata)
            print(referencia)
            print(self.pontuacao_bleu[-1])
        #print avg

        print("A pontuação média Bleu foi de: " + str(self.avg_pontuacao_bleu/len(self.pontuacao_bleu)))




    #Métodos Públicos
    def treinar(self):
        """
        Método utilizado para organizar e treinar o modelo
        """
        # Iniciando o  word embeddings
        self.embedding = nn.Embedding(self.dataset.voc.num_words, self.hidden_size)
        # Iniciando o encoder e decoder models
        self.encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
        self.decoder = LuongAttnDecoderRNN(self.attn_model, self.embedding,
                                           self.hidden_size, self.dataset.voc.num_words,
                                           self.decoder_n_layers, self.dropout)
        # Utilizando o equipamento necessário
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.encoder.train()
        self.decoder.train()

        # Iniciando os otimizadores
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate * self.decoder_learning_ratio)

        # Caso tenha placa de ´video - utiliza ela
        for state in self.encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in self.decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        print("-*Iniciando as rodadas de treinamento*-")
        self.__trainIters()

    def interar(self):
        """
        Método para preparar a interação com o chatbot
        """
        self.encoder.eval()
        self.decoder.eval()
        self.searcher = GreedySearchDecoder(self.encoder, self.decoder)
        print("Interação com Chatbot TIA")
        # Após configurar - realiza a interação com o chabot
        self.__interacao_entrada()

    def avaliar(self):
        """
        Método utilizado para avaliar as perguntas e respostas separadas para testes
        """
        #ler perguntas de teste
        voc, pairs = self.dataset.carregar_base_testes()
        self.encoder.eval()
        self.decoder.eval()
        # Inicializa o modulo de busca
        self.searcher = GreedySearchDecoder(self.encoder, self.decoder)
        print("Interação Automatizada com Chatbot TIA")
        self.__avaliar_questoes_testes(pairs)
        self.__calcular_pontuacao_respostas()