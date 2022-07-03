from common import unicodeToAscii, normalizeString, readVocs, trimRareWords
from common import indexesFromSentence, zeroPadding,binaryMatrix,inputVar,outputVar, batch2TrainData
from voc import Voc
import os,codecs,csv, random,re
from datetime import datetime
from sklearn.model_selection import train_test_split
MAX_LENGTH = 20  # Maximum sentence length to consider

from nltk.util import ngrams

class Data_set:
    def __init__(self):
        self.corpus_name = "cornell movie-dialogs corpus"
        self.corpus = os.path.join("data", self.corpus_name)
        self.datafile = os.path.join(self.corpus, "formatted_movie_lines.txt")
        self.datafile_teste = os.path.join(self.corpus, "formatted_movie_lines_teste.txt")
        self.datafile_resultados = os.path.join(self.corpus, "chatbot_resultados.txt")
        self.delimiter = '\t'
        self.delimiter = str(codecs.decode(self.delimiter, "unicode_escape"))
        self.save_dir = os.path.join("data", "save")
        self.pairs = []
        self.voc = []
        self.usar_base_geral = True
        self.usar_base_especialista = True
        #fator usado para separar base treino/teste
        self.treino = 10
        self.pairs_teste = []
        self.tam_base_geral = 20000

    # Funções utilizadas na base do cornell para carregar linhas e conversas
    def __carregar_linhas(self,fileName, fields):
        """
        Método utilizado para carregar as linhas de acordo com a estrutura do diálogo
        :param fileName:
        :param fields:
        :return:
        """
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
        return lines
    def __carregar_conversas(self,fileName, lines, fields):
        """
        Método utilizado para carregar as conversas.
        A conversa da base do cornell é estrurada de acordo com o arquivo de conversas.
        :param fileName:
        :param lines:
        :param fields:
        :return:
        """
        conversations = []
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")

                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]

                utterance_id_pattern = re.compile('L[0-9]+')
                lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(lines[lineId])
                conversations.append(convObj)
        return conversations

    # Métodos utilizados para filtrar e extrair os pares de perguntas/respostas
    def __extrair_pares(self,conversations):
        """
        Método utilizado para extrair os pares
        :param conversations:
        :return:
        """
        qa_pairs = []
        for conversation in conversations:
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i + 1]["text"].strip()
                # Retirar aquelas que estão vazias
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        return qa_pairs
    def __filtrar_par(self,p):
        """
        Função utilizada para retirar as frases maiores que o MAX_LENGTH
        :param p:
        :return:
        """
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
    def __filtrar_pares(self,pairs):
        """
        Função utilizada para filtrar as frases da base
        :return:
        """
        return [pair for pair in pairs if self.__filtrar_par(pair)]

    # Funções utilizadas para obter o vocabulario da base
    def __carregar_dados_preparados(self):
        """
        Método utilizado para ler os dados previamente carregados e separados das bases.
        """
        print("-*Iniciando a preparação dos dados de treino*-")
        self.voc, self.pairs = readVocs(self.datafile, self.corpus_name)
        print("-> {!s} Pares carregados".format(len(self.pairs)))
        self.pairs = self.__filtrar_pares(self.pairs)
        print("->Filtrado para {!s} pares".format(len(self.pairs)))
        print("->Contando palavras")
        for pair in self.pairs:
            self.voc.addSentence(pair[0])
            self.voc.addSentence(pair[1])
        print("Palavras Contadas:", self.voc.num_words)
    def __contagem_palavras(self):
        """
        Método utilizado para fazer a contagem de vezes que uma palavra é utilizada
        Caso a contagem seja inferior a 3 a palavra será retirada
        """
        MIN_COUNT = 3  # Minimum word count threshold for trimming
        self.pairs = trimRareWords(self.voc, self.pairs, MIN_COUNT)
    def __preparar_dados(self):
        """
        Função utilizada para preparar os dados de acordo com a base carregada
        """
        small_batch_size = 5
        batches = batch2TrainData(self.voc, [random.choice(self.pairs) for _ in range(small_batch_size)])

    #separação Treino Teste
    def separar_treino_teste(self,base):
        """
        Método utilizado para fazer a separação treino e testes
        :return: Retorna a base de testes
        """
        #prepara os dados para separação
        X=[]
        y=[]
        for i in base:
            X.append(i[0])
            y.append(i[1])
        #realiza a separação
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(self.treino*0.01), random_state=42)
        base_treino = []
        base_teste = []
        for i in range(len(X_train)):
            base_treino.append([X_train[i], y_train[i]])
        base_teste = []
        for i in range(len(X_test)):
            base_teste.append([X_test[i], y_test[i]])

        return base_treino,base_teste

    #bloco de carregamento
    def __carregar_base_geral(self):
        """
        Método utilizado para carregar a base de dados gerais
        Após carregar e organizar as informações
        Salva os dados nos arquivos que serão utilizados posteriormente
        Este método faz a separação dos dados de testes e salva em um arquivo a ser
        utilizado posteriormente.
        """
        lines = {}
        conversations = []
        MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

        print("\n-*Carregando Base de Perguntas Gerais*-")
        print("->Carregando Linhas")
        lines = self.__carregar_linhas(os.path.join(self.corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
        print("->Carregando conversas")
        conversations = self.__carregar_conversas(os.path.join(self.corpus, "movie_conversations.txt"),
                                          lines, MOVIE_CONVERSATIONS_FIELDS)

        self.pairs = []
        base =[]
        print("->Gravando perguntas e respostas gerais")
        with open(self.datafile, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=self.delimiter, lineterminator='\n')
            for pair in self.__extrair_pares(conversations):
                self.pairs.append(pair)
                base.append([pair[0], pair[1][:-1]])
            self.pairs = self.pairs[:self.tam_base_geral]
            self.pairs, base_teste = self.separar_treino_teste(base)
            for pair in self.pairs:
                writer.writerow(pair)

        print("->Gravando perguntas e respostas gerais para Testes posteriores")
        with open(self.datafile_teste, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=self.delimiter, lineterminator='\n')
            for pair in base_teste:
                writer.writerow(pair)
        del (base_teste)

        self.pairs=[]
        print("")
    def __carregar_base_geral_dialogpt(self):
        """
        Método utilizado para carregar a base de dados especialistas
        Após carregar e organizar as informações
        Salva os dados nos arquivos que serão utilizados posteriormente
        Este método faz a separação dos dados de testes e salva em um arquivo a ser
        utilizado posteriormente.
        """
        print("\n-*Carregando Base de Perguntas Especialistas*-")
        arquivo = "base.csv"
        base=[]
        with open(arquivo) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                base.append([row[0], row[1][:-2]])

        pares, base_teste = self.separar_treino_teste(base)
        self.pairs += pares
        #self.pairs,base_teste = self.separar_treino_teste(base)

        #tipo_abertura_arquivo = "w"

        if (self.usar_base_geral):
            tipo_abertura_arquivo = "a"
        else:
            tipo_abertura_arquivo = "w"

        print("->Gravando perguntas e respostas especialistas")
        with open(self.datafile, tipo_abertura_arquivo, encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=self.delimiter, lineterminator='\n')
            for pair in self.pairs:
                writer.writerow(pair)

        print("->Gravando perguntas e respostas especialistas para Testes posteriores")
        with open(self.datafile_teste, tipo_abertura_arquivo, encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=self.delimiter, lineterminator='\n')
            for pair in base_teste:
                writer.writerow(pair)
        del (base_teste,base)
    def __carregar_base_especialista(self):
        """
        Método utilizado para carregar a base de dados especialistas
        Após carregar e organizar as informações
        Salva os dados nos arquivos que serão utilizados posteriormente
        Este método faz a separação dos dados de testes e salva em um arquivo a ser
        utilizado posteriormente.
        """
        print("\n-*Carregando Base de Perguntas Especialistas*-")
        arquivo = "especialista.csv"
        base = []
        base_treino=[]
        with open(arquivo) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                base.append([row[0], row[1]])

        base_treino,base_teste = self.separar_treino_teste(base)
        self.pairs+=base_treino

        if(self.usar_base_geral):
            tipo_abertura_arquivo = "a"
        else:
            tipo_abertura_arquivo = "w"

        print("->Gravando perguntas e respostas especialistas")
        with open(self.datafile, tipo_abertura_arquivo, encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=self.delimiter, lineterminator='\n')
            for pair in self.pairs:
                writer.writerow(pair)

        print("->Gravando perguntas e respostas especialistas para Testes posteriores")
        with open(self.datafile_teste, tipo_abertura_arquivo, encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=self.delimiter, lineterminator='\n')
            for pair in base_teste:
                writer.writerow(pair)
        del (base_teste,base,base_treino)

    def carregar_base_testes(self):
        """
        Função utilizada para carregar as perguntas e respostas de testes
        para avaliação do chatbot
        :return: Retorna o vocabulário das perguntas de testes e as perguntas e respostas
        """
        print("Start preparing training data ...")
        voc, pairs = readVocs(self.datafile_teste, self.corpus_name)
        print("Read {!s} sentence pairs".format(len(self.pairs)))
        pairs = self.__filtrar_pares(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counted words:", voc.num_words)
        return voc, pairs



    #funções gerais da classe
    def carregar_bases(self):
        """
        Método utilizado para estruturar o carregamento das bases de dados
        De acordo com o "usar_base_geral" e "usar_base_especialista" carrega dos arquivos as bases de dados
        Prepara os dados posteriormete a contagem de palavras
        Por fim prepara os dados para processamento pelo modelo.
        """
        if self.usar_base_geral==True :
            self.__carregar_base_geral()
            self.__carregar_base_geral_dialogpt()
        if self.usar_base_especialista==True :
            self.__carregar_base_especialista()
        self.__carregar_dados_preparados()
        self.__contagem_palavras()
        #self.__preparar_dados()
    def salvar_resultados(self,resultados):
        """
        Método utilizado para salvar os resultados das consultas
        :param resultados:
        """
        print("->Salvando resultados")
        with open(self.datafile_resultados, 'w', encoding='utf-8') as outputfile:
            #writer = csv.writer(outputfile, delimiter=self.delimiter, lineterminator='\n')
            for pair in resultados:
                #print(pair)
                outputfile.write(pair+"\n")
                #writer.writerow(pair)
    def salvar_resultados_bleu(self,resultados,media):
        """
        Método utilizado para salvar os resultados das consultas
        :param resultados:
        """
        print("->Salvando resultados_bleu")
        now = str(datetime.now().year)+"_"+str(datetime.now().month)+"_"+str(datetime.now().day)+"_"+str(datetime.now().hour)+"_"+str(datetime.now().minute)
        arquivo_resultado_bleu = os.path.join(self.corpus, "result_bleu_"+str(now)+".txt")
        with open(arquivo_resultado_bleu, 'w', encoding='utf-8') as outputfile:
            #writer = csv.writer(outputfile, delimiter=self.delimiter, lineterminator='\n')
            outputfile.write("A média do Bleu para esse teste/treino foi: "+str(media)+"\n")
            for pair in resultados:
                outputfile.write(str(pair)+"\n")