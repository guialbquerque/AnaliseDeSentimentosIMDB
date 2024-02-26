# **Análise de Sentimentos da base de dados IMDB**

O dataset IMDB é um dos mais famosos benchmarks para modelos de classificação de textos que existe. Contando com mais de 49.000 avaliações de filmes e séries, e etiquetados com sentimento *positivo* e *negativo*. [Link do dataset](https://www.kaggle.com/luisfredgs/imdb-ptbr).

![Screenshot 2024-02-24 202913](https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/assets/67128202/01c438cd-391b-4700-8189-7a9cd79cf8bb)

A imagem mostra a distribuição das classes do dataset, e features com as reviews em português e inglês, estando completamente balanceado. 

## **Proposta de análise**

Aplicação de técnicas do [NLTK](https://www.nltk.org/), junto com a métrica de TF-IDF (term frequency-inverse document frequency), a fim de vetorizar e tratar os textos. Por fim, aplicar algoritmos de Machine Learning adequados para classificação de textos e conseguir avaliar de forma mais robusta o sentimento das avaliações em três categorias: Positivo, Negativo e Neutro.

## **Dependências usadas**

Como boa prática de projeto, a última versão mais estável do Python foi utilizada utilizando a ferramenta Pyenv, e a ferramenta [Poetry](https://python-poetry.org/) foi utilizada em conjunto com o ambiente virtual criado para gerenciar e adicionar todas ferramentas necessárias para o projeto. Bibliotecas e Frameworks que não foram possíveis instalar com Poetry foram instaladas com utilizando [Pip][https://pypi.org/project/pip/], como por exemplo: > ```pip install wordcloud```.

| Ferramenta | Versão    |
| -------------  | --- |
| python        | "3.12.1"
| numpy | "1.26.3"
| pandas         | "2.2.0"
| seaborn | "0.13.2"
| nltk | "3.8.1"
| scikit-learn | "1.4.0"
| unidecod | "1.3.8"
| matplotlib | "3.8.2"
| wordcloud | "1.9.3"

## **Organização do conteúdo** 

1. [notebooks](https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/tree/main/notebooks) - Organizado por tópicos, com diretórios para análise exploratória, pré-processamento, modelagem.
2. [reports](https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/tree/main/reports) - Contém visualizações e relatórios.
3. [src](https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/tree/main/src) - Código-fonte organizado por funcionalidade, incluindo preparação de dados, engenharia de recursos, modelagem.

### **Principais Recursos utilizados**

1. Bag of Words - Abordagem que consiste em realizar uma contagem de palavras, ou seja, transformar o texto em um vetor de números. Processo de extração de características de texto para que uma forma estruturada possa ser utilizada para algoritmos de Machine Learning.
2. Tokenização - Processo de dividir o texto em palavras ou frases menores, que são chamadas de tokens. Existem muitas abordagens para tokenização, para os propósitos deste projeto, a tokenização considerando espaços em branco entre as palavras será suficiente, e a biblioteca *NLTK* será utilizada em conjunto. Ferramentas do NLTK implementadas:
   - Remoção de stopwords
   - Eliminação de pontuações
   - Normalização do corpus
   - Stemming do corpus
3. TF-IDF - técnica de processamento de linguagem natural que é usada para avaliar a importância de uma palavra em um documento. A sigla TF-IDF significa "Term Frequency-Inverse Document Frequency" e é uma métrica que avalia a importância de uma palavra em um documento. A técnica é composta por duas partes: a primeira parte é a frequência do termo (TF) e a segunda parte é a frequência inversa do documento (IDF).

### **Algoritmos de Machine Learning Aplicados**

1. Regressão Logística
2. Naive Bayes Gaussiano
3. Naive Bayes Multinomial
4. Random Forest
5. Support Vector Machine

### **Métricas de análise**

1. Curva ROC
2. Matriz de Confusão
3. Acurácia
4. Precisão
5. Recall

### **Visualizações**

Foi utilizada a ferramenta wordcloud com o propósito de se obter nuvens de palavras com as avaliações obtidas dos algoritmos.

### **Resumo dos principais resultados obtidos**

#### Curva ROC

<img src="https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/blob/main/reports/figures/roc_curve.png" width=50% height=50%>

#### Análise da curva ROC

A curva ROC é uma ferramenta que permite avaliar a capacidade discriminativa de um classificador binário. A área sob a curva (AUC) é uma métrica que permite comparar a capacidade discriminativa de diferentes classificadores. Quanto maior a AUC, melhor a capacidade discriminativa do classificador. E além disso, a curva ROC permite visualizar a taxa de verdadeiros positivos (sensibilidade) em função da taxa de falsos positivos (1-especificidade). Valore acentuados para o lado esquerdo superior indicam que os classificadores acertam mais verdadeiros positivos e erram menos falsos positivos.

Do gráfico plotado e valores acima mostrados, os modelos de regressão logística e support vector machine apresentaram os melhores resultados, com AUC de 0.948 e 0.951, respectivamente.

### **Matriz de Confusão**

A matriz de confusão é uma tabela que mostra as frequências de classificação para cada classe de um modelo. A tabela tem duas linhas e duas colunas que reportam o número de:

- Verdadeiros positivos (VP) - classificações corretas da classe positiva
- Falsos positivos (FP) - classificações incorretas da classe positiva
- Verdadeiros negativos (VN) - classificações corretas da classe negativa
- Falsos negativos (FN) - classificações incorretas da classe negativa

#### Matriz de confusão - Regressão Logística

<img src="https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/blob/main/reports/figures/cm_lg.png" width=50% height=50%>

A tabela abaixo mostra a montagem da tabela acima em termos das frequências:

| VN  | FP |
| --- | -- |
| FN  | VP |

|Métricas      |                |               |
|------------- | ---------------| ------------- |
| Acurácia   |    Precisão  | Recall    |
| 87.86%    | 87.18%       | 88.54%      |

Considerando o contexto de classificação de textos, os resultados obtidos para o melhor classificador são bem satisfatórios, mostrando que o algoritmo consegue classificar corretamente positivos que são realmente positivos, e negativos que são realmente negativos.

### WordCloud de palavras


#### Nuvem positiva

<img src="https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/blob/main/reports/figures/positive_cloud_tfidf.png" width=50% height=50%>

#### Nuvem negativa

<img src="https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/blob/main/reports/figures/negative_cloud_tfidf.png" width=50% height=50%>

#### Análise das WordClouds

As nuvens de palavras acima mostram de forma satisfatória que as palavras mais relevantes para avaliações positivas e negativas estão bem construídas, sendo as palavras encontradas em sua forma raiz, base, após processo de stemming.

Para mais detalhes sobre todos os resultados obtidos:
- [Bag of Words](https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/blob/main/notebooks/modeling/models_bow.ipynb)
- [Tratamento dos textos](https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/tree/main/notebooks/preprocessing)
- [Resultados após tratamentos](https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/blob/main/notebooks/modeling/models_nltk.ipynb)
- [Resultados após TF-IDF](https://github.com/guialbquerque/AnaliseDeSentimentosIMDB/blob/main/notebooks/modeling/models_tfidf.ipynb)






