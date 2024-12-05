Titanic Survival Prediction:

Descrição do Projeto
O objetivo deste projeto é prever a sobrevivência de passageiros do Titanic com base em dados históricos. 
Utilizando técnicas de aprendizado de máquina, explorei diferentes modelos de classificação para realizar a previsão, após um processo de pré-processamento dos dados. 
A análise e modelagem envolvem a extração de informações dos dados brutos, a normalização de variáveis contínuas, 
a codificação de variáveis categóricas, a imputação de valores ausentes e a seleção dos melhores parâmetros do modelo por meio de busca de hiperparâmetros.

Etapas do Projeto

1. Leitura e Análise dos Dados
   
Os dados foram carregados a partir de dois arquivos CSV: um contendo as informações de treinamento (train.csv) e outro de teste (test.csv). 

A primeira etapa consistiu na leitura desses arquivos e na análise inicial da estrutura dos dados, verificando as dimensões (número de linhas e colunas) de cada conjunto.

2. Pré-processamento dos Dados
O pré-processamento envolveu várias etapas cruciais:

Extração de Títulos: A partir do nome dos passageiros, extraí os títulos (como "Sr.", "Sra.") presentes nos dados.

Extração da Primeira Letra da Cabine: Para lidar com valores ausentes na coluna "Cabin", extraí a primeira letra de cada valor e usei como característica.

Remoção de Colunas Desnecessárias: Algumas colunas como Name, Cabin, Ticket e PassengerId foram descartadas, pois não contribuem diretamente para a análise.

Codificação de Variáveis Categóricas: Variáveis como sexo, título, letra da cabine e embarque foram codificadas utilizando a técnica de One-Hot Encoding.

Normalização dos Dados Contínuos: Variáveis numéricas como Age, SibSp, Parch e Fare foram normalizadas para o intervalo [0, 1] usando o MinMaxScaler.

Imputação de Valores Ausentes: A imputação foi realizada nos dados ausentes utilizando o IterativeImputer, que preencheu os valores faltantes com base em outros dados.

3. Modelos de Classificação e Validação
Diversos modelos de classificação foram testados para avaliar a acurácia do modelo preditivo:

Regressão Logística

Máquinas de Vetores de Suporte (SVC)

K-Nearest Neighbors (KNN)

Naive Bayes

Gradient Boosting Classifier

Para avaliar o desempenho de cada modelo, utilizei a validação cruzada, que divide os dados em subconjuntos e treina os modelos em diferentes combinações de dados, 
permitindo uma avaliação mais robusta.

4. Busca de Hiperparâmetros
5. 
Para melhorar a performance do modelo final, realizei uma busca de hiperparâmetros utilizando o RandomizedSearchCV.
Essa técnica permitiu encontrar os melhores parâmetros para o modelo GradientBoostingClassifier, variando os valores de hiperparâmetros como o número de estimadores,
a taxa de aprendizado, a profundidade máxima das árvores, entre outros.

7. Treinamento e Previsão
8. 
Após encontrar os melhores parâmetros, o modelo foi treinado com o conjunto de dados de treinamento.
Com o modelo ajustado, fiz previsões sobre o conjunto de dados de teste, determinando se cada passageiro sobreviveu ou não.

9. Resultados e Saída
As previsões geradas pelo modelo foram associadas ao PassengerId e mapeadas de valores binários (0 ou 1) para rótulos legíveis,
como "Survived" (Sobreviveu) e "Not Survived" (Não Sobreviveu).
O resultado final foi um DataFrame contendo as previsões para cada passageiro do conjunto de teste.

Tecnologias Utilizadas
Pandas: Para manipulação de dados e pré-processamento.
Scikit-learn: Para modelagem preditiva, validação cruzada e busca de hiperparâmetros.
NumPy: Para operações numéricas e manipulação de arrays.
SciPy: Para distribuição estatística e geração de números aleatórios.
Conclusão
Neste projeto, apliquei técnicas de aprendizado de máquina para prever a sobrevivência dos passageiros do Titanic. 
A partir de um conjunto de dados históricos, o modelo foi treinado e otimizado para realizar previsões sobre um novo conjunto de dados. 
Através de uma série de transformações nos dados, validação de modelos e busca de hiperparâmetros, 
o objetivo foi criar um modelo robusto e eficaz para classificar os passageiros com base em suas características.
