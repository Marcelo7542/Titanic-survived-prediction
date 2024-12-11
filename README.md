Titanic Survival Prediction:

English description below

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
   
Para melhorar a performance do modelo final, realizei uma busca de hiperparâmetros utilizando o RandomizedSearchCV.
Essa técnica permitiu encontrar os melhores parâmetros para o modelo GradientBoostingClassifier, variando os valores de hiperparâmetros como o número de estimadores,
a taxa de aprendizado, a profundidade máxima das árvores, entre outros.

7. Treinamento e Previsão
   
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



English

Project Description

The objective of this project is to predict the survival of Titanic passengers based on historical data. Using machine learning techniques, 
I explored different classification models to make the prediction after a preprocessing step of the data. 

The analysis and modeling involved extracting information from raw data, normalizing continuous variables, encoding categorical variables, imputing missing values, and selecting the best model parameters through hyperparameter tuning.

Project Steps

Data Reading and Analysis

The data was loaded from two CSV files: one containing the training information (train.csv) and another for testing (test.csv).

The first step was reading these files and performing an initial analysis of the data structure, checking the dimensions (number of rows and columns) of each dataset.

Data Preprocessing

The preprocessing involved several crucial steps:

Title Extraction: 

From the passenger names, I extracted the titles (e.g., "Mr.", "Mrs.") present in the data.

First Letter of the Cabin: 

To handle missing values in the "Cabin" column, I extracted the first letter of each value and used it as a feature.

Removal of Unnecessary Columns: 

Some columns like Name, Cabin, Ticket, and PassengerId were discarded as they did not directly contribute to the analysis.

Encoding Categorical Variables: 

Variables like gender, title, cabin letter, and embarkation were encoded using One-Hot Encoding.

Normalization of Continuous Data: 

Numeric variables such as Age, SibSp, Parch, and Fare were normalized to the [0, 1] range using MinMaxScaler.

Imputation of Missing Values: 

Imputation was performed on the missing data using the IterativeImputer, which filled in the missing values based on other available data.

Classification Models and Validation

Various classification models were tested to assess the accuracy of the predictive model:

Logistic Regression

Support Vector Machines (SVC)

K-Nearest Neighbors (KNN)

Naive Bayes

Gradient Boosting Classifier

To evaluate the performance of each model, I used cross-validation, which splits the data into subsets and trains the models on different data combinations, providing a more robust evaluation.

Hyperparameter Tuning

To improve the performance of the final model, I performed a hyperparameter search using RandomizedSearchCV. This technique allowed me to find the best parameters for the GradientBoostingClassifier model by varying hyperparameter values such as the number of estimators, learning rate, and maximum tree depth, among others.

Training and Prediction

After finding the best parameters, the model was trained with the training dataset. Once the model was tuned, I made predictions on the test dataset to determine whether each passenger survived or not.

Results and Output

The predictions generated by the model were associated with the PassengerId and mapped from binary values (0 or 1) to readable labels, such as "Survived" and "Not Survived". The final output was a DataFrame containing the predictions for each passenger in the test dataset.

Technologies Used

Pandas: For data manipulation and preprocessing.

Scikit-learn: For predictive modeling, cross-validation, and hyperparameter search.

NumPy: For numerical operations and array manipulation.

SciPy: For statistical distributions and random number generation.

Conclusion

In this project, I applied machine learning techniques to predict the survival of Titanic passengers. 
From a historical dataset, the model was trained and optimized to make predictions on a new dataset. Through a series of data transformations, model validation, and hyperparameter search, the goal was to create a robust and effective model to classify passengers based on their characteristics.
