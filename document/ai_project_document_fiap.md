
<img src="../assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=30% height=30%>

# Implementando algoritmos de Machine Learning com Scikit-learn

## Nome do Grupo
Deniz Feital Armanhe - individual

#### Nomes dos integrantes do grupo
Deniz Feital Armanhe


## Sumário

[1. Introdução](#c1)

[2. Visão Geral do Projeto](#c2)

[3. Desenvolvimento do Projeto](#c3)

[4. Resultados e Avaliações](#c4)

[5. Conclusões e Trabalhos Futuros](#c5)

[6. Referências](#c6)

[Anexos](#c7)

<br>

# <a name="c1"></a>1. Introdução

## 1.1. Escopo do Projeto

### 1.1.1. Cap 3 - Implementando algoritmos de Machine Learning com Scikit-learn

Nesta nova fase do projeto, Fase 4, Cap -3, o objetivo é efetuar a análise de um dataset pré-definido, gerando estatísticas descritivas, gráficos e normalização dos dados. Efetuar treinamento utilizando soluções diferentes e analisar os resultados. 

### 1.1.2. Descrição da Solução Desenvolvida

A solução foi desenvolvida utilizando Python e diversos módulos como Pandas, numpy, sklearn, seaborn e matplotlib.pyplot. 

# <a name="c2"></a>2. Visão Geral do Projeto

## 2.1. Objetivos do Projeto

Demonstrar as novas funcionalidades deste módulo com a adição das mesmas ao ambiente e analisar o dataset fornecido, ajustando-o para que a análise dos dados possa ser feita como esperado.

## 2.2. Público-Alvo

Bem, isto é um trabalho da FIAP, mas obviamente a idea é que o aprendizado seja aplicado na análise de dados, como mencionamos acima, diversas funcionalidades foram utilizadas que podem ser úteis para quem está aprendendo Machine Learning. Este trabalho é voltado para o ramo agrícola, mas o conhecimento se aplica a todas as áreas.

## 2.3. Metodologia

*A metodologia foi utilizar toda a nova documentação fornecida no módulo 4 e também nos módulos anteriores. Como base deste trabalho específico, seguimos o CRISP-DM.

O CRISP-DM tem 6 grandes passos. São eles:
Etapa 1: Entendimento do Negócio
Etapa 2: Entendimento dos dados
Etapa 3: Preparação dos dados
Etapa 4: Desenvolvimento do estudo ou análise
Etapa 5: Validação
Etapa 6: Implantação do projeto e acompanhamento

![image](https://github.com/user-attachments/assets/cf2f47ae-d048-41d4-8304-a2c8ece2bbc3)

fonte: https://www.preditiva.ai/blog/entenda-o-crisp-dm-suas-etapas-e-como-de-fato-gerar-valor-com-essa-metodologia


# <a name="c3"></a>3. Desenvolvimento do Projeto

## 3.1. Tecnologias Utilizadas

  Python
  
  Visual Studio Code
  
  Jupiter Notebook
  
  ChatGPT
  
  Além das bibliotecas que contans no arquivo requirements.txt


## 3.2. Modelagem e Algoritmos

*Foram utilizados principalmente o Scikit-learn e o Stremlit, além de bibliotecas Python para criação de datasets.

## 3.3. Treinamento e Teste

Utilizamos o Python e três modelos: K-Nearest Neighbors (KNN), Random Forest e Logistic Regression.

# <a name="c4"></a>4. Resultados e Avaliações

## 4.1. Análise dos Resultados

Temos bastante coisa para mostrar aqui, vamos lá.

Dataset:

Abaixo uma amostra do dataset:

![image](https://github.com/user-attachments/assets/e5b8b1d4-fefb-49c1-b0b3-4e68dc092e22)


As atividades foram divididas em quatro grandes atividades:

Para a **Atividade 1** o objetivo foi analisar e pré-processar os dados fornecidos no dataset.

Abaixo alguns resultados:

Histograma dos componentes:

Aqui podemos ver a distribuição dos mesmos. Este gráfico neste momento é apenas informativo.
![image](https://github.com/user-attachments/assets/2c194f0f-41b4-46c0-a8f9-368e8a908907)

Boxplots dos componentes:

Aqui percebemos também que o campo "assimetria" e o campo "compacidade" apresentaram outliers mais gritantes. Os demais possuem uma certa distruição, porém área e perímetro e distribuição é maior, ou seja, os dados não são tão homogêneos.

![image](https://github.com/user-attachments/assets/9c32f233-99a3-4ab7-bfc8-aec79db44664)


Gráfico de dispersão dos componentes:

Aqui percebemos que, no geral, existe uma homogeneidade estre as classes (target 1, 2 e 3), o que corrobora com o boxsplot.

![image](https://github.com/user-attachments/assets/bc485aad-d6e5-4b09-8067-f9fcc0628c7a)

Informações Gerais:

![image](https://github.com/user-attachments/assets/e59d769d-b335-47cf-8a21-115618e8151f)

Aqui um resumo das estatísticas descritivas

Mediana de cada componente.

Nenhum valor faltante, ou seja, o dataset está preenchido completamente.


![image](https://github.com/user-attachments/assets/a98b6e9e-2b2e-4a25-b10f-a0805873568b)


Informações sobre normalização e padronização

![image](https://github.com/user-attachments/assets/05b7804a-348a-452a-b9a5-a9c3c5392b72)




![image](https://github.com/user-attachments/assets/75ab2330-d154-4e60-a759-1e1f2c361131)

Para a **Atividade 2** o objetivo foi implementar e comparar diferentes algoritmos de classificação: Os algoritmos escolhidos foram: 

K-Nearest Neighbors (KNN);

Random Forest;

Logistic Regression.

Comparando os três temos os seguintes resultados:

Vale notar que a acurácia para o Random Forest foi um pouco abaixo, mas honestamente, os três apresentam resultados bem semelhantes.

![image](https://github.com/user-attachments/assets/a5100091-c480-41f7-b389-699be5ff32ee)

Sobre a matrix de confusão, tambem percebemos resultados semelhantes para os três, conforme abaixo:


![image](https://github.com/user-attachments/assets/32e2c31b-ea0d-48bf-8a87-ea4ae1e6ea8e)

![image](https://github.com/user-attachments/assets/22674852-0f3c-4098-8900-eee54c11f2cd)

![image](https://github.com/user-attachments/assets/97102060-4ba5-4f9a-9b20-0b50bf41d864)

Para a **Atividade 3** o objetivo foi otimizar os modelos para melhorar o desempenho (se necessário).

Percebemos que neste caso, o ganho foi bem pequeno, comparado ao anterior, como vemos abaixo:

![image](https://github.com/user-attachments/assets/f32fb20a-59de-468b-8f4c-9a09be833f7c)

![image](https://github.com/user-attachments/assets/2b5e00dc-bbf9-43c8-b2d3-feb39b0aa924)

![image](https://github.com/user-attachments/assets/25246c36-9a5a-447d-811f-ce3afeca029c)

Como comparação, podemos realmente verificar que a diferença não foi grande:
![image](https://github.com/user-attachments/assets/ece8ef1f-4c2d-4138-bc22-5b94ee52fc43)

Para a **Atividade 4** o objetivo foi interpretar os resultados e extrair insights relevantes, que foi exatamente o que demonstrei neste documento.


# <a name="c4"></a>5. Demais documentos e anexos

## 4.1 Documentação.

No diretório documents podemos encontrar também um arquivo PDF sobre o Jupiter notebook e o resultado apresentado.
