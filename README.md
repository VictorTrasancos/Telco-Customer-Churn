# Telco Customer Churn: análise e previsão de cancelamento de clientes

Projeto da disciplina **Introdução à Ciência de Dados** (ICMC-USP, prof. Francisco Rodrigues). O objetivo é identificar as características associadas ao cancelamento de clientes de uma empresa de telecomunicações e construir modelos de classificação para apontar os clientes com maior propensão ao cancelamento (*churn*).
## Sobre o conjunto de dados
Os dados são do [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle): 7.043 clientes, com informações demográficas, serviços contratados, tempo de permanência (`tenure`), tipo de contrato, forma de pagamento e valores cobrados. A variável resposta, `Churn`, indica se o cliente deixou a empresa no último mês.

| Classe | Clientes | Proporção |
|---|---|---|
| Não cancelou (`No`) | 5.174 | 73,5% |
| Cancelou (`Yes`) | 1.869 | 26,5% |

A base é **desbalanceada**, o que orienta toda a escolha de métricas: a acurácia não é adequada (um modelo que sempre responde "não cancela" acertaria 73,5%), então o projeto usa precisão, sensibilidade, F1, ROC-AUC e precisão média (*Average Precision*, AP) da classe de cancelamento.

> O arquivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` **não está incluído** neste repositório. Baixe-o pelo link acima e coloque-o na mesma pasta do notebook.
`
## Etapas do projeto

O notebook está dividido em partes, cada uma com código e um comentário interpretativo:

| Parte | Conteúdo |
|---|---|
| 0 | Bibliotecas |
| 1 | Preparação dos dados: tipos, ausentes, duplicados, inconsistências, conversão de `TotalCharges`, exclusão de `customerID` e divisão treino/teste estratificada |
| 2 | Análise exploratória (somente no treino): proporção de cancelamentos, medidas descritivas, taxas por grupo com o tamanho dos grupos, gráficos e correlações |
| 3 | Seleção e transformação das variáveis: codificação, padronização, redundâncias e criação de variáveis |
| 4 | Construção de quatro modelos, com hiperparâmetros escolhidos por validação cruzada |
| 5 | Comparação dos métodos sob o mesmo protocolo e efeito da ponderação de classes |
| 6 | Avaliação no teste (matriz de confusão, precisão, sensibilidade, F1, ROC-AUC e AP) e efeito do limiar de decisão |
| 7 | Estratégias de balanceamento: undersampling, oversampling aleatório e SMOTE |
| 8 | Interpretação: importância das variáveis, consistência entre modelos, razões de chances, priorização de ações, limitações e distinção entre associação e causalidade |

## Metodologia

- **Preparação.** As 11 linhas com `TotalCharges` em branco têm `tenure = 0` (clientes ainda não cobrados), e receberam `TotalCharges = 0`. `SeniorCitizen` foi recodificada como `Yes`/`No`. `customerID` foi removido dos preditores: é um identificador único, sem informação generalizável, e permitiria ao modelo memorizar clientes (sobreajuste).
- **Divisão.** 80% treino e 20% teste, **estratificada** pela variável resposta (`random_state=42`), feita antes de qualquer análise. O teste mantém a distribuição original das classes e só é usado na avaliação final.
- **Pré-processamento.** One-hot nas variáveis categóricas (`drop='if_binary'`) e padronização (`StandardScaler`) nas numéricas, necessária para a regressão logística regularizada e para o KNN.
- **Modelos.** Regressão Logística, Árvore de Decisão, KNN e Random Forest.
- **Seleção de hiperparâmetros.** `GridSearchCV` com validação cruzada estratificada de 5 dobras, usando a **precisão média (AP)** como critério, por ser independente do limiar e adequada a classes desbalanceadas.
- **Sem vazamento de dados.** Todas as transformações dependentes dos dados (padronização, one-hot e reamostragem) ficam dentro de um `Pipeline` e são ajustadas apenas no treino de cada dobra. O modelo é escolhido pela validação cruzada e não pelo teste, e o limiar de decisão é calculado com predições fora-da-dobra do treino.
- **Balanceamento.** Ponderação de classes (`class_weight='balanced'`) e três técnicas de reamostragem (`RandomUnderSampler`, `RandomOverSampler` e `SMOTE`, da biblioteca [imbalanced-learn](https://imbalanced-learn.org)), aplicadas apenas dentro das dobras de treino.

## Principais resultados

### Desempenho no conjunto de teste (1.409 clientes, 374 cancelamentos; limiar 0,5)

| Modelo | Precisão | Sensibilidade | F1 | ROC-AUC | AP |
|---|---|---|---|---|---|
| Regressão Logística | 0,657 | 0,559 | 0,604 | 0,842 | 0,634 |
| Random Forest | 0,649 | 0,524 | 0,580 | 0,838 | 0,655 |
| Árvore de Decisão | 0,635 | 0,567 | 0,599 | 0,830 | 0,622 |
| KNN | 0,589 | 0,575 | 0,582 | 0,829 | 0,614 |

A **Regressão Logística** foi escolhida pela validação cruzada (AP = 0,6615). Os quatro modelos ficaram próximos, e o "melhor" depende da métrica (a Regressão Logística lidera em precisão, F1 e ROC-AUC; a Random Forest, em AP; o KNN, em sensibilidade).

### Efeito do limiar de decisão (Regressão Logística)

| Limiar | Precisão | Sensibilidade | F1 | Falsos negativos | Falsos positivos |
|---|---|---|---|---|---|
| 0,50 | 0,657 | 0,559 | 0,604 | 165 | 109 |
| 0,33 (ótimo no treino) | 0,539 | 0,722 | 0,617 | 104 | 231 |

Reduzir o limiar detecta 61 cancelamentos a mais ao custo de 122 falsos alarmes a mais. A escolha depende do custo da ação de retenção em relação ao valor de um cliente perdido.

### Contato com os 10% de clientes de maior risco

Contatando 141 clientes (10% do teste), a Regressão Logística identifica **28,3%** de todos os cancelamentos (106 de 374), com precisão de 75,2% e *lift* de 2,83 sobre o contato aleatório. O máximo possível com esse número de contatos seria 37,7%.

### Variáveis mais relevantes

Nos três modelos analisados (Regressão Logística, Árvore e Random Forest), as variáveis mais importantes por permutação foram **`tenure`**, **`Contract`**, **`InternetService`** e **`TotalCharges`**, seguidas por `TechSupport`, `OnlineSecurity` e `PaymentMethod`. A correlação de Spearman entre os rankings de importância foi de 0,67 a 0,88.

Fatores associados a **maior** risco: contrato mensal, permanência curta, internet de fibra óptica e pagamento por cheque eletrônico. Fatores associados a **menor** risco: contrato de dois anos, maior permanência, DSL e presença de suporte técnico e segurança online.

## Limitações

- Os dados são **observacionais** e de um único momento, sem datas dos eventos, então não é possível validar o modelo ao longo do tempo.
- O cancelamento refere-se apenas ao último mês, e não há informação sobre custos das ações de retenção nem sobre o valor de cada cliente.
- O teste tem apenas 374 cancelamentos, e os resultados vêm de uma única divisão, sem intervalos de confiança. Diferenças pequenas entre modelos não devem ser consideradas relevantes.
- Há forte colinearidade entre algumas variáveis (como `tenure` e `TotalCharges`), o que torna a importância e os coeficientes individuais menos confiáveis.
- **Os resultados mostram associações, não causalidade.** Não é possível concluir que ações de retenção reduziriam os cancelamentos sem um experimento controlado (por exemplo, um teste A/B).
rn](https://scikit-learn.org) e [imbalanced-learn](https://imbalanced-learn.org)
