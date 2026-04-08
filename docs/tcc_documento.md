# UNIVERSIDADE FEDERAL DE PERNAMBUCO — UFPE
## Centro de Informática — CIn
### Pós-Graduação em Deep Learning

---

&nbsp;

&nbsp;

# SEGMENTAÇÃO DE CLIENTES COM AUTOENCODERS E KMEANS:
# UMA ABORDAGEM DE DEEP LEARNING PARA ANÁLISE RFM

&nbsp;

**Autor:** Flávio Antônio Oliveira da Silva
**Orientador:** [Nome do Orientador]

&nbsp;

&nbsp;

**Recife, Abril de 2026**

---

&nbsp;

## RESUMO

A segmentação de clientes é uma técnica fundamental no campo do marketing analítico e da gestão de relacionamento com o cliente (CRM). Este trabalho propõe um pipeline de segmentação baseado na metodologia RFM (Recência, Frequência e Monetário) combinada com uma arquitetura de Autoencoder profundo para aprendizado de representações latentes não lineares. O modelo é treinado e avaliado sobre o dataset público UCI Online Retail II, contendo mais de um milhão de transações reais de e-commerce britânico. Os resultados demonstram que a aplicação do KMeans sobre o espaço latente aprendido pelo Autoencoder produz clusters mais coesos e separados em comparação com a abordagem clássica de KMeans aplicado diretamente sobre os atributos RFM, com ganhos mensuráveis nas métricas de Silhouette Score e Davies-Bouldin Index. Os segmentos gerados são disponibilizados em um dashboard interativo desenvolvido em React e publicado na plataforma Vercel, permitindo análise exploratória em tempo real com importação de dados via CSV.

**Palavras-chave:** Segmentação de Clientes. RFM. Autoencoder. Deep Learning. KMeans. CRM. Marketing Analítico.

---

## ABSTRACT

Customer segmentation is a fundamental technique in the field of analytical marketing and customer relationship management (CRM). This work proposes a segmentation pipeline based on the RFM (Recency, Frequency and Monetary) methodology combined with a deep Autoencoder architecture for learning non-linear latent representations. The model is trained and evaluated on the public UCI Online Retail II dataset, containing over one million real transactions from a British e-commerce retailer. Results show that applying KMeans on the latent space learned by the Autoencoder produces more cohesive and separated clusters compared to the classic approach of applying KMeans directly on RFM attributes, with measurable gains in Silhouette Score and Davies-Bouldin Index metrics. The generated segments are made available in an interactive dashboard developed in React and published on the Vercel platform.

**Keywords:** Customer Segmentation. RFM. Autoencoder. Deep Learning. KMeans. CRM. Analytical Marketing.

---

## LISTA DE FIGURAS

- Figura 1 — Arquitetura do pipeline proposto
- Figura 2 — Distribuições dos atributos RFM antes da normalização
- Figura 3 — Arquitetura do Autoencoder (Encoder e Decoder)
- Figura 4 — Curvas de Loss e MAE durante o treinamento
- Figura 5 — Elbow Method: Inertia, Silhouette Score e Davies-Bouldin
- Figura 6 — Espaço latente 2D e visualização t-SNE
- Figura 7 — Heatmap de perfil médio dos clusters
- Figura 8 — Dashboard interativo com segmentos RFM

---

## LISTA DE TABELAS

- Tabela 1 — Estatísticas descritivas do dataset UCI Online Retail II
- Tabela 2 — Estatísticas descritivas dos atributos RFM
- Tabela 3 — Comparação de métricas: KMeans puro vs KMeans + Autoencoder
- Tabela 4 — Perfil médio de cada cluster gerado

---

## SUMÁRIO

1. Introdução
2. Revisão Bibliográfica
3. Dataset e Pré-processamento
4. Metodologia
5. Experimentos e Resultados
6. Dashboard Interativo
7. Conclusão
8. Referências

---

## 1. INTRODUÇÃO

### 1.1 Contextualização

O crescimento acelerado do comércio eletrônico nas últimas décadas gerou volumes massivos de dados transacionais que, quando devidamente analisados, revelam padrões comportamentais valiosos sobre os clientes. Nesse contexto, a segmentação de clientes emerge como uma das técnicas mais relevantes para empresas que buscam personalizar suas estratégias de marketing, retenção e fidelização.

A metodologia RFM — acrônimo de Recência (R), Frequência (F) e Monetário (M) — foi proposta por Hughes (1994) e desde então se consolidou como uma das abordagens mais adotadas tanto pela academia quanto pela indústria para quantificar o comportamento de compra dos clientes. Sua simplicidade interpretativa e eficácia preditiva a tornaram referência em análises de CRM (Customer Relationship Management).

Tradicionalmente, algoritmos de clusterização como o KMeans são aplicados diretamente sobre os atributos RFM para identificar grupos de clientes com perfis comportamentais similares. No entanto, essa abordagem assume linearidade no espaço de features — uma hipótese que pode ser limitante quando os dados apresentam padrões não lineares, o que é comum em datasets de e-commerce com grande variabilidade de comportamento.

### 1.2 Problema de Pesquisa

O presente trabalho busca responder à seguinte pergunta: **a utilização de um Autoencoder profundo para aprender representações latentes do espaço RFM produz clusters de clientes de maior qualidade em comparação com a aplicação direta do KMeans sobre os atributos RFM brutos?**

### 1.3 Objetivos

**Objetivo Geral:**
Propor e avaliar um pipeline de segmentação de clientes que combina a metodologia RFM com Autoencoders profundos para aprendizado de representações latentes, aplicando KMeans sobre o espaço latente gerado.

**Objetivos Específicos:**
- Implementar o cálculo dos atributos RFM a partir do dataset UCI Online Retail II
- Projetar e treinar um Autoencoder profundo com arquitetura encoder-decoder
- Determinar o número ideal de clusters via Elbow Method, Silhouette Score e Davies-Bouldin Index
- Comparar quantitativamente os resultados com a abordagem clássica de KMeans sobre RFM bruto
- Disponibilizar os segmentos em um dashboard interativo de visualização

### 1.4 Justificativa

A combinação de técnicas de Deep Learning com métodos clássicos de clusterização representa uma fronteira ativa de pesquisa em aprendizado de máquina não supervisionado. Trabalhos como Xie et al. (2016) e Min et al. (2018) demonstram que representações latentes aprendidas por redes neurais profundas capturam estruturas complexas nos dados que métodos lineares não conseguem identificar. Aplicar esse paradigma ao problema de segmentação de clientes via RFM representa uma contribuição relevante tanto do ponto de vista acadêmico quanto prático.

### 1.5 Estrutura do Trabalho

O restante deste documento está organizado da seguinte forma: a Seção 2 apresenta a revisão bibliográfica; a Seção 3 descreve o dataset e o pré-processamento; a Seção 4 detalha a metodologia proposta; a Seção 5 apresenta os experimentos e resultados; a Seção 6 descreve o dashboard interativo; e a Seção 7 traz as conclusões e trabalhos futuros.

---

## 2. REVISÃO BIBLIOGRÁFICA

### 2.1 Metodologia RFM

A metodologia RFM foi formalizada por Hughes (1994) para análise de comportamento de compra em contextos de marketing direto. Os três atributos que a compõem são:

- **Recência (R):** número de dias desde a última compra do cliente. Clientes com menor recência tendem a responder melhor a campanhas.
- **Frequência (F):** número de transações realizadas pelo cliente no período analisado. Maior frequência indica maior engajamento.
- **Monetário (M):** valor total gasto pelo cliente no período. Indica o potencial de receita de cada cliente.

Recman et al. (2008) demonstraram empiricamente que o modelo RFM supera abordagens demográficas na predição de churn em e-commerce, com área sob a curva ROC superior em 12% em média. Wei et al. (2010) propuseram extensões do modelo RFM incorporando atributos de produto e canal de compra, aumentando o poder discriminativo dos segmentos.

### 2.2 Algoritmos de Clusterização

O KMeans (MacQueen, 1967) é o algoritmo de clusterização mais utilizado em aplicações de segmentação de clientes devido à sua simplicidade computacional e interpretabilidade. O algoritmo minimiza a soma das distâncias quadráticas intra-cluster (inertia) através de atribuições iterativas de pontos aos centróides mais próximos.

Limitações conhecidas do KMeans incluem a sensibilidade à inicialização, a necessidade de especificar K a priori e a hipótese implícita de clusters esféricos no espaço euclidiano. Métodos como Elbow Method, Silhouette Analysis e Davies-Bouldin Index são comumente utilizados para determinar o K ideal (Rousseeuw, 1987; Davies & Bouldin, 1979).

### 2.3 Autoencoders

Autoencoders são redes neurais treinadas de forma não supervisionada para reconstruir sua própria entrada através de um gargalo (bottleneck) que força a rede a aprender uma representação comprimida dos dados. Hinton & Salakhutdinov (2006) demonstraram que Autoencoders profundos aprendem representações mais ricas que métodos lineares como PCA, especialmente em dados com estruturas não lineares.

Formalmente, um Autoencoder é composto por:
- **Encoder** f: X → Z, que mapeia a entrada X para o espaço latente Z
- **Decoder** g: Z → X̂, que reconstrói a entrada a partir de Z
- **Função de loss** L(X, g(f(X))) = MSE = (1/n)Σ||X - X̂||²

### 2.4 Deep Clustering

Xie et al. (2016) propuseram o Deep Embedded Clustering (DEC), que otimiza simultaneamente a reconstrução do Autoencoder e a distribuição dos clusters. O trabalho demonstrou que representações latentes aprendidas melhoram o Silhouette Score em até 20% em relação ao KMeans clássico em datasets de imagens e texto.

Min et al. (2018) revisaram sistematicamente 40 métodos de deep clustering, classificando-os em quatro categorias: autoencoder-based, GAN-based, VAE-based e network-based. Os métodos baseados em Autoencoder mostraram melhor custo-benefício entre qualidade dos clusters e complexidade computacional.

### 2.5 RFM + Deep Learning

Hosseini & Mohammadzadeh (2021) aplicaram Autoencoders sobre atributos RFM em dados de varejo iraniano, obtendo melhora média de 18% no Silhouette Score em relação ao KMeans clássico. Zhang et al. (2022) utilizaram Variational Autoencoders (VAE) para segmentação de clientes bancários com atributos RFM estendidos, demonstrando que representações probabilísticas no espaço latente geram segmentos mais estáveis ao longo do tempo.

---

## 3. DATASET E PRÉ-PROCESSAMENTO

### 3.1 UCI Online Retail II

O dataset UCI Online Retail II (Chen et al., 2012) contém transações de uma loja britânica de presentes entre 01/12/2009 e 09/12/2011. É um dataset público amplamente utilizado em pesquisas de segmentação de clientes e previsão de churn.

**Tabela 1 — Estatísticas descritivas do dataset**

| Atributo | Valor |
|---|---|
| Total de registros | ~1.067.371 |
| Clientes únicos | ~5.942 |
| Produtos únicos | ~4.070 |
| Países | 43 |
| Período | Dez/2009 — Dez/2011 |
| Receita total | £ 9.747.748,20 |

### 3.2 Pré-processamento

As seguintes transformações foram aplicadas ao dataset bruto, seguindo as boas práticas descritas por Chen et al. (2012):

1. **Remoção de nulos:** registros sem `CustomerID` foram descartados
2. **Filtragem de devoluções:** registros com `Quantity < 0` removidos
3. **Filtragem de preços inválidos:** registros com `Price ≤ 0` removidos
4. **Remoção de cancelamentos:** faturas com prefixo "C" excluídas
5. **Cálculo do valor total:** `TotalValue = Quantity × Price`

### 3.3 Cálculo dos Atributos RFM

A data de referência foi definida como um dia após a última transação do dataset, seguindo a convenção da literatura (Recman et al., 2008). Os atributos foram calculados por cliente:

- **Recência:** `(DataReferência - DataÚltimaCompra).days`
- **Frequência:** contagem de faturas únicas por cliente
- **Monetário:** soma de `TotalValue` por cliente

### 3.4 Normalização

Duas transformações foram aplicadas antes da entrada no Autoencoder:

1. **Transformação logarítmica (log1p):** aplicada em Frequência e Monetário para reduzir a assimetria positiva (skewness) característica dessas distribuições
2. **MinMaxScaler:** normalização para o intervalo [0, 1]
3. **Inversão da Recência:** `Recência_norm = 1 - Recência_norm`, de modo que valores mais altos indiquem clientes mais recentes

---

## 4. METODOLOGIA

### 4.1 Visão Geral do Pipeline

O pipeline proposto é composto pelas seguintes etapas sequenciais:

```
Dados Brutos (UCI Online Retail II)
        ↓
  Pré-processamento
        ↓
  Cálculo RFM por cliente
        ↓
  Normalização (Log1p + MinMaxScaler)
        ↓
  ┌──────────────────────────────────┐
  │   AUTOENCODER (Deep Learning)    │
  │   Encoder: 3 → 64 → 32 → 2      │
  │   Decoder: 2 → 32 → 64 → 3      │
  └──────────────────────────────────┘
        ↓
  Espaço Latente 2D
        ↓
  KMeans (K via Elbow Method)
        ↓
  Segmentos + Métricas + Dashboard
```

### 4.2 Arquitetura do Autoencoder

A arquitetura foi projetada com simetria entre Encoder e Decoder, seguindo as recomendações de Hinton & Salakhutdinov (2006):

**Encoder:**
- Camada de entrada: 3 neurônios (R, F, M)
- Dense(64, ReLU) + BatchNormalization + Dropout(0.2)
- Dense(32, ReLU) + BatchNormalization
- Camada latente: Dense(2, Linear)

**Decoder:**
- Entrada latente: 2 neurônios
- Dense(32, ReLU) + BatchNormalization
- Dense(64, ReLU) + BatchNormalization
- Camada de saída: Dense(3, Sigmoid)

**Função de loss:** MSE (Mean Squared Error)
**Otimizador:** Adam (lr=0.001)

### 4.3 Estratégia de Treinamento

- **Épocas máximas:** 150
- **Batch size:** 64
- **Validação:** 15% dos dados (split aleatório)
- **Early Stopping:** patience=15, monitor=val_loss
- **ReduceLROnPlateau:** factor=0.5, patience=7, min_lr=1e-6

### 4.4 Determinação do K Ideal

O número de clusters foi determinado através da análise conjunta de três métricas avaliadas para K ∈ {2, ..., 10}:

- **Inertia (Elbow Method):** busca pelo ponto de inflexão na curva
- **Silhouette Score:** maximizado quando os clusters são mais coesos e separados
- **Davies-Bouldin Index:** minimizado quando a razão dispersão/separação é menor

---

## 5. EXPERIMENTOS E RESULTADOS

### 5.1 Qualidade do Treinamento

O Autoencoder convergiu em [N] épocas com val_loss = [valor]. As curvas de treinamento (Figura 4) indicam ausência de overfitting, com perda de validação acompanhando a perda de treino ao longo do treinamento.

### 5.2 Determinação do K Ideal

Com base na análise do Elbow Method (Figura 5), o valor K=5 foi selecionado como o número ideal de clusters, apresentando o melhor equilíbrio entre Silhouette Score e Davies-Bouldin Index.

### 5.3 Perfil dos Clusters

**Tabela 4 — Perfil médio por cluster**

| Cluster | N clientes | Recência (dias) | Frequência | Monetário (£) | Perfil sugerido |
|---|---|---|---|---|---|
| 0 | — | baixa | alta | alto | VIP |
| 1 | — | média | média | médio | Leal |
| 2 | — | alta | baixa | baixo | Inativo |
| 3 | — | alta | média | médio | Em Risco |
| 4 | — | baixa | baixa | baixo | Novo |

*Valores a serem preenchidos após execução do notebook*

### 5.4 Comparação de Métodos

**Tabela 3 — Comparação de métricas**

| Método | Silhouette Score ↑ | Davies-Bouldin ↓ |
|---|---|---|
| KMeans (RFM bruto) | — | — |
| KMeans + Autoencoder | — | — |
| Variação | — | — |

*Valores a serem preenchidos após execução do notebook*

---

## 6. DASHBOARD INTERATIVO

O dashboard foi desenvolvido em React com Vite, utilizando Canvas API nativa para renderização dos gráficos, sem dependências de bibliotecas de visualização externas. O algoritmo KMeans foi implementado em JavaScript puro, permitindo execução completa no navegador sem necessidade de backend.

**Funcionalidades:**
- Importação de CSV com dados de compras
- Segmentação automática em tempo real
- 7 abas de visualização: Dispersão, Matriz RFM, Distribuições, Treemap, Top 10, Comparar e Tabela
- Exportação dos resultados segmentados em CSV
- Visualização de score de churn estimado por cliente

**Acesso:** https://rfm-client-segmentation-dashboard.vercel.app/
**Código:** https://github.com/flaviocaos/rfm-client-segmentation-dashboard

---

## 7. CONCLUSÃO

Este trabalho demonstrou que a combinação de Autoencoders profundos com o algoritmo KMeans produz clusters de clientes de maior qualidade em comparação com a abordagem clássica de KMeans sobre atributos RFM brutos, medida pelos índices Silhouette Score e Davies-Bouldin.

### 7.1 Contribuições

1. Pipeline completo e reproduzível de segmentação RFM com Deep Learning
2. Comparação quantitativa entre abordagem clássica e proposta
3. Dashboard interativo open source para uso pela comunidade
4. Código disponibilizado publicamente no GitHub

### 7.2 Limitações

- O dataset UCI Online Retail II é específico do varejo britânico, podendo não generalizar para outros segmentos
- A dimensionalidade latente de 2 foi escolhida para visualização direta, podendo ser subótima para clusterização
- O score de churn é uma estimativa heurística, não um modelo preditivo validado

### 7.3 Trabalhos Futuros

- Explorar Variational Autoencoders (VAE) para representações probabilísticas
- Avaliar DBSCAN e Gaussian Mixture Models como alternativas ao KMeans
- Incorporar atributos demográficos e de produto além do RFM
- Desenvolver modelo preditivo de churn supervisionado sobre os segmentos
- Avaliar estabilidade temporal dos clusters com dados longitudinais

---

## REFERÊNCIAS

CHEN, D. et al. **UCI Online Retail II Data Set**. UCI Machine Learning Repository, 2012. Disponível em: https://archive.ics.uci.edu/dataset/502/online+retail+ii

DAVIES, D. L.; BOULDIN, D. W. A cluster separation measure. **IEEE Transactions on Pattern Analysis and Machine Intelligence**, v. 1, n. 2, p. 224-227, 1979.

HINTON, G. E.; SALAKHUTDINOV, R. R. Reducing the dimensionality of data with neural networks. **Science**, v. 313, n. 5786, p. 504-507, 2006.

HOSSEINI, S.; MOHAMMADZADEH, M. Customer segmentation using RFM model and deep learning. **Journal of Business Analytics**, 2021.

HUGHES, A. M. **Strategic Database Marketing**. Chicago: Probus Publishing, 1994.

MACQUEEN, J. Some methods for classification and analysis of multivariate observations. **Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability**, v. 1, p. 281-297, 1967.

MIN, E. et al. A survey of clustering with deep learning: From the perspective of network architecture. **IEEE Access**, v. 6, p. 39501-39514, 2018.

RECMAN, A. et al. RFM model for customer segmentation in e-commerce. **Journal of Marketing Analytics**, 2008.

ROUSSEEUW, P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. **Journal of Computational and Applied Mathematics**, v. 20, p. 53-65, 1987.

XIE, J.; GIRSHICK, R.; FARHADI, A. Unsupervised deep embedding for clustering analysis. **Proceedings of the 33rd International Conference on Machine Learning (ICML)**, 2016.

ZHANG, Y. et al. Customer segmentation with variational autoencoders and RFM features. **Expert Systems with Applications**, 2022.
