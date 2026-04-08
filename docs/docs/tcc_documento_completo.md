# UNIVERSIDADE FEDERAL DE PERNAMBUCO — UFPE
## Centro de Informática — CIn
### Pós-Graduação em Deep Learning

---

&nbsp;

# SEGMENTAÇÃO DE CLIENTES COM AUTOENCODERS E KMEANS:
# UMA ABORDAGEM DE DEEP LEARNING PARA ANÁLISE RFM

&nbsp;

**Autor:** Flávio Antônio Oliveira da Silva
**Orientador:** [Nome do Orientador]
**Data:** Abril de 2026

---

## RESUMO

A segmentação de clientes é uma técnica fundamental no campo do marketing analítico e da gestão de relacionamento com o cliente (CRM). Este trabalho propõe um pipeline de segmentação baseado na metodologia RFM (Recência, Frequência e Monetário) combinada com uma arquitetura de Autoencoder profundo para aprendizado de representações latentes não lineares. O modelo é treinado e avaliado sobre o dataset público UCI Online Retail II, contendo mais de um milhão de transações reais de e-commerce britânico.

**Resultados obtidos:**
- Silhouette Score: **0.3992** (+45% vs KMeans puro)
- Davies-Bouldin Index: **0.7382** (-14% vs KMeans puro)
- Clientes segmentados: **4.338**
- Clusters identificados: **5**

**Palavras-chave:** Segmentação de Clientes. RFM. Autoencoder. Deep Learning. KMeans. CRM. Marketing Analítico.

---

## ABSTRACT

Customer segmentation is a fundamental technique in the field of analytical marketing and customer relationship management (CRM). This work proposes a segmentation pipeline based on the RFM (Recency, Frequency and Monetary) methodology combined with a deep Autoencoder architecture for learning non-linear latent representations. Results show that applying KMeans on the latent space learned by the Autoencoder produces more cohesive and separated clusters compared to the classic approach, with measurable gains in Silhouette Score (+45%) and Davies-Bouldin Index (-14%).

**Keywords:** Customer Segmentation. RFM. Autoencoder. Deep Learning. KMeans. CRM.

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

A metodologia RFM — acrônimo de Recência (R), Frequência (F) e Monetário (M) — foi proposta por Hughes (1994) e desde então se consolidou como uma das abordagens mais adotadas tanto pela academia quanto pela indústria para quantificar o comportamento de compra dos clientes.

### 1.2 Problema de Pesquisa

**A utilização de um Autoencoder profundo para aprender representações latentes do espaço RFM produz clusters de clientes de maior qualidade em comparação com a aplicação direta do KMeans sobre os atributos RFM brutos?**

### 1.3 Objetivos

**Objetivo Geral:**
Propor e avaliar um pipeline de segmentação de clientes que combina a metodologia RFM com Autoencoders profundos para aprendizado de representações latentes.

**Objetivos Específicos:**
- Implementar o cálculo dos atributos RFM a partir do dataset UCI Online Retail II
- Projetar e treinar um Autoencoder profundo com arquitetura encoder-decoder
- Determinar o número ideal de clusters via Elbow Method, Silhouette Score e Davies-Bouldin Index
- Comparar quantitativamente os resultados com a abordagem clássica de KMeans
- Disponibilizar os segmentos em um dashboard interativo de visualização

### 1.4 Justificativa

A combinação de técnicas de Deep Learning com métodos clássicos de clusterização representa uma fronteira ativa de pesquisa. Xie et al. (2016) e Min et al. (2018) demonstram que representações latentes aprendidas por redes neurais profundas capturam estruturas complexas nos dados que métodos lineares não conseguem identificar.

---

## 2. REVISÃO BIBLIOGRÁFICA

### 2.1 Metodologia RFM

A metodologia RFM foi formalizada por Hughes (1994) para análise de comportamento de compra. Os três atributos são:

- **Recência (R):** dias desde a última compra
- **Frequência (F):** número de transações realizadas
- **Monetário (M):** valor total gasto pelo cliente

### 2.2 Algoritmos de Clusterização

O KMeans (MacQueen, 1967) é o algoritmo mais utilizado em segmentação de clientes. Limitações incluem sensibilidade à inicialização e hipótese implícita de clusters esféricos.

### 2.3 Autoencoders

Autoencoders são redes neurais treinadas para reconstruir sua entrada através de um bottleneck. Hinton & Salakhutdinov (2006) demonstraram que Autoencoders profundos aprendem representações mais ricas que PCA.

**Função de loss:** L(X, g(f(X))) = MSE = (1/n)Σ||X - X̂||²

### 2.4 Deep Clustering

Xie et al. (2016) propuseram o Deep Embedded Clustering (DEC), demonstrando melhora de até 20% no Silhouette Score. Min et al. (2018) revisaram sistematicamente 40 métodos de deep clustering.

### 2.5 RFM + Deep Learning

Hosseini & Mohammadzadeh (2021) aplicaram Autoencoders sobre atributos RFM, obtendo melhora média de 18% no Silhouette Score — resultado superado neste trabalho (+45%).

---

## 3. DATASET E PRÉ-PROCESSAMENTO

### 3.1 UCI Online Retail II

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

1. Remoção de registros sem `CustomerID`
2. Filtragem de devoluções (`Quantity < 0`)
3. Filtragem de preços inválidos (`Price ≤ 0`)
4. Remoção de cancelamentos (faturas com prefixo "C")
5. Cálculo do valor total: `TotalValue = Quantity × Price`

### 3.3 Cálculo RFM

- **Recência:** `(DataReferência - DataÚltimaCompra).days`
- **Frequência:** contagem de faturas únicas por cliente
- **Monetário:** soma de `TotalValue` por cliente

### 3.4 Normalização

1. **Log1p** em Frequência e Monetário para reduzir assimetria
2. **MinMaxScaler** para normalização em [0, 1]
3. **Inversão da Recência:** `R_norm = 1 - R_norm`

---

## 4. METODOLOGIA

### 4.1 Pipeline Proposto

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
  KMeans (K=5 via Elbow Method)
        ↓
  5 Segmentos + Métricas + Dashboard
```

### 4.2 Arquitetura do Autoencoder

**Encoder:**
- Entrada: 3 neurônios (R, F, M)
- Dense(64, ReLU) + BatchNormalization + Dropout(0.2)
- Dense(32, ReLU) + BatchNormalization
- Camada latente: Dense(2, Linear)

**Decoder:**
- Entrada latente: 2 neurônios
- Dense(32, ReLU) + BatchNormalization
- Dense(64, ReLU) + BatchNormalization
- Saída: Dense(3, Sigmoid)

**Otimizador:** Adam (lr=0.001) | **Loss:** MSE

### 4.3 Treinamento

- Épocas máximas: 150
- Batch size: 64
- Validação: 15%
- Early Stopping: patience=15
- ReduceLROnPlateau: factor=0.5, patience=7

---

## 5. EXPERIMENTOS E RESULTADOS

### 5.1 Comparação de Métodos

**Tabela 3 — Comparação de métricas**

| Método | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|
| KMeans (RFM bruto) | 0.2748 | 0.8590 |
| **KMeans + Autoencoder** | **0.3992** | **0.7382** |
| **Melhora** | **+45%** | **-14%** |

### 5.2 Perfil dos Clusters

**Tabela 4 — Perfil médio por cluster**

| Cluster | N clientes | Recência (dias) | Frequência | Monetário (£) | Perfil |
|---|---|---|---|---|---|
| 0 | 522 | 240.9 | 1.38 | 372.8 | Inativo |
| 1 | 1.205 | 128.1 | 2.59 | 505.4 | Novo |
| 2 | 291 | 20.3 | 29.5 | 8.509.3 | VIP |
| 3 | 600 | 38.5 | 11.0 | 8.539.1 | Leal |
| 4 | 221 | 343.7 | 1.97 | 573.9 | Inativo |

---

## 6. DASHBOARD INTERATIVO

O dashboard foi desenvolvido em **React + Vite**, utilizando Canvas API nativa para renderização dos gráficos, sem dependências externas de visualização. O KMeans foi implementado em JavaScript puro.

**Funcionalidades:**
- Importação de CSV com dados de compras
- Segmentação automática em tempo real
- 4 abas: Visão Geral, Dispersão RFM, Clientes, Top Clientes
- Exportação dos resultados segmentados em CSV
- Score de churn estimado por cliente
- Sidebar retrátil e design responsivo

**Acesso:** https://rfm-client-segmentation-dashboard.vercel.app/
**Código:** https://github.com/flaviocaos/rfm-pro-dashboard

---

## 7. CONCLUSÃO

### 7.1 Contribuições

1. Pipeline completo e reproduzível de segmentação RFM com Deep Learning
2. Melhora de +45% no Silhouette Score vs KMeans clássico
3. Dashboard interativo open source para uso pela comunidade
4. Código disponibilizado publicamente no GitHub

### 7.2 Limitações

- Dataset específico do varejo britânico
- Dimensão latente 2D escolhida para visualização direta
- Score de churn é estimativa heurística

### 7.3 Trabalhos Futuros

- Explorar Variational Autoencoders (VAE)
- Avaliar DBSCAN e Gaussian Mixture Models
- Incorporar atributos demográficos além do RFM
- Desenvolver modelo preditivo de churn supervisionado
- Avaliar estabilidade temporal dos clusters

---

## REFERÊNCIAS

CHEN, D. et al. **UCI Online Retail II Data Set**. UCI Machine Learning Repository, 2012.

DAVIES, D. L.; BOULDIN, D. W. A cluster separation measure. **IEEE Transactions on Pattern Analysis and Machine Intelligence**, v. 1, n. 2, p. 224-227, 1979.

HINTON, G. E.; SALAKHUTDINOV, R. R. Reducing the dimensionality of data with neural networks. **Science**, v. 313, n. 5786, p. 504-507, 2006.

HOSSEINI, S.; MOHAMMADZADEH, M. Customer segmentation using RFM model and deep learning. **Journal of Business Analytics**, 2021.

HUGHES, A. M. **Strategic Database Marketing**. Chicago: Probus Publishing, 1994.

MACQUEEN, J. Some methods for classification and analysis of multivariate observations. **Proceedings of the 5th Berkeley Symposium**, v. 1, p. 281-297, 1967.

MIN, E. et al. A survey of clustering with deep learning. **IEEE Access**, v. 6, p. 39501-39514, 2018.

ROUSSEEUW, P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. **Journal of Computational and Applied Mathematics**, v. 20, p. 53-65, 1987.

XIE, J.; GIRSHICK, R.; FARHADI, A. Unsupervised deep embedding for clustering analysis. **ICML**, 2016.
