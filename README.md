# 📊 RFM Pro Dashboard

> Plataforma de segmentação de clientes com KMeans, Autoencoder (Deep Learning) e dashboard interativo em React.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![React](https://img.shields.io/badge/React-18-61dafb.svg)
![Python](https://img.shields.io/badge/Python-3.11-3776ab.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00.svg)
![Vite](https://img.shields.io/badge/Vite-5-646cff.svg)

---

## 🌐 Demo ao vivo

**Dashboard:** [rfm-client-segmentation-dashboard.vercel.app](https://rfm-client-segmentation-dashboard.vercel.app)

---

## 📌 Sobre o projeto

O **RFM Pro** é uma solução completa para análise e segmentação de clientes baseada na metodologia **RFM (Recência, Frequência e Monetário)**, combinada com um **Autoencoder profundo** para aprendizado de representações latentes não lineares.

O projeto é composto por dois módulos principais:

- **Dashboard React** — interface interativa para visualização e análise dos segmentos
- **Pipeline ML/DL** — pipeline Python com Autoencoder + KMeans para segmentação avançada

---

## 🧠 Metodologia

```
Dados Brutos (CSV / UCI Online Retail II)
        ↓
  Pré-processamento
        ↓
  Cálculo RFM por cliente
  (Recência · Frequência · Monetário)
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
  5 Segmentos + Métricas + Dashboard
```

### Segmentos identificados

| Segmento | Perfil | Risco Churn |
|---|---|---|
| 👑 VIP | Alta freq., alto valor, compra recente | 5% |
| 💎 Leal | Frequência média-alta, bom valor | 15% |
| ⚠️ Em Risco | Foram bons clientes, recência caindo | 55% |
| 💤 Inativo | Baixa freq., baixo valor, sem compras recentes | 80% |
| 🌱 Novo | Poucas compras, potencial a explorar | 30% |

---

## 🏗️ Estrutura do Repositório

```
📦 rfm-pro-dashboard/
│
├── 📁 dashboard/                  ← React + Vite
│   ├── src/
│   │   └── App.jsx                ← Componente principal
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── 📁 ml-pipeline/                ← Python / Google Colab
│   ├── tcc_autoencoder.py         ← Pipeline completo
│   └── tcc_notebook.py            ← Notebook acadêmico
│
├── 📁 docs/                       ← Documentação acadêmica
│   └── tcc_documento.md           ← TCC completo
│
├── 📁 data/                       ← Dados de exemplo
│   └── base_clientes_teste.csv    ← CSV de teste (50 clientes)
│
└── 📄 README.md
```

---

## 🚀 Como usar

### Dashboard (React)

```bash
# Clone o repositório
git clone https://github.com/flaviocaos/rfm-pro-dashboard.git
cd rfm-pro-dashboard/dashboard

# Instale as dependências
npm install

# Rode localmente
npm run dev

# Acesse em http://localhost:5173
```

### Pipeline ML/DL (Python)

**Opção 1 — Google Colab (recomendado):**
1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Faça upload do arquivo `ml-pipeline/tcc_autoencoder.py`
3. Execute todas as células

**Opção 2 — Local (Python 3.11):**
```bash
cd ml-pipeline

# Instale as dependências
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn openpyxl

# Baixe o dataset UCI Online Retail II
# https://archive.ics.uci.edu/dataset/502/online+retail+ii

# Execute o pipeline
python tcc_autoencoder.py
```

---

## 📊 Dashboard — Funcionalidades

| Aba | Descrição |
|---|---|
| 👁️ Visão Geral | KPIs, cards de segmento, Top 5 e risco de churn |
| ⬡ Dispersão RFM | Gráfico interativo com hover por cliente |
| ≡ Clientes | Tabela completa com ordenação e exportação |
| ★ Top Clientes | Pódio + Top 10 com score de churn individual |

**Recursos:**
- ✅ Importar CSV com dados reais
- ✅ Exportar segmentos em CSV
- ✅ KMeans implementado em JavaScript puro
- ✅ Score de churn estimado por cliente
- ✅ Sidebar retrátil
- ✅ Responsivo

---

## 🧪 Pipeline ML — Resultados

Resultados obtidos sobre o dataset **UCI Online Retail II** (4.338 clientes):

| Método | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|
| KMeans (RFM bruto) | 0.2748 | 0.8590 |
| **KMeans + Autoencoder** | **0.3992** | **0.7382** |
| Melhora | **+45%** | **-14%** |

---

## 🗂️ Formato do CSV

O dashboard aceita arquivos CSV com as seguintes colunas:

```csv
customer_id,purchase_date,purchase_value
C0001,2024-12-10,1250.00
C0001,2024-10-05,980.50
C0002,2024-11-28,320.00
```

> Um arquivo de teste com 50 clientes está disponível em `data/base_clientes_teste.csv`

---

## 🛠️ Tecnologias

### Dashboard
| Tecnologia | Uso |
|---|---|
| React 18 | Framework UI |
| Vite 5 | Build tool |
| Canvas API | Gráficos nativos sem dependências |
| JavaScript puro | KMeans implementado do zero |

### Pipeline ML/DL
| Tecnologia | Uso |
|---|---|
| Python 3.11 | Linguagem principal |
| TensorFlow/Keras | Autoencoder |
| Scikit-learn | KMeans, métricas, normalização |
| Pandas / NumPy | Manipulação de dados |
| Matplotlib / Seaborn | Visualizações científicas |

---

## 📚 Referências Acadêmicas

- Hughes, A.M. (1994). *Strategic Database Marketing*. Probus Publishing.
- Hinton, G.E. & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786).
- Xie, J., Girshick, R. & Farhadi, A. (2016). Unsupervised deep embedding for clustering analysis. *ICML*.
- Min, E. et al. (2018). A survey of clustering with deep learning. *IEEE Access*.
- Chen, D. et al. (2012). UCI Online Retail II Dataset. UCI Machine Learning Repository.

---

## 👤 Autor

**Flávio Antônio Oliveira da Silva**

[![GitHub](https://img.shields.io/badge/GitHub-flaviocaos-181717?logo=github)](https://github.com/flaviocaos)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Flávio-0077b5?logo=linkedin)](https://linkedin.com/in/flaviocaos)

---

## 📜 Licença

Este projeto está licenciado sob a licença **MIT** — veja o arquivo [LICENSE](LICENSE) para detalhes.

---

<div align="center">
  <sub>Desenvolvido como parte do TCC da Pós-Graduação em Deep Learning — UFPE</sub>
</div>
