 # 📊 RFMPro Analytics

> Plataforma de segmentação de clientes com Deep Learning rodando 100% no navegador — zero backend, zero instalação.

🔗 **[Acesse ao vivo → rfm-pro-dashboard.vercel.app](https://rfm-pro-dashboard.vercel.app)**

![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![TensorFlow](https://img.shields.io/badge/TensorFlow.js-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white)

---

## 🚀 O que é

RFMPro Analytics é uma plataforma web completa para análise e segmentação de clientes baseada na metodologia **RFM (Recência, Frequência e Monetário)**, com modelo de **Deep Learning exportado para TensorFlow.js** rodando diretamente no navegador do usuário — sem servidor, sem instalação, sem custo.

---

## 🧠 Deep Learning no Navegador

O modelo foi treinado no **Google Colab** com dados reais do **Online Retail II (UCI Machine Learning Repository)** — 4.338 clientes reais. Após o treinamento, foi exportado para **TensorFlow.js** e carregado diretamente no navegador, onde a predição acontece na GPU/CPU do dispositivo via **WebGL**.

Google Colab → Treina modelo → Exporta TF.js
↓
Navegador carrega model.json
↓
Predição 100% local via WebGL

**Métricas do modelo:**

| Métrica | Valor |
|---------|-------|
| Silhouette Score | 0.84 |
| Davies-Bouldin Index | 0.07 |
| Clientes segmentados | 4.338 |
| Clusters | 5 |

---

## 📊 Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| 🎯 Segmentação RFM | VIP, Leal, Em Risco, Inativo e Novo |
| 📈 Histogramas RFM | Distribuição de Recência, Frequência e Monetário |
| 🔵 Método do Cotovelo | K ideal de clusters automaticamente |
| 🌐 Espaço Latente 2D | Projeção visual dos clusters |
| 🔥 Matriz Heatmap | Cruzamento de scores R×F |
| 🔵 Dispersão Interativa | Hover com detalhes do cliente |
| ⚠️ Risco de Churn | Probabilidade estimada por segmento |
| 🤖 Preditor com IA | TensorFlow.js em tempo real |
| 📋 Estatísticas | Mín, Máx, Média, Mediana, Desvio Padrão |
| ↓ Exportação CSV | Dados segmentados para download |

---

## 📁 Formato do CSV

```csv
customer_id,purchase_date,purchase_value
CLI001,2024-12-20,8500.00
CLI001,2024-11-15,6200.00
CLI002,2024-12-28,12000.00
CLI003,2024-09-01,3200.00
```

---

## 🏢 Áreas de Aplicação

- 🛒 **Varejo e e-commerce** — identificar clientes VIP e em risco de churn
- 📣 **Marketing** — personalizar campanhas por perfil comportamental
- 🤝 **CRM** — priorizar ações de retenção com base em dados reais
- 💰 **Financeiro** — identificar padrões de consumo
- 🏥 **Saúde** — segmentar pacientes por frequência de atendimento

---

## ⚙️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| React + Vite | Frontend |
| TensorFlow.js | Deep Learning no navegador |
| Python + Scikit-learn + Keras | Treinamento no Google Colab |
| Canvas API | Gráficos customizados |
| Vercel | Deploy gratuito |

---

## 👨‍💻 Autor

**Flávio Antonio Oliveira da Silva**  
Engenheiro Cartógrafo | Data Science | Machine Learning

---

## 🔬 Próximo Projeto

Este projeto é a base para o **GeoRFM Analytics** — plataforma de segmentação geoespacial com Deep Learning para:

- 🏙️ Análise Urbana
- 🌿 Monitoramento Ambiental  
- 🚜 Agricultura de Precisão
- 🏥 Saúde Pública Espacial
- 🛰️ Sensoriamento Remoto

---

## 📄 Licença

MIT License — livre para usar, modificar e distribuir.
