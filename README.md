📊 RFMPro Analytics

Plataforma de segmentação de clientes com Deep Learning rodando 100% no navegador — zero backend, zero instalação.

🔗 Acesse ao vivo

🚀 O que é
RFMPro Analytics é uma plataforma web completa para análise e segmentação de clientes baseada na metodologia RFM (Recência, Frequência e Monetário), com modelo de Deep Learning exportado para TensorFlow.js e rodando diretamente no navegador do usuário.

🧠 Como funciona o Deep Learning
O modelo foi treinado no Google Colab com dados reais do dataset Online Retail II (UCI Machine Learning Repository) — mais de 4.300 clientes reais. Após o treinamento, foi exportado para TensorFlow.js e carregado diretamente no navegador, onde a predição acontece na GPU/CPU do dispositivo via WebGL — zero latência, zero custo de servidor.

📊 Funcionalidades

✅ Segmentação automática em 5 perfis: VIP, Leal, Em Risco, Inativo e Novo
✅ 7 gráficos analíticos gerados automaticamente com qualquer base de dados
✅ Método do Cotovelo (Elbow) — k ideal de clusters
✅ Espaço Latente 2D — projeção visual dos clusters
✅ Matriz RFM Heatmap — cruzamento de scores R×F
✅ Dispersão RFM Interativa — hover com detalhes do cliente
✅ Perfis por Segmento — comparação de médias normalizadas
✅ Risco de Churn — probabilidade estimada por segmento
✅ Preditor em tempo real com TensorFlow.js
✅ Estatísticas descritivas completas
✅ Exportação CSV dos dados segmentados
✅ Upload de qualquer base no formato CSV


📁 Formato do CSV
csvcustomer_id,purchase_date,purchase_value
CLI001,2024-12-20,8500.00
CLI001,2024-11-15,6200.00
CLI002,2024-12-28,12000.00

🏢 Áreas de Aplicação

Varejo e e-commerce — identificar clientes VIP e em risco de churn
Marketing — personalizar campanhas por perfil comportamental
CRM — priorizar ações de retenção com base em dados reais
Financeiro — identificar padrões de consumo
Saúde — segmentar pacientes por frequência de atendimento
Qualquer negócio com histórico de transações de clientes


📈 Resultados que pode trazer

Redução de churn com ações direcionadas ao segmento certo
Aumento do ticket médio em clientes Leal e Novo
ROI maior em campanhas ao focar nos clientes VIP
Decisões baseadas em dados, não em intuição


⚙️ Stack Tecnológica
TecnologiaUsoReact + ViteFrontendTensorFlow.jsDeep Learning no navegadorPython + Scikit-learn + KerasTreinamento no ColabCanvas APIGráficos customizadosVercelDeploy gratuito

🧪 Métricas do Modelo
MétricaValorSilhouette Score0.84Davies-Bouldin Index0.07Clientes segmentados4.338Clusters5

👨‍💻 Autor
Flávio Antonio Oliveira da Silva
Engenheiro Cartógrafo | Data Science | Machine Learning

📄 Licença
MIT License — livre para usar, modificar e distribuir.


🔬 Este projeto é a base para o GeoRFM Analytics — plataforma de segmentação geoespacial com Deep Learning para Análise Urbana, Monitoramento Ambiental, Agricultura de Precisão, Saúde Pública Espacial e Sensoriamento Remoto. Em breve!
