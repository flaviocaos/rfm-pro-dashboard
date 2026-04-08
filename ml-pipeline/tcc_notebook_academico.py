# ╔══════════════════════════════════════════════════════════════════╗
# ║  Para converter em .ipynb rode no terminal:                      ║
# ║  pip install jupytext                                            ║
# ║  jupytext --to notebook tcc_notebook.py                         ║
# ╚══════════════════════════════════════════════════════════════════╝

# %% [markdown]
"""
# Segmentação de Clientes com Autoencoders e KMeans
## Uma Abordagem de Deep Learning para Análise RFM

**Autor:** Flávio Antônio Oliveira da Silva
**Instituição:** Universidade Federal de Pernambuco — UFPE
**Curso:** Pós-Graduação em Deep Learning
**Data:** Abril de 2026

---

## Resumo

Este trabalho propõe um pipeline de segmentação de clientes baseado
na metodologia RFM (Recência, Frequência e Monetário) combinada com
uma arquitetura de Autoencoder profundo para aprendizado de
representações latentes não lineares. O modelo é avaliado sobre o
dataset público UCI Online Retail II, contendo mais de 1 milhão de
transações reais de e-commerce britânico.

**Resultados obtidos:**
- Silhouette Score: 0.3992 (+45% vs KMeans puro)
- Davies-Bouldin Index: 0.7382 (-14% vs KMeans puro)
- Clientes segmentados: 4.338
- Clusters: 5

**Palavras-chave:** Segmentação de Clientes, RFM, Autoencoder,
Deep Learning, KMeans, CRM.
"""

# %% [markdown]
"""
---
## 1. Introdução

A segmentação de clientes é uma das técnicas mais relevantes no
campo do marketing analítico e da gestão de relacionamento com o
cliente (CRM). A metodologia RFM — proposta por Hughes (1994) —
quantifica o comportamento de compra em três dimensões:

- **Recência (R):** dias desde a última compra
- **Frequência (F):** número de transações no período
- **Monetário (M):** valor total gasto pelo cliente

Este trabalho propõe o uso de um **Autoencoder profundo** para
aprender uma representação latente comprimida do espaço RFM,
preservando estruturas não lineares que o KMeans clássico não
consegue capturar (Xie et al., 2016).
"""

# %% [markdown]
"""
---
## 2. Configuração do Ambiente
"""

# %% Célula 1 — Instalação
# !pip install scikit-learn pandas numpy matplotlib seaborn openpyxl -q
# TensorFlow já vem instalado no Google Colab

# %% Célula 2 — Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.rcParams.update({"figure.dpi":120,"axes.spines.top":False,"axes.spines.right":False})
print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ Pandas:     {pd.__version__}")

# %% [markdown]
"""
---
## 3. Dataset e Pré-processamento

### 3.1 UCI Online Retail II
- **Fonte:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **Registros:** ~1.067.371 transações
- **Clientes únicos:** ~5.942
- **Período:** Dez/2009 — Dez/2011
"""

# %% Célula 3 — Upload (Google Colab)
from google.colab import files
print("📂 Faça o upload do arquivo online_retail_II.xlsx")
uploaded = files.upload()

# %% Célula 4 — Carregamento
df = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011", engine="openpyxl")
print(f"Shape original: {df.shape}")
print(f"Colunas: {df.columns.tolist()}")
df.head()

# %% Célula 5 — Limpeza e padronização
df = df.rename(columns={
    'Invoice':'invoice','StockCode':'stock_code','Description':'description',
    'Quantity':'quantity','InvoiceDate':'invoice_date','Price':'price',
    'Customer ID':'customer_id','Country':'country'
})
df = df.dropna(subset=["customer_id"])
df = df[df["quantity"] > 0]
df = df[df["price"] > 0]
df = df[~df["invoice"].astype(str).str.startswith("C")]
df["total_value"] = df["quantity"] * df["price"]
df["invoice_date"] = pd.to_datetime(df["invoice_date"])
df["customer_id"] = df["customer_id"].astype(int)

print(f"Shape após limpeza:  {df.shape}")
print(f"Clientes únicos:     {df['customer_id'].nunique():,}")
print(f"Receita total:       £{df['total_value'].sum():,.2f}")

# %% Célula 6 — Análise exploratória
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Análise Exploratória — Distribuições", fontsize=13, fontweight="bold")
for ax, col, color, title in zip(axes,
    ["quantity","price","total_value"],
    ["#7F77DD","#1D9E75","#378ADD"],
    ["Quantity","Price (£)","Total Value (£)"]):
    ax.hist(df[col].clip(upper=df[col].quantile(0.99)), bins=40, color=color, alpha=0.8)
    ax.set_title(title); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("eda_distribuicoes.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
"""
---
## 4. Cálculo dos Scores RFM
"""

# %% Célula 7 — RFM
ref_date = df["invoice_date"].max() + pd.Timedelta(days=1)
rfm = df.groupby("customer_id").agg(
    recency   = ("invoice_date", lambda x: (ref_date - x.max()).days),
    frequency = ("invoice",       "nunique"),
    monetary  = ("total_value",   "sum")
).reset_index()
print("Estatísticas descritivas RFM:")
rfm[["recency","frequency","monetary"]].describe().round(2)

# %% Célula 8 — Distribuição RFM
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Distribuição dos Atributos RFM", fontsize=13, fontweight="bold")
for ax, col, color, title in zip(axes,
    ["recency","frequency","monetary"],
    ["#7F77DD","#1D9E75","#378ADD"],
    ["Recência (dias)","Frequência (compras)","Monetário (£)"]):
    ax.hist(rfm[col].clip(upper=rfm[col].quantile(0.99)), bins=40, color=color, alpha=0.8)
    ax.set_title(title); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("rfm_distribuicao.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
"""
---
## 5. Normalização
"""

# %% Célula 9 — Normalização
rfm_norm = rfm.copy()
rfm_norm["monetary"]  = np.log1p(rfm_norm["monetary"])
rfm_norm["frequency"] = np.log1p(rfm_norm["frequency"])
scaler = MinMaxScaler()
features = ["recency","frequency","monetary"]
rfm_norm[features] = scaler.fit_transform(rfm_norm[features])
rfm_norm["recency"] = 1 - rfm_norm["recency"]
X = rfm_norm[features].values
print(f"Shape: {X.shape} | Min: {X.min():.4f} | Max: {X.max():.4f}")

# %% [markdown]
"""
---
## 6. Autoencoder (Deep Learning)

### Arquitetura
```
Encoder: 3 → Dense(64,ReLU) → BN → Dropout(0.2) → Dense(32,ReLU) → BN → Dense(2,Linear)
Decoder: 2 → Dense(32,ReLU) → BN → Dense(64,ReLU) → BN → Dense(3,Sigmoid)
```
**Referência:** Hinton & Salakhutdinov (2006)
"""

# %% Célula 10 — Construção
LATENT_DIM = 2

enc_in  = layers.Input(shape=(3,), name="encoder_input")
x       = layers.Dense(64, activation="relu")(enc_in)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.2)(x)
x       = layers.Dense(32, activation="relu")(x)
x       = layers.BatchNormalization()(x)
latent  = layers.Dense(LATENT_DIM, activation="linear", name="latent")(x)
encoder = Model(enc_in, latent, name="encoder")

dec_in  = layers.Input(shape=(LATENT_DIM,), name="decoder_input")
x       = layers.Dense(32, activation="relu")(dec_in)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(64, activation="relu")(x)
x       = layers.BatchNormalization()(x)
dec_out = layers.Dense(3, activation="sigmoid", name="decoder_output")(x)
decoder = Model(dec_in, dec_out, name="decoder")

ae_in  = layers.Input(shape=(3,))
ae_out = decoder(encoder(ae_in))
autoencoder = Model(ae_in, ae_out, name="autoencoder")
autoencoder.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
autoencoder.summary()

# %% Célula 11 — Treinamento
cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=15,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=7, min_lr=1e-6, verbose=1),
]
history = autoencoder.fit(
    X, X, epochs=150, batch_size=64,
    validation_split=0.15, callbacks=cbs, verbose=1
)

# %% Célula 12 — Curvas de loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Histórico de Treinamento — Autoencoder", fontsize=13, fontweight="bold")
axes[0].plot(history.history["loss"],     label="Train", color="#7F77DD", lw=2)
axes[0].plot(history.history["val_loss"], label="Val",   color="#E24B4A", lw=2)
axes[0].set_title("Loss (MSE)"); axes[0].set_xlabel("Época")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history.history["mae"],     label="Train MAE", color="#1D9E75", lw=2)
axes[1].plot(history.history["val_mae"], label="Val MAE",   color="#EF9F27", lw=2)
axes[1].set_title("MAE"); axes[1].set_xlabel("Época")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Melhor val_loss: {min(history.history['val_loss']):.6f}")

# %% [markdown]
"""
---
## 7. Espaço Latente e Elbow Method
"""

# %% Célula 13 — Espaço latente
X_latent = encoder.predict(X, verbose=0)
print(f"Espaço latente: {X_latent.shape}")

# %% Célula 14 — Elbow Method
resultados_k = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_latent)
    resultados_k.append({
        "k":k,"inertia":km.inertia_,
        "silhouette":silhouette_score(X_latent,labels),
        "davies_bouldin":davies_bouldin_score(X_latent,labels)
    })
df_k = pd.DataFrame(resultados_k)
print(df_k.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Elbow Method — Seleção do K Ideal", fontsize=13, fontweight="bold")
axes[0].plot(df_k["k"],df_k["inertia"],       "bo-",lw=2); axes[0].set_title("Inertia")
axes[1].plot(df_k["k"],df_k["silhouette"],    "go-",lw=2); axes[1].set_title("Silhouette ↑")
axes[2].plot(df_k["k"],df_k["davies_bouldin"],"ro-",lw=2); axes[2].set_title("Davies-Bouldin ↓")
for ax in axes: ax.set_xlabel("K"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
"""
---
## 8. Clustering no Espaço Latente
"""

# %% Célula 15 — KMeans
K_IDEAL = 5
km_final = KMeans(n_clusters=K_IDEAL, random_state=SEED, n_init=20)
rfm["cluster"] = km_final.fit_predict(X_latent)
sil = silhouette_score(X_latent, rfm["cluster"])
dbi = davies_bouldin_score(X_latent, rfm["cluster"])
print(f"✅ Silhouette Score:     {sil:.4f}")
print(f"✅ Davies-Bouldin Index: {dbi:.4f}")

# %% Célula 16 — Visualização espaço latente
colors = ["#7F77DD","#1D9E75","#EF9F27","#E24B4A","#378ADD"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Espaço Latente do Autoencoder", fontsize=13, fontweight="bold")
for i in range(K_IDEAL):
    mask = rfm["cluster"] == i
    axes[0].scatter(X_latent[mask,0],X_latent[mask,1],
                   c=colors[i],label=f"Cluster {i}",alpha=0.6,s=15)
axes[0].set_title("Espaço Latente 2D"); axes[0].legend(); axes[0].grid(alpha=0.2)
X_tsne = TSNE(n_components=2,random_state=SEED,perplexity=30).fit_transform(X_latent)
for i in range(K_IDEAL):
    mask = rfm["cluster"] == i
    axes[1].scatter(X_tsne[mask,0],X_tsne[mask,1],
                   c=colors[i],label=f"Cluster {i}",alpha=0.6,s=15)
axes[1].set_title("t-SNE"); axes[1].legend(); axes[1].grid(alpha=0.2)
plt.tight_layout()
plt.savefig("latent_space.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
"""
---
## 9. Resultados
"""

# %% Célula 17 — Perfil dos clusters
profile = rfm.groupby("cluster").agg(
    n_clientes = ("customer_id","count"),
    recencia   = ("recency",    "mean"),
    frequencia = ("frequency",  "mean"),
    monetario  = ("monetary",   "mean")
).round(2)
print("Perfil médio por cluster:")
profile

# %% Célula 18 — Heatmap
fig, ax = plt.subplots(figsize=(8,4))
profile_norm = (profile[["recencia","frequencia","monetario"]] -
                profile[["recencia","frequencia","monetario"]].min()) / \
               (profile[["recencia","frequencia","monetario"]].max() -
                profile[["recencia","frequencia","monetario"]].min())
sns.heatmap(profile_norm, annot=profile[["recencia","frequencia","monetario"]],
            fmt=".1f", cmap="RdYlGn", ax=ax, linewidths=0.5)
ax.set_title("Heatmap de Perfil dos Clusters (normalizado)", fontweight="bold")
plt.tight_layout()
plt.savefig("heatmap_perfil.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Célula 19 — Comparação de métodos
km_raw  = KMeans(n_clusters=K_IDEAL, random_state=SEED, n_init=20).fit(X)
sil_raw = silhouette_score(X, km_raw.labels_)
dbi_raw = davies_bouldin_score(X, km_raw.labels_)
df_comp = pd.DataFrame({
    "KMeans (RFM bruto)":   {"Silhouette ↑":sil_raw,"Davies-Bouldin ↓":dbi_raw},
    "KMeans + Autoencoder": {"Silhouette ↑":sil,    "Davies-Bouldin ↓":dbi},
}).T.round(4)
print("="*50)
print("COMPARAÇÃO DE MÉTODOS")
print("="*50)
print(df_comp.to_string())
print(f"\nMelhora Silhouette:      +{((sil-sil_raw)/sil_raw*100):.1f}%")
print(f"Melhora Davies-Bouldin:  {((dbi-dbi_raw)/dbi_raw*100):.1f}%")
df_comp.to_csv("comparacao_metodos.csv")

# %% Célula 20 — Exportar resultados
rfm["latent_1"] = X_latent[:, 0]
rfm["latent_2"] = X_latent[:, 1]
rfm.to_csv("rfm_clusters_final.csv", index=False)

print("\n✅ Arquivos gerados:")
print("   📊 eda_distribuicoes.png")
print("   📊 rfm_distribuicao.png")
print("   📈 training_history.png")
print("   🗺️  latent_space.png")
print("   📋 heatmap_perfil.png")
print("   📐 elbow_method.png")
print("   ⚖️  comparacao_metodos.csv")
print("   💾 rfm_clusters_final.csv")
print(f"\n✅ {len(rfm):,} clientes segmentados com sucesso!")

# %% [markdown]
"""
---
## 10. Conclusão

Este trabalho demonstrou que a combinação de Autoencoders profundos
com o algoritmo KMeans produz clusters de clientes mais coesos e
separados em comparação com a abordagem clássica.

### Resultados finais

| Método | Silhouette ↑ | Davies-Bouldin ↓ |
|---|---|---|
| KMeans (RFM bruto) | 0.2748 | 0.8590 |
| **KMeans + Autoencoder** | **0.3992** | **0.7382** |
| **Melhora** | **+45%** | **-14%** |

### Referências
- Hughes, A.M. (1994). Strategic Database Marketing.
- Hinton & Salakhutdinov (2006). Science, 313(5786).
- Xie et al. (2016). ICML — Deep Embedded Clustering.
- Min et al. (2018). IEEE Access — Deep Clustering Survey.
- Chen et al. (2012). UCI Online Retail Dataset.
"""
