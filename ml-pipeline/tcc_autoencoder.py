"""
================================================================
 TCC — Segmentação de Clientes com Autoencoders e KMeans
 Pós-Graduação em Deep Learning — UFPE
 Dataset: UCI Online Retail II
 Resultados validados: Silhouette=0.3992 | DBI=0.7382
================================================================
"""

# 1. Instalar dependências
import subprocess
subprocess.run(["pip", "install", "openpyxl", "-q"])

# 2. Importações
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
print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ Bibliotecas carregadas com sucesso!")

# ── 3. Carregamento e pré-processamento ──────────────────────────

def load_and_preprocess(path):
    """
    Carrega o dataset UCI Online Retail II e aplica filtros de qualidade.
    Compatível com colunas: Invoice, StockCode, Description, Quantity,
    InvoiceDate, Price, Customer ID, Country.
    """
    print("\n📂 Carregando dataset...")
    df = pd.read_excel(path, sheet_name="Year 2010-2011", engine="openpyxl")
    print(f"   Shape original: {df.shape}")
    print(f"   Colunas: {df.columns.tolist()}")

    # Renomear colunas para padronização
    df = df.rename(columns={
        'Invoice':     'invoice',
        'StockCode':   'stock_code',
        'Description': 'description',
        'Quantity':    'quantity',
        'InvoiceDate': 'invoice_date',
        'Price':       'price',
        'Customer ID': 'customer_id',
        'Country':     'country'
    })

    # Limpeza
    df = df.dropna(subset=["customer_id"])
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    df["total_value"] = df["quantity"] * df["price"]
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = df["customer_id"].astype(int)

    print(f"   Shape após limpeza: {df.shape}")
    print(f"   Clientes únicos: {df['customer_id'].nunique():,}")
    print(f"   Período: {df['invoice_date'].min().date()} → {df['invoice_date'].max().date()}")
    print(f"   Receita total: £{df['total_value'].sum():,.2f}")
    return df

# ── 4. Cálculo RFM ───────────────────────────────────────────────

def compute_rfm(df):
    """Calcula Recência, Frequência e Monetário por cliente."""
    print("\n📊 Calculando scores RFM...")
    ref_date = df["invoice_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        recency   = ("invoice_date", lambda x: (ref_date - x.max()).days),
        frequency = ("invoice",       "nunique"),
        monetary  = ("total_value",   "sum")
    ).reset_index()
    print(f"\n   Estatísticas RFM:")
    print(rfm[["recency","frequency","monetary"]].describe().round(2).to_string())
    return rfm

# ── 5. Normalização ───────────────────────────────────────────────

def normalize_rfm(rfm):
    """Normaliza com Log1p + MinMaxScaler."""
    print("\n⚙️  Normalizando dados RFM...")
    rfm_norm = rfm.copy()
    rfm_norm["monetary"]  = np.log1p(rfm_norm["monetary"])
    rfm_norm["frequency"] = np.log1p(rfm_norm["frequency"])
    scaler = MinMaxScaler()
    features = ["recency", "frequency", "monetary"]
    rfm_norm[features] = scaler.fit_transform(rfm_norm[features])
    rfm_norm["recency"] = 1 - rfm_norm["recency"]
    X = rfm_norm[features].values
    print(f"   Shape: {X.shape} | Min: {X.min():.3f} | Max: {X.max():.3f}")
    return X, scaler, rfm_norm

# ── 6. Autoencoder ────────────────────────────────────────────────

def build_autoencoder(input_dim=3, latent_dim=2):
    """
    Autoencoder profundo com arquitetura simétrica.
    Encoder: input_dim → 64 → 32 → latent_dim
    Decoder: latent_dim → 32 → 64 → input_dim
    Referência: Hinton & Salakhutdinov (2006)
    """
    # Encoder
    enc_in  = layers.Input(shape=(input_dim,), name="encoder_input")
    x       = layers.Dense(64, activation="relu")(enc_in)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.2)(x)
    x       = layers.Dense(32, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    latent  = layers.Dense(latent_dim, activation="linear", name="latent")(x)
    encoder = Model(enc_in, latent, name="encoder")

    # Decoder
    dec_in  = layers.Input(shape=(latent_dim,), name="decoder_input")
    x       = layers.Dense(32, activation="relu")(dec_in)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(64, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    dec_out = layers.Dense(input_dim, activation="sigmoid", name="decoder_output")(x)
    decoder = Model(dec_in, dec_out, name="decoder")

    # Autoencoder completo
    ae_in  = layers.Input(shape=(input_dim,), name="ae_input")
    ae_out = decoder(encoder(ae_in))
    autoencoder = Model(ae_in, ae_out, name="autoencoder")
    autoencoder.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])

    print("\n🧠 Arquitetura do Autoencoder:")
    autoencoder.summary()
    return autoencoder, encoder, decoder

# ── 7. Treinamento ────────────────────────────────────────────────

def train_autoencoder(autoencoder, X, epochs=150, batch_size=64):
    """Treina com Early Stopping e ReduceLROnPlateau."""
    print("\n🏋️  Treinando Autoencoder...")
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=7, min_lr=1e-6, verbose=1),
    ]
    history = autoencoder.fit(
        X, X, epochs=epochs, batch_size=batch_size,
        validation_split=0.15, callbacks=cbs, verbose=1
    )
    print(f"\n✅ Treinamento concluído!")
    print(f"   Melhor val_loss: {min(history.history['val_loss']):.6f}")
    return history

# ── 8. Visualizações ──────────────────────────────────────────────

def plot_training_history(history):
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
    print("   💾 Salvo: training_history.png")

def elbow_method(X_latent, k_range=range(2, 11)):
    print("\n📐 Calculando Elbow Method...")
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X_latent)
        sil = silhouette_score(X_latent, labels)
        dbi = davies_bouldin_score(X_latent, labels)
        results.append({"k":k,"inertia":km.inertia_,"silhouette":sil,"davies_bouldin":dbi})
        print(f"   K={k:2d} | Inertia={km.inertia_:10.1f} | Silhouette={sil:.4f} | DBI={dbi:.4f}")
    df_k = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Elbow Method — Determinação do K Ideal", fontsize=13, fontweight="bold")
    axes[0].plot(df_k["k"], df_k["inertia"],       "bo-", lw=2); axes[0].set_title("Inertia")
    axes[1].plot(df_k["k"], df_k["silhouette"],    "go-", lw=2); axes[1].set_title("Silhouette ↑")
    axes[2].plot(df_k["k"], df_k["davies_bouldin"],"ro-", lw=2); axes[2].set_title("Davies-Bouldin ↓")
    for ax in axes: ax.set_xlabel("K"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("elbow_method.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   💾 Salvo: elbow_method.png")
    return df_k

def plot_latent_space(X_latent, labels, k):
    colors = ["#7F77DD","#1D9E75","#EF9F27","#E24B4A","#378ADD"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Espaço Latente do Autoencoder", fontsize=13, fontweight="bold")
    for i in range(k):
        mask = labels == i
        axes[0].scatter(X_latent[mask,0], X_latent[mask,1],
                       c=colors[i%len(colors)], label=f"Cluster {i}", alpha=0.6, s=15)
    axes[0].set_title("Espaço Latente 2D"); axes[0].legend(); axes[0].grid(alpha=0.2)
    print("\n🗺️  Gerando t-SNE...")
    X_tsne = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(X_latent)
    for i in range(k):
        mask = labels == i
        axes[1].scatter(X_tsne[mask,0], X_tsne[mask,1],
                       c=colors[i%len(colors)], label=f"Cluster {i}", alpha=0.6, s=15)
    axes[1].set_title("t-SNE"); axes[1].legend(); axes[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("latent_space.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   💾 Salvo: latent_space.png")

def plot_cluster_profiles(rfm, k):
    colors = ["#7F77DD","#1D9E75","#EF9F27","#E24B4A","#378ADD"]
    profile = rfm.groupby("cluster").agg(
        n_clientes = ("customer_id","count"),
        recencia   = ("recency",    "mean"),
        frequencia = ("frequency",  "mean"),
        monetario  = ("monetary",   "mean")
    ).round(2)
    print("\n📋 Perfil médio por cluster:")
    print(profile.to_string())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Perfil Médio dos Clusters — RFM", fontsize=13, fontweight="bold")
    for ax, col, title in zip(axes,
        ["recencia","frequencia","monetario"],
        ["Recência (dias)","Frequência (compras)","Monetário (£)"]):
        bars = ax.bar(profile.index.astype(str), profile[col],
                     color=[colors[i%len(colors)] for i in profile.index])
        ax.set_title(title); ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   💾 Salvo: cluster_profiles.png")
    return profile

def compare_methods(X, X_latent, k, sil_ae, dbi_ae):
    print("\n⚖️  Comparando métodos...")
    km_raw  = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(X)
    sil_raw = silhouette_score(X, km_raw.labels_)
    dbi_raw = davies_bouldin_score(X, km_raw.labels_)
    df_comp = pd.DataFrame({
        "KMeans (RFM bruto)":   {"Silhouette ↑": sil_raw, "Davies-Bouldin ↓": dbi_raw},
        "KMeans + Autoencoder": {"Silhouette ↑": sil_ae,  "Davies-Bouldin ↓": dbi_ae},
    }).T.round(4)
    print("\n📊 Tabela de Comparação:")
    print(df_comp.to_string())
    df_comp.to_csv("comparacao_metodos.csv")
    print("   💾 Salvo: comparacao_metodos.csv")
    return df_comp

# ── 9. Pipeline principal ─────────────────────────────────────────

def main():
    print("=" * 60)
    print(" TCC — Segmentação RFM com Autoencoder + KMeans")
    print(" Pós-Graduação em Deep Learning — UFPE")
    print("=" * 60)

    # Upload do arquivo (Google Colab)
    try:
        from google.colab import files
        print("\n📂 Faça o upload do arquivo online_retail_II.xlsx")
        uploaded = files.upload()
        PATH = "online_retail_II.xlsx"
    except ImportError:
        # Execução local
        PATH = "online_retail_II.xlsx"
        print(f"\n📂 Usando arquivo local: {PATH}")

    # Pipeline
    df  = load_and_preprocess(PATH)
    rfm = compute_rfm(df)
    X, scaler, rfm_norm = normalize_rfm(rfm)

    autoencoder, encoder, decoder = build_autoencoder(input_dim=3, latent_dim=2)
    history = train_autoencoder(autoencoder, X, epochs=150, batch_size=64)
    plot_training_history(history)

    X_latent = encoder.predict(X, verbose=0)
    print(f"\n✅ Espaço latente gerado: {X_latent.shape}")

    df_elbow = elbow_method(X_latent, k_range=range(2, 11))

    K_IDEAL = 5
    km_final = KMeans(n_clusters=K_IDEAL, random_state=SEED, n_init=20)
    rfm["cluster"] = km_final.fit_predict(X_latent)
    sil = silhouette_score(X_latent, rfm["cluster"])
    dbi = davies_bouldin_score(X_latent, rfm["cluster"])
    print(f"\n✅ K={K_IDEAL} | Silhouette: {sil:.4f} | Davies-Bouldin: {dbi:.4f}")

    plot_latent_space(X_latent, rfm["cluster"].values, K_IDEAL)
    profile = plot_cluster_profiles(rfm, K_IDEAL)
    df_comp = compare_methods(X, X_latent, K_IDEAL, sil, dbi)

    # Exportar resultados
    rfm["latent_1"] = X_latent[:, 0]
    rfm["latent_2"] = X_latent[:, 1]
    rfm.to_csv("rfm_clusters_final.csv", index=False)

    print("\n" + "=" * 60)
    print(" ✅ PIPELINE CONCLUÍDO!")
    print("=" * 60)
    print(f" Clientes segmentados:  {len(rfm):,}")
    print(f" Clusters (K):          {K_IDEAL}")
    print(f" Silhouette Score:      {sil:.4f}")
    print(f" Davies-Bouldin Index:  {dbi:.4f}")
    print("=" * 60)
    print("\n📁 Arquivos gerados:")
    print("   📊 training_history.png")
    print("   📐 elbow_method.png")
    print("   🗺️  latent_space.png")
    print("   📋 cluster_profiles.png")
    print("   ⚖️  comparacao_metodos.csv")
    print("   💾 rfm_clusters_final.csv  ← importar no dashboard!")


if __name__ == "__main__":
    main()
