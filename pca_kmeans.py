import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CONFIGURAÇÃO ---
nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
N_CLUSTERS = 5 # O k do K-Means, como sugerido (4 ou 5)
N_COMPONENTS_PCA = 15 # Quantos componentes do PCA usaremos para o novo modelo

# --- 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
try:
    print("Carregando e preparando os dados...")
    df = pd.read_csv(nome_arquivo_entrada, sep=';')

    # a) Preparação do alvo (y) e features (X)
    df = df[df['ano_in_grad'] >= 1990].copy()
    df['tempo_permanencia'] = df['nu_ano_enade'] - df['ano_in_grad']
    df = df[(df['tempo_permanencia'] >= 2) & (df['tempo_permanencia'] <= 15)]

    y = df['tempo_permanencia']
    X = df.drop(columns=['tempo_permanencia', 'nu_ano_enade', 'ano_in_grad', 'nt_ger', 'nt_fg'])

    # b) Converter categóricas e garantir que tudo é numérico
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"Dataset preparado com {X.shape[0]} amostras e {X.shape[1]} features.\n")

    # --- 3. PADRONIZAÇÃO DOS DADOS (CRUCIAL PARA K-MEANS E PCA) ---
    print("Padronizando a escala das features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # --- 4. K-MEANS CLUSTERING ---
    print(f"Executando K-Means com k={N_CLUSTERS}...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    print("Clusters K-Means identificados para cada aluno.\n")


    # --- 5. PCA (ANÁLISE DE COMPONENTES PRINCIPAIS) ---
    print("Executando PCA para redução de dimensionalidade...")
    pca = PCA(n_components=None, random_state=42) # n_components=None para ver todos
    X_pca = pca.fit_transform(X_scaled)
    
    # Explicar a variância dos primeiros componentes
    explained_variance = pca.explained_variance_ratio_
    print(f"Variância explicada por PC1: {explained_variance[0]:.2%}")
    print(f"Variância explicada por PC2: {explained_variance[1]:.2%}")
    print(f"Variância acumulada pelos {N_COMPONENTS_PCA} primeiros componentes: {np.sum(explained_variance[:N_COMPONENTS_PCA]):.2%}\n")


    # --- 6. VISUALIZAÇÃO E VALIDAÇÃO (K-MEANS vs PCA) ---
    print("Gerando gráfico de validação (Clusters do K-Means no espaço do PCA)...")
    
    df_plot = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    df_plot['cluster_kmeans'] = kmeans_labels
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='cluster_kmeans',
        palette=sns.color_palette("viridis", N_CLUSTERS),
        data=df_plot,
        legend="full",
        alpha=0.6
    )
    plt.title('Visualização dos Clusters K-Means nos 2 Primeiros Componentes Principais do PCA', pad=20)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()


    # --- 7. TREINAR NOVO RANDOM FOREST COM DADOS REDUZIDOS (PCA) ---
    print(f"\nTreinando um novo Random Forest usando os top {N_COMPONENTS_PCA} componentes do PCA...")
    
    # Usando os N primeiros componentes como nossas novas features
    X_pca_reduced = X_pca[:, :N_COMPONENTS_PCA]

    X_train, X_test, y_train, y_test = train_test_split(X_pca_reduced, y, test_size=0.2, random_state=42)

    modelo_pca = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo_pca.fit(X_train, y_train)

    print("Avaliando o novo modelo...")
    y_pred = modelo_pca.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # --- 8. COMPARAÇÃO FINAL ---
    print("\n" + "="*60)
    print("--- RESULTADO FINAL DA ANÁLISE ---")
    print("\nPerformance do Modelo Original (com todas as features):")
    print("MAE: ~0.98 anos | RMSE: ~1.40 anos | R²: ~31.2%")
    
    print("\nPerformance do Novo Modelo (com features do PCA):")
    print(f"MAE: {mae:.2f} anos | RMSE: {rmse:.2f} anos | R²: {r2:.2%}")
    print("="*60)

    # Adicione este código ao final do seu script, antes do bloco "except"

    print("\n\n" + "="*60)
    print("--- ANÁLISE DESCRITIVA DOS CLUSTERS K-MEANS ---")

    # Adiciona os rótulos do cluster de volta ao DataFrame original
    df['cluster'] = kmeans_labels

    # Agrupa por cluster e analisa a variável alvo (tempo_permanencia)
    analise_clusters = df.groupby('cluster')['tempo_permanencia'].agg(['mean', 'std', 'count'])
    analise_clusters = analise_clusters.rename(columns={
        'mean': 'tempo_medio_permanencia',
        'std': 'desvio_padrao_tempo',
        'count': 'numero_de_alunos'
    })

    print("\nTempo de permanência médio para cada perfil de aluno encontrado:")
    print(analise_clusters)
    print("="*60)

   # Substitua o antigo bloco "PROFILING" por este:

    print("\n\n" + "="*60)
    print("--- PROFILING (DESCRIÇÃO) DOS CLUSTERS K-MEANS ---")

    # Adiciona os rótulos do cluster de volta ao DataFrame original para análise
    df['cluster'] = kmeans_labels

    # Vamos focar nos clusters com mais de 100 pessoas para uma análise mais robusta
    clusters_principais = df[df['cluster'].isin([1, 3, 4])]

    # --- Análise de Variáveis Numéricas ---
    print("\n--- Perfil Numérico Médio por Cluster ---\n")
    # Selecionamos as features numéricas de interesse
    features_numericas = ['faixa_etaria', 'in_capital_curso'] 
    # Agrupamos por cluster e calculamos a média
    perfil_numerico = clusters_principais.groupby('cluster')[features_numericas].mean()
    print(perfil_numerico)


    # --- Análise de Variáveis Categóricas ---
    print("\n\n--- Perfil Categórico (Distribuição de Respostas) por Cluster ---\n")

    # Lista de perguntas categóricas importantes para analisar
    features_categoricas = ['tp_sexo', 'q11', 'q16']

    for feature in features_categoricas:
        # pd.crosstab é perfeito para ver a distribuição de respostas por cluster
        # normalize='index' nos dá a porcentagem de cada resposta por linha (por cluster)
        tabela_distribuicao = pd.crosstab(clusters_principais['cluster'], clusters_principais[feature], normalize='index')
        
        print(f"\n--- Distribuição de Respostas para '{feature}' por Cluster ---\n")
        # Multiplicamos por 100 para ver como porcentagem e arredondamos
        print(tabela_distribuicao.multiply(100).round(2))

    print("\n" + "="*60)



except FileNotFoundError:
    print(f"ERRO: O arquivo '{nome_arquivo_entrada}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")