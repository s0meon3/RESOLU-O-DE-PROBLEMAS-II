import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm

# --- Função para calcular ATT via PSM ---
def att_psm(Y, D, X, n_neighbors=1):
    # Estima propensity score
    logit = LogisticRegression(max_iter=1000)
    logit.fit(X, D)
    propensity = logit.predict_proba(X)[:, 1]

    # Índices de tratados e controles
    treated_idx = np.where(D == 1)[0]
    control_idx = np.where(D == 0)[0]

    # Pareamento 1:1
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(propensity[control_idx].reshape(-1, 1))
    _, indices = nn.kneighbors(propensity[treated_idx].reshape(-1, 1))
    matched_controls = control_idx[indices.flatten()]

    # ATT = diferença de médias
    return Y[treated_idx].mean() - Y[matched_controls].mean()

# --- 1. CONFIGURAÇÃO E PREPARAÇÃO DOS DADOS ---
try:
    nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
    print(f"Carregando o arquivo '{nome_arquivo_entrada}'...")
    df_original = pd.read_csv(nome_arquivo_entrada, sep=';', low_memory=False)

    # a) Preparação dos dados
    df = df_original.copy()
    df = df[df['ano_in_grad'] >= 1990]
    df['tempo_permanencia'] = df['nu_ano_enade'] - df['ano_in_grad']
    df = df[(df['tempo_permanencia'] >= 2) & (df['tempo_permanencia'] <= 15)]

    # b) Criar variável de tratamento
    if 'tp_categoria_administrativa_ies' not in df.columns:
        raise KeyError("A coluna 'tp_categoria_administrativa_ies' não foi encontrada.")
    df['tratamento_publica'] = df['tp_categoria_administrativa_ies'].isin([1, 2, 3]).astype(int)

    # c) Selecionar outcome e covariáveis
    Y_pd = df['tempo_permanencia']
    D_pd = df['tratamento_publica']
    covariates_cols = ['faixa_etaria', 'in_capital_curso', 'tp_sexo'] + [f'q{i}' for i in range(1, 69)]
    X_df = pd.get_dummies(df[covariates_cols], drop_first=True)

    final_df = pd.concat([Y_pd, D_pd, X_df], axis=1)
    for col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    final_df = final_df.dropna()

    Y = final_df['tempo_permanencia'].values
    D = final_df['tratamento_publica'].values
    X = final_df.drop(columns=['tempo_permanencia', 'tratamento_publica']).values

    print(f"\nDados prontos. Total de {len(Y)} amostras.")
    print(f"Grupo Tratamento (Pública): {D.sum()} alunos.")
    print(f"Grupo Controle (Privada): {len(D) - D.sum()} alunos.\n")

    # --- 2. ESTIMATIVA DO ATT ---
    ATT = att_psm(Y, D, X)
    print(f"ATT estimado: {ATT:.4f}")

    # --- 3. BOOTSTRAPPING PARA ERRO PADRÃO E P-VALOR ---
    print("Executando bootstrapping...")
    B = 200  # número de reamostragens
    atts = []
    n = len(Y)
    rng = np.random.default_rng(42)

    for _ in range(B):
        sample_idx = rng.choice(n, n, replace=True)
        att_b = att_psm(Y[sample_idx], D[sample_idx], X[sample_idx])
        atts.append(att_b)

    atts = np.array(atts)
    se = atts.std(ddof=1)  # erro padrão
    z = ATT / se
    p_value = 2 * (1 - norm.cdf(abs(z)))

    print("\n--- Resultados ---")
    print(f"Efeito médio do tratamento (ATT): {ATT:.4f}")
    print(f"Erro padrão (bootstrap): {se:.4f}")
    print(f"z-score: {z:.2f}")
    print(f"p-valor: {p_value:.4f}")

except KeyError as e:
    print(f"\nERRO DE COLUNA: {e}. Verifique se a coluna está presente no seu arquivo CSV final.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")