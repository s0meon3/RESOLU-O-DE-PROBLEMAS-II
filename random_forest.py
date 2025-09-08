import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CONFIGURAÇÃO ---
nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
nome_arquivo_resultados = 'analise_completa_random_forest.txt'

# Mapeamento dos códigos de região para nomes
REGIAO_MAP = {
    1: 'Norte',
    2: 'Nordeste',
    3: 'Sudeste',
    4: 'Sul',
    5: 'Centro-Oeste'
}

# --- 2. FUNÇÃO PARA TREINAR, AVALIAR E ANALISAR FEATURES ---
def processar_modelo(df_dados, nome_modelo):
    """
    Recebe um DataFrame, treina um modelo, e retorna métricas e feature importances.
    """
    print(f"Processando: {nome_modelo}...")
    
    # a) Preparação do alvo (y)
    df_dados = df_dados[df_dados['ano_in_grad'] >= 1990].copy()
    df_dados['tempo_permanencia'] = df_dados['nu_ano_enade'] - df_dados['ano_in_grad']
    df_dados = df_dados[(df_dados['tempo_permanencia'] >= 2) & (df_dados['tempo_permanencia'] <= 15)]
    
    if len(df_dados) < 50:
        print(f"-> Dados insuficientes para {nome_modelo}. ({len(df_dados)} amostras)")
        return None

    y = df_dados['tempo_permanencia']
    
    # b) Preparação das features (X)
    colunas_para_remover = ['tempo_permanencia', 'nu_ano_enade', 'ano_in_grad', 'nt_ger', 'nt_fg']
    X = df_dados.drop(columns=colunas_para_remover)
    X = pd.get_dummies(X, drop_first=True)
    
    # c) Divisão, Treinamento e Avaliação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    metricas = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'N_Amostras': len(df_dados)
    }
    
    # d) Análise de Features
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    top_features = importancias.nlargest(15)
    
    print(f"-> Concluído: {nome_modelo}")
    return {'metricas': metricas, 'top_features': top_features}

# --- 3. EXECUÇÃO PRINCIPAL ---
try:
    # Lista para armazenar todos os resultados como texto
    output_lines = ["ANÁLISE PREDITIVA DO TEMPO DE PERMANÊNCIA COM RANDOM FOREST\n"]
    output_lines.append("="*60 + "\n")

    print("Carregando e preparando os dados...")
    df_principal = pd.read_csv(nome_arquivo_entrada, sep=';')
    
    if 'co_regiao_curso' not in df_principal.columns:
         raise KeyError("A coluna 'co_regiao_curso' não foi encontrada. Por favor, gere novamente o arquivo CSV final incluindo esta coluna.")

    # --- Modelo Nacional ---
    resultado_nacional = processar_modelo(df_principal, "Modelo Nacional (Brasil)")
    todos_os_resultados = {'Nacional': resultado_nacional}

    # --- Modelos Regionais ---
    for codigo, nome_regiao in REGIAO_MAP.items():
        df_regiao = df_principal[df_principal['co_regiao_curso'] == codigo].copy()
        # Removemos colunas que seriam constantes dentro da região
        df_regiao = df_regiao.drop(columns=['co_regiao_curso', 'sigla_uf_curso'], errors='ignore')
        
        resultado_regional = processar_modelo(df_regiao, f"Modelo Regional ({nome_regiao})")
        if resultado_regional:
            todos_os_resultados[nome_regiao] = resultado_regional

    # --- 4. FORMATAÇÃO E COMPARAÇÃO DOS RESULTADOS ---
    print("\nFormatando e salvando os resultados...")
    
    # Formatando tabela de métricas
    metricas_finais = {nome: res['metricas'] for nome, res in todos_os_resultados.items()}
    df_metricas = pd.DataFrame.from_dict(metricas_finais, orient='index')
    
    output_lines.append("--- TABELA COMPARATIVA DE PERFORMANCE DOS MODELOS ---\n")
    output_lines.append(df_metricas.sort_values(by='R2', ascending=False).to_string())
    output_lines.append("\n\n" + "="*60 + "\n")
    
    # Formatando análise de features para cada modelo
    output_lines.append("--- ANÁLISE DAS VARIÁVEIS MAIS IMPORTANTES (FEATURE IMPORTANCE) ---\n")
    for nome, res in todos_os_resultados.items():
        output_lines.append(f"\n--- Top 15 Features para o {nome} ---\n")
        output_lines.append(res['top_features'].to_string())
        output_lines.append("\n")

    # --- 5. SALVANDO TUDO NO ARQUIVO DE TEXTO ---
    with open(nome_arquivo_resultados, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print(f"\nPROCESSO CONCLUÍDO!\nAnálise completa salva no arquivo: '{nome_arquivo_resultados}'")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{nome_arquivo_entrada}' não foi encontrado.")
except KeyError as e:
    print(f"ERRO DE COLUNA: {e}. Verifique se a coluna está presente no seu arquivo CSV final.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")