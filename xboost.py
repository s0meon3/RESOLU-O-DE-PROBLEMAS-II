import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CONFIGURAÇÃO ---
nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
nome_arquivo_resultados = 'analise_completa_xgboost.txt'

REGIAO_MAP = {
    1: 'Norte',
    2: 'Nordeste',
    3: 'Sudeste',
    4: 'Sul',
    5: 'Centro-Oeste'
}

# --- 2. FUNÇÃO PARA TREINAR E AVALIAR O MODELO XGBOOST ---
def processar_modelo_xgb(df_dados, nome_modelo):
    print(f"Processando: {nome_modelo} com XGBoost...")
    
    df_dados = df_dados[df_dados['ano_in_grad'] >= 1990].copy()
    df_dados['tempo_permanencia'] = df_dados['nu_ano_enade'] - df_dados['ano_in_grad']
    df_dados = df_dados[(df_dados['tempo_permanencia'] >= 2) & (df_dados['tempo_permanencia'] <= 15)]
    
    if len(df_dados) < 50:
        print(f"-> Dados insuficientes para {nome_modelo}. ({len(df_dados)} amostras)")
        return None

    y = df_dados['tempo_permanencia']
    
    colunas_para_remover = ['tempo_permanencia', 'nu_ano_enade', 'ano_in_grad', 'nt_ger', 'nt_fg']
    X = df_dados.drop(columns=colunas_para_remover)
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- CORREÇÃO APLICADA AQUI ---
    # Convertendo DataFrames para arrays NumPy para máxima compatibilidade
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values
    y_test_np = y_test.values
    # --------------------------------
    
    modelo = xgb.XGBRegressor(objective='reg:squarederror',
                              n_estimators=1000,
                              learning_rate=0.05,
                              early_stopping_rounds=10,
                              random_state=42,
                              n_jobs=-1)
    
    # Usamos os arrays NumPy na chamada .fit()
    modelo.fit(X_train_np, y_train_np,
               eval_set=[(X_test_np, y_test_np)],
               verbose=False)
    
    y_pred = modelo.predict(X_test_np)
    
    metricas = {
        'MAE': mean_absolute_error(y_test_np, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_np, y_pred)),
        'R2': r2_score(y_test_np, y_pred),
        'N_Amostras': len(df_dados)
    }
    
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    top_features = importancias.nlargest(15)
    
    print(f"-> Concluído: {nome_modelo}")
    return {'metricas': metricas, 'top_features': top_features}

# --- 3. EXECUÇÃO PRINCIPAL ---
try:
    output_lines = ["ANÁLISE PREDITIVA DO TEMPO DE PERMANÊNCIA COM XGBOOST\n"]
    output_lines.append("="*60 + "\n")

    print("Carregando e preparando os dados...")
    df_principal = pd.read_csv(nome_arquivo_entrada, sep=';')
    
    if 'co_regiao_curso' not in df_principal.columns:
         raise KeyError("A coluna 'co_regiao_curso' não foi encontrada.")

    resultado_nacional = processar_modelo_xgb(df_principal, "Modelo Nacional (Brasil)")
    todos_os_resultados = {'Nacional': resultado_nacional}

    for codigo, nome_regiao in REGIAO_MAP.items():
        df_regiao = df_principal[df_principal['co_regiao_curso'] == codigo].copy()
        df_regiao = df_regiao.drop(columns=['co_regiao_curso', 'sigla_uf_curso'], errors='ignore')
        
        resultado_regional = processar_modelo_xgb(df_regiao, f"Modelo Regional ({nome_regiao})")
        if resultado_regional:
            todos_os_resultados[nome_regiao] = resultado_regional

    # --- 4. FORMATAÇÃO E COMPARAÇÃO DOS RESULTADOS ---
    print("\nFormatando e salvando os resultados...")
    
    metricas_finais = {nome: res['metricas'] for nome, res in todos_os_resultados.items() if res}
    df_metricas = pd.DataFrame.from_dict(metricas_finais, orient='index')
    
    output_lines.append("--- TABELA COMPARATIVA DE PERFORMANCE DOS MODELOS (XGBOOST) ---\n")
    output_lines.append(df_metricas.sort_values(by='R2', ascending=False).to_string())
    output_lines.append("\n\n" + "="*60 + "\n")
    
    output_lines.append("--- ANÁLISE DAS VARIÁVEIS MAIS IMPORTANTES (XGBOOST) ---\n")
    for nome, res in todos_os_resultados.items():
        if res:
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