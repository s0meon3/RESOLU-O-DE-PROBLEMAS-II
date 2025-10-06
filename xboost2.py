import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CONFIGURAÇÃO ---
nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
nome_arquivo_resultados = 'analise_xgboost_feature_selection.txt'
LIMIAR_IMPORTANCIA = 0.01 # Define o ponto de corte para a seleção de features

REGIAO_MAP = {
    1: 'Norte',
    2: 'Nordeste',
    3: 'Sudeste',
    4: 'Sul',
    5: 'Centro-Oeste'
}

# --- 2. FUNÇÃO ATUALIZADA PARA O PROCESSO DE DUAS ETAPAS ---
def processar_modelo_com_feature_selection(df_dados, nome_modelo):
    print(f"\n--- Processando: {nome_modelo} com XGBoost (Processo de 2 Etapas) ---")
    
    df_dados = df_dados[df_dados['ano_in_grad'] >= 1990].copy()
    df_dados['tempo_permanencia'] = df_dados['nu_ano_enade'] - df_dados['ano_in_grad']
    df_dados = df_dados[(df_dados['tempo_permanencia'] >= 2) & (df_dados['tempo_permanencia'] <= 15)]
    
    if len(df_dados) < 50:
        print(f"-> Dados insuficientes. ({len(df_dados)} amostras)")
        return None

    y = df_dados['tempo_permanencia']
    X = df_dados.drop(columns=['tempo_permanencia', 'nu_ano_enade', 'ano_in_grad', 'nt_ger', 'nt_fg'])
    X = pd.get_dummies(X, drop_first=True)
    
    # --- ETAPA 1: TREINAR MODELO COMPLETO PARA DESCOBRIR FEATURES ---
    print("Etapa 1: Treinando modelo com todas as features para obter importâncias...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo_full = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                                   early_stopping_rounds=10, random_state=42, n_jobs=-1)
    
    modelo_full.fit(X_train.values, y_train.values, eval_set=[(X_test.values, y_test.values)], verbose=False)
    
    y_pred_full = modelo_full.predict(X_test.values)
    metricas_full = {
        'MAE': mean_absolute_error(y_test, y_pred_full), 'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_full)),
        'R2': r2_score(y_test, y_pred_full), 'N_Amostras': len(df_dados)
    }
    
    importancias = pd.Series(modelo_full.feature_importances_, index=X.columns)
    
    # --- SELEÇÃO DAS FEATURES ---
    features_selecionadas = importancias[importancias >= LIMIAR_IMPORTANCIA].index.tolist()
    print(f"-> {len(features_selecionadas)} features selecionadas com importância >= {LIMIAR_IMPORTANCIA}")

    if not features_selecionadas:
        print("-> Nenhuma feature atingiu o limiar. Retornando apenas o resultado do modelo completo.")
        return {'full_model': {'metricas': metricas_full}, 'reduced_model': None, 'features_used': []}

    # --- ETAPA 2: TREINAR NOVO MODELO APENAS COM AS FEATURES SELECIONADAS ---
    print("Etapa 2: Treinando novo modelo com features selecionadas...")
    X_reduzido = X[features_selecionadas]
    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduzido, y, test_size=0.2, random_state=42)
    
    modelo_reduced = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                                      early_stopping_rounds=10, random_state=42, n_jobs=-1)
    
    modelo_reduced.fit(X_train_red.values, y_train_red.values, eval_set=[(X_test_red.values, y_test_red.values)], verbose=False)
    
    y_pred_reduced = modelo_reduced.predict(X_test_red.values)
    metricas_reduced = {
        'MAE': mean_absolute_error(y_test_red, y_pred_reduced), 'RMSE': np.sqrt(mean_squared_error(y_test_red, y_pred_reduced)),
        'R2': r2_score(y_test_red, y_pred_reduced), 'N_Amostras': len(df_dados)
    }
    
    print(f"-> Concluído: {nome_modelo}")
    return {'full_model': {'metricas': metricas_full}, 'reduced_model': {'metricas': metricas_reduced}, 'features_used': features_selecionadas}

# --- 3. EXECUÇÃO PRINCIPAL ---
try:
    output_lines = ["ANÁLISE PREDITIVA COM XGBOOST E SELEÇÃO DE FEATURES\n"]
    output_lines.append("="*60 + "\n")

    print("Carregando e preparando os dados...")
    df_principal = pd.read_csv(nome_arquivo_entrada, sep=';')
    
    if 'co_regiao_curso' not in df_principal.columns:
         raise KeyError("A coluna 'co_regiao_curso' não foi encontrada.")

    # Executa o processo de 2 etapas para o Brasil e cada região
    todos_os_resultados = {}
    resultado_nacional = processar_modelo_com_feature_selection(df_principal, "Modelo Nacional (Brasil)")
    todos_os_resultados['Nacional'] = resultado_nacional

    for codigo, nome_regiao in REGIAO_MAP.items():
        df_regiao = df_principal[df_principal['co_regiao_curso'] == codigo].copy()
        df_regiao = df_regiao.drop(columns=['co_regiao_curso', 'sigla_uf_curso'], errors='ignore')
        
        resultado_regional = processar_modelo_com_feature_selection(df_regiao, f"Modelo Regional ({nome_regiao})")
        if resultado_regional:
            todos_os_resultados[nome_regiao] = resultado_regional

    # --- 4. FORMATAÇÃO E COMPARAÇÃO DOS RESULTADOS ---
    print("\nFormatando e salvando os resultados...")
    
    # Extraindo os resultados para as tabelas
    metricas_full_finais = {nome: res['full_model']['metricas'] for nome, res in todos_os_resultados.items() if res}
    metricas_reduced_finais = {nome: res['reduced_model']['metricas'] for nome, res in todos_os_resultados.items() if res and res['reduced_model']}
    
    df_metricas_full = pd.DataFrame.from_dict(metricas_full_finais, orient='index')
    df_metricas_reduced = pd.DataFrame.from_dict(metricas_reduced_finais, orient='index')
    
    output_lines.append("--- FEATURES SELECIONADAS (IMPORTÂNCIA >= 0.01) ---\n")
    for nome, res in todos_os_resultados.items():
        if res:
            output_lines.append(f"\n--- Features para o {nome} ({len(res['features_used'])} selecionadas) ---\n")
            output_lines.append(str(res['features_used']))
            output_lines.append("\n")
    
    output_lines.append("\n" + "="*60 + "\n")
    
    output_lines.append("--- PERFORMANCE DOS MODELOS COM TODAS AS FEATURES ---\n")
    output_lines.append(df_metricas_full.sort_values(by='R2', ascending=False).to_string())
    output_lines.append("\n\n" + "="*60 + "\n")

    output_lines.append("--- PERFORMANCE DOS MODELOS COM FEATURES SELECIONADAS ---\n")
    output_lines.append(df_metricas_reduced.sort_values(by='R2', ascending=False).to_string())

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