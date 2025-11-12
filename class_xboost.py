import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- 1. CONFIGURAÇÃO ---
nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
nome_arquivo_resultados = 'analise_xgboost_CLASSIFICACAO.txt'

REGIAO_MAP = {
    1: 'Norte',
    2: 'Nordeste',
    3: 'Sudeste',
    4: 'Sul',
    5: 'Centro-Oeste'
}

# --- Função para categorizar tempo de permanência ---
def categorizar_tempo(anos):
    if anos in [3, 4]:
        return "3-4 anos"
    elif anos == 5:
        return "5 anos"
    elif anos == 6:
        return "6 anos"
    elif anos == 7:
        return "7 anos"
    else:
        return "8 ou mais anos"

# --- 2. FUNÇÃO PRINCIPAL ---
def processar_modelo_classificacao_com_fs(df_dados, nome_modelo):
    print(f"\n--- Processando: {nome_modelo} com XGBoost (Classificação 2 Etapas) ---")
    
    df_dados = df_dados[df_dados['ano_in_grad'] >= 1990].copy()
    df_dados['tempo_permanencia'] = df_dados['nu_ano_enade'] - df_dados['ano_in_grad']
    df_dados = df_dados[(df_dados['tempo_permanencia'] >= 2) & (df_dados['tempo_permanencia'] <= 15)]
    
    if len(df_dados) < 50:
        print(f"-> Dados insuficientes. ({len(df_dados)} amostras)")
        return None

    # --- CATEGORIZAR TEMPO DE PERMANÊNCIA ---
    df_dados['classe_tempo'] = df_dados['tempo_permanencia'].apply(categorizar_tempo)
    y_labels, y_map = pd.factorize(df_dados['classe_tempo'])
    y = pd.Series(y_labels, index=df_dados.index)
    num_classes = len(y_map)
    class_mapping = dict(zip(range(num_classes), y_map))
    
    print(f"-> Mapeamento de classes: {class_mapping}")
    
    if num_classes < 2:
        print(f"-> Menos de 2 classes de tempo de permanência. Pulando.")
        return None
    
    X = df_dados.drop(columns=['tempo_permanencia', 'classe_tempo', 'nu_ano_enade', 'ano_in_grad', 'nt_ger', 'nt_fg'])
    X = pd.get_dummies(X, drop_first=True)
    
    # --- ETAPA 1: MODELO COMPLETO ---
    print("Etapa 1: Treinando modelo com todas as features para obter importâncias...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo_full = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        n_estimators=1000, 
        learning_rate=0.05,
        early_stopping_rounds=10, 
        random_state=42, 
        n_jobs=-1
    )
    
    modelo_full.fit(X_train.values, y_train.values, eval_set=[(X_test.values, y_test.values)], verbose=False)
    
    y_pred_full = modelo_full.predict(X_test.values)
    
    metricas_full = {
        'accuracy': accuracy_score(y_test, y_pred_full),
        'f1_weighted': f1_score(y_test, y_pred_full, average='weighted'),
        'precision_weighted': precision_score(y_test, y_pred_full, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred_full, average='weighted'),
        'N_Amostras': len(df_dados)
    }
    
    importancias = pd.Series(modelo_full.feature_importances_, index=X.columns)
    top_features = importancias.sort_values(ascending=False).head(15)
    features_selecionadas = top_features.index.tolist()
    
    if not features_selecionadas:
        print("-> Nenhuma feature atingiu relevância. Retornando apenas o resultado do modelo completo.")
        return {'full_model': {'metricas': metricas_full}, 'reduced_model': None, 'features_used': top_features}
    
    # --- ETAPA 2: MODELO REDUZIDO ---
    print("Etapa 2: Treinando novo modelo com features selecionadas...")
    X_reduzido = X[features_selecionadas]
    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduzido, y, test_size=0.2, random_state=42)
    
    modelo_reduced = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        n_estimators=1000, 
        learning_rate=0.05,
        early_stopping_rounds=10, 
        random_state=42, 
        n_jobs=-1
    )
    
    modelo_reduced.fit(X_train_red.values, y_train_red.values, eval_set=[(X_test_red.values, y_test_red.values)], verbose=False)
    
    y_pred_reduced = modelo_reduced.predict(X_test_red.values)
    
    metricas_reduced = {
        'accuracy': accuracy_score(y_test_red, y_pred_reduced),
        'f1_weighted': f1_score(y_test_red, y_pred_reduced, average='weighted'),
        'precision_weighted': precision_score(y_test_red, y_pred_reduced, average='weighted'),
        'recall_weighted': recall_score(y_test_red, y_pred_reduced, average='weighted'),
        'N_Amostras': len(df_dados)
    }
    
    print(f"-> Concluído: {nome_modelo}")
    return {
        'full_model': {'metricas': metricas_full},
        'reduced_model': {'metricas': metricas_reduced},
        'features_used': top_features
    }

# --- 3. EXECUÇÃO PRINCIPAL ---
try:
    output_lines = ["ANÁLISE PREDITIVA (CLASSIFICAÇÃO) COM XGBOOST E SELEÇÃO DE FEATURES\n"]
    output_lines.append("="*70 + "\n")

    print("Carregando e preparando os dados...")
    df_principal = pd.read_csv(nome_arquivo_entrada, sep=';')
    
    if 'co_regiao_curso' not in df_principal.columns:
         raise KeyError("A coluna 'co_regiao_curso' não foi encontrada.")
    
    todos_os_resultados = {}
    resultado_nacional = processar_modelo_classificacao_com_fs(df_principal, "Modelo Nacional (Brasil)")
    todos_os_resultados['Nacional'] = resultado_nacional
    
    for codigo, nome_regiao in REGIAO_MAP.items():
        df_regiao = df_principal[df_principal['co_regiao_curso'] == codigo].copy()
        df_regiao = df_regiao.drop(columns=['co_regiao_curso', 'sigla_uf_curso'], errors='ignore')
        
        resultado_regional = processar_modelo_classificacao_com_fs(df_regiao, f"Modelo Regional ({nome_regiao})")
        if resultado_regional:
            todos_os_resultados[nome_regiao] = resultado_regional
    
    # --- 4. FORMATAÇÃO E COMPARAÇÃO DOS RESULTADOS ---
    print("\nFormatando e salvando os resultados...")
    
    # Coleta das métricas
    metricas_full_finais = {
        nome: res['full_model']['metricas']
        for nome, res in todos_os_resultados.items() if res
    }
    metricas_reduced_finais = {
        nome: res['reduced_model']['metricas']
        for nome, res in todos_os_resultados.items() if res and res['reduced_model']
    }
    
    df_metricas_full = pd.DataFrame.from_dict(metricas_full_finais, orient='index')
    df_metricas_reduced = pd.DataFrame.from_dict(metricas_reduced_finais, orient='index')
    
    # Top 15 features
    output_lines.append("--- TOP 15 FEATURES SELECIONADAS ---\n")
    for nome, res in todos_os_resultados.items():
        if res and res['features_used'] is not None:
            output_lines.append(f"\n--- Top 15 Features para o {nome} ---\n")
            output_lines.append(res['features_used'].to_string(float_format=lambda x: f"{x:,.4f}"))
            output_lines.append("\n")
    
    output_lines.append("\n" + "="*70 + "\n")
    
    # Métricas dos modelos completos
    output_lines.append("--- PERFORMANCE DOS MODELOS COM TODAS AS FEATURES (CLASSIFICAÇÃO) ---\n")
    output_lines.append(
        df_metricas_full.sort_values(by='accuracy', ascending=False)
        .to_string(float_format=lambda x: f"{x:,.4f}")
    )
    output_lines.append("\n\n" + "="*70 + "\n")
    
    # Métricas dos modelos reduzidos
    output_lines.append("--- PERFORMANCE DOS MODELOS COM FEATURES SELECIONADAS (CLASSIFICAÇÃO) ---\n")
    output_lines.append(
        df_metricas_reduced.sort_values(by='accuracy', ascending=False)
        .to_string(float_format=lambda x: f"{x:,.4f}")
    )
    
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
           