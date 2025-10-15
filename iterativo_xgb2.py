import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CONFIGURAÇÃO ---
nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
nome_arquivo_resultados = 'analise_iterativa_final.txt'

REGIAO_MAP = {
    1: 'Norte',
    2: 'Nordeste',
    3: 'Sudeste',
    4: 'Sul',
    5: 'Centro-Oeste'
}

# Parâmetros para o modelo XGBoost
XGB_PARAMS = dict(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=15,
    random_state=42,
    n_jobs=-1
)

# Parâmetros para a árvore de decisão que busca regras de qualidade
TREE_PARAMS = dict(
    max_depth=3,
    min_samples_leaf=30 
)

# Configurações do loop iterativo
MAX_ITERS = 5
MIN_IMPROVEMENT_RMSE = 0.001
MIN_MEAN_RESIDUAL_MAG = 0.1 # Filtro de qualidade para as regras

# --- 2. FUNÇÕES AUXILIARES ---
def avaliar_modelo(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

# --- 3. FUNÇÃO PRINCIPAL DO PROCESSO ITERATIVO ---
def processo_iterativo_robusto(df_dados, nome_modelo):
    print(f"Processando: {nome_modelo}...")

    # a. Preparação dos dados
    df = df_dados.copy()
    df = df[df['ano_in_grad'] >= 1990]
    df['tempo_permanencia'] = df['nu_ano_enade'] - df['ano_in_grad']
    df = df[(df['tempo_permanencia'] >= 2) & (df['tempo_permanencia'] <= 15)]

    if len(df) < 100:
        print(f"-> Dados insuficientes. ({len(df)} amostras)")
        return None

    y = df['tempo_permanencia']
    colunas_para_remover = ['tempo_permanencia', 'nu_ano_enade', 'ano_in_grad', 'nt_ger', 'nt_fg']
    X = df.drop(columns=colunas_para_remover)
    X = pd.get_dummies(X, drop_first=True)

    # b. Divisão Tripla: Treino (60%), Validação (20%), Teste (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    historico_validacao = []
    
    # c. Iteração 0: Modelo Baseline
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred_val = model.predict(X_val)
    metrics_val = avaliar_modelo(y_val, y_pred_val)
    best_rmse = metrics_val['RMSE']
    best_model = model
    
    historico_validacao.append({'iter': 0, **metrics_val})

    X_train_acumulado = X_train.copy()
    X_val_acumulado = X_val.copy()
    transformadores_uteis = []

    # d. Loop de Iterações
    for iter_num in range(1, MAX_ITERS + 1):
        y_pred_train = model.predict(X_train_acumulado)
        residuos_train = y_train - y_pred_train
        
        arvore = DecisionTreeRegressor(**TREE_PARAMS, random_state=42 + iter_num)
        arvore.fit(X_train_acumulado, residuos_train)

        # II. Identificar e filtrar as folhas relevantes (Controle de Qualidade)
        leaf_ids_train = arvore.apply(X_train_acumulado)
        df_leaf_stats = pd.DataFrame({'leaf_id': leaf_ids_train, 'resid': residuos_train})
        stats = df_leaf_stats.groupby('leaf_id')['resid'].agg(['mean', 'count'])
        
        folhas_validas = stats[
            (stats['count'] >= TREE_PARAMS['min_samples_leaf']) & 
            (stats['mean'].abs() >= MIN_MEAN_RESIDUAL_MAG)
        ].index.tolist()

        if not folhas_validas:
            print(f"Iter {iter_num}: Nenhuma regra de alta qualidade encontrada. Parando.")
            break
            
        # III. Criar features candidatas
        novas_cols = [f"regra_iter{iter_num}_{lid}" for lid in folhas_validas]
        
        X_train_cand = X_train_acumulado.copy()
        X_val_cand = X_val_acumulado.copy()
        
        leaf_ids_train_filtered = np.isin(leaf_ids_train, folhas_validas)
        leaf_ids_val = arvore.apply(X_val_acumulado)

        for lid in folhas_validas:
            col_name = f"regra_iter{iter_num}_{lid}"
            X_train_cand[col_name] = (leaf_ids_train == lid).astype(np.int8)
            X_val_cand[col_name] = (leaf_ids_val == lid).astype(np.int8)
        
        # IV. Treinar modelo candidato e verificar melhora
        model_cand = xgb.XGBRegressor(**XGB_PARAMS)
        model_cand.fit(X_train_cand, y_train, eval_set=[(X_val_cand, y_val)], verbose=False)
        
        y_pred_val_cand = model_cand.predict(X_val_cand)
        metrics_val_cand = avaliar_modelo(y_val, y_pred_val_cand)
        rmse_cand = metrics_val_cand['RMSE']
        improvement = best_rmse - rmse_cand
        
        print(f"Iter {iter_num}: RMSE anterior={best_rmse:.4f}, RMSE novo={rmse_cand:.4f}, melhoria={improvement:+.4f}")
        historico_validacao.append({'iter': iter_num, **metrics_val_cand})
        
        # V. Lógica de Aceitar/Rejeitar
        if improvement >= MIN_IMPROVEMENT_RMSE:
            print("-> Melhoria aceita. Atualizando modelo e features.")
            best_rmse = rmse_cand
            best_model = model_cand
            model = model_cand
            X_train_acumulado = X_train_cand
            X_val_acumulado = X_val_cand
            transformadores_uteis.append({'tree': arvore, 'folhas_validas': folhas_validas, 'cols': novas_cols})
        else:
            print("-> Melhoria insuficiente. Rejeitando features e encerrando iterações.")
            break
            
    # e. Avaliação Final no Conjunto de Teste
    print("Aplicando transformações finais no conjunto de teste...")
    X_test_acumulado = X_test.copy()
    
    for transformador in transformadores_uteis:
        leaf_ids_test = transformador['tree'].apply(X_test_acumulado)
        for i, lid in enumerate(transformador['folhas_validas']):
            col_name = transformador['cols'][i]
            X_test_acumulado[col_name] = (leaf_ids_test == lid).astype(np.int8)

    X_test_final = X_test_acumulado.reindex(columns=best_model.get_booster().feature_names, fill_value=0)
    
    y_pred_test = best_model.predict(X_test_final)
    metricas_finais = avaliar_modelo(y_test, y_pred_test)
    metricas_finais['N_Amostras'] = len(df)
    
    importancias = pd.Series(best_model.feature_importances_, index=best_model.get_booster().feature_names)
    top_features = importancias[importancias > 0].nlargest(20)

    print(f"-> Concluído: {nome_modelo}\n")
    return {
        'metricas_finais': metricas_finais, 
        'top_features': top_features, 
        'historico_validacao': historico_validacao
    }

# --- 4. EXECUÇÃO PRINCIPAL ---
try:
    print("Carregando dados...")
    df_principal = pd.read_csv(nome_arquivo_entrada, sep=';', low_memory=False)
    
    if 'co_regiao_curso' not in df_principal.columns:
        raise KeyError("A coluna 'co_regiao_curso' não foi encontrada.")

    todos_os_resultados = {}
    resultado_nacional = processo_iterativo_robusto(df_principal, "Modelo Nacional (Brasil)")
    if resultado_nacional:
        todos_os_resultados['Nacional'] = resultado_nacional
        
    for codigo, nome_regiao in REGIAO_MAP.items():
        df_regiao = df_principal[df_principal['co_regiao_curso'] == codigo].copy()
        resultado_regional = processo_iterativo_robusto(df_regiao, f"Modelo Regional ({nome_regiao})")
        if resultado_regional:
            todos_os_resultados[nome_regiao] = resultado_regional

    # --- 5. FORMATAÇÃO DO RELATÓRIO FINAL ---
    print("\nFormatando e salvando o relatório final...")
    output_lines = ["ANÁLISE PREDITIVA COM XGBOOST ITERATIVO (LÓGICA OTIMIZADA)\n", "="*80 + "\n"]

    metricas_resumo = {nome: res['metricas_finais'] for nome, res in todos_os_resultados.items()}
    df_resumo = pd.DataFrame.from_dict(metricas_resumo, orient='index')
    
    output_lines.append("--- TABELA DE PERFORMANCE FINAL (NO CONJUNTO DE TESTE) ---\n")
    output_lines.append(df_resumo.sort_values(by='R2', ascending=False).to_string(float_format=lambda x: f"{x:,.4f}"))
    output_lines.append("\n\n" + "="*80 + "\n")

    output_lines.append("--- HISTÓRICO DE PERFORMANCE (NO CONJUNTO DE VALIDAÇÃO) ---\n")
    for nome, res in todos_os_resultados.items():
        output_lines.append(f"\n--- {nome} ---\n")
        df_hist = pd.DataFrame(res['historico_validacao']).set_index('iter')
        output_lines.append(df_hist.to_string(float_format=lambda x: f"{x:,.4f}"))
        output_lines.append("\n")

    output_lines.append("\n" + "="*80 + "\n")
    
    output_lines.append("--- TOP FEATURES DO MELHOR MODELO FINAL ---\n")
    for nome, res in todos_os_resultados.items():
        output_lines.append(f"\n--- {nome} ---\n")
        output_lines.append(res['top_features'].to_string())
        output_lines.append("\n")

    with open(nome_arquivo_resultados, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print(f"\nPROCESSO CONCLUÍDO!\nRelatório salvo em: '{nome_arquivo_resultados}'")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{nome_arquivo_entrada}' não foi encontrado.")
except KeyError as e:
    print(f"ERRO DE COLUNA: {e}. Verifique se a coluna está presente no seu arquivo CSV final.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")