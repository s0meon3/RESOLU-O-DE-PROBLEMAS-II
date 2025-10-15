# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import shap
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CONFIGURAÇÃO ---
nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
nome_arquivo_resultados = 'analise_completa_stacking_rf_xgb_shap.txt'

REGIAO_MAP = {
    1: 'Norte',
    2: 'Nordeste',
    3: 'Sudeste',
    4: 'Sul',
    5: 'Centro-Oeste'
}

# --- 2. FUNÇÕES AUXILIARES ---
def preparar_dados(df):
    """Filtra, cria alvo e dummies; retorna X, y, cols."""
    df = df[df['ano_in_grad'] >= 1990].copy()
    df['tempo_permanencia'] = df['nu_ano_enade'] - df['ano_in_grad']
    df = df[(df['tempo_permanencia'] >= 2) & (df['tempo_permanencia'] <= 15)]
    if len(df) < 50:
        return None, None, None
    y = df['tempo_permanencia']
    colunas_para_remover = ['tempo_permanencia', 'nu_ano_enade', 'ano_in_grad', 'nt_ger', 'nt_fg']
    X = df.drop(columns=colunas_para_remover)
    X = pd.get_dummies(X, drop_first=True)
    return X, y, X.columns

def avaliar_modelo(y_true, y_pred, n_amostras):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'N_Amostras': n_amostras
    }

def top15(series):
    return series.nlargest(15)

def shap_importance_tree(modelo, X_train_df, X_test_df, cols):
    """
    SHAP para modelos de árvore (RF/XGB): converte DataFrame -> NumPy para evitar erro de dtype.
    Retorna Series com top-15 da média dos valores absolutos de SHAP por feature.
    """
    try:
        # Background amostrado para performance
        if len(X_train_df) > 1000:
            background_df = X_train_df.sample(1000, random_state=42)
        else:
            background_df = X_train_df
        background = background_df.values
        X_test_np = X_test_df.values

        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_test_np)  # shape: (n_amostras, n_features)
        shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
        return top15(pd.Series(shap_abs_mean, index=cols))
    except Exception as e:
        print(f"Erro no SHAP (árvore): {e}")
        return pd.Series(dtype=float, index=cols)

def shap_importance_meta(meta_model, base_train_df, base_test_df):
    """
    SHAP para meta-modelo (Ridge) usando as entradas RF_pred e XGB_pred.
    Retorna importância relativa das duas entradas.
    """
    meta_cols = ['RF_pred', 'XGB_pred']
    try:
        X_bg = base_train_df.values
        if len(X_bg) > 2000:
            X_bg = base_train_df.sample(2000, random_state=42).values
        explainer = shap.Explainer(meta_model.predict, X_bg)
        shap_values = explainer(base_test_df.values)  # (n_amostras, 2)
        shap_abs_mean = np.mean(np.abs(shap_values.values), axis=0)
        return pd.Series(shap_abs_mean, index=meta_cols).sort_values(ascending=False)
    except Exception as e:
        print(f"Erro no SHAP (meta): {e}")
        return pd.Series(dtype=float, index=meta_cols)

# --- 3. PIPELINE STACKING ---
def processar_stacking(df_dados, nome_modelo):
    print(f"Processando: {nome_modelo} com Stacking (RF + XGB)...")

    X, y, cols = preparar_dados(df_dados)
    if X is None:
        print(f"-> Dados insuficientes para {nome_modelo}.")
        return None

    # Split inicial: treino + teste
    X_train_full, X_test_df, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Split adicional: treino + validação (apenas dentro do treino)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # NumPy para compatibilidade com XGB/SHAP
    X_train = X_train_df.values
    X_val = X_val_df.values
    X_test = X_test_df.values
    y_train_np = y_train.values
    y_val_np = y_val.values
    y_test_np = y_test.values

    # Modelos base
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=800,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=6,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    # Treinamento RF
    rf.fit(X_train_df, y_train)

    # Treinamento XGB com early stopping usando validação
    try:
        xgb_reg.fit(
            X_train, y_train_np,
            eval_set=[(X_val, y_val_np)],
            verbose=False,
            early_stopping_rounds=20
        )
    except TypeError:
        xgb_reg.fit(
            X_train, y_train_np,
            eval_set=[(X_val, y_val_np)],
            verbose=False
        )

    # Predições base
    rf_pred_train = rf.predict(X_train_df)
    xgb_pred_train = xgb_reg.predict(X_train)
    rf_pred_test = rf.predict(X_test_df)
    xgb_pred_test = xgb_reg.predict(X_test)

    # Stacking (meta-modelo Ridge)
    stack = StackingRegressor(
        estimators=[('rf', rf), ('xgb', xgb_reg)],
        final_estimator=Ridge(alpha=1.0),
        passthrough=False,
        n_jobs=-1
    )
    stack.fit(X_train_df, y_train)
    stack_pred = stack.predict(X_test_df)

    # Métricas
    metricas = {
        'Random Forest': avaliar_modelo(y_test, rf_pred_test, len(X)),
        'XGBoost': avaliar_modelo(y_test, xgb_pred_test, len(X)),
        'Stacking (RF+XGBoost)': avaliar_modelo(y_test, stack_pred, len(X))
    }

    # Importâncias nativas
    fi_rf = top15(pd.Series(rf.feature_importances_, index=cols))
    fi_xgb = top15(pd.Series(xgb_reg.feature_importances_, index=cols))

    # SHAP dos modelos base
    shap_rf = shap_importance_tree(rf, X_train_df, X_test_df, cols)
    shap_xgb = shap_importance_tree(xgb_reg, X_train_df, X_test_df, cols)

    # SHAP do meta-modelo
    base_train_df = pd.DataFrame({'RF_pred': rf_pred_train, 'XGB_pred': xgb_pred_train})
    base_test_df = pd.DataFrame({'RF_pred': rf_pred_test, 'XGB_pred': xgb_pred_test})
    shap_meta = shap_importance_meta(stack.final_estimator_, base_train_df, base_test_df)

    print(f"-> Concluído: {nome_modelo}")
    return {
        'metricas': metricas,
        'feature_importance': {'Random Forest': fi_rf, 'XGBoost': fi_xgb},
        'shap': {
            'Random Forest': shap_rf,
            'XGBoost': shap_xgb,
            'Meta (RF_pred vs XGB_pred)': shap_meta
        },
        'n_features': len(cols)
    }

# --- 4. EXECUÇÃO PRINCIPAL ---
try:
    output_lines = ["ANÁLISE PREDITIVA DO TEMPO DE PERMANÊNCIA COM STACKING (RF + XGBoost) + SHAP\n"]
    output_lines.append("="*80 + "\n")

    print("Carregando e preparando os dados...")
    # Leitura robusta para evitar DtypeWarning
    df_principal = pd.read_csv(nome_arquivo_entrada, sep=';', low_memory=False)

    if 'co_regiao_curso' not in df_principal.columns:
        raise KeyError("A coluna 'co_regiao_curso' não foi encontrada.")

    # Modelo Nacional
    resultado_nacional = processar_stacking(df_principal, "Modelo Nacional (Brasil)")
    todos_os_resultados = {'Nacional': resultado_nacional}

    # Modelos Regionais
    for codigo, nome_regiao in REGIAO_MAP.items():
        df_regiao = df_principal[df_principal['co_regiao_curso'] == codigo].copy()
        # Removemos colunas constantes por região
        df_regiao = df_regiao.drop(columns=['co_regiao_curso', 'sigla_uf_curso'], errors='ignore')

        resultado_regional = processar_stacking(df_regiao, f"Modelo Regional ({nome_regiao})")
        if resultado_regional:
            todos_os_resultados[nome_regiao] = resultado_regional

    # --- 5. RELATÓRIO: Tabela comparativa de performance (RF vs XGB vs Stacking) ---
    metricas_aggregate = {}
    for nome, res in todos_os_resultados.items():
        if res:
            metricas_aggregate[nome] = {
                'MAE_RF': res['metricas']['Random Forest']['MAE'],
                'RMSE_RF': res['metricas']['Random Forest']['RMSE'],
                'R2_RF': res['metricas']['Random Forest']['R2'],

                'MAE_XGB': res['metricas']['XGBoost']['MAE'],
                'RMSE_XGB': res['metricas']['XGBoost']['RMSE'],
                'R2_XGB': res['metricas']['XGBoost']['R2'],

                'MAE_STACK': res['metricas']['Stacking (RF+XGBoost)']['MAE'],
                'RMSE_STACK': res['metricas']['Stacking (RF+XGBoost)']['RMSE'],
                'R2_STACK': res['metricas']['Stacking (RF+XGBoost)']['R2'],

                'N_Amostras': res['metricas']['Stacking (RF+XGBoost)']['N_Amostras'],
                'N_Features': res['n_features']
            }

    df_metricas = pd.DataFrame.from_dict(metricas_aggregate, orient='index')
    output_lines.append("--- TABELA COMPARATIVA DE PERFORMANCE (RF vs XGB vs STACKING) ---\n")
    output_lines.append(
        df_metricas.sort_values(by='R2_STACK', ascending=False).to_string(float_format=lambda x: f"{x:,.4f}")
    )
    output_lines.append("\n\n" + "="*80 + "\n")

    # --- 6. RELATÓRIO: Feature importance ---
    output_lines.append("--- ANÁLISE DE VARIÁVEIS MAIS IMPORTANTES (FEATURE IMPORTANCE) ---\n")
    for nome, res in todos_os_resultados.items():
        if res:
            output_lines.append(f"\n--- Top 15 Features (Random Forest) para {nome} ---\n")
            output_lines.append(res['feature_importance']['Random Forest'].to_string())
            output_lines.append("\n")

            output_lines.append(f"--- Top 15 Features (XGBoost) para {nome} ---\n")
            output_lines.append(res['feature_importance']['XGBoost'].to_string())
            output_lines.append("\n")

    # --- 7. RELATÓRIO: SHAP por modelo e meta-modelo ---
    output_lines.append("\n" + "="*80 + "\n")
    output_lines.append("--- INTERPRETABILIDADE COM SHAP (média de |SHAP| por feature) ---\n")
    for nome, res in todos_os_resultados.items():
        if res:
            output_lines.append(f"\n--- Top 15 SHAP (Random Forest) para {nome} ---\n")
            if res['shap']['Random Forest'].empty:
                output_lines.append("SHAP indisponível para RF neste escopo.\n")
            else:
                output_lines.append(res['shap']['Random Forest'].sort_values(ascending=False).to_string())
                output_lines.append("\n")

            output_lines.append(f"--- Top 15 SHAP (XGBoost) para {nome} ---\n")
            if res['shap']['XGBoost'].empty:
                output_lines.append("SHAP indisponível para XGB neste escopo.\n")
            else:
                output_lines.append(res['shap']['XGBoost'].sort_values(ascending=False).to_string())
                output_lines.append("\n")

            output_lines.append(f"--- Importância SHAP das entradas do meta-modelo (RF_pred vs XGB_pred) para {nome} ---\n")
            if res['shap']['Meta (RF_pred vs XGB_pred)'].empty:
                output_lines.append("SHAP indisponível para o meta-modelo neste escopo.\n")
            else:
                output_lines.append(res['shap']['Meta (RF_pred vs XGB_pred)'].to_string())
                output_lines.append("\n")

    # --- 8. SALVAR RELATÓRIO ---
    with open(nome_arquivo_resultados, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\nPROCESSO CONCLUÍDO!\nAnálise completa salva no arquivo: '{nome_arquivo_resultados}'")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{nome_arquivo_entrada}' não foi encontrado.")
except KeyError as e:
    print(f"ERRO DE COLUNA: {e}. Verifique se a coluna está presente no seu arquivo CSV final.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")