import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. CONFIGURAÇÃO E PREPARAÇÃO DOS DADOS ---
try:
    # --- NOVAS CONFIGURAÇÕES ---
    nome_pasta_graficos = 'aed_ampliada'
    nome_arquivo_tabelas = 'aed_analise_tabelas.txt'
    output_lines = ["ANÁLISE EXPLORATÓRIA DE DADOS (EDA) - TABELAS\n\n"]
    os.makedirs(nome_pasta_graficos, exist_ok=True)
    print(f"Os gráficos serão salvos na pasta: '{nome_pasta_graficos}/'")
    print(f"As tabelas serão salvas no arquivo: '{nome_arquivo_tabelas}'")
    # ----------------------------------------------------------------

    nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
    df = pd.read_csv(nome_arquivo_entrada, sep=';', low_memory=False)

    df = df[df['ano_in_grad'] >= 1990].copy()
    df['tempo_permanencia'] = df['nu_ano_enade'] - df['ano_in_grad']
    df = df[(df['tempo_permanencia'] >= 2) & (df['tempo_permanencia'] <= 15)]
    
    TITULOS_MAP = { 'tempo_permanencia': 'Tempo de Permanência', 'faixa_etaria': 'Faixa Etária', 'ano_in_grad': 'Ano de Início da Graduação', 'tp_sexo': 'Sexo', 'co_regiao_curso': 'Região do Curso', 'q2': 'Cor/Raça', 'q8': 'Estado Civil', 'q11': 'Renda Familiar', 'q16': 'Escolaridade Familiar', 'q25': 'Motivo Escolha do Curso' }
    REGIAO_MAP = { 1: 'Norte', 2: 'Nordeste', 3: 'Sudeste', 4: 'Sul', 5: 'Centro-Oeste' }
    df['nome_regiao'] = df['co_regiao_curso'].map(REGIAO_MAP)

    output_lines.append("="*80 + "\n--- 2.1. ANÁLISE UNIVARIADA: VARIÁVEIS NUMÉRICAS ---\n" + "="*80 + "\n")
    
    vars_numericas = ['tempo_permanencia', 'faixa_etaria', 'ano_in_grad']
    
    descritivas = df[vars_numericas].describe().T
    output_lines.append("Estatísticas Descritivas:\n")
    output_lines.append(descritivas.to_string() + "\n")
    
    for var in vars_numericas:
        plt.figure(figsize=(10, 6)); sns.histplot(df[var], kde=True, bins=15)
        plt.title(f'Distribuição de: {TITULOS_MAP.get(var, var)}', fontsize=16)
        caminho_salvar = os.path.join(nome_pasta_graficos, f'hist_{var}.png'); plt.savefig(caminho_salvar); plt.close()
        print(f"-> Gráfico salvo em: {caminho_salvar}")

    output_lines.append("\n" + "="*80 + "\n--- 2.2. ANÁLISE UNIVARIADA: VARIÁVEIS CATEGÓRICAS ---\n" + "="*80 + "\n")
    
    vars_categoricas = ['tp_sexo', 'nome_regiao', 'q2', 'q8', 'q11', 'q16', 'q25']
    
    for var in vars_categoricas:
        titulo_var = TITULOS_MAP.get(var, var)
        output_lines.append(f"\n--- Frequência para a variável: {titulo_var} ---\n")
        proporcao = df[var].value_counts(normalize=True).multiply(100).round(2)
        output_lines.append("Proporção de Respostas (%):\n" + proporcao.to_string() + "\n")
        
        plt.figure(figsize=(12, 7))
        # Correção do FutureWarning
        sns.countplot(y=var, data=df, order=df[var].value_counts().index, hue=var, legend=False)
        plt.title(f'Distribuição de: {titulo_var}', fontsize=16); plt.tight_layout()
        caminho_salvar = os.path.join(nome_pasta_graficos, f'bar_{var}.png'); plt.savefig(caminho_salvar); plt.close()
        print(f"-> Gráfico salvo em: {caminho_salvar}")

    output_lines.append("\n" + "="*80 + "\n--- 3.1. ANÁLISE BIVARIADA: NUMÉRICA vs. NUMÉRICA ---\n" + "="*80 + "\n")

    correlacao = df[vars_numericas].corr()
    output_lines.append("Matriz de Correlação:\n" + correlacao.to_string() + "\n")
    
    plt.figure(figsize=(8, 6)); sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Mapa de Calor da Correlação', fontsize=16)
    caminho_salvar = os.path.join(nome_pasta_graficos, 'heatmap_correlacao_numericas.png'); plt.savefig(caminho_salvar); plt.close()
    print(f"-> Gráfico salvo em: {caminho_salvar}")

    output_lines.append("\n" + "="*80 + "\n--- 3.2. ANÁLISE BIVARIADA: CATEGÓRICA vs. NUMÉRICA ---\n" + "="*80 + "\n")
    
    # Lista de variáveis para o boxplot (removido q16)
    vars_bivariada_cat = ['nome_regiao', 'q11', 'q8'] 
    
    for var in vars_bivariada_cat:
        titulo_var = TITULOS_MAP.get(var, var)
        plt.figure(figsize=(14, 8))
        order_cat = sorted(df[var].dropna().unique().astype(str))
        # Correção do FutureWarning
        sns.boxplot(x='tempo_permanencia', y=var, data=df, order=order_cat, hue=var, legend=False)
        plt.title(f'Tempo de Permanência vs. {titulo_var}', fontsize=16); plt.tight_layout()
        caminho_salvar = os.path.join(nome_pasta_graficos, f'boxplot_tempo_vs_{var}.png'); plt.savefig(caminho_salvar); plt.close()
        print(f"-> Gráfico salvo em: {caminho_salvar}")
        
    # --- SALVANDO O ARQUIVO DE TEXTO COM AS TABELAS ---
    with open(nome_arquivo_tabelas, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\n-> Tabelas de análise salvas em: {nome_arquivo_tabelas}")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{nome_arquivo_entrada}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")