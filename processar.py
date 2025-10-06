import pandas as pd

nome_arquivo_entrada = 'SoU_enade_2019.csv'
nome_arquivo_saida = 'enade_computacao_2019_limpo_final.csv'

cursos_para_filtrar = [
    40, 72, 79, 4003, 4004, 4005, 4006, 4007, 5809, 
    6409, 5811, 5813, 5814, 5815
]
grau_academico_para_filtrar = [1, 2, 3, 4]
in_gratuito_validos = [0, 1]
regiao_curso_validas = [1, 2, 3, 4, 5]
tp_pres_presente = 555

colunas_para_selecionar = [
    'nu_ano_enade', 'in_capital_curso', 'tp_categoria_administrativa_ies','faixa_etaria', 'tp_sexo', 'ano_in_grad',
    'nt_ger', 'nt_fg', 'sigla_uf_curso', 'co_regiao_curso',
    'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
    'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20',
    'q21', 'q22', 'q23', 'q24', 'q25', 'q26', 'q27', 'q28', 'q29', 'q30',
    'q31', 'q32', 'q33', 'q34', 'q35', 'q36', 'q37', 'q38', 'q39', 'q40',
    'q41', 'q42', 'q43', 'q44', 'q45', 'q46', 'q47', 'q48', 'q49', 'q50',
    'q51', 'q52', 'q53', 'q54', 'q55', 'q56', 'q57', 'q58', 'q59', 'q60',
    'q61', 'q62', 'q63', 'q64', 'q65', 'q66', 'q67', 'q68'
]

valores_para_apagar = ['Ausente', 'SEM INFORMACAO']

try:
    print(f"Carregando o arquivo '{nome_arquivo_entrada}'...")
    df = pd.read_csv(nome_arquivo_entrada, sep=';', encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.lower()
    print(f"Arquivo carregado. Shape original: {df.shape}\n")


    print("Iniciando a filtragem...")
    df_filtrado = df[
        (df['co_grupo'].isin(cursos_para_filtrar)) &
        (df['tp_grau_academico'].isin(grau_academico_para_filtrar)) &
        (df['in_gratuito'].isin(in_gratuito_validos)) &
        (df['co_regiao_curso'].isin(regiao_curso_validas)) &
        (df['tp_pres'] == tp_pres_presente)
    ]
    print(f"Shape após a filtragem: {df_filtrado.shape}\n")

    print("Selecionando colunas...")
    df_selecionado = df_filtrado[colunas_para_selecionar]
    print(f"Shape após a seleção de colunas: {df_selecionado.shape}\n")

    print(f"Limpando linhas com os valores: {valores_para_apagar}...")
    mascara_remocao = df_selecionado.isin(valores_para_apagar).any(axis=1)
    df_limpo_strings = df_selecionado[~mascara_remocao].copy()
    print(f"Shape após limpeza de strings: {df_limpo_strings.shape}")
    
    print("Removendo linhas com células vazias...")
    df_final = df_limpo_strings.dropna(how='any')
    print(f"Shape final do DataFrame após remover vazios: {df_final.shape}\n")

    print("--- AMOSTRA DO RESULTADO ---")
    print("Amostra do DataFrame final e limpo:")
    print(df_final.head())
    print(f"\nResumo: De {df.shape[0]} linhas originais, o DataFrame final possui {df_final.shape[0]} linhas.")

    if not df_final.empty:
        print(f"\nSalvando o DataFrame limpo em '{nome_arquivo_saida}'...")
        df_final.to_csv(nome_arquivo_saida, sep=';', encoding='utf-8-sig', index=False)
        print("Arquivo salvo com sucesso!")
    else:
        print("\nO DataFrame final está vazio. Nenhum arquivo foi salvo.")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{nome_arquivo_entrada}' não foi encontrado.")
except KeyError as e:
    print(f"ERRO: A coluna {e} não foi encontrada no arquivo CSV.")