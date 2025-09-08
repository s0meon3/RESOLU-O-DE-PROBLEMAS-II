import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURAÇÃO ---
nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'

# Dicionário para traduzir os códigos de faixa etária
FAIXA_ETARIA_MAP = {
    1: 'Até 17 anos',
    2: '18 a 21 anos',
    3: '22 a 25 anos',
    4: '26 a 29 anos',
    5: '30 a 33 anos',
    6: '34 a 37 anos',
    7: '38 a 41 anos',
    8: '42 a 45 anos',
    9: '46 a 49 anos',
    10: '50 a 53 anos',
    11: '54 a 57 anos',
    12: '58 a 61 anos',
    13: '62 a 65 anos',
    14: '66 anos ou mais'
}

# --- 2. EXECUÇÃO DA ANÁLISE ---
try:
    print("Carregando o arquivo de dados limpo...")
    df = pd.read_csv(nome_arquivo_entrada, sep=';')

    # a) Recalcular a variável 'tempo_permanencia' para a análise
    df = df[df['ano_in_grad'] >= 1990].copy()
    df['tempo_permanencia'] = df['nu_ano_enade'] - df['ano_in_grad']
    df = df[(df['tempo_permanencia'] >= 2) & (df['tempo_permanencia'] <= 15)]
    
    # b) Agrupar por faixa etária e calcular a média e a contagem
    print("Analisando o tempo de permanência por faixa etária...")
    analise_idade = df.groupby('faixa_etaria')['tempo_permanencia'].agg(['mean', 'count']).reset_index()

    # c) Mapear os códigos para descrições legíveis e renomear colunas
    analise_idade['faixa_etaria_desc'] = analise_idade['faixa_etaria'].map(FAIXA_ETARIA_MAP).fillna('Desconhecida')
    analise_idade = analise_idade.rename(columns={
        'mean': 'tempo_medio_permanencia_anos',
        'count': 'numero_de_alunos'
    })
    
    # Reordenar colunas para melhor visualização
    analise_idade = analise_idade[['faixa_etaria', 'faixa_etaria_desc', 'numero_de_alunos', 'tempo_medio_permanencia_anos']]
    
    # --- 3. EXIBIÇÃO DOS RESULTADOS ---
    print("\n--- Tabela: Tempo Médio de Permanência por Faixa Etária ---")
    print(analise_idade.to_string(index=False))

    # --- 4. VISUALIZAÇÃO GRÁFICA ---
    print("\nGerando gráfico de barras...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    barplot = sns.barplot(
        x='faixa_etaria_desc',
        y='tempo_medio_permanencia_anos',
        data=analise_idade.sort_values('faixa_etaria'),
        palette='viridis'
    )
    
    plt.title('Tempo Médio de Permanência em Cursos de Computação por Faixa Etária', fontsize=16, pad=20)
    plt.xlabel('Faixa Etária do Aluno', fontsize=12)
    plt.ylabel('Tempo Médio de Permanência (em anos)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout() # Ajusta o layout para evitar que os rótulos se sobreponham
    
    # Adicionar os valores no topo das barras
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.2f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')

    plt.show()


except FileNotFoundError:
    print(f"ERRO: O arquivo '{nome_arquivo_entrada}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")