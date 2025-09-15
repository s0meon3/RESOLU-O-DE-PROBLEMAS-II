import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CONFIGURAÇÃO E PREPARAÇÃO DOS DADOS ---
nome_arquivo_entrada = 'enade_computacao_2019_limpo_final.csv'
SOM_X_DIM = 40
SOM_Y_DIM = 40
NUM_ITERATIONS = 2000

print("Carregando e preparando os dados...")
df = pd.read_csv(nome_arquivo_entrada, sep=';')
df = df[df['ano_in_grad'] >= 1990].copy()
df['tempo_permanencia'] = df['nu_ano_enade'] - df['ano_in_grad']
df = df[(df['tempo_permanencia'] >= 2) & (df['tempo_permanencia'] <= 15)]

y = df['tempo_permanencia']
X = df.drop(columns=['tempo_permanencia', 'nu_ano_enade', 'ano_in_grad', 'nt_ger', 'nt_fg'])
X_dummies = pd.get_dummies(X, drop_first=True)

print("Padronizando a escala das features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummies)

# --- 2. TREINAMENTO DO SOM (como antes) ---
print(f"Treinando o SOM de {SOM_X_DIM}x{SOM_Y_DIM}...")
som = MiniSom(x=SOM_X_DIM, y=SOM_Y_DIM,
              input_len=X_scaled.shape[1],
              sigma=3.0, learning_rate=0.5,
              random_seed=42)
som.random_weights_init(X_scaled)
som.train_random(data=X_scaled, num_iteration=NUM_ITERATIONS)
print("Treinamento do SOM concluído.\n")


# --- 3. ENGENHARIA DE FEATURES COM O SOM ---
print("Criando novas features a partir do SOM...")

# Coordenadas (x,y) do neurônio vencedor para cada aluno
winners = np.array([som.winner(x) for x in X_scaled])
som_x = winners[:, 0]
som_y = winners[:, 1]

# Erro de quantização para cada aluno
q_error = np.linalg.norm(som.quantization_error(X_scaled), axis=1)

# Criando um novo DataFrame com as features do SOM
df_som_features = pd.DataFrame({
    'som_x': som_x,
    'som_y': som_y,
    'erro_quantizacao': q_error
}, index=X_dummies.index) # Garantir que os índices batem

# Juntando as features originais com as novas features do SOM
X_enriquecido = pd.concat([X_dummies, df_som_features], axis=1)
print(f"Dataset enriquecido criado com {X_enriquecido.shape[1]} features.\n")


# --- 4. TREINAR E AVALIAR O NOVO RANDOM FOREST ---
print("Treinando o Random Forest com o dataset enriquecido...")
X_train, X_test, y_train, y_test = train_test_split(X_enriquecido, y, test_size=0.2, random_state=42)

modelo_enriquecido = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_enriquecido.fit(X_train, y_train)

print("Avaliando o novo modelo...")
y_pred = modelo_enriquecido.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --- 5. COMPARAÇÃO FINAL ---
print("\n" + "="*60)
print("--- RESULTADO FINAL DA ANÁLISE ---")
print("\nPerformance do Modelo Original (com features originais):")
print("MAE: ~0.98 anos | RMSE: ~1.40 anos | R²: ~31.2%")

print("\nPerformance do Novo Modelo (features originais + features do SOM):")
print(f"MAE: {mae:.2f} anos | RMSE: {rmse:.2f} anos | R²: {r2:.2%}")
print("="*60)