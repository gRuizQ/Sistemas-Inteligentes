import utils
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 1. Preparação dos dados
arquivo_treino = '02_treino_sinais_vitais_com_label.txt'
X_train_s, X_test_s, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler = utils.preparar_dados(arquivo_treino)

# 2. Definindo os dicionários de modelos para testar (Diferentes algoritmos e parâmetros)
modelos_para_regressao = {
    "MLP_10_5": MLPRegressor(hidden_layer_sizes=(10, 5), solver='lbfgs', max_iter=5000, random_state=42),
    "MLP_10_5_5": MLPRegressor(hidden_layer_sizes=(10, 5, 5), solver='lbfgs', max_iter=5000, random_state=42),
    "RandomForest_Default": RandomForestRegressor(random_state=42),
    "RandomForest_100_Trees": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
}

modelos_para_classificacao = {
    "MLP_10_5": MLPClassifier(hidden_layer_sizes=(10, 5), solver='lbfgs', max_iter=5000, random_state=42),
    "MLP_10_5_5": MLPClassifier(hidden_layer_sizes=(10, 5, 5), solver='lbfgs', max_iter=5000, random_state=42),
    "RandomForest_Default": RandomForestClassifier(random_state=42),
}

# 3. Treino e Avaliação em Lote
resultados_regressao = utils.comparar_modelos(
    modelos_para_regressao, X_train_s, y_reg_train, X_test_s, y_reg_test, tarefa='regressao'
)

resultados_classificacao = utils.comparar_modelos(
    modelos_para_classificacao, X_train_s, y_clf_train, X_test_s, y_clf_test, tarefa='classificacao'
)

# 4. Geração do teste cego (Selecionando o melhor modelo manualmente após ver o console)
arquivo_cego = '01_treino_sinais_vitais_sem_label.txt'
arquivo_saida = 'resultados_predicao.csv'

# Suponha que ao analisar o console, a MLP (10, 5, 5) foi a melhor na regressão 
# e a Random Forest foi a melhor na classificação:
melhor_modelo_reg = resultados_regressao["MLP_10_5_5"]
melhor_modelo_clf = resultados_classificacao["RandomForest_Default"]

utils.gerar_arquivo_teste_cego(
    filepath_entrada=arquivo_cego,
    filepath_saida=arquivo_saida,
    scaler=scaler,
    modelo_reg=melhor_modelo_reg,
    modelo_clf=melhor_modelo_clf
)
