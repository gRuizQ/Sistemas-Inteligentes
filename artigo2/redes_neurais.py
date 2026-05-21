import os

import utils
from sklearn.neural_network import MLPRegressor, MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = os.path.dirname(__file__)

arquivo_treino = os.path.join(BASE_DIR, '02_treino_sinais_vitais_com_label.txt')
X_train_s, X_test_s, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler = utils.preparar_dados(arquivo_treino)

# 2. Instanciando os modelos
mlp_reg_model = MLPRegressor(hidden_layer_sizes=(10, 5, 5), solver='lbfgs', max_iter=5000, random_state=42)
mlp_clf_model = MLPClassifier(hidden_layer_sizes=(10, 5, 5), solver='lbfgs', max_iter=5000, random_state=42)
# Instanciando os modelos
mlp_reg_model = MLPRegressor(hidden_layer_sizes=(9, 9, 7), tol=0.001, solver='lbfgs', max_iter=2000, learning_rate_init=0.01, random_state=42, activation="logistic")
mlp_clf_model = MLPClassifier(hidden_layer_sizes=(8, 6), tol=0.001, solver='lbfgs', max_iter=15000, learning_rate_init=0.01, random_state=42, activation="tanh")

# 3. Treino e Avaliação
# Treino e Avaliação
modelo_reg_treinado = utils.treinar_e_avaliar_regressao(
    mlp_reg_model, X_train_s, y_reg_train, X_test_s, y_reg_test, nome_modelo="MLP"
)

modelo_clf_treinado = utils.treinar_e_avaliar_classificacao(
    mlp_clf_model, X_train_s, y_clf_train, X_test_s, y_clf_test, nome_modelo="MLP"
)

# 4. Geração do teste cego
# Gerando as predições nos dados de teste para alimentar os gráficos
predicoes_reg_teste = modelo_reg_treinado.predict(X_test_s)
predicoes_clf_teste = modelo_clf_treinado.predict(X_test_s)

# Chamando as funções de plotagem do utils
utils.plotar_analise_regressao(y_reg_test, predicoes_reg_teste, nome_modelo="MLP Regressão")
utils.plotar_analise_classificacao(y_clf_test, predicoes_clf_teste, nome_modelo="MLP Classificação")

# Geração do teste cego
arquivo_cego = os.path.join(BASE_DIR, '01_treino_sinais_vitais_sem_label.txt')
arquivo_saida = os.path.join(BASE_DIR, 'resultados_predicao.csv')

utils.gerar_arquivo_teste_cego(
    filepath_entrada=arquivo_cego,
    filepath_saida=arquivo_saida,
    scaler=scaler,
    modelo_reg=modelo_reg_treinado,
    modelo_clf=modelo_clf_treinado
)