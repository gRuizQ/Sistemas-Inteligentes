import utils
from sklearn.neural_network import MLPRegressor, MLPClassifier

arquivo_treino = '02_treino_sinais_vitais_com_label.txt'
X_train_s, X_test_s, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler = utils.preparar_dados(arquivo_treino)

# 2. Instanciando os modelos
mlp_reg_model = MLPRegressor(hidden_layer_sizes=(10, 5, 5), solver='lbfgs', max_iter=5000, random_state=42)
mlp_clf_model = MLPClassifier(hidden_layer_sizes=(10, 5, 5), solver='lbfgs', max_iter=5000, random_state=42)

# 3. Treino e Avaliação
modelo_reg_treinado = utils.treinar_e_avaliar_regressao(
    mlp_reg_model, X_train_s, y_reg_train, X_test_s, y_reg_test, nome_modelo="MLP"
)

modelo_clf_treinado = utils.treinar_e_avaliar_classificacao(
    mlp_clf_model, X_train_s, y_clf_train, X_test_s, y_clf_test, nome_modelo="MLP"
)

# 4. Geração do teste cego
arquivo_cego = '01_treino_sinais_vitais_sem_label.txt'
arquivo_saida = 'resultados_predicao.csv'

utils.gerar_arquivo_teste_cego(
    filepath_entrada=arquivo_cego,
    filepath_saida=arquivo_saida,
    scaler=scaler,
    modelo_reg=modelo_reg_treinado,
    modelo_clf=modelo_clf_treinado
)
