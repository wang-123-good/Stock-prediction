import numpy as np
import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")
import builtins
from tensorflow.keras import backend as K
import tensorflow as tf

K.clear_session()

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

for typename in ['list', 'dict', 'set', 'sum', 'input', 'print']:
    if not isinstance(getattr(builtins, typename), type(eval(typename))):
        raise TypeError(f" ...`")

print("...")

# ========== 路径配置 ==========
data_dir = './processed_data'
os.makedirs(data_dir, exist_ok=True)
os.makedirs('train_results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ========== 加载数据 ==========
X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
scaler = joblib.load(os.path.join(data_dir, 'scaler.save'))

# ========== 超参数边界定义 ==========
param_bounds = {
    'units': (128, 256),
    'dropout': (0.05, 0.3),
    'batch_size': (16, 128),
    'learning_rate': (1e-4, 1e-2)
}

def decode_individual(ind):
    return {
        'units': int(np.clip(round(ind[0]), *param_bounds['units'])),
        'dropout': float(np.clip(ind[1], *param_bounds['dropout'])),
        'batch_size': int(np.clip(round(ind[2]), *param_bounds['batch_size'])),
        'learning_rate': float(np.clip(ind[3], *param_bounds['learning_rate']))
    }

def evaluate_lstm(individual, X_train, y_train):
    params = decode_individual(individual)
    K.clear_session()

    model = Sequential([
        LSTM(params['units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(params['dropout']),
        LSTM(params['units']),
        Dropout(params['dropout']),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')


    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=params['batch_size'],
        validation_split=0.1,
        verbose=0,
        callbacks=[early_stopping] 
    )

    val_loss = min(history.history['val_loss'])

   
    y_val_pred = model.predict(X_train).flatten()
    y_val_pred_inv = y_val_pred * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
    y_val_inv = y_train * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]

    r2 = r2_score(y_val_inv, y_val_pred_inv)
    r2 = max(r2, 1e-6)

    epoch_ratio = len(history.history['val_loss']) / 30
    w_loss = 0.7 - 0.4 * epoch_ratio
    w_r2 = 1 - w_loss

    score = w_loss * val_loss + w_r2 * (1 - r2)
    return score

def initialize_population(size):
    return [[random.uniform(*param_bounds[k]) for k in param_bounds] for _ in range(size)]

def adaptive_k_range(generation, max_gen, base_range=(2, 4), final_range=(4, 5)):

    alpha = generation / max_gen
    k_min = int(round((1 - alpha) * base_range[0] + alpha * final_range[0]))
    k_max = int(round((1 - alpha) * base_range[1] + alpha * final_range[1]))
    # 确保 k_min ≤ k_max
    k_min = max(base_range[0], min(k_min, k_max))
    return (k_min, k_max)

def tournament_selection(pop, fitness, k_range=(2, 5)):
    selected = []
    for _ in range(len(pop)):
        k = random.randint(*k_range)
        idx = random.sample(range(len(pop)), k)
        best_idx = min(idx, key=lambda i: fitness[i])
        selected.append(pop[best_idx])
    return selected

def crossover(p1, p2):
    return [(v1 + v2) / 2 for v1, v2 in zip(p1, p2)]

def mutation(ind, generation, max_gen):
    rate = 0.5 - 0.4 * (generation / max_gen) 
    return [
        val + np.random.normal(0, rate * (param_bounds[k][1] - param_bounds[k][0])) if random.random() < rate else val
        for val, k in zip(ind, param_bounds)
    ]

def woa_update(pop, best, a):
    new_pop = []
    for ind in pop:
        r = random.random()
        A = 2 * a * r - a
        C = 2 * random.random()
        D = [abs(C * b - x) for b, x in zip(best, ind)]
        l = random.uniform(-1, 1)
        p = random.random()
        if p < 0.33:
            new_ind = [b - A * d for b, d in zip(best, D)]
        elif p < 0.66:
            new_ind = [d * np.exp(1 * l) * np.cos(2 * np.pi * l) + b for d, b in zip(D, best)]
        else:
            rand_ind = random.choice(pop)
            D_rand = [abs(C * r - x) for r, x in zip(rand_ind, ind)]
            new_ind = [r - A * d for r, d in zip(rand_ind, D_rand)]
        new_pop.append([
            np.clip(x, *param_bounds[k]) for x, k in zip(new_ind, param_bounds)
        ])
    return new_pop

def ga_woa_optimize(X_train, y_train, pop_size=20, generations=15):
    pop = initialize_population(pop_size)
    fitness = [evaluate_lstm(ind, X_train, y_train) for ind in pop]
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    best_loss = fitness[best_idx]
    all_losses = [best_loss]

    for g in range(generations):
        print(f"Generation {g+1} | Best Score: {best_loss:.4f}")
        k_range = adaptive_k_range(g, generations)
        selected = tournament_selection(pop, fitness, k_range=k_range)
        next_gen = []
        for i in range(0, len(selected) - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            child = mutation(crossover(p1, p2), g, generations)
            next_gen.append(child)
        if len(selected) % 2 != 0:
            child = mutation(selected[-1], g, generations)
            next_gen.append(child)

        next_gen[0] = best.copy() 
        a = 2 - g * (2 / generations)
        fusion_ratio = 0.3 + 0.4 * (g / generations)
        ga_part = int(pop_size * (1 - fusion_ratio))
        woa_part = pop_size - ga_part

        woa_gen = woa_update(next_gen, best, a)
        combined_pop = next_gen[:ga_part] + woa_gen[:woa_part]

        fitness = [evaluate_lstm(ind, X_train, y_train) for ind in combined_pop]
        pop = combined_pop

        current_best_idx = np.argmin(fitness)
        current_best_loss = fitness[current_best_idx]
        if current_best_loss < best_loss:
            best = pop[current_best_idx]
            best_loss = current_best_loss
        all_losses.append(best_loss)

    plt.figure()
    plt.plot(all_losses, marker='o')
    plt.title("GA-WOA Multi-Objective Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Score (Lower is Better)")
    plt.tight_layout()
    plt.savefig("figures/GA_WOA_multiobj_convergence_curve.png")
    plt.close()

    print("Best Hyperparameters:", decode_individual(best))
    return decode_individual(best)


best_params = ga_woa_optimize(X_train, y_train)

K.clear_session()
model = Sequential([
    LSTM(best_params['units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(best_params['dropout']),
    LSTM(best_params['units']),
    Dropout(best_params['dropout']),
    Dense(1)
])
optimizer = Adam(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=80,
    batch_size=best_params['batch_size'],
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)


pd.DataFrame({
    'units': [best_params['units']],
    'dropout': [best_params['dropout']],
    'batch_size': [best_params['batch_size']],
    'learning_rate': [best_params['learning_rate']]
}).to_csv('train_results/GA_WOA_LSTM_best_params.csv', index=False)


plt.figure(figsize=(16, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('GA-WOA-LSTM Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('figures/GA_WOA_LSTM_training_loss.png')
plt.show()

loss_data = pd.DataFrame({
    'epoch': range(1, len(history.history['loss']) + 1),
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})
loss_data.to_csv('train_results/GA_WOA_LSTM_loss_curve.csv', index=False)


y_train_pred = model.predict(X_train).flatten()
y_train_pred_inv = y_train_pred * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
y_train_inv = y_train * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]

train_mae = mean_absolute_error(y_train_inv, y_train_pred_inv)
train_mape = np.mean(np.abs((y_train_inv - y_train_pred_inv) / y_train_inv)) * 100
train_rmse = np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))
train_r2 = r2_score(y_train_inv, y_train_pred_inv)

print("\n训练集评估结果：")
print(f"Train MAE  = {train_mae:.4f}")
print(f"Train MAPE = {train_mape:.2f}%")
print(f"Train RMSE = {train_rmse:.4f}")
print(f"Train R²   = {train_r2:.4f}")

y_pred = model.predict(X_test).flatten()
y_pred_inv = y_pred * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
y_test_inv = y_test * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]

mae = mean_absolute_error(y_test_inv, y_pred_inv)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print("\n测试集评估结果：")
print(f"Test MAE  = {mae:.4f}")
print(f"Test MAPE = {mape:.2f}%")
print(f"Test RMSE = {rmse:.4f}")
print(f"Test R²   = {r2:.4f}")


plt.figure(figsize=(16, 6))
plt.plot(y_train_inv, label='Actual Train',color='green')
plt.plot(y_train_pred_inv, label='Predicted Train',color='purple')
plt.title('GA-WOA-LSTM Train Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('figures/GA_WOA_LSTM_train_fit_plot.png')
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(y_test_inv, label='Actual Test', color='blue',)
plt.plot(y_pred_inv, label='Predicted Test', color='red')
plt.legend()
plt.title('GA-WOA-LSTM Test Prediction')
plt.tight_layout()
plt.savefig('figures/GA_WOA_LSTM_test_fit_plot.png')
plt.show()


pd.DataFrame({
    'Actual_Close_Train': y_train_inv,
    'Predicted_Close_Train': y_train_pred_inv
}).to_csv('train_results/GA_WOA_LSTM_train_prediction.csv', index=False)

pd.DataFrame({
    'Actual_Close': y_test_inv,
    'Predicted_Close': y_pred_inv
}).to_csv('train_results/GA_WOA_LSTM_test_prediction.csv', index=False)

pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'MAE': [train_mae, mae],
    'MAPE (%)': [train_mape, mape],
    'RMSE': [train_rmse, rmse],
    'R2': [train_r2, r2]
}).to_csv('train_results/GA_WOA_LSTM_evaluation_metrics.csv', index=False)

model.save('train_results/GA_WOA_LSTM_model.h5')
np.save("train_results/y_pred_train_GA_WOA_LSTM.npy", y_train_pred_inv)
np.save("train_results/y_train_GA_WOA_LSTM.npy", y_train_inv)
np.save("train_results/y_pred_GA_WOA_LSTM.npy", y_pred_inv)
np.save("train_results/y_test_GA_WOA_LSTM.npy", y_test_inv)
