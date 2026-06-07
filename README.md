# Smart-Building-Energy-Forecasting
# Adaptive Time-Series Forecasting for Smart Building Energy Consumption

An end-to-end predictive machine learning pipeline designed to forecast hourly energy consumption loads under non-stationary conditions. This framework targets data drift by implementing structural change-point detection to isolate operational regimes, building localized models that outperform rigid global neural architectures.

## 🚀 Key Engineering Highlights
* **Dynamic Regime Segmentation:** Integrated the `ruptures` library to run structural change-point detection, identifying latent operational shifts caused by institutional schedules and environmental metrics.
* **Leakage-Insulated Validation:** Implemented a robust forward-chaining evaluation framework. Feature scaling boundaries (`StandardScaler`) are calculated strictly inside localized training windows, preventing forward-looking data leakage.
* **Architecture Pragmatism:** Conducted empirical trade-off evaluations proving that compact, optimized Dense ANNs matched deep sequential LSTMs in predictive accuracy given sample constraints, drastically reducing training latency and compute overhead.

## 📊 Core Performance Metrics
* **Predictive Accuracy:** The adaptive pipeline achieved a real-world Mean Absolute Error (MAE) of ~6.x (representing a tight ~7.5% average variance against the baseline operational range of 53–99). This corresponds to an optimized latent-space MAE of 0.37 when evaluated on normalized scales during localized window training.

## 🛠️ Stack & Dependencies
* **Time-Series Analysis:** `ruptures` (Change-point detection)
* **Deep Learning Frameworks:** `TensorFlow` / `Keras`
* **Core Data Science:** `Python`, `Pandas`, `NumPy`, `Scikit-Learn`, `Matplotlib`
