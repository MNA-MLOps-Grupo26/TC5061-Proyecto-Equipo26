## P1 (selecci├│n: val_wape, menor es mejor)

| run | etapa | modelo | learning_rate | max_iter | max_leaf_nodes | alpha | val_wape | test_wape | val_mae | test_mae | val_rmse | test_rmse | fit_time_s | mejora_vs_baseline_% | brecha_val_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p1-hgb-lr0.05-it300 | candidate | HistGradientBoostingRegressor | 0.05 | 300 | 31 |  | 0.1781 | 0.2158 | 4.8514 | 7.4770 | 8.1715 | 13.6736 | 2.9279 | 12.9669 | 0.0376 |
| p1-hgb-lr0.07-it350 | search | HistGradientBoostingRegressor | 0.07 | 350 | 31 |  | 0.1791 | 0.2170 | 4.8780 | 7.5190 | 8.2563 | 13.7208 | 5.7456 | 12.4885 | 0.0379 |
| p1-hgb-lr0.03-it400 | search | HistGradientBoostingRegressor | 0.03 | 400 | 31 |  | 0.1805 | 0.2163 | 4.9152 | 7.4962 | 8.3256 | 13.7309 | 4.3089 | 11.8214 | 0.0358 |
| p1-hgb-lr0.1-it200 | search | HistGradientBoostingRegressor | 0.1 | 200 | 31 |  | 0.1813 | 0.2189 | 4.9369 | 7.5844 | 8.4843 | 13.7795 | 3.5689 | 11.4327 | 0.0376 |
| p1-baseline-promedio-semanal | baseline | BaselinePromedioSemanal |  |  |  |  | 0.2047 | 0.2493 | 5.5741 | 8.6392 | 10.1199 | 15.4204 |  | -0.0000 | 0.0446 |
| p1-glm-alpha1.0 | candidate | PoissonRegressor |  | 1000 |  | 1.0 | 0.4911 | 0.4933 | 13.3739 | 17.0955 | 30.2068 | 53.4765 | 0.7550 | -139.9271 | 0.0022 |

## P2 (selecci├│n: val_recall_at_30, mayor es mejor)

| run | etapa | modelo | learning_rate | max_iter | max_leaf_nodes | C | val_recall_at_30 | test_recall_at_30 | val_precision_at_30 | test_precision_at_30 | val_f1_macro | test_f1_macro | fit_time_s | mejora_vs_baseline_% | brecha_val_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p2-hgb-lr0.1-leaf31 | search | HistGradientBoostingClassifier | 0.1 | 300 | 31 |  | 0.2217 | 0.1930 | 0.9108 | 0.9046 | 0.7206 | 0.7128 | 17.1709 | 5.5638 | -0.0287 |
| p2-hgb-lr0.05-leaf63 | search | HistGradientBoostingClassifier | 0.05 | 300 | 63 |  | 0.2205 | 0.1956 | 0.9075 | 0.9149 | 0.7254 | 0.7150 | 24.1028 | 4.9851 | -0.0248 |
| p2-hgb-lr0.05-leaf31 | candidate | HistGradientBoostingClassifier | 0.05 | 300 | 31 |  | 0.2204 | 0.1962 | 0.9065 | 0.9161 | 0.7256 | 0.7145 | 19.1706 | 4.9702 | -0.0243 |
| p2-hgb-lr0.03-leaf15 | search | HistGradientBoostingClassifier | 0.03 | 300 | 15 |  | 0.2193 | 0.1951 | 0.9043 | 0.9103 | 0.7277 | 0.7167 | 22.4265 | 4.4257 | -0.0242 |
| p2-logreg-C5.0 | search | LogisticRegression |  | 2000 |  | 5.0 | 0.2145 | 0.1918 | 0.8882 | 0.8977 | 0.7227 | 0.7132 | 2.6707 | 2.1605 | -0.0228 |
| p2-logreg-C0.5 | search | LogisticRegression |  | 2000 |  | 0.5 | 0.2145 | 0.1918 | 0.8882 | 0.8977 | 0.7226 | 0.7134 | 2.1950 | 2.1605 | -0.0228 |
| p2-logreg-C0.1 | search | LogisticRegression |  | 2000 |  | 0.1 | 0.2145 | 0.1920 | 0.8882 | 0.8989 | 0.7227 | 0.7137 | 2.2772 | 2.1605 | -0.0225 |
| p2-logreg-C1.0 | candidate | LogisticRegression |  | 2000 |  | 1.0 | 0.2145 | 0.1920 | 0.8882 | 0.8989 | 0.7226 | 0.7132 | 1.6382 | 2.1605 | -0.0226 |
| p2-baseline-semana-anterior | baseline | BaselineSemanaAnterior |  |  |  |  | 0.2100 | 0.1891 | 0.8753 | 0.8885 | 0.6542 | 0.6390 |  | 0.0000 | -0.0209 |

## P3 (selecci├│n: val_ari_vs_train, mayor es mejor)

| run | etapa | modelo | max_iter | k | val_ari_vs_train | test_ari_vs_train | val_fraccion_estable | test_fraccion_estable | val_silhouette_asignacion_fija | test_silhouette_asignacion_fija | train_silhouette | fit_time_s | mejora_vs_baseline_% | brecha_val_test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p3-kmeans-k2-historia180 | search | KMeans | 300 | 2 | 0.6898 | 0.6548 | 0.9155 | 0.9049 | 0.0501 | 0.0486 | 0.1323 | 0.0998 | 0.7194 | -0.0350 |
| p3-kmeans-k3-historia180 | baseline | KMeans | 300 | 3 | 0.6849 | 0.6163 | 0.8965 | 0.8750 | 0.0489 | 0.0533 | 0.1433 | 0.3108 | 0.0000 | -0.0686 |
| p3-kmeans-k4-historia180 | search | KMeans | 300 | 4 | 0.6117 | 0.6375 | 0.8420 | 0.8587 | -0.0062 | 0.0026 | 0.0792 | 0.0353 | -10.6805 | 0.0257 |
| p3-kmeans-k5-historia180 | search | KMeans | 300 | 5 | 0.5965 | 0.5437 | 0.8229 | 0.7962 | -0.0175 | -0.0095 | 0.0672 | 0.0983 | -12.9112 | -0.0528 |
