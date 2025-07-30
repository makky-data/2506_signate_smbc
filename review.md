## 1. TL;DR（140字以内）

特徴量エンジニアリングとアンサンブル（GDBT, SVM, CatBoost, XGBoost, LGBM）を試行。
最良Public LBは10.85（最終順位は要確認）。ラグ特徴量・冬モデル分割・逐次予測が大きく寄与。

---

## 2. Experiment Timeline（時系列表）
| 日付      | 仮説 / 変更点                                   | Public LB   | 気づき・メモ                                 |
|-----------|------------------------------------------------|-------------|----------------------------------------------|
| 2025/06/14| ベースライン構築・特徴量仮説                    | 16.09       | 時系列分割重要、外挿注意                     |
| 2025/06/15| EDA・特徴量追加・バリデーション改善             | 15.03       | 時間帯特徴量が有効、週末/季節は微妙          |
| 2025/06/16| Q4直近バリデーション・Seed Averaging            | 14.12       | 過学習傾向、直近Q4バリデ有効                 |
| 2025/06/17| 冬モデル分割・祝日特徴量                        | 11台        | 冬特化で大幅改善、OOFとLBの差縮小            |
| 2025/06/18| Adversarial Validation・重み付け                | 14          | drift大、重み付け効果薄                     |
| 2025/06/19| 特徴量追加・バリデパターン比較                  | 13.22       | Q4 holdoutが最もLBに近い                    |
| 2025/06/20| 全特徴量実装・パラメータ調整                    | 13.23       | generation_sum等は効果薄                    |
| 2025/06/21| 他人手法再現・期間カット                        | 13台        | 2015/8-2016/3カットで微改善                 |
| 2025/06/22| PCA+SVR提出                                    | 8.5         | 他人手法で大幅改善（詳細要追試）            |
| 2025/06/23| ラグ特徴量・逐次更新                            | 11台→24     | ラグ特徴量が劇的に効く、逐次更新でLB11台     |
| 2025/06/24| log1p変換・ラグ推測                             | 12.85       | log1pはVal良いがLB悪化、ラグ推測は有効       |
| 2025/06/25| SHAPで特徴量選定・アンサンブル                  | 12.62       | SHAPで寄与度確認、アンサンブルは要工夫      |
| 2025/06/26| 2015/8-2016/3除外・アンサンブル                 | 10.85       | 除外+アンサンブルでベストスコア             |
| 2025/06/27| sin/cos特徴量追加                               | -           | わずかにスコアアップ                        |
| 2025/06/29| GDBT+SVMアンサンブル                            | -           | 最終アンサンブル試行                        |

---

## 3. Key Learnings（3〜5 個）

- **ラグ特徴量（特に逐次更新）が最も効果的**  
  → price_actual_lag1, lag24, lag168 など。逐次更新でリーク防止しつつ精度向上。

- **冬モデル分割で季節性のギャップを吸収**  
  → 冬だけRMSEが跳ねる問題を分割で解消。OOFとLBの差が縮小。

- **特徴量の選択はSHAPで可視化・検証**  
  → 上位30特徴量で再学習、寄与度の高いものを厳選。

- **log1p変換はValで効くがLBで悪化する場合あり**  
  → 分布の歪み補正には有効だが、LBとのギャップに注意。

- **アンサンブルは単純加重平均より逐次予測型が有効**  
  → モデルごとの出力を工夫して組み合わせる必要あり。

---

## 4. Reusable Tips / Code Snippets

- **ラグ特徴量生成（逐次更新）**
  ```python
  for lag in [1, 24, 168]:
      df[f'price_actual_lag{lag}'] = df['price_actual'].shift(lag)
  # 逐次予測時はtestにも逐次的に値を入れる
  ```

- **祝日・連休特徴量**
  ```python
  import holidays
  es_holidays = holidays.Spain(years=[2015, 2016, 2017, 2018])
  df["is_holiday"] = df["time"].dt.date.map(lambda d: 1 if d in es_holidays else 0)
  for s in [-1, 1]:
      df[f"hol_adj{s}"] = df["is_holiday"].shift(s).fillna(0)
  df["is_long_wend"] = (df["is_holiday"].rolling(3, min_periods=1).sum() >= 2).astype(int)
  ```

- **SHAPによる特徴量寄与度可視化**
  ```python
  import shap
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X)
  shap.summary_plot(shap_values, X)
  ```

---

## 5. Next Actions（優先度付き TODO）

1. **アンサンブル手法の最適化**  
   - 逐次予測型アンサンブルの安定化・自動化
2. **特徴量の自動選択・生成パイプライン構築**  
   - SHAPやPermutation Importanceを組み込む
3. **外挿リスクの低減**  
   - テスト分布とのdrift検知・補正
4. **他人手法（PCA+SVR等）の再現・検証**
5. **バリデーション戦略のさらなる最適化**  
   - Q4 holdout以外のfold設計も検討

---

## 6. Skill-Building Plan

| スキル                | 現状レベル | 習得ステップ                           | リソース                        |
|-----------------------|------------|----------------------------------------|---------------------------------|
| 特徴量エンジニアリング| 中級       | SHAP/Permutationで寄与度分析           | kaggle, SHAP公式, fast.ai       |
| アンサンブル設計      | 初級       | 逐次予測型・stackingの実装練習         | kaggle, ensemble入門記事        |
| drift検知・補正       | 初級       | adversarial validationの自動化         | kaggle, Qiita, Zenn             |
| バリデーション設計    | 中級       | 時系列CV・季節分割の実装・評価         | kaggle, sklearn, techブログ     |
| モデル解釈            | 初級       | SHAP, LIMEの可視化・解釈               | SHAP公式, LIME公式, kaggle      |
