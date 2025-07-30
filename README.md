# SMBC Electricity Price Forecast (Signate, 2025/06)

## 📁 Repository Structure

📂 notebooks/ # 実験用 Jupyter Notebook を置く
📂 src/ # 再利用 Python スクリプト（data.py, train.py など）
.gitignore # キャッシュ/チェックポイント除外設定
competition_daily.md # 日々の進捗メモ（ラフな “今日やったこと” ログ）
competition-index.md # ポートフォリオ目次（他コンペへのリンク集）
review_competition_daily.md # コンペ終了後の学び・振り返りまとめ
README.md # リポジトリ概要（このファイル）


## 1. Result
| Metric | Public LB | Private LB | Rank |
|--------|-----------|------------|------|
| RMSE   | **8.596441007544344**Discussionコピペ  | **8.498687536995755**   | 92 / 2,082 |
| RMSE   | **10.79338205924817**自分の力  | None | None |


## 2. Task & Data
- **Goal**: 2018年の電力価格予測  
- **Period**: 2015-01-01 ～ 2017-12-31（train） / 2018-01-01～ 2018-12-31（test）  
- **Features**: 発電実績データ (generation ...)	時刻から24時間前のバイオマス、各種化石燃料（褐炭、天然ガス、石炭、石油など）、原子力、水力（揚水消費、自流式、貯水池式）、太陽光、風力（陸上・洋上）といった多様な電源別の詳細な発電実績量。　　
総電力需要実績 (total load actual)	時刻から24時間前の実際の電力需要量。　　
電力価格実績 (price actual：目的変数)	
時刻と同時刻の実際の電力価格（目的変数）。電力価格は、主に需要と供給のバランスによって決定される。　　
気象データ	時刻と同時刻のスペインの主要5都市（バレンシア、マドリード、ビルバオ、バルセロナ、セビリア）における詳細な気象情報。　　
具体的には、気温（現在・最低・最高）、気圧、湿度、風速、風向、降雨量（過去1時間・3時間）、降雪量（過去3時間）、雲量、および天候を示す各種記述データなどが含まれます。

## 3. Approach
1. 需要・発電量・気象・時刻情報から多様な特徴量（ラグ特徴量、sin/cos時系列、祝日/連休フラグ等）を作成
2. XGBoost, LightGBM, CatBoost, SVM, GDBT等の回帰モデルを活用し、アンサンブルも実施
3. バリデーションは「直近Q4 holdout」や「冬/非冬分割」など時系列・季節性を考慮して設計
4. SHAPで特徴量寄与度を可視化し、重要特徴量を厳選

## 4. Reproduce
TBD