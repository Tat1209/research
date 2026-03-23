# PyTorch Model Profiling Metrics


### 1. MACs (Multiply-Accumulate Operations: 積和演算回数)
モデルの理論的な計算量を示す静的指標です。`fvcore` (Meta Research) による解析に基づき、テンソルの積和演算の総数を算出します。

* **MACs_Forward_Train**: 学習モード (`model.train()`) でのForwardパスの計算量。推論時にはバイパスされる演算（特定のDropout処理など）が含まれる場合があります。
* **MACs_Forward_Eval**: 推論モード (`model.eval()`) でのForwardパスの計算量。純粋な推論コストを示します。
* **MACs_Total_Train**: `NaN` (未定義)
  * **根拠**: PyTorchのAutograd機能は、Forward実行時に動的計算グラフ (Dynamic Computational Graph) を構築してBackwardを実行します。Backwardパスの演算は、グラフの動的解放処理や勾配計算の最適化手順に強く依存するため、静的解析ツールで数学的に正確な計算量を算出することは原理上不可能です。推測による不正確な値を排除するため、意図的に計算不可としています。

### 2. Throughput (スループット: サンプル処理速度)
1秒間に処理できるデータ（サンプル）の数です。`バッチサイズ / 処理時間(秒)` で算出されます。

* **Throughput_Forward_Train**: 学習時のForwardパスにおけるデータ処理速度。
* **Throughput_Forward_Eval**: 推論時のデータ処理速度。実運用環境におけるシステムのレイテンシ要件や最大処理能力を評価する主要な基準となります。
* **Throughput_Total_Train**: 学習1ステップ全体（Forward + Loss計算 + Backward）の実効処理速度。ハードウェア間での計算効率を比較するための最も標準的な客観的指標です。

### 3. Time (処理時間: ミリ秒)
実際の実行にかかった物理的な時間です。`torch.utils.benchmark` を用い、CUDAカーネルの非同期実行による誤差を排除（計測前後の `synchronize` を強制）した厳密な平均値を算出します。

* **Time_Forward_Train**: 学習モードにおけるForwardパス単体の実行時間。
* **Time_Forward_Eval**: 推論モードにおけるForwardパス単体の実行時間。
* **Time_Total_Train**: 学習1ステップ全体（Forward + Loss計算 + Backward）の実行時間。

### 4. VRAM (VRAM使用量: MB)
GPUデバイス上でテンソルに実際に割り当てられたピークメモリ量 (Allocated Memory) です。アロケータが保持するだけの予約済み (Reserved) メモリを排除し、アルゴリズムが要求する純粋なメモリフットプリントを示します。

* **VRAM_Forward_Train**: 学習モードでのForwardパス完了時点のピークメモリ。Backward計算のために動的計算グラフと中間テンソル (Activations) を保持するため、推論時より大きく増加します。
* **VRAM_Forward_Eval**: 推論モードでのピークメモリ。`torch.no_grad()` コンテキスト内で実行されるため、中間テンソルが保存されず、理論上最小のメモリ消費量となります。
* **VRAM_Total_Train**: Backwardパス完了までの全工程を含めたピークメモリ。勾配計算によってパラメータと同サイズの勾配テンソル (`.grad`) が新たに割り当てられるため、原則として `VRAM_Forward_Train` の値からさらに増加します。