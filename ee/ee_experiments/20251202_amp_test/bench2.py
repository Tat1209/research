import torch
import torch.nn as nn
import time
import statistics

# --- 設定 ---
# Transformerで一般的な設定（BERT-base相当のサイズ感）
BATCH_SIZE = 64
SEQ_LEN = 128     # シーケンス長
D_MODEL = 768     # 埋め込み次元
NHEAD = 12        # ヘッド数
NUM_LAYERS = 6    # レイヤー数
ITERATIONS = 50   # 計測回数
WARMUP = 10       # ウォームアップ回数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TF32有効化（Ampere以降のGPUで必須）
torch.set_float32_matmul_precision('high')

print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- モデル定義 ---
class TransformerBenchmark(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch標準のTransformerEncoder
        # 内部で MultiHeadAttention -> Add -> LayerNorm -> FeedForward -> Add -> LayerNorm
        # という「メモリ移動が多い」処理が行われるため、コンパイルの効果が高い
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=NHEAD, 
            dim_feedforward=D_MODEL*4, 
            dropout=0.1, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

    def forward(self, x):
        return self.encoder(x)

def run_benchmark(model, input_tensor, mode_name="Eager", use_amp=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    target = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device)
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"\n--- Starting Benchmark: {mode_name} (AMP={'ON' if use_amp else 'OFF'}) ---")

    # ウォームアップ
    print(f"Warming up...")
    for _ in range(WARMUP):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            output = model(input_tensor)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize()

    # 本計測
    timings = []
    print(f"Measuring ({ITERATIONS} steps)...")
    
    for i in range(ITERATIONS):
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        start_event.record()
        
        # Mixed Precision (AMP) コンテキスト
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            output = model(input_tensor)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    median_time = statistics.median(timings)
    print(f"[{mode_name}] Median Time: {median_time:.3f} ms")
    return median_time

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA not found. Comparison requires NVIDIA GPU.")
        return

    # データ準備
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device)

    # 1. Eager Mode (通常)
    model_eager = TransformerBenchmark().to(device)
    time_eager = run_benchmark(model_eager, x, mode_name="Eager Mode")

    # 2. Compile Mode
    # メモリリセット
    del model_eager
    torch.cuda.empty_cache()
    
    model_compile = TransformerBenchmark().to(device)
    
    print("\nCompiling model (using max-autotune)...")
    # max-autotune: Tritonを使用して最適なカーネル構成を探索する（コンパイル時間は長いが最速）
    opt_model = torch.compile(model_compile, mode="max-autotune")
    
    time_compile = run_benchmark(opt_model, x, mode_name="Compile Mode")

    # 結果表示
    speedup = time_eager / time_compile
    print("\n" + "="*40)
    print(f"Transformer Speedup: {speedup:.2f}x")
    print("="*40)
    
    if speedup < 1.1:
        print("Note: 差が小さい場合、GPU世代(Volta以前)やドライバの影響の可能性があります。")

if __name__ == "__main__":
    main()