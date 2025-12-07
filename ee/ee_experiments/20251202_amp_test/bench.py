import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time

# --- 設定 ---
BATCH_SIZE = 128
NUM_ITER = 100
WARMUP = 10
DEVICE = "cuda"

def run_benchmark(label, precision_mode, enable_amp, amp_dtype=None):
    """
    precision_mode: 'highest' (FP32), 'high' (TF32), or 'medium'
    """
    print(f"--- Running: {label} ---")
    
    # 【変更点】 新しいAPIで精度を設定
    # これによりWarningが消え、TF32の挙動が明示的に制御されます
    torch.set_float32_matmul_precision(precision_mode)
    
    # Note: CuDNNに関しては現状 allow_tf32 フラグがまだ一般的ですが、
    # 行列演算(matmul)のprecision設定が学習全体の速度に大きく寄与します。
    # 念のためCuDNNも連動させたい場合は以下を残しますが、Warningが出る場合は
    # 上記の set_float32_matmul_precision だけで十分効果があります。
    if precision_mode == 'high':
        torch.backends.cudnn.allow_tf32 = True 
    else:
        torch.backends.cudnn.allow_tf32 = False

    # モデルとデータの準備
    model = models.resnet50().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)
    targets = torch.randint(0, 1000, (BATCH_SIZE,), device=DEVICE)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # ウォームアップ
    model.train()
    for _ in range(WARMUP):
        optimizer.zero_grad()
        if enable_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    
    # 本番計測
    start_event.record()
    for _ in range(NUM_ITER):
        optimizer.zero_grad()
        if enable_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        optimizer.step()
        
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / NUM_ITER
    print(f"Average time per batch: {avg_time_ms:.2f} ms")
    if 'baseline_time' in globals():
        print(f"Speedup vs Baseline: x{baseline_time / avg_time_ms:.2f}")
    print("-" * 30)
    
    return avg_time_ms

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not found")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Baseline: FP32 (TF32 OFF -> 'highest')
    baseline_time = run_benchmark(
        label="1. Default (FP32, TF32=OFF)", 
        precision_mode='highest',  # ここを変更
        enable_amp=False
    )
    
    # 2. TF32 Only (TF32 ON -> 'high')
    tf32_time = run_benchmark(
        label="2. TF32 Only", 
        precision_mode='high',     # ここを変更
        enable_amp=False
    )
    
    # 3. Best Practice: TF32 + BFloat16 AMP
    bf16_time = run_benchmark(
        label="3. Best Practice (TF32 + BF16 AMP)", 
        precision_mode='high',     # ここを変更
        enable_amp=True,
        amp_dtype=torch.bfloat16
    )
    
    print("\n=== Summary ===")
    print(f"Default (FP32): {baseline_time:.2f} ms")
    print(f"TF32 Only     : {tf32_time:.2f} ms (x{baseline_time/tf32_time:.2f})")
    print(f"Best Practice : {bf16_time:.2f} ms (x{baseline_time/bf16_time:.2f})")