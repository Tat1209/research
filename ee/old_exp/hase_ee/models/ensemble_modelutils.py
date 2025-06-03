from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from itertools import permutations, combinations, product
import random
from collections import defaultdict

def straight_get_lambda(lam, total_epochs, epoch, L_avg, L_ens):
    return lam

def dynamic_get_lambda(lam, total_epochs, epoch, L_avg, L_ens):
    return (L_ens / (L_avg + L_ens))

def increasing_get_lambda(lam, total_epochs, epoch, L_avg, L_ens):
    # epochの比率に応じてラムダを増やす．上限はlam
    return min(lam, lam * (epoch / total_epochs))
    
def decreasing_get_lambda(lam, total_epochs, epoch, L_avg, L_ens):
    return max(0.0, lam - lam * (epoch / total_epochs))

def cosine_decay_get_lambda(lam, total_epochs, epoch, L_avg, L_ens):
    return lam / 2 * (1 + np.cos(np.pi * epoch / total_epochs))

# より勾配の激しいS字関数
def sigmoid_get_lambda(lam, total_epochs, epoch, L_avg, L_ens):
    return lam / (1 + np.exp(-10 * (epoch - total_epochs / 2) / total_epochs))

def inv_sigmoid_get_lambda(lam, total_epochs, epoch, L_avg, L_ens):
    return lam - lam / (1 + np.exp(-10 * (epoch - total_epochs / 2) / total_epochs))

# 0 - total_epochs/2の間でコサインカーブで減衰し，lamにwarmupし，残りのtotal_epochs/2の期間で再度コサインカーブで減衰する．
def warmup_cosine_get_lambda(lam, total_epochs, epoch, L_avg, L_ens):
    if epoch < total_epochs / 2:
        return lam / 2 * (1 + np.cos(np.pi * epoch / (total_epochs / 2)))
    else:
        return lam / 2 * (1 + np.cos(np.pi * (epoch - total_epochs / 2) / (total_epochs / 2)))
    
def calc_mse_logits(logits_list):
    """
    各モデルのロジットのペアワイズ二乗誤差を計算し、平均を返すサンプル関数
    多様性を「大きく」したい場合は、損失に「-epsilon * diversity」のように加えることが多い
    """
    div_sum = 0.0
    count = 0
    M = len(logits_list)
    if M < 2:
        return 0.0  # モデルが1つだけなら多様性を計算できない
    
    for i in range(M):
        for j in range(i+1, M):
            # バッチ方向(N)に対して平均をとり、モデル間の差異を計算
            diff = (logits_list[i] - logits_list[j]) ** 2
            div_sum += diff.mean()
            count += 1
    return div_sum / count

def js_div_logit(logit_p, logit_q, epsilon=1e-10):
    """
    Jensen-Shannon Divergence を計算する関数
    """
    p = F.softmax(torch.clamp(logit_p, min=-100, max=100), dim=-1)
    q = F.softmax(torch.clamp(logit_q, min=-100, max=100), dim=-1)
    return js_div(p, q, epsilon)

def js_div(p, q, epsilon=1e-10):
    """
    Jensen-Shannon Divergence を計算する関数
    """
    p = p + epsilon
    q = q + epsilon
    m = 0.5 * (p + q)
    #return 0.5 * (F.kl_div(p, m) + F.kl_div(q, m))
    # 入力は torch.log(p) として KL-divergence を計算
    return 0.5 * (F.kl_div(torch.log(m), p, reduction='batchmean') + 
                  F.kl_div(torch.log(m), q, reduction='batchmean'))

def dkd_divergence(logit_p, logit_q, labels, alpha=1.0, beta=8.0, T=1.0, epsilon=1e-10):
    """
    Decoupled Knowledge Distillation を計算する関数
    """
    N, C = logit_p.shape
    logit_p, logit_q = torch.clamp(logit_p, min=-100, max=100), torch.clamp(logit_q, min=-100, max=100)

    tgt_mask = torch.zeros_like(logit_p, dtype=torch.bool)  # 全 False
    tgt_mask[range(N), labels] = True                             # 正解クラスの列だけ True
    ntgt_mask = torch.ones_like(logit_p, dtype=torch.bool)  # 全 True
    ntgt_mask[range(N), labels] = False                            # 正解クラスの列だけ False

    p = F.softmax(logit_p/T, dim=-1) + epsilon
    q = F.softmax(logit_q/T, dim=-1) + epsilon
    
    p_tgt, p_ntgt = p[tgt_mask].view(N, 1), p[ntgt_mask].view(N, C-1).sum(dim=-1)
    q_tgt, q_ntgt = q[tgt_mask].view(N, 1), q[ntgt_mask].view(N, C-1).sum(dim=-1)

    p_nctgt = F.softmax(logit_p[ntgt_mask].view(N, C-1)/T, dim=-1) + epsilon
    q_nctgt = F.softmax(logit_q[ntgt_mask].view(N, C-1)/T, dim=-1) + epsilon

    # KL Divergence の計算
    tckd_kl = F.kl_div(p_tgt, q_tgt) + F.kl_div(p_ntgt, q_ntgt)
    nckd_kl = F.kl_div(p_nctgt, q_nctgt)

    # JS Divergence の計算
    tckd_js = js_div(p_tgt, q_tgt) + js_div(p_ntgt, q_ntgt)
    nckd_js = js_div(p_nctgt, q_nctgt)

    dkd_kl = (alpha * tckd_kl + beta * nckd_kl) * T**2
    dkd_js = (alpha * tckd_js + beta * nckd_js) * T**2

    return dkd_kl, dkd_js

def calc_jsd_logits(logits_list, labels, alpha=1.0, beta=8.0):
    """
    各モデルのロジットのペアワイズ Jensen-Shannon Divergence を計算し、平均を返す関数
    """
    jsd_sum, dkdkl_sum, dkdjs_sum = 0.0, 0.0, 0.0
    count = 0
    M = len(logits_list)
    if M < 2:
        return 0.0  # モデルが1つだけなら多様性を計算できない

    for i in range(M):
        for j in range(i + 1, M):
            # Jensen-Shannon Divergence
            jsd = js_div_logit(logits_list[i], logits_list[j])
            dkd_kl, dkd_js = dkd_divergence(logits_list[i], logits_list[j], labels, alpha, beta)
            
            # バッチ方向 (N) に対して平均をとる
            jsd_sum += jsd.mean() / torch.log(torch.tensor(2.0))  # log2 で正規化
            dkdkl_sum += torch.clamp(dkd_kl.mean(), max=10.0)/10.0 # KL Divergence は上限無限なので正規化せず，上限を設ける
            dkdjs_sum += dkd_js.mean() / torch.log(torch.tensor(2.0)) # JS Divergence はlog2で正規化（正確にはalpha,betaを考慮する必要ありだが計算がむずい）
            count += 1

    if count == 0:  # ペアが存在しない場合は 0.0 を返す
        return 0.0

    return jsd_sum/count, dkdkl_sum/count, dkdjs_sum/count

def calc_mse_features(features_list):
    """
    各モデルの特徴マップのペアワイズ二乗誤差を計算し、平均を返すサンプル関数
    """
    div_sum = 0.0
    count = 0
    M = len(features_list)
    if M < 2:
        return 0.0
    
    for i in range(M):
        for j in range(i+1, M):
            diff = (features_list[i] - features_list[j]) ** 2
            div_sum += diff.mean()
            count += 1
    return div_sum / count

def calc_cossim_features(features_list):
    """
    特徴マップのペアワイズコサイン距離を計算し、平均を返す
    """
    div_sum = 0.0
    count = 0
    M = len(features_list)
    if M < 2:
        return 0.0

    for i in range(M):
        for j in range(i + 1, M):
            # 特徴マップを正規化（ゼロノルム対策）
            norm_i_value = features_list[i].norm(p=2, dim=1, keepdim=True)
            norm_j_value = features_list[j].norm(p=2, dim=1, keepdim=True)
            if (norm_i_value == 0).any() or (norm_j_value == 0).any():
                continue  # ゼロノルムのベクトルはスキップ

            norm_i = features_list[i] / norm_i_value
            norm_j = features_list[j] / norm_j_value
            
            # コサイン類似度計算
            cos_sim = F.cosine_similarity(norm_i, norm_j, dim=1)  # バッチ方向で計算
            cos_distance = 1 - cos_sim.mean()  # コサイン距離 (0:同一　～　2：逆向き)
            
            div_sum += cos_distance / 2.0  # 0~1 に正規化
            count += 1

    if count == 0:  # ペアが存在しない場合は 0.0 を返す
        return 0.0

    return div_sum / count


def calc_partial_logit_diversity_mc(features_list, models):
    """
    多クラス分類を想定し、各モデル i の:
        - 中間特徴 z_i(x)  (shape: (batch_size, feature_dim))
        - 最終線形層: models[i].fc (shape: (num_classes, feature_dim))
    を用いて 'partial logit vector' を計算し、ペアワイズコサイン距離の平均を返す。

    【引数】
    - features_list: List[Tensor]
        * 各モデル i が出力する特徴 z_i(x) のリスト
        * z_i(x) は形状 (batch_size, feature_dim)
    - models: List[nn.Module]
        * models[i].fc.weight の形状は (num_classes, feature_dim)
        * モデル数は len(models) = len(features_list)
    - num_classes: int
        * クラス数

    【戻り値】
    - avg_diversity: float
        * モデル間のペアワイズなコサイン距離 (0~1 に正規化) の平均
        * 値が大きいほど多様性が高い
    """
    M = len(features_list)
    if M < 2:
        return 0.0  # 比較対象がない

    div_sum = 0.0
    count = 0

    # まず、各モデル i に対して partial logit vector を作成する
    # shape のイメージ:
    #   partial_list[i] -> (batch_size, num_classes, feature_dim)
    partial_list = []
    for i in range(M):
        z_i = features_list[i]  # shape: (batch_size, feature_dim)
        fc_weight = models[i].fc.weight  # shape: (num_classes, feature_dim)

        # (num_classes, feature_dim) -> (1, num_classes, feature_dim) にしてブロードキャスト
        # z_i を (batch_size, 1, feature_dim)
        z_i_expand = z_i.unsqueeze(1)               # (batch_size, 1, feature_dim)
        w_expand = fc_weight.unsqueeze(0)           # (1, num_classes, feature_dim)
        # 要素積 -> (batch_size, num_classes, feature_dim)
        partial_i = z_i_expand * w_expand

        partial_list.append(partial_i)

    # ペアワイズにコサイン距離を計算
    for i in range(M):
        for j in range(i + 1, M):
            # partial_i, partial_j: (batch_size, num_classes, feature_dim)
            partial_i = partial_list[i]
            partial_j = partial_list[j]

            # バッチ・クラス方向をまとめて (batch_size, num_classes * feature_dim)
            # -> shape (batch_size, num_classes * feature_dim)
            # これにより各サンプルごとに 1次元ベクトルとしてコサイン類似度を計算可能
            shape_i = partial_i.shape
            shape_j = partial_j.shape
            # safety check
            assert shape_i == shape_j, f"Shape mismatch: {shape_i} vs {shape_j}"

            batch_size = shape_i[0]
            # flatten: (batch_size, num_classes * feature_dim)
            partial_i_flat = partial_i.view(batch_size, -1)
            partial_j_flat = partial_j.view(batch_size, -1)

            # ノルム計算
            norm_i_value = partial_i_flat.norm(p=2, dim=1, keepdim=True)
            norm_j_value = partial_j_flat.norm(p=2, dim=1, keepdim=True)

            # 有効サンプル(ノルム>0)だけを対象
            valid_mask = (norm_i_value.squeeze() > 1e-12) & (norm_j_value.squeeze() > 1e-12)
            if not torch.any(valid_mask):
                continue

            p_i_valid = partial_i_flat[valid_mask]
            p_j_valid = partial_j_flat[valid_mask]

            # 正規化
            norm_i_valid = p_i_valid.norm(p=2, dim=1, keepdim=True)
            norm_j_valid = p_j_valid.norm(p=2, dim=1, keepdim=True)
            p_i_unit = p_i_valid / norm_i_valid
            p_j_unit = p_j_valid / norm_j_valid

            # コサイン類似度 -> 平均 -> コサイン距離
            cos_sim = F.cosine_similarity(p_i_unit, p_j_unit, dim=1)
            cos_sim_mean = cos_sim.mean()
            cos_dist = 1.0 - cos_sim_mean  # 0~2

            # 0~1 に正規化
            cos_dist_0to1 = cos_dist / 2.0

            div_sum += cos_dist_0to1.item()
            count += 1

    if count == 0:
        return 0.0

    avg_diversity = div_sum / count
    return avg_diversity

def decode_perm_3(n, rank):
    """
    0 <= rank < n*(n-1)*(n-2) の整数を，
    辞書順 (lexicographic order) における 3要素の順列 (i, j, k) に変換する。
    
    ※ここでは n=1000 前後を想定。
    """
    # 総数 = n*(n-1)*(n-2)
    # 1. まず「1つ目の要素」を決める
    block_size = (n-1)*(n-2)         # i が固定されたときの残りのパターン数
    i_index = rank // block_size     # 0 <= i_index < n
    rank_rem = rank % block_size

    first_val = i_index  # 0..n-1 のうち i_index 番目 (= i_index 自身) とする
                         # (今回は range(n) = [0,1,2,...,n-1] としてみる)

    # 2. 「2つ目の要素」を決める
    block_size_2 = (n-2)             # i, j が固定されたときの残りパターン数
    j_index = rank_rem // block_size_2  # 0 <= j_index < (n-1)
    rank_rem = rank_rem % block_size_2

    # j_index は [0..(n-2)] 相当だが，すでに使った first_val を飛ばして値を決める
    if j_index < first_val:
        second_val = j_index
    else:
        second_val = j_index + 1

    # 3. 「3つ目の要素」を決める (さらに first_val, second_val を飛ばす)
    k_index = rank_rem  # 0 <= k_index < (n-2)

    # ここでは 2つ飛ばす必要がある
    # 例: n=5, first_val=2, second_val=4 とかいう場合に
    #      k_index=0 => 実際には 0
    #      k_index=1 => 実際には 1
    #      k_index=2 => 実際には 3 (なぜなら 2 は first_val, 4 は second_val で使用済み)
    #
    # 汎用的には「使った要素を飛ばして k_index 番目を取る」ロジックを組む
    def skip_used(x, used, n):
        """0 <= x < n-len(used) の整数xを
           range(n) \ used の中で x番目に小さい要素に写す。
        """
        count = -1
        for val in range(n):
            if val not in used:
                count += 1
                if count == x:
                    return val
        raise ValueError("Unexpected")  # 理論上ここには来ない

    third_val = skip_used(k_index, {first_val, second_val}, n)

    return (first_val, second_val, third_val)

import bisect

def prepare_distinct_3(labels):
    # クラスごとのインデックス
    groups = defaultdict(list)
    for i, c in enumerate(labels):
        groups[c].append(i)
    classes = sorted(groups.keys())
    K = len(classes)
    # 各クラスのサイズ配列と累積和
    sizes = [len(groups[c]) for c in classes]  # index i に対応するクラス classes[i]
    ps = [0]*(K+1)
    for i in range(K):
        ps[i+1] = ps[i] + sizes[i]
    
    # (c1, c2)ごとのブロックサイズを計算し prefix sums を作成
    pair_info = []       # [(c1_idx, c2_idx, block_size), ...]
    pair_prefix = []     # 累積和
    running_sum = 0
    for i in range(K):
        sz_i = sizes[i]
        for j in range(i+1, K):
            sz_j = sizes[j]
            # c3は jより大きいクラス
            sum_c3 = ps[K] - ps[j+1]  
            if sum_c3 == 0 or sz_i == 0 or sz_j == 0:
                continue
            block_size = 6 * sz_i * sz_j * sum_c3
            running_sum += block_size
            pair_info.append((i, j, block_size))
            pair_prefix.append(running_sum)
    
    return running_sum, pair_info, pair_prefix, groups, classes, sizes, ps

def decode_distinct_3(rank, pair_info, pair_prefix, groups, classes, sizes, ps):
    # 1) ペア (c1_idx, c2_idx) を二分探索
    pair_idx = bisect.bisect_left(pair_prefix, rank+1)
    c1_idx, c2_idx, block_size = pair_info[pair_idx]
    start_of_block = pair_prefix[pair_idx-1] if pair_idx > 0 else 0
    offset_in_block = rank - start_of_block
    
    # 2) 順列6通りのどれか
    sz_i = sizes[c1_idx]
    sz_j = sizes[c2_idx]
    sum_c3 = (ps[len(sizes)] - ps[c2_idx+1])
    sub_size = sz_i * sz_j * sum_c3  # この *6 = block_size
    
    perm_index = offset_in_block // sub_size  # 0..5
    offset2 = offset_in_block % sub_size
    
    # 3) c3 の特定 (offset2 // (sz_i*sz_j)) を二分探索
    c3_offset = offset2 // (sz_i * sz_j)
    offset_in_sub_ij = offset2 % (sz_i * sz_j)
    
    # c3 を探す: c2_idx < c3_idx, ps[c3_idx+1] - ps[c2_idx+1] > c3_offset となる最小c3_idx
    # バイナリサーチ用に「c3_offset を ps 上でずらす」
    target = c3_offset + ps[c2_idx+1]
    # bisect_leftで ps[c3_idx] >= target となる最小 c3_idx を探す
    # ただし実際に欲しいのは "ps[c3_idx] > target" なので1引き算に気を付ける
    c3_idx = bisect.bisect_left(ps, target)
    # もし ps[c3_idx] <= target なら一個進める
    if c3_idx < len(ps) and ps[c3_idx] <= target:
        c3_idx += 1
    
    # (c3_idx は c2_idx+1 以上になっているはず)
    # c3_offset内でのクラス c3_idx 内インデックス
    offset_in_class_c3 = c3_offset - (ps[c3_idx-1] - ps[c2_idx+1])
    
    # iA, iB: c1_idx, c2_idx 内でのインデックス
    iA = offset_in_sub_ij // sz_j
    iB = offset_in_sub_ij % sz_j
    
    idxA = groups[classes[c1_idx]][iA]
    idxB = groups[classes[c2_idx]][iB]
    idxC = groups[classes[c3_idx-1]][offset_in_class_c3]  # c3_idx-1 に注意 (上で+1 進めた可能性があるため)
    
    # 4) perm_index に応じて並び替え
    perm_map = [
        (idxA, idxB, idxC),
        (idxA, idxC, idxB),
        (idxB, idxA, idxC),
        (idxB, idxC, idxA),
        (idxC, idxA, idxB),
        (idxC, idxB, idxA),
    ]
    return perm_map[perm_index]


def compute_centroids(feature_lists, labels):
    """
    feature_lists: List[Tensor], それぞれ shape (N, d)
      - M 個のモデルに対する特徴マップ
    labels: shape (N,) のクラスラベル
    戻り値: shape (M, C, d) の重心テンソル
    """
    device = feature_lists[0].device
    # 1. feature_listsをスタック => [M, N, d]
    stacked = torch.stack(feature_lists, dim=0)  # (M, N, d)
    M, N, d = stacked.shape

    # 2. ラベルのユニークと逆引きを取得
    unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)
    C = unique_labels.size(0)
    if C == 0:
        return torch.tensor(0.0, device=device)

    # 3. one-hot ベクトルを作る => [N, C]
    one_hot = F.one_hot(inverse_indices, C).float()

    # 4. 行列演算でクラスごとの合計を求める
    #    まず stacked を [M, d, N] に permute
    stacked_t = stacked.permute(0, 2, 1)  # (M, d, N)

    # one_hot を [M, N, C] にブロードキャスト
    #  (単に batch 次元 M を合わせるだけ)
    one_hot_batched = one_hot.unsqueeze(0).expand(M, -1, -1)  # (M, N, C)
    sum_tensors = torch.bmm(stacked_t, one_hot_batched)  # (M, d, C)
    counts = one_hot.sum(dim=0)  # (C,)

    # countsを [M, 1, C] にブロードキャストして割り算 => (M, d, C)
    #  0サンプルのクラスがあれば0割りになるので適宜対処
    #  （ここでは単純に / counts としており、0 クラスは inf になる）
    centroids = sum_tensors / counts.view(1, 1, C)
    centroids = centroids.permute(0, 2, 1)  # (M, C, d)

    return centroids

def compute_angle_distance_loss(centroids_list, num_iter, labels=None, device=None):
    """
    centroids_list: [M, C, d] のテンソル、もしくは List[Tensor (C, d)] でも可
        （C はクラス数, d は特徴次元, M はモデル数）
    num_iter: サンプリングする順列数の上限
    """
    # centroids_list がもしリストなら Tensor スタックする（[M, C, d]化）
    if isinstance(centroids_list, list):
        centroids_list = torch.stack(centroids_list, dim=0)  # (M, C, d)

    # shape 確認
    M, C, d = centroids_list.shape
    if device is None:
        device = centroids_list.device
    
    # labelsが与えられたときは，labelsをまたぐように組み合わせを作成する．
    if labels is not None:
        total, pair_info, pair_prefix, groups, classes, sizes, ps = prepare_distinct_3(labels)
        num_samples = min(num_iter, total)
        ranks = random.sample(range(total), num_samples)
        sampled_permutations = [decode_distinct_3(r, pair_info, pair_prefix, groups, classes, sizes, ps) for r in ranks]
    else:  # そうでない場合は，全てのクラスの組み合わせを列挙する
        total = C*(C-1)*(C-2)
        num_samples = min(num_iter, total)
        ranks = random.sample(range(total), num_samples)
        sampled_permutations = [decode_perm_3(C, r) for r in ranks]

    if len(sampled_permutations) == 0:
        # クラス数が2以下の場合など
        return torch.tensor(0.0, device=device)

    # 以下のようにベクトル演算へ置き換える。
    # 1) permutations をテンソル化: shape [num_samples, 3]
    permutations_tensor = torch.tensor(sampled_permutations, dtype=torch.long, device=device)
    # 各列が c1, c2, c3
    c1 = permutations_tensor[:, 0]  # [num_samples]
    c2 = permutations_tensor[:, 1]
    c3 = permutations_tensor[:, 2]

    # angles_per_model を格納するリスト
    angles_per_model_list = []

    # M はそれほど大きくない想定で、モデルごとに処理
    for m in range(M):
        # centroids: shape [C, d]
        centroids = centroids_list[m]  # あるモデルのクラス重心

        # 2) v1, v2 = (c1 - c2), (c2 - c3) をまとめて計算 => shape [num_samples, d]
        v1 = centroids[c1] - centroids[c2]  # (num_samples, d)
        v2 = centroids[c2] - centroids[c3]  # (num_samples, d)

        # 3) 正規化
        v1_norm = v1.norm(dim=1, p=2, keepdim=True) + 1e-8
        v2_norm = v2.norm(dim=1, p=2, keepdim=True) + 1e-8
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm

        # 4) 内積 => clamp => acos => shape [num_samples]
        dot = (v1 * v2).sum(dim=1)  # row-wise dot
        dot = torch.clamp(dot, -1.0 + 1e-8, 1.0 - 1e-8)
        angles = torch.acos(dot)

        # 5) angles (num_samples,) を格納
        angles_per_model_list.append(angles)

    # モデル数が 0 の場合（理論的にはない想定だが保険）
    if len(angles_per_model_list) == 0:
        return torch.tensor(0.0, device=device)

    # [M, num_samples] にスタック
    angles_per_model = torch.stack(angles_per_model_list, dim=0)  # (M, num_samples)

    # --- 以下、モデル間のペアワイズ距離を計算 ---
    # pairwise 距離行列 => shape [M, M]
    # cdist には 「(M, D) vs (M, D)」を入力すると各行ベクトル間の距離が出る
    # ここでは D = num_samples
    pairwise_dist = torch.cdist(angles_per_model, angles_per_model, p=2)  # (M, M)

    # 上三角 (i < j) の要素を取り出して平均
    # triu_indicesで (row, col) が得られるのでそれでインデックス
    if M > 1:
        idx = torch.triu_indices(M, M, offset=1)
        upper = pairwise_dist[idx[0], idx[1]]  # (M*(M-1)/2,)
        mean_dist = upper.mean()
    else:
        # M=1 ならペアは存在しないので距離0という扱いでOKか
        mean_dist = torch.tensor(0.0, device=device)

    # 正規化: 最大距離は要素数が num_samples 次元ベクトル同士のユークリッド距離なので
    #   - 各成分の最大差は π (acos の最大値)
    #   - num_samples個要素のベクトルなら最大ノルムは π * sqrt(num_samples)
    max_distance = torch.sqrt(torch.tensor(float(num_samples), device=device)) * torch.pi
    normalized_mean_distance = mean_dist / max_distance

    return normalized_mean_distance

def compute_pairwise_distance_loss(centroids_list, num_iter, device=None):
    """
    centroids_list:  [M, C, d] あるいは List[Tensor(C, d)] の形のクラス重心
       - M: モデル数, C: クラス数, d: 特徴量次元
    num_iter: ランダムサンプリングするクラスペア数の上限
    device: 返り値のデバイス指定 (Noneの場合はcentroids_listから取得)

    戻り値: 正規化した距離 (tensor, shape=[])
    """
    # -- centroids_list がリストならテンソルにスタック --
    if isinstance(centroids_list, list):
        centroids_list = torch.stack(centroids_list, dim=0)  # => (M, C, d)

    M, C, d = centroids_list.shape
    if device is None:
        device = centroids_list.device

    if C < 2:
        # クラスが1つ以下ならペアが作れないので0を返す
        return torch.tensor(0.0, device=device)

    # 1. 全てのクラス間ペアを取得
    all_pairs = list(combinations(range(C), 2))  # 長さは C*(C-1)/2
    if len(all_pairs) == 0:
        return torch.tensor(0.0, device=device)

    # 2. ランダムサンプリング
    num_samples = min(num_iter, len(all_pairs))
    sampled_pairs = random.sample(all_pairs, num_samples)
    sampled_pairs_tensor = torch.tensor(sampled_pairs, dtype=torch.long, device=device)
    c1 = sampled_pairs_tensor[:, 0]
    c2 = sampled_pairs_tensor[:, 1]

    # 3. centroids_list から一度に差分を取る
    v = centroids_list[:, c1, :] - centroids_list[:, c2, :]  # (M, num_samples, d)

    # 4. 距離(ノルム)を計算 => shape (M, num_samples)
    distances = v.norm(p=2, dim=2)  # ユークリッド距離
    D_max = 10.0
    distances_clamped = torch.clamp(distances, max=D_max)

    # --- 以下、モデル間のペアワイズ距離を計算 ---
    # pairwise 距離行列 => shape [M, M]
    # cdist には 「(M, D) vs (M, D)」を入力すると各行ベクトル間の距離が出る
    # ここでは D = num_samples
    pairwise_dist = torch.cdist(distances_clamped, distances_clamped, p=2)  # (M, M)

    # 上三角 (i < j) の要素を取り出して平均
    # triu_indicesで (row, col) が得られるのでそれでインデックス
    if M > 1:
        idx = torch.triu_indices(M, M, offset=1)
        upper = pairwise_dist[idx[0], idx[1]]  # (M*(M-1)/2,)
        mean_dist = upper.mean()
    else:
        # M=1 ならペアは存在しないので距離0という扱いでOKか
        mean_dist = torch.tensor(0.0, device=device)

    # 6. 正規化
    normalized_mean_distance = mean_dist / (D_max * torch.sqrt(torch.tensor(float(num_samples), device=device)))

    return normalized_mean_distance

def calc_centroid_diversity(feature_lists, logits, labels, calctype="CCA", num_iter=1000):
    """
    クラス重心間角の多様性制約を計算する。

    引数：
        feature_lists (list of torch.Tensor): 各モデルの特徴マップ (shape: [batch_size, feature_dim])
        logits (torch.Tensor): モデルのロジット (shape: [batch_size, num_classes])
        labels (torch.Tensor): ミニバッチ内の正解ラベル (shape: [batch_size])
        calctype (str): 計算方法を指定。"CCA"（角度ベース）または "CCD"（ペアワイズ距離ベース）

    戻り値：
        torch.Tensor: モデル間のペアワイズ距離の正規化平均値（0-1 に収まる）
    """
    # 1. 各モデルのクラスごとの重心を計算
    centroids_list = compute_centroids(feature_lists, labels)  # (M, N, d) -> (M, C. d)

    stacked = torch.stack(feature_lists, dim=0)  # (M, N, d)
    if calctype == "CCA":   # Class-Centroid Angle
        # 2. 異なる3クラスの順列をランダムサンプリングし、角度を計算
        return compute_angle_distance_loss(centroids_list, num_iter, device=None)
    elif calctype == "IA":  # Instance-Angle
        return compute_angle_distance_loss(stacked, num_iter, device=None)
    elif calctype == "CIA":  # Class-Instance Angle
        return compute_angle_distance_loss(stacked, num_iter, labels=labels, device=None)
    elif calctype == "CCD":
        # 2. 全てのクラス間ペアの距離を計算
        return compute_pairwise_distance_loss(centroids_list, num_iter, device=None)

    else:
        raise ValueError("Invalid calctype. Choose 'CCA' or 'CCD'.")

class EnsembleTrainer:
    """
    - models: List[nn.Module]
        1~M 個のモデル（すべて 32x32x3 -> 10クラス の入出力を満たす）
    - lambda_: float
        損失関数の comb と dist を混合する際に使用する重み
    - epsilon: float
        正則化項の重み
    - type_: str
        損失関数の計算方法 ("dist", "comb", "lambda")
    - type_regularizer: str
        正則化項の種類 ("none", "divL", "divF")
    - lr: float
        学習率 (デフォルト 0.1)
    - T_max: int
        コサインアニーリングの最大イテレーション数 (デフォルト 100)
    """
    def __init__(
        self,
        models,
        lambda_=0.5,
        epsilon=0.1,
        type_="lambda",
        type_regularizer="none",
        lr=0.1,
        T_max=100,
        device=None,
        total_epochs=100,
        get_lambda=straight_get_lambda,
        get_epsilon=None,
        num_iter=1000,
        alpha=1.0,
        beta=8.0,
    ):
        self.models = models
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.type_ = type_
        self.type_regularizer = type_regularizer
        self.lr = lr
        self.T_max = T_max
        self.total_epochs = total_epochs
        self.get_lambda = get_lambda
        self.get_epsilon = get_epsilon
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta = beta
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # モデルを device へ
        for model in self.models:
            model.to(self.device)
        
        # 全モデルのパラメータをまとめて最適化する
        params = []
        for model in self.models:
            params += list(model.parameters())
        
        # SGD と CosineAnnealingLR を設定
        self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.T_max)
    
    def _compute_loss(self, logits_list, y, epoch=0):
        """
        ロジットとラベルから、type_ に従った分類損失を計算する。
         - dist: 各モデル個別の CE を平均 (L_avg)
         - comb: 各モデル出力の平均をとって CE (L_ens)
         - lambda: λL_ens + (1-λ)L_avg
        """
        # logits_list: list of [N, num_classes]
        # y: [N]
        
        # L_avg (dist)
        ce_losses = []
        for logits in logits_list:
            ce_losses.append(F.cross_entropy(logits, y))
        L_avg = torch.stack(ce_losses).mean()  # 各モデルの CE を平均
        
        # L_ens (comb)
        mean_logits = torch.mean(torch.stack(logits_list), dim=0)  # [N, num_classes]
        L_ens = F.cross_entropy(mean_logits, y)

        # 動的λの実装
        lam = self.lambda_
        #if lam < 0.0:  # λ が負の場合は 動的λ にする
        #    lam = (L_ens / (L_avg + L_ens)).item()  # ここ，微分切るか否かでも挙動が変わりそう．．
        if self.get_lambda is not None:
            lam = self.get_lambda(self.lambda_, self.total_epochs, epoch, L_avg, L_ens)

        res = {"L_avg": L_avg, "L_ens": L_ens, 
               "L_lambda": lam * L_ens + (1.0 - lam) * L_avg,
               "lambda": lam}
        return res
    
    def _compute_regularizer(self, logits_list, features_list, labels, targets=["divF", "divCCA", "divCCD"]):
        """
        type_regularizer に応じて正則化項を計算する。
        多様性を大きくする場合は損失に「- epsilon * diversity」を加えるのが一例。
        """
        jsd, dkdkl, dkdjs = calc_jsd_logits(logits_list, labels, alpha=self.alpha, beta=self.beta)
        res = {"divL": jsd,
               "divDKDKL": dkdkl,
               "divDKDJS": dkdjs,
               "divF": calc_cossim_features(features_list) if "divF" in targets else 0.0, 
               #"divPL": calc_partial_logit_diversity_mc(features_list, self.models),
               "divPL": 0.0,
               "divCCA": calc_centroid_diversity(features_list, logits_list, labels, calctype="CCA", num_iter=self.num_iter) if "divCCA" in targets else 0.0,
               #"divIA": calc_centroid_diversity(features_list, logits_list, labels, calctype="IA", num_iter=self.num_iter),
               #"divCIA": calc_centroid_diversity(features_list, logits_list, labels, calctype="CIA", num_iter=self.num_iter),
               "divIA": 0.0, 
               "divCIA": 0.0,
               "divCCD": calc_centroid_diversity(features_list, logits_list, labels, calctype="CCD", num_iter=self.num_iter) if "divCCD" in targets else 0.0,
               }
        return res
    
    def forward(self, x):
        """
        1 つの入力 x に対して、各モデルの出力を返す。
        """
        logits_list = []
        for model in self.models:
            logits = model(x)
            logits_list.append(logits)
        # logits_listの平均を返す
        return torch.mean(torch.stack(logits_list), dim=0)
    
    def train_one_epoch(self, dataloader, train=True, regularize=True, epoch=0):
        """
        1 epoch 分の学習を行う。
        戻り値として以下のような情報を格納した dict を返す例:
        {
            "loss": (平均)損失,
            "ce_loss": 分類損失,
            "reg_loss": 正則化項,
        }
        """
        for model in self.models:
            if train:
                model = model.train().to(self.device)
            else:
                model = model.eval().to(self.device)
        
        loss_targets = ["loss", "ce_loss", "reg_loss", "L_avg", "L_ens", "L_lambda", "lambda", "epsilon"]
        reg_targets = ["divL", "divDKDKL", "divDKDJS", "divF", "divPL", "divCCA", "divCCD", "divIA", "divCIA"]
        losses = {key: 0.0 for key in loss_targets+reg_targets}

        num_batches = 0
        # Accuracy算出用の予測成功可否に関する情報
        ensemble_corrected = []
        corrected = [[] for _ in range(len(self.models))]
        
        for x, y in tqdm(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward 各モデル
            logits_list = []
            features_list = []
            for model in self.models:
                # ここでは例として、"model" が logits を返す想定。
                # 特徴マップを得たい場合は、モデルに工夫が必要 (中間出力を返すようにする等)
                # たとえば model(x) -> (features, logits) のように書き換えてもよい。
                features, logits = model.get_features_and_logits(x)
                logits_list.append(logits)
                features_list.append(features)
            
            # 損失計算
            classification_losses = self._compute_loss(logits_list, y, epoch=epoch)
            if self.type_ == "dist":
                classification_loss = classification_losses["L_avg"]
            elif self.type_ == "comb":
                classification_loss = classification_losses["L_ens"]
            elif self.type_ == "lambda":
                classification_loss = classification_losses["L_lambda"]
            else:
                raise ValueError(f"Invalid type_: {self.type_}")
            
            # 正則化
            if regularize:
                regularizers = self._compute_regularizer(logits_list, features_list, y, targets=[self.type_regularizer])
                regularizer = 0.0
                for key in reg_targets:
                    if key in self.type_regularizer:
                        regularizer += regularizers[key]
            else:
                regularizer = 0.0
                regularizers = {key: 0.0 for key in reg_targets}
            
            eps = self.epsilon
            regloss_temp = regularizer if isinstance(regularizer, float) else regularizer.item()
            if self.get_epsilon is not None:
                eps = self.get_epsilon(eps, self.total_epochs, epoch, 
                                       classification_loss.item(), regloss_temp)
            
            loss = classification_loss - eps * regularizer
            
            # Backward
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # ログ計算
            losses["loss"] += loss.item()
            losses["ce_loss"] += classification_loss.item()
            losses["reg_loss"] += regloss_temp
            losses["L_avg"] += classification_losses["L_avg"].item()
            losses["L_ens"] += classification_losses["L_ens"].item()
            losses["L_lambda"] += classification_losses["L_lambda"].item()
            losses['lambda'] += classification_losses['lambda']
            losses['epsilon'] += eps
            for key in reg_targets:
                losses[key] += regularizers[key] if isinstance(regularizers[key], float) else regularizers[key].item()
            
            # Accuracy算出用
            mean_logits = torch.mean(torch.stack(logits_list), dim=0)  # [N, num_classes]
            preds = mean_logits.argmax(dim=1)
            ensemble_corrected.extend((preds == y).cpu().numpy())
            for num, logits in enumerate(logits_list):
                pred = logits.argmax(dim=1)
                corrected[num].extend((pred == y).cpu().numpy())
            
            num_batches += 1
            torch.cuda.empty_cache()
            del x, y, logits_list, features_list, loss, classification_loss, regularizer, mean_logits, preds
            del classification_losses, regularizers
        
        for key in losses.keys():
            losses[key] /= num_batches  # 平均を計算

        losses['ensemble_acc'] = sum(ensemble_corrected) / len(ensemble_corrected)
        for num, corr in enumerate(corrected):
            losses[f'm{num}_acc'] = sum(corr) / len(corr)
        
        # 1 バッチごとに step() しても良いし, 1 epoch 終わりで step() してもよい
        # CosineAnnealingLR の使い方に合わせて適宜調整
        if train:
            self.scheduler.step()
        
        return losses
    
class WeightedEnsembleTrainer(EnsembleTrainer):
    def __init__(self, *args, num_classes=10, cw_trainable=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.cw_trainable = cw_trainable

        # (num_classes, M) 形状で乱数初期化
        # M は self.models の個数
        self.raw_class_weights = nn.Parameter(
            torch.randn(self.num_classes, len(self.models)).to(self.device)
        )
        
        if not self.cw_trainable:
            self.raw_class_weights.requires_grad = False  # 学習しない場合は勾配計算を無効化

        # 既存の optimizer に本パラメータを追加登録し，学習可能とする
        self.optimizer.add_param_group({
            "params": [self.raw_class_weights],
            "lr": self.lr  # 必要に応じて別の学習率にしてもよい
        })
    
    def _compute_loss(self, logits_list, y, epoch=0):
        """
        class_weights を活かしたクロスエントロピーを計算する．
         - dist (L_avg): 各モデルの CE を個別に計算 -> 平均
         - comb (L_ens): 各モデル出力の平均 (mean_logits) -> 通常の CE
         - lambda: lam * L_ens + (1 - lam) * L_avg
        """
        # class_weights をモデル方向に softmax で正規化
        # (num_classes, M)
        class_weights = F.softmax(self.raw_class_weights, dim=1)
        
        # L_avg (dist) では 各モデル i ごとに
        #   CE_i = -(1/N) * sum_{n=1..N} [ class_weights[y_n, i] * log p_i(y_n|x_n) ]
        # の平均をとる
        ce_losses = []
        for i, logits in enumerate(logits_list):
            # logits: [N, num_classes]
            log_probs = F.log_softmax(logits, dim=1)  # [N, num_classes]
            
            # バッチ内それぞれのサンプル n について
            #   w_{n} = class_weights[y_n, i]
            w_n = class_weights[y, i]  # shape: (N,)
            # -log_probs[range(N), y]: shape (N,)
            # => サンプルごとの NLL
            nll = -log_probs[range(len(y)), y]
            
            ce_loss = (w_n * nll).mean()
            ce_losses.append(ce_loss)
        
        L_avg = torch.stack(ce_losses).mean()  # 各モデルのCEをさらに平均
        
        # L_ens (comb) は 通常どおり各モデルの平均ロジットを利用
        mean_logits = torch.mean(torch.stack(logits_list), dim=0)  # [N, num_classes]
        L_ens = F.cross_entropy(mean_logits, y)
        
        # lambda モード
        lam = self.lambda_
        if self.get_lambda is not None:
            lam = self.get_lambda(self.lambda_, self.total_epochs, epoch, L_avg, L_ens)
        
        return {
            "L_avg": L_avg,
            "L_ens": L_ens,
            "L_lambda": lam * L_ens + (1.0 - lam) * L_avg,
            "lambda": lam
        }

from .modelutils import create_model
def create_ensemble_models(ensembles, type_, type_regularizer, lambda_, epsilon, 
                           total_epochs, get_lambda, get_epsilon, num_classes,
                           random_class_weight=False, cw_trainable=False, 
                           linear_sharing=False, num_iter=1000, alpha=1.0, beta=8.0, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = 1 # 仮

    models = []
    for i in range(ensembles):
        m = create_model(for_cifar_customize=True, num_classes=num_classes, **kwargs)
        m = m.to(device)
        if num_gpus > 1:  # multi-gpu
            m = torch.nn.DataParallel(m)
        models.append(m)
    
    # Linearを共有する
    if linear_sharing:
        for i in range(1, ensembles):
            models[i].fc = models[0].fc

    if random_class_weight:
        return  WeightedEnsembleTrainer(models, 
                                lambda_=lambda_,
                                epsilon=epsilon,
                                type_=type_,
                                type_regularizer=type_regularizer,
                                lr=0.1,
                                T_max=100,
                                device=device,
                                total_epochs=total_epochs,
                                get_lambda=get_lambda,
                                get_epsilon=get_epsilon,
                                num_classes=num_classes,
                                cw_trainable=cw_trainable,
                                num_iter=num_iter,
                                alpha=alpha,
                                beta=beta,
                                )
    else:
        return  EnsembleTrainer(models, 
                                lambda_=lambda_,
                                epsilon=epsilon,
                                type_=type_,
                                type_regularizer=type_regularizer,
                                lr=0.1,
                                T_max=100,
                                device=device,
                                total_epochs=total_epochs,
                                get_lambda=get_lambda,
                                get_epsilon=get_epsilon,
                                num_iter=num_iter,
                                alpha=alpha,
                                beta=beta,
                                )
    
def get_module_by_name(model, name):
    """Get a module by its name in the model's named_modules."""
    for n, m in model.named_modules():
        if n == name:
            return m
    return None

# モデルの重みの転移：ただし対応が取れているのか？は検証が必要
# scale = Trueの場合はin_channelsが減少する際に，重みをスケールする
# cross = Trueの場合は，i,jの位置を交互に入れ替える
def transfer_weights(model, ensemble, div, scale=False, cross=False):
    counter = 0
    for parent_name, parent_mod in model.named_modules():
        if isinstance(parent_mod, nn.Conv2d):
            weight = parent_mod.weight.data
            outs, ins, _, _ = weight.shape
            for i in range(div):
                for j in range(div):
                    num = i*div + j
                    mod = get_module_by_name(ensemble.models[num], parent_name)
                    # in_channelsが3の場合はdiv回繰り返し
                    outr, inr = outs // div, ins // div
                    scale_factor = 1.0
                    if scale:
                        scale_factor = div

                    if mod.in_channels == 3:
                        mod.weight.data = weight[j*outr:(j+1)*outr,:,:,:].clone()
                    else:
                        if cross:
                            if counter % 2 == 0:
                                mod.weight.data = weight[j*outr:(j+1)*outr, i*inr:(i+1)*inr, :, :].clone() * scale_factor
                            else:
                                mod.weight.data = weight[i*outr:(i+1)*outr, j*inr:(j+1)*inr, :, :].clone() * scale_factor
                        else:
                            mod.weight.data = weight[i*outr:(i+1)*outr, j*inr:(j+1)*inr, :, :].clone() * scale_factor
            counter += 1
        elif isinstance(parent_mod, nn.Linear):
            weight = parent_mod.weight.data
            outs, ins = weight.shape
            for i in range(div):
                for j in range(div):
                    num = i*div + j
                    # in_featuresが3の場合はdiv回繰り返し
                    mod = get_module_by_name(ensemble.models[num], parent_name)
                    inr = ins // div
                    scale_factor = 1.0
                    if scale:
                        scale_factor = div
                    if cross and counter % 2 == 0:
                        mod.weight.data = weight[:, j*inr:(j+1)*inr].clone() * scale_factor
                    else:
                        mod.weight.data = weight[:, i*inr:(i+1)*inr].clone() * scale_factor
        elif isinstance(parent_mod, nn.BatchNorm2d):
            weight = parent_mod.weight.data
            ins = weight.shape[0]
            for i in range(div):
                for j in range(div):
                    num = i*div + j
                    mod = get_module_by_name(ensemble.models[num], parent_name)
                    inr = ins // div
                    if cross and counter % 2 == 0:
                        mod.weight.data = weight[j*inr:(j+1)*inr].clone()
                    else:
                        mod.weight.data = weight[i*inr:(i+1)*inr].clone()