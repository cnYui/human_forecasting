# SoMoFormer 与当前双人预测模型组件图

## 图 1：SoMoFormer 原始模型总览

```mermaid
flowchart TD
    A["多人 3D 关节序列<br/>raw joints: [B,N,F,J,3]"] --> B["collate_batch<br/>padding variable people"]
    B --> C["batch_process_joints"]
    C --> C1["以最后观测帧 pelvis/neck 为中心<br/>joints -= in_joints_pelvis"]
    C1 --> C2["reshape<br/>[B,N,F,J,3] -> [B,F,N*J,3]"]
    C2 --> D["输入片段 in_joints<br/>[B,in_F,N*J,3]"]
    C2 --> E["监督未来 out_joints<br/>[B,out_F,N*J,3]"]
    D --> F["last-pose padding<br/>补到 seq_len = in_F + out_F"]
    F --> G["DCT over time<br/>[F,B,N*J*3] -> [N*J*3,B,dct_n]"]
    G --> H["Linear fc_in<br/>DCT token -> hidden"]
    H --> I["Joint embedding + Person embedding"]
    I --> J["Grid / location embedding<br/>来自 pelvis/neck 位置"]
    J --> K["Transformer Encoder layers"]
    K --> L["Linear fc_out<br/>hidden -> dct_n"]
    L --> M["Residual on DCT coeffs"]
    M --> N["IDCT<br/>还原完整序列 [B,seq_len,N*J,3]"]
    N --> O["取未来帧<br/>pred[:, in_F:]"]
    O --> P["masked keypoint MSE / VIM / MPJPE"]
    E --> P
```

## 图 2：SoMoFormer 的 token 化与 DCT 细节

```mermaid
flowchart LR
    A["输入观测<br/>[B,in_F,N*J,3]"] --> B["复制最后一帧<br/>补未来占位"]
    B --> C["完整伪序列<br/>[B,F,N*J,3]"]
    C --> D["flatten joints+xyz<br/>[F,B,N*J*3]"]
    D --> E["DCT matrix @ time axis"]
    E --> F["tokens<br/>[N*J*3,B,dct_n]"]
    F --> G["每个 token 是<br/>某个人-某关节-某坐标<br/>的一整段时间轨迹"]
    G --> H["Transformer attention<br/>跨 joint / person / coordinate token"]
    H --> I["预测 DCT coeffs"]
    I --> J["IDCT"]
    J --> K["完整未来 3D joints"]
```

## 图 3：当前 ReGenNet 双人 forecasting 数据与评估闭环

```mermaid
flowchart TD
    A["InterHuman SMPL H5<br/>[T,25,12]"] --> B["extract_active_motion"]
    B --> C["active motion<br/>[T,2,147]"]
    C --> D["固定窗口裁剪<br/>window_len=150"]
    D --> E["obs<br/>[B,30,2,147]"]
    D --> F["target<br/>[B,120,2,147]"]
    E --> G["ForecastingNormalizer<br/>train-only mean/std"]
    F --> G
    G --> H["训练模型<br/>normalized active-vector MSE"]
    H --> I["pred_normalized<br/>[B,120,2,147]"]
    I --> J["denormalize"]
    J --> K["pred original-scale<br/>[B,120,2,147]"]
    K --> L["compute_forecasting_metrics"]
    F --> L
    E --> L
    L --> M["future_mse / rotation_mse / translation_mse<br/>short / mid / long_mse<br/>relative root distance / orientation / consistency"]
```

## 图 4：当前 independent baseline 组件图

```mermaid
flowchart TD
    A["obs<br/>[B,30,2,147]"] --> B["按 person 拆分并共享权重<br/>[B*2,30,147]"]
    B --> C["GRU person encoder<br/>input=147, hidden=H, layers=L"]
    C --> D["last hidden<br/>[B*2,H]"]
    D --> E["MLP future decoder<br/>H -> 120*147"]
    E --> F["reshape<br/>[B,2,120,147]"]
    F --> G["permute<br/>pred [B,120,2,147]"]
    G --> H["normalized MSE vs target"]
```

## 图 5：当前 concat no-relation baseline 组件图

```mermaid
flowchart TD
    A["obs<br/>[B,30,2,147]"] --> B["concat two persons<br/>[B,30,294]"]
    B --> C["GRU joint encoder<br/>input=294, hidden=H, layers=L"]
    C --> D["last hidden<br/>[B,H]"]
    D --> E["MLP future decoder<br/>H -> 120*294"]
    E --> F["reshape<br/>pred [B,120,2,147]"]
    F --> G["normalized MSE vs target"]
```

## 图 6：当前 relation-aware model 组件图

```mermaid
flowchart TD
    A["obs<br/>[B,30,2,147]"] --> B["person branch"]
    A --> C["relation branch"]

    B --> B1["按 person 展开<br/>[B*2,30,147]"]
    B1 --> B2["shared GRU person encoder"]
    B2 --> B3["person hidden concat<br/>[B,2H]"]

    C --> C1["extract_relation_features"]
    C1 --> C2["relative root translation<br/>relative root velocity<br/>root distance<br/>relative root orientation"]
    C2 --> C3["relation sequence<br/>[B,30,16]"]
    C3 --> C4["GRU or mean+Linear relation encoder"]
    C4 --> C5["relation hidden<br/>[B,R]"]

    B3 --> D["fusion MLP<br/>[B,2H+R] -> [B,H]"]
    C5 --> D
    D --> E["future decoder<br/>H -> 120*294"]
    E --> F["reshape<br/>pred [B,120,2,147]"]
    F --> G["normalized active-vector MSE"]
```

## 图 7：SoMoFormer-style 适配到当前任务的推荐路线

```mermaid
flowchart TD
    A["当前 obs active<br/>[B,30,2,147]"] --> B{"选择适配层级"}

    B --> C["P7.1 joint-space SoMoFormer baseline"]
    C --> C1["active -> SMPL xyz<br/>[B,T,2,24,3]"]
    C1 --> C2["SoMoFormer DCT token Transformer"]
    C2 --> C3["pred_xyz<br/>[B,120,2,24,3]"]
    C3 --> C4["joint_mse / MPJPE-like<br/>relative distance / long joint error"]
    C4 --> C5["优点：最接近 SoMoFormer 原始框架<br/>限制：不能直接进入 active-vector 主表"]

    B --> D["P7.2 somoformer_active"]
    D --> D1["active token / joint-aware token encoder"]
    D1 --> D2["DCT over 150-frame padded sequence"]
    D2 --> D3["Transformer over person/joint/channel tokens"]
    D3 --> D4["active future decoder"]
    D4 --> D5["pred active<br/>[B,120,2,147]"]
    D5 --> D6["复用 P2/P5 metrics 和 aggregate"]
    D6 --> D7["优点：可同口径比较<br/>风险：设计变量更多"]
```

## 关键边界

- SoMoFormer 原版是 3D joint coordinate trajectory model，不是 rot6d / SMPL parameter model。
- 当前 ReGenNet 主表是 active-vector original-scale 指标；joint-space SoMoFormer baseline 不能直接替代 P5 主表。
- 如果目标是论文可比较结果，最终需要 `somoformer_active` 输出 `[B,120,2,147]`，并和 independent / concat / relation-aware 同口径比较。
