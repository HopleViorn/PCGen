 vbre# BrepGen: 基于潜在扩散模型的B-rep CAD实体生成项目文档

## 1. 项目概述

### 1.1 功能与作用

BrepGen是一个基于深度学习的生成式CAD项目。其核心功能是从噪声开始，逐步生成复杂的三维实体模型。与处理点云或网格（Mesh）的模型不同，本项目直接生成**B-rep（Boundary Representation, 边界表示）**数据。B-rep是CAD领域的标准表示法，它使用精确的数学曲面（如NURBS）和曲线来描述模型，保留了模型的拓扑结构（面、边、顶点的连接关系）。

这种方法的优势在于：
*   **精度高**：生成的是数学上精确的CAD模型，而非近似的网格。
*   **可编辑性强**：输出的STEP文件可以在主流CAD软件（如SolidWorks, CATIA, Fusion 360）中直接打开和编辑。
*   **拓扑有效**：模型隐式地学习并保证了生成实体的拓扑有效性（如水密性）。

### 1.2 核心技术：潜在扩散模型 (LDM)

项目采用了**潜在扩散模型（Latent Diffusion Model, LDM）**作为核心生成引擎。LDM的工作原理不是在原始的高维数据空间（如控制点坐标）上进行去噪，而是在一个由**变分自编码器（VAE）**学习到的低维、紧凑的**潜在空间（Latent Space）**中进行。

**整体生成流程**:
**噪声 -> [扩散模型] -> 潜在向量 `z` -> [VAE解码器] -> 控制点 -> [CAD库] -> B-rep模型 (STEP) -> [CAD库] -> 可视化网格 (STL)**

## 2. 实现原理与技术细节

### 2.1 分层、自回归的生成策略

BrepGen将一个复杂CAD模型的生成任务分解为四个连续的、有条件的阶段。每个阶段都由一个独立的、基于Transformer的扩散模型负责，并以前一阶段的输出作为条件。

**生成顺序**: `surfpos` -> `surfz` -> `edgepos` -> `edgez`

1.  **`surfpos` (Surface Position)**: 生成模型中所有**曲面**的**位置和尺寸**（由3D包围盒表示）。
    *   **训练入口**: [`ldm.py:19`](ldm.py:19)
    *   **训练器**: [`SurfPosTrainer` in `trainer.py:267`](trainer.py:267)
    *   **网络模型**: [`SurfPosNet` in `network.py:1066`](network.py:1066)

2.  **`surfz` (Surface Latent Geometry)**: 在给定曲面位置的条件下，生成每个曲面的**精确几何形状**（由一个低维潜在向量`z`表示）。
    *   **训练入口**: [`ldm.py:24`](ldm.py:24)
    *   **训练器**: [`SurfZTrainer` in `trainer.py:416`](trainer.py:416)
    *   **网络模型**: [`SurfZNet` in `network.py:1129`](network.py:1129)

3.  **`edgepos` (Edge Position)**: 在给定所有曲面信息（位置和几何）的条件下，生成附着在这些曲面上的**边的位置和尺寸**。
    *   **训练入口**: [`ldm.py:29`](ldm.py:29)
    *   **训练器**: [`EdgePosTrainer` in `trainer.py:612`](trainer.py:612)
    *   **网络模型**: [`EdgePosNet` in `network.py:1203`](network.py:1203)

4.  **`edgez` (Edge Latent Geometry & Vertices)**: 在给定所有曲面和边的位置信息后，生成**边的精确几何形状**（潜在向量`z`）以及连接它们的**顶点的精确三维坐标**。
    *   **训练入口**: [`ldm.py:34`](ldm.py:34)
    *   **训练器**: [`EdgeZTrainer` in `trainer.py:807`](trainer.py:807)
    *   **网络模型**: [`EdgeZNet` in `network.py:1289`](network.py:1289)

### 2.2 数据表示：将几何视为“图像”和“序列”

项目不处理非结构化的点云，而是处理B-rep曲面/曲线的**控制点**。

*   **曲面表示**: 一个NURBS曲面由一个控制点网格定义（例如 `16x16` 个三维点）。在代码中，这个 `(16, 16, 3)` 的张量被重排为 `(3, 16, 16)`，并被视为一张**3通道的“几何图像”**，其中3个通道分别代表所有控制点的X, Y, Z坐标。
    *   **相关代码**: [`trainer.py:75`](trainer.py:75) `surf_uv.to(self.device).permute(0,3,1,2)`

*   **边表示**: 一条NURBS曲线由一个控制点序列定义（例如 `32` 个三维点）。这个 `(32, 3)` 的张量被视为一个**一维序列**。
    *   **相关代码**: [`trainer.py:202`](trainer.py:202) `edge_u.to(self.device).permute(0,2,1)`

这种表示方法使得项目可以利用成熟的2D CNN（用于曲面）和1D CNN（用于边）架构来构建VAE。

### 2.3 VAE：几何信息的压缩与解压

项目的核心前提是使用VAE将高维的控制点数据压缩到低维潜在空间。

*   **训练入口**: [`vae.py`](vae.py) 脚本用于独立训练VAE。
*   **曲面VAE**: 使用 `diffusers` 库的 `AutoencoderKL`，这是一个为2D图像设计的标准VAE。
    *   **实例化代码**: [`trainer.py:20`](trainer.py:20)
*   **边VAE**: 使用项目自定义的 `AutoencoderKL1D`，它将2D VAE的组件（如`ResConvBlock`）替换为1D版本。
    *   **模型定义**: [`AutoencoderKL1D` in `network.py:316`](network.py:316)
    *   **实例化代码**: [`trainer.py:146`](trainer.py:146)

### 2.4 扩散过程：从噪声到几何

扩散模型的核心是**去噪**。训练过程如下（以`SurfZTrainer`为例）：

1.  **获取干净数据**: 将曲面控制点通过VAE编码器得到干净的潜在向量 `surfZ`。 ([`trainer.py:518-524`](trainer.py:518))
2.  **加噪**: 随机选择一个时间步`t`，向 `surfZ` 添加对应强度的高斯噪声，得到带噪的 `surfZ_diffused`。 ([`trainer.py:529-531`](trainer.py:529))
3.  **预测噪声**: 将 `surfZ_diffused`、时间步`t`以及条件信息（如`surfPos`）输入到Transformer模型（`SurfZNet`），模型的目标是预测出添加的噪声。 ([`trainer.py:534`](trainer.py:534))
4.  **计算损失**: 计算预测噪声与真实噪声之间的MSE损失，并反向传播更新模型。 ([`trainer.py:537`](trainer.py:537))

### 2.5 拓扑保证：隐式学习与条件生成

项目不使用硬编码的几何约束来保证边面相交。它通过以下方式**隐式地学习**拓扑关系：

*   **条件生成**: 如2.1节所述，边的生成严格以面的信息为条件，迫使模型学习它们之间的空间关系。
*   **显式生成顶点**: 在最后阶段，模型直接生成连接边的**顶点坐标** (`vertPos`)。这些顶点作为几何“锚点”，极大地增强了拓扑连接的准确性。
    *   **相关代码**: [`trainer.py:930`](trainer.py:930) `joint_data = torch.concat([edgeZ, vertPos], -1)`
*   **后处理优化**: 在采样阶段，通过 `joint_optimize` 函数对生成的控制点进行联合优化，进一步确保几何上的一致性。
    *   **相关代码**: [`sample.py:356`](sample.py:356)

### 2.6 数据预处理流程

模型的性能高度依赖于输入数据的质量。项目采用了一个专业且关键的两阶段数据预处理流程，该流程在 `data_process/` 目录下通过 `process.sh` 和 `deduplicate.sh` 脚本执行。此流程明确支持 **DeepCAD**, **ABC** 和 **Furniture** 数据集。

**核心思想**: 为两种不同的训练任务（VAE 和 LDM）准备两种不同的、经过优化的数据集。

1.  **阶段一：解析 (Parsing)** - `process.sh`
    *   **目的**: 将原始的 `.step` B-rep文件解析为模型可以理解的NURBS几何采样点数据。
    *   **核心脚本**: [`data_process/process_brep.py`](data_process/process_brep.py)
    *   **机制**: 脚本以并行方式遍历原始CAD文件。具体流程如下：
        1.  **加载 `.step` 文件**: 使用 [`occwl.io.load_step`](occwl.io.load_step) 函数加载 `.step` 文件，得到 `occwl.solid` 对象。这在 [`data_process/process_brep.py:172`](data_process/process_brep.py:172) 中完成。
        2.  **提取几何采样点**: 调用 [`extract_primitive`](data_process/convert_utils.py:252-317) 函数（位于 [`data_process/convert_utils.py`](data_process/convert_utils.py) 中）来从 `occwl.solid` 对象中提取曲面和边的几何采样点。
            *   **曲面采样点**: 对于每个曲面，使用 [`occwl.uvgrid.uvgrid`](occwl.uvgrid.uvgrid) 函数（在 [`data_process/convert_utils.py:292`](data_process/convert_utils.py:292) 行）以 `method="point"` 采样 `32x32` 的 UV 网格点。这些点是**在曲面几何实体上均匀采样的三维坐标点**，而非NURBS的数学控制顶点。
            *   **边采样点**: 对于每条边，使用 [`occwl.uvgrid.ugrid`](occwl.uvgrid.ugrid) 函数（在 [`data_process/convert_utils.py:308`](data_process/convert_utils.py:308) 行）以 `method="point"` 采样 `32` 个 U 网格点。这些点是**在边几何实体上均匀采样的三维坐标点**，而非NURBS的数学控制顶点。同时，还会提取每条边的起始和结束顶点。
        3.  **保存中间格式**: 这些提取出的采样点随后会经过归一化处理，并保存为中间格式（例如 `.pkl` 文件，保存在 `deepcad_parsed/` 目录中），供后续的 VAE 和 LDM 训练使用。

2.  **阶段二：去重 (Deduplication)** - `deduplicate.sh`
    *   **目的**: 消除数据冗余，为VAE和LDM提供高质量、多样化的训练样本。
    *   **为VAE训练去重**:
        *   **脚本**: [`data_process/deduplicate_surfedge.py`](data_process/deduplicate_surfedge.py)
        *   **目标**: 对**单个曲面（Surface）**和**单个边（Edge）**进行去重。VAE的任务是学习如何压缩和解压独立的几何图元，因此它需要一个不重复、多样化的曲面/边样本库。
    *   **为LDM训练去重**:
        *   **脚本**: [`data_process/deduplicate_cad.py`](data_process/deduplicate_cad.py)
        *   **目标**: 对**整个CAD模型**进行去重。LDM的任务是学习生成完整的、拓扑有效的CAD实体，因此它需要看到多样化的、不重复的完整模型样本。

这个“关注点分离”的设计（VAE学习局部几何，LDM学习全局组合）是项目成功的关键之一。

## 3. 关键代码文件与功能定位

*   **`sample.py`**: **生成/采样脚本**。这是运行训练好的模型以生成新CAD模型的入口。
    *   **核心函数**: `sample(eval_args)` ([`sample.py:35`](sample.py:35))
    *   **模型加载**: [`sample.py:56-99`](sample.py:56)
    *   **分步去噪循环**: [`sample.py:123-287`](sample.py:123)
    *   **VAE解码**: [`sample.py:289-294`](sample.py:289)
    *   **B-rep构建与保存**: [`sample.py:358-368`](sample.py:358)

*   **`trainer.py`**: **训练器定义**。包含所有VAE和扩散模型训练阶段的训练循环逻辑。
    *   **核心类**: `SurfVAETrainer`, `EdgeVAETrainer`, `SurfPosTrainer`, `SurfZTrainer`, `EdgePosTrainer`, `EdgeZTrainer`。
    *   **去噪步骤**: 在每个LDM训练器的 `train_one_epoch` 方法中，例如 [`SurfZTrainer.train_one_epoch`](trainer.py:488)。

*   **`network.py`**: **网络模型定义**。定义了所有VAE和Transformer模型的架构。
    *   **VAE模型**: `AutoencoderKL1D`, `AutoencoderKLFastEncode`, `AutoencoderKLFastDecode`。
    *   **Transformer模型**: `SurfPosNet`, `SurfZNet`, `EdgePosNet`, `EdgeZNet`。

*   **`ldm.py` / `vae.py`**: **训练入口脚本**。用于启动LDM和VAE的训练过程。

*   **`utils.py`**: **辅助函数**。包含B-rep构建 (`construct_brep`)、拓扑检测 (`detect_shared_vertex`, `detect_shared_edge`)、联合优化 (`joint_optimize`) 等关键的几何处理函数。

*   **`dataset.py`**: **数据加载器**。负责从文件中读取预处理好的B-rep控制点数据，并提供给训练器。
    *   **功能概述**: 包含 `SurfData`, `EdgeData`, `SurfPosData`, `SurfZData`, `EdgePosData`, `EdgeZData` 等数据集类，分别对应VAE和LDM训练的不同阶段。
    *   **数据处理**: 负责数据加载、并行过滤（`filter_data`）、数据增强（随机旋转）和填充（`pad_repeat`, `pad_zero`）。
    *   **批次组织**: `DataLoader` 会将 `__getitem__` 返回的单个样本在第一个维度（batch维度）上堆叠。每个数据集类返回的数据结构和维度不同，例如 `SurfData` 返回 `(batch_size, 32, 32, 3)` 的曲面UV采样点。
    *   **`surf_mask`**: 一个布尔掩码，用于指示填充后的序列中哪些是有效数据，哪些是填充数据，主要用于损失计算和注意力机制。

## 4. 如何使用和修改

*   **生成新模型**: 运行 `python sample.py --mode [abc/deepcad/furniture]`。配置文件在 `eval_config.yaml`。
*   **可视化控制点**: 在 [`sample.py:358`](sample.py:358) 之前，可以插入代码保存解码后的控制点（`surf_ncs_cad`, `edge_ncs_cad`, `unique_vertices`）为点云文件（如.xyz）进行可视化。
*   **调整输出网格精度**: 在 [`sample.py:368`](sample.py:368) 的 `write_stl_file` 函数中，减小 `linear_deflection` 和 `angular_deflection` 的值可以得到更平滑的STL网格。
*   **训练新模型**:
    1.  **准备数据**:
        *   将您的原始数据集（例如DeepCAD的`.step`文件）放置在一个目录下，例如 `datasets/deepcad_step/`。
        *   修改 `data_process/process.sh`，将其中的 `--input` 路径指向您的数据集目录。
        *   运行 `bash data_process/process.sh` 来解析数据。
        *   运行 `bash data_process/deduplicate.sh` 来为VAE和LDM训练准备去重后的数据集。
    2.  **开始训练**:
        *   运行 `train_vae.sh` 脚本独立训练曲面和边的VAE模型。
        *   运行 `train_ldm.sh` 脚本训练四个阶段的扩散模型。
