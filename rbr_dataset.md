你现在需要为我的 InSAR Phase Unwrapping 深度学习研究构建一个完整的数据集生成项目。

目标：
构建一个真实地形驱动、可重复、可扩展的 InSAR phase unwrapping dataset generator。

不要使用 InSAR-DLPU。
数据集主要用于训练 U-Net、Transformer、Diffusion 等相位解缠模型。

==================================================
1. 总体数据生成思想
==================================================

不要只使用纯随机 Gaussian surface。

使用真实 DEM / LiCSAR 等公开地学数据作为真实地形和 InSAR 场景基础，
同时允许加入物理可控的模拟形变、相干性和失相干噪声。

最终生成：

clean unwrapped phase:
    phi_gt

clean wrapped phase:
    phi_wrapped_clean = angle(exp(1j * phi_gt))

noisy wrapped phase:
    phi_wrapped_noisy

coherence:
    gamma

noise map:
    noise

以及必要的 metadata。

基本关系必须满足：

phi_wrapped = wrap(phi_gt)

其中：

wrap(phi) = angle(exp(1j * phi))

范围统一为：

[-pi, pi)

绝对禁止直接对 wrapped phase 做普通线性截断来模拟 wrapping。

==================================================
2. 研究区域设计
==================================================

不要只使用一个区域。

建立多区域、多地貌数据集，建议至少包含：

Region A：祁连山
特点：
- 高山
- 大地形梯度
- 密集 phase fringes
- 高难度 phase unwrapping

Region B：兰州/定西黄土高原
特点：
- 黄土沟壑
- 中等至较强地形起伏
- 复杂连续地形

Region C：华北平原
特点：
- 平坦区域
- 低 phase gradient
- 稀疏 fringe
- easy samples

Region D：矿区，例如淮南/太原等
特点：
- mining subsidence
- localized deformation
- high phase gradient
- deformation-driven phase

Region E：四川/云南植被山区
特点：
- mountainous terrain
- vegetation
- low coherence
- strong decorrelation noise

程序不要把区域写死。
使用 YAML 配置文件：

configs/regions.yaml

每个 region 包含：

name
bbox
dem_source
insar_source
landcover_source
target_resolution
scene_type

方便以后增加新的研究区域。

==================================================
3. DEM 数据
==================================================

优先使用公开真实 DEM。

程序应设计统一 DEM loader，例如：

load_dem(region_config)

支持：

Copernicus DEM
SRTM
ALOS AW3D30
本地 GeoTIFF DEM

所有 DEM 最终：

1. reproject
2. resample
3. crop
4. remove invalid values
5. optional smoothing
6. normalize only when required by the model

保留原始 elevation 单位：meter。

不要在数据生成阶段破坏真实地形比例关系。

==================================================
4. InSAR phase 生成
==================================================

需要实现两种模式。

-------------------------
Mode A：DEM-driven simulation
-------------------------

根据真实 DEM 和 InSAR imaging geometry 模拟 topographic phase。

至少考虑：

radar wavelength lambda
incidence angle theta
slant range R
perpendicular baseline B_perp
satellite altitude / geometry parameters

生成：

phi_topo

参数不要固定。

每个样本随机采样合理的：

B_perp
theta
lambda / satellite type
phase offset

从而使同一个 DEM 可以产生不同 fringe density。

-------------------------
Mode B：Real InSAR assisted
-------------------------

允许读取：

LiCSAR interferogram
coherence
unwrapped phase（如果存在）

用于：

1. 提取真实 noise/coherence statistics
2. 提取真实 phase spatial characteristics
3. 后续构建 real-data-based reconstruction samples

代码结构必须预留接口：

load_licsar_data()
extract_real_phase_statistics()
extract_coherence_statistics()

即使第一版暂时只完成 Mode A，也必须保留 Mode B 的接口。

==================================================
5. deformation phase
==================================================

不要让所有样本只有 topographic phase。

增加 deformation phase：

phi_gt =
    phi_topo
    + phi_deformation
    + phase_offset

设计多种 deformation models：

1. Gaussian subsidence bowl
2. elliptical subsidence
3. multiple deformation centers
4. linear deformation gradient
5. nonlinear smooth deformation
6. localized high-gradient deformation
7. optional Mogi-like deformation

例如 Gaussian deformation：

d(x,y) =
A * exp(
    -((x-x0)^2/(2*sigma_x^2)
    +(y-y0)^2/(2*sigma_y^2))
)

转换 LOS displacement 到 phase：

phi_def =
4*pi/lambda * d_LOS

注意符号约定必须在项目文档中明确。

随机改变：

A
x0
y0
sigma_x
sigma_y
orientation

部分样本：

terrain only

部分样本：

terrain + deformation

部分样本：

deformation dominant

==================================================
6. coherence simulation
==================================================

生成空间变化的 coherence map：

gamma(x,y)

范围：

0 < gamma <= 1

不要简单地让整幅图 coherence 是一个常数。

coherence 应受到：

terrain
slope
land cover
spatial random field

影响。

如果存在真实 land-cover 数据，可以考虑：

water -> low coherence
dense vegetation -> low/medium coherence
bare land -> medium/high coherence
urban -> generally high coherence

但必须保留随机性。

第一版可以实现：

gamma =
f(
    landcover,
    slope,
    spatial_random_field
)

spatial_random_field 必须具有空间相关性，
不要使用完全独立的 pixel-wise random noise。

可以使用：

Gaussian random field
correlated noise
Sequential Gaussian Simulation（如果实现成本合理）

生成不同 coherence difficulty：

easy:
0.7 - 0.95

medium:
0.5 - 0.8

hard:
0.3 - 0.6

extreme:
0.15 - 0.4

注意这些区间用于数据生成配置，不要求每个像素严格限制在同一个常数。

==================================================
7. decorrelation noise
==================================================

这是非常重要的一部分。

不要简单：

wrapped_phase += GaussianNoise

必须尽可能模拟 InSAR complex-domain noise。

推荐流程：

clean complex interferogram:

z_clean = exp(1j * phi_gt)

根据 coherence gamma 生成 correlated complex noise / noisy interferogram：

z_noisy = simulate_insar_noise(
    z_clean,
    gamma
)

然后：

phi_wrapped_noisy = angle(z_noisy)

这样得到的 phase noise 应随着 coherence 降低而增强。

必须保证：

high coherence -> low phase noise
low coherence -> strong phase noise

同时实现一个简单 Gaussian phase noise 模式作为 baseline，
但默认使用 complex-domain InSAR noise。

==================================================
8. phase wrapping
==================================================

统一函数：

def wrap_phase(phi):
    return np.angle(np.exp(1j * phi))

生成：

phi_gt
phi_wrapped_clean
phi_wrapped_noisy

必须做自动 consistency test：

abs(
angle(exp(1j*phi_gt))
-
phi_wrapped_clean
)

应接近数值误差。

==================================================
9. patch generation
==================================================

训练数据不要直接保存整景。

默认：

patch_size = 128

支持：

64
128
256

使用 sliding-window cropping。

支持：

stride
overlap
random crop

例如：

128x128
stride = 64

必须过滤：

NaN
NoData
大量无效区域
几乎完全平坦且无有效信息的 patch

但不要过度过滤 easy samples。

==================================================
10. dataset balance
==================================================

必须控制数据分布。

建议：

30% easy
30% medium
25% hard
15% extreme

同时控制：

flat terrain
moderate terrain
steep terrain

以及：

terrain-only
terrain + deformation
deformation-dominant

避免数据集全部由复杂山区组成。

==================================================
11. data augmentation
==================================================

允许：

horizontal flip
vertical flip
90/180/270 rotation

对于 phase：

禁止普通 interpolation 后不处理 phase periodicity。

如果进行需要插值的空间变换，
应优先在 complex representation：

cos(phi)
sin(phi)

上进行，再恢复：

phi = atan2(sin_phi, cos_phi)

GT unwrapped phase的空间变换需要保持连续相位值。

==================================================
12. train / val / test split
==================================================

这是关键要求。

不要简单地：

shuffle all patches
random 80/10/10 split

因为相邻 patch 高度相关，会导致 spatial leakage。

必须支持：

Spatial Split

以及：

Cross-Region Split

推荐：

Train:
Qilian
Lanzhou/Loess
North China Plain
Sichuan subset

Validation:
这些区域中完全独立的 spatial blocks

Test-1:
训练区域中的独立 spatial blocks

Test-2:
完全没有参与训练的新区域

Test-3:
real LiCSAR interferograms

这样可以评估：

in-domain performance
cross-region generalization
real-data generalization

==================================================
13. 输出格式
==================================================

每个 sample 至少保存：

phi_gt
phi_wrapped_clean
phi_wrapped_noisy
coherence
dem
deformation_phase
noise_map

metadata 包含：

region
longitude/latitude or source window
DEM source
satellite
wavelength
incidence_angle
B_perp
coherence_level
deformation_type
random_seed
patch_size
resolution

推荐：

dataset/
    train/
    val/
    test_in_domain/
    test_cross_region/
    metadata/

不要生成数万个独立 PNG 作为主要训练格式。

推荐：

HDF5
Zarr
NPZ shards

同时提供少量 PNG/TIFF visualization samples。

==================================================
14. dataset statistics
==================================================

构建完成后自动统计：

sample count

phase:
min
max
mean
std

coherence:
mean
std
histogram

terrain:
elevation range
slope distribution

phase gradient:
mean
max
histogram

不同 region 数量

不同 difficulty 数量

不同 deformation type 数量

生成：

dataset_statistics.json

以及：

figures/
    coherence_distribution.png
    phase_gradient_distribution.png
    terrain_distribution.png
    region_distribution.png

==================================================
15. 数据质量检查
==================================================

自动随机抽取至少 100 个样本检查：

1. wrapped/unwrapped consistency
2. NaN / Inf
3. phase range
4. coherence range
5. empty patches
6. abnormal gradient
7. duplicate samples

生成 QA report。

特别检查：

wrap(phi_gt) ≈ phi_wrapped_clean

以及：

phi_wrapped_noisy ∈ [-pi, pi)

==================================================
16. 可复现性
==================================================

所有随机操作必须支持：

seed

每个 sample 保存自己的：

random_seed

配置全部放 YAML。

例如：

configs/
    dataset.yaml
    regions.yaml
    simulation.yaml

不要在 Python 文件中硬编码实验参数。

==================================================
17. 项目结构
==================================================

建议：

insar_dataset/
│
├── configs/
│   ├── dataset.yaml
│   ├── regions.yaml
│   └── simulation.yaml
│
├── src/
│   ├── data/
│   │   ├── dem.py
│   │   ├── licsar.py
│   │   └── landcover.py
│   │
│   ├── simulation/
│   │   ├── geometry.py
│   │   ├── phase.py
│   │   ├── deformation.py
│   │   ├── coherence.py
│   │   └── noise.py
│   │
│   ├── dataset/
│   │   ├── patch.py
│   │   ├── split.py
│   │   ├── writer.py
│   │   └── statistics.py
│   │
│   └── utils/
│
├── scripts/
│   ├── download_data.py
│   ├── generate_dataset.py
│   ├── validate_dataset.py
│   └── visualize_dataset.py
│
├── tests/
│
├── README.md
└── requirements.txt

==================================================
18. 实现原则
==================================================

第一阶段不要一次实现所有复杂功能。

按以下顺序开发：

Stage 1:
真实 DEM -> phase -> wrapping -> patch

Stage 2:
deformation simulation

Stage 3:
spatial coherence simulation

Stage 4:
complex-domain decorrelation noise

Stage 5:
multi-region dataset generation

Stage 6:
LiCSAR real-data-assisted statistics

Stage 7:
dataset QA + visualization

每完成一个 Stage：

1. 写 unit test
2. 生成 10-100 个 samples
3. visualization
4. 检查物理合理性
5. 再进入下一阶段

==================================================
19. 最重要的原则
==================================================

这个项目的目标不是单纯生成很多图片。

核心目标是构建：

Real terrain
+
Physical InSAR geometry
+
Controllable deformation
+
Spatially varying coherence
+
Realistic decorrelation noise

从而得到：

Realistic InSAR Phase Unwrapping Dataset

数据集必须能够支持后续论文中的：

1. supervised phase unwrapping
2. diffusion phase unwrapping
3. noise robustness experiments
4. cross-region generalization
5. cross-coherence experiments
6. real InSAR validation

优先保证：

物理合理性 > 数据数量
区域独立性 > 随机切分
真实噪声特性 > 简单 Gaussian noise
可复现性 > 临时脚本

==================================================
20. 开始工作方式
==================================================

先不要直接大量下载数据或生成完整数据集。

第一步：

1. 检查当前项目目录。
2. 根据以上要求生成完整项目架构。
3. 编写 README.md。
4. 编写 configs/*.yaml。
5. 实现 Stage 1。
6. 使用一个小 DEM 区域生成约 20 个测试样本。
7. 自动生成 visualization。
8. 运行 unit tests。
9. 汇报：
   - 当前完成内容
   - 数据尺寸
   - phase 范围
   - DEM 范围
   - consistency test
   - visualization 文件位置
   - 下一阶段计划

不要在 Stage 1 验证通过之前直接开始批量生成完整数据集。