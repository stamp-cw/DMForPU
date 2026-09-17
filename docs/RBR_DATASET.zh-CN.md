# RBR v1：真实地形驱动相位解缠数据集

RBR 定义为 Real-terrain Based Reconstruction。v1 使用 Copernicus GLO-30 DSM 驱动可控 InSAR 仿真；不使用 InSAR-DLPU，也不把模拟样本描述成真实干涉图。

## 数据来源与区域

真实高程来自 Copernicus DEM 2021 GLO-30 公共 COG。选取祁连山、兰州/定西黄土高原、华北平原、淮南矿区和川滇植被山区五个区域。前四个区域用于空间分离的训练、验证和区域内测试；川滇区域完全留出，只进入跨区域测试。

源数据声明：produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

## 与本项目的兼容协议

- RBR128、RBR64、RBR32 各含 5,000 张训练样本，其中最后 500 张标记为独立空间验证块。
- 每个尺寸包含 0、5、10、20、30 dB 五个测试文件，每档 1,000 张。
- 测试集每档包含 700 张区域内独立空间块和 300 张川滇跨区域场景。
- `psi`、`phi`、`snr`、`scene_id` 与 GFS/RME 完全同接口；相位范围限制在模型当前使用的 [-14π,14π]。
- 同一尺寸的五个 SNR 文件共享场景、DEM、真值、相干性和物理参数，只重新生成复数域噪声。

## 物理与噪声模型

地形相位采用 `-4π B_perp h / (λ R sinθ)`，随机采样 Sentinel-1 C、ALOS-2 L 和 TerraSAR-X X 波段参数。正 LOS 形变定义为朝向传感器，并以 `+4π d_LOS/λ` 转换为形变相位。

形变包括地形-only、Gaussian/椭圆沉降、多中心、线性梯度、非线性平滑和局部高梯度。相干性由难度档位、真实地形坡度、区域类型和空间相关随机场生成。默认噪声在复数干涉域加入，局部幅度由相干性和名义 SNR 共同控制。

HDF5 还保存 `phi_wrapped_clean`、`coherence`、`dem`、`deformation_phase`、`noise_map`，以及区域、空间分区、卫星、成像几何、形变类别、随机种子和源窗口坐标。

## 可复现生成

```powershell
.\.venv\Scripts\python.exe -B -u experiments/generate_rbr_dataset.py --update-manifest
```

## 区域矢量与图册

- `data/rbr_sources/regions/rbr_regions.shp`：五个完整研究区边界，EPSG:4326。
- `data/rbr_sources/regions/rbr_spatial_splits.shp`：训练、验证、域内测试与跨区域测试的 13 个空间分区面，EPSG:4326。
- 同目录提供 UTF-8 GeoJSON、PRJ、CPG 和生成摘要，便于 QGIS、ArcGIS 与脚本读取。
- `output/pdf/rbr_dataset_atlas.pdf`：九页 RBR 数据集图册；逐页 PNG 位于 `output/pdf/rbr_dataset_atlas_pages/`。

重新生成矢量和图册：

```powershell
.\.venv\Scripts\python.exe experiments/create_rbr_region_vectors.py
.\.venv\Scripts\python.exe experiments/make_rbr_dataset_atlas.py
```

配置位于 `configs/rbr/`。LiCSAR v1 只保留明确抛出 `NotImplementedError` 的接口，等真实数据获取和参考真值协议确定后再启用。
