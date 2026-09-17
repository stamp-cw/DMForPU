# GFS128 与 RME128 数据集图册

- `dataset_atlas_gfs128_rme128.pdf`：A4 横版图册，共 8 页。
- `dataset_atlas_gfs128_rme128_pages/`：与 PDF 对应的 8 张高分辨率 PNG，便于论文挑图和幻灯片排版。
- `dataset_atlas_gfs128_rme128.json`：代表样本索引、生成时间和统计量，保证选择过程可复现。

## 页面内容

1. 数据协议与代表样本索引
2. GFS128 干净展开相位形态
3. RME128 干净展开相位形态
4. GFS128 相同场景的 30/20/10/5/0 dB 对照
5. RME128 相同场景的 30/20/10/5/0 dB 对照
6. GFS128 训练输入与真值样本对
7. RME128 训练输入与真值样本对
8. 相位跨度、相位标准差、空间梯度和噪声圆周误差统计

重新生成：

```powershell
.\.venv\Scripts\python.exe -B experiments\make_dataset_atlas.py
```
