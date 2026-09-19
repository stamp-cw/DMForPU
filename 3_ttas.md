为了增强低频结构信息与高频细节信息之间的交互能力，本文提出一种基于小波分解的窗口频分交叉注意力模块（Window Wavelet Frequency Cross Attention, WWFCA）。该模块先在局部窗口内进行小波频域分解，然后通过交叉注意力机制实现不同频带之间的信息融合。设输入特征图为：
\begin{gathered}\mathrm{X}\in {\mathrm{R}}^{\mathrm{H}\times \mathrm{W}\times \mathrm{C}}\#\left(16\right)\end{gathered}
其中，H、W 和 C 分别表示特征图的高度、宽度和通道数。先通过窗口划分操作（Split Window, SW）将特征图划分为多个不重叠的局部窗口：
\begin{gathered}{\mathrm{X}}_{\mathrm{w}}=\mathrm{SW}\left(\mathrm{X}\right)\#\left(17\right)\end{gathered}
每个窗口特征表示为：
\begin{gathered}{\mathrm{X}}_{\mathrm{w}}\in {\mathrm{R}}^{\mathrm{M}\times \mathrm{M}\times \mathrm{C}}\#\left(18\right)\end{gathered}
其中 M 表示窗口大小。为了获得不同尺度的频率信息，对每个窗口特征 X_{w} 进行二维离散小波变换（Discrete Wavelet Transform, DWT），得到四个子频带：
\begin{gathered}{\mathrm{X}}_{\mathrm{LL}},{\mathrm{X}}_{\mathrm{LH}},{\mathrm{X}}_{\mathrm{HL}},{\mathrm{X}}_{\mathrm{HH}}=\mathrm{DWT}\left({\mathrm{X}}_{\mathrm{w}}\right)\#\left(19\right)\end{gathered}
其中，X_{LL} 表示低频子带，X_{LH} 和 X_{HL} 分别表示水平和垂直高频子带，X_{HH} 表示对角线高频子带。各子频带特征尺寸为:
\begin{gathered}{\mathrm{X}}_{\mathrm{LL}},{\mathrm{X}}_{\mathrm{LH}},{\mathrm{X}}_{\mathrm{HL}},{\mathrm{X}}_{\mathrm{HH}}\in {\mathrm{R}}^{\frac{\mathrm{M}}{2}\times \frac{\mathrm{M}}{2}\times \mathrm{C}}\#\left(20\right)\end{gathered}
为了进行频域信息交互，将三个高频分量进行拼接为： 
\begin{gathered}{\mathrm{X}}_{\mathrm{H}}=\mathrm{Concat}\left({\mathrm{X}}_{\mathrm{LH}},{\mathrm{X}}_{\mathrm{HL}},{\mathrm{X}}_{\mathrm{HH}}\right)\#\left(21\right)\end{gathered}
在频域注意力计算中，本文以低频分量作为 Query，高频分量作为 Key 和 Value，从而实现结构信息对细节信息的自适应融合。首先进行线性投影：
\begin{gathered}{\mathrm{Q}}_{\mathrm{LL}}={\mathrm{W}}_{\mathrm{Q}}{\mathrm{X}}_{\mathrm{LL}},{\mathrm{K}}_{\mathrm{H}}={\mathrm{W}}_{\mathrm{K}}{\mathrm{X}}_{\mathrm{H}},{\mathrm{V}}_{\mathrm{H}}={\mathrm{W}}_{\mathrm{V}}{\mathrm{X}}_{\mathrm{H}}\#\left(22\right)\end{gathered}
其中，W_{Q}、W_{K} 和 W_{V} 分别为对应的线性变换矩阵。然后利用缩放点积注意力机制计算交叉注意力：
\begin{gathered}\text{Attn}\left({\mathrm{Q}}_{\mathrm{LL}},{\mathrm{K}}_{\mathrm{H}},{\mathrm{V}}_{\mathrm{H}}\right)=\text{Softmax}\left(\frac{{\mathrm{Q}}_{\mathrm{LL}}{\mathrm{K}}_{\mathrm{H}}^{\top }}{\sqrt{\mathrm{d}}}\right){\mathrm{V}}_{\mathrm{H}}\#\left(23\right)\end{gathered}
其中d表示特征维度的缩放因子。通过上述机制，低频结构信息能够自适应地从不同高频子带中聚合有用的细节特征，从而提升特征表达能力。最后，对注意力输出进行归一化与正则化处理：
\begin{gathered}\mathrm{Z}=\text{Dropout}\left(\text{LayerNorm}\left(\text{Attn}\left({\mathrm{Q}}_{\mathrm{LL}},{\mathrm{K}}_{\mathrm{H}},{\mathrm{V}}_{\mathrm{H}}\right)\right)\right)\#\left(24\right)\end{gathered}
其中，LayerNorm 表示层归一化操作，Dropout 用于防止过拟合。通过引入WWFCA模块，模型能够在不同空间位置之间建立全局关联，并有效提升对复杂相位结构与不连续区域的表达能力，从而增强相位解缠模型在复杂 InSAR 场景下的性能与泛化能力。
