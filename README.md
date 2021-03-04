# Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach-Pytorch

TensorFlow 版本：https://github.com/hanchenchen/Attention-A-Lightweight-2D-Hand-Pose-Estimation-Approach-master/tree/dev

在手部关键点检测任务中，使用论文 **Attention! A Lightweight 2D Hand Pose Estimation Approach**  中提出的Attention Augmented Inverted Bottleneck Block等结构。

 - 尝试将 aug_block的输出与 conv的输出 concatenate，而非 add
 - 加入CPM中使用的Heatmap
 - 加入NSRM中提出的LPM

