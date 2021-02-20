# slam_evaluation

基于EVO的slam评估脚本。

results包含A-loam,lego-loam,sc-lego-loam的 pose和times数据序列（kitti vo的GT作为ref）

基于evo的APE和RPE对总体和局部结果进行统计结果生成。

输入数据包含过滤动态目标(_d)和完整数据（_raw）。

可调参数主要针对RPE的计算单位和计算窗口：

"""
    frame 一般不涉及单位，可以进行逐帧对比 取值delta=1，不太影响评估误差数量
    使用frame，1，利用最小粒度单位进行详细评估，寻找异常帧位置，从而修改代码。
    meters 可以通过设置10/20/50/100 m ，会缩小评估误差数量
    degrees=radians, filter_pairs_by_angle
"""
