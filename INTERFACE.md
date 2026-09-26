# 接口契约(成员A 按此实现)

## detect()
- 输入:图片文件路径(jpg/png,路径可能含中文)
- 输出:检测结果列表,**无检测时返回空列表 []**
- 每个元素格式:
  {"class": "crack", "conf": 0.87, "bbox_xyxy": [x1, y1, x2, y2],
   "length_px": 44.8, "width_px": 30.0, "area_px": 999.0}
  - class:英文小写类别名
  - conf:置信度,0~1 的浮点数
  - bbox_xyxy:原图像素坐标,xyxy = 左上角 + 右下角
  - length_px:裂缝长度(像素,对角线近似);旧服务无此字段时为 None
  - width_px:裂缝宽度(像素,短边近似);旧服务无此字段时为 None
  - area_px:检测框面积(像素²,剥落建议看此值);旧服务无此字段时为 None

## get_quantification()
- 输入:图片文件路径
- 输出:顶层量化汇总 dict,形如
  {"crack_count": 5, "spalling_count": 5, "total_defects": 10,
   "max_crack_length_px": 67.5, "avg_crack_length_px": 48.4,
   "max_spalling_area_px": 4487.2}
- 服务未返回时为 {}

## get_heatmap()
- 输入:图片文件路径
- 输出:缺陷密度热力图(PIL.Image);服务未返回热力图时为 None

## generate_report()
- 输入:detect() 的输出
- 输出:Markdown 格式巡检报告文本
- 无检测时输出「未发现明显缺陷」的报告;有检测时包含数量、类型、置信度、风险等级、整改建议

## generate_video_report()
- 输入:视频抽帧检测结果 [{"time_sec": 1.0, "detections": [...]}, ...]
- 输出:Markdown 格式视频巡检报告(抽检帧数、检出帧数、时间分布、风险评级、整改建议)

## 变更规则

改函数签名或返回格式前,先在群里通知对方并更新本文件。

（2026-09:检测服务新增 length_px/width_px/area_px 量化字段及
get_quantification/get_heatmap 两个辅助函数,已由成员A 同步到此契约。）
