# Qwen 采样与模式默认值

服务的采样器支持 temperature、top-k、top-p、min-p、presence 和 repetition penalty；temperature 为 0 时使用 greedy。模型包的 `generation_config.json` 和请求覆盖决定生效参数。CUDA 的设备贪心捷径只用于无惩罚的 greedy；随机采样和惩罚使用完整 logits 在主机采样，不能把设备贪心路径的速度等同于随机采样。

[Qwen3 原版混合模型](https://huggingface.co/Qwen/Qwen3-0.6B)建议 thinking 使用 temperature 0.6、top-p 0.95、top-k 20，non-thinking 使用 0.7、0.8、20；thinking 不建议 greedy。[Qwen3.8 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)给出的 thinking 参数为 1.0、0.95、20 和 presence 0，non-thinking 为 0.7、0.8、20 和 presence 1.5。两者均建议 min-p 0、repetition penalty 1。不同任务和变体没有一个经过本项目实测的通用最优值，显式请求可以覆盖参数。

原版 Qwen3 生成配置通常提供 thinking 参数，服务此前却默认关闭 thinking。现按模型卡 front matter 中原版 Qwen3 的已知 base_model 标识启用默认 thinking，关闭 thinking 时应用其温度和 top-p 建议；模型包显式关闭采样时保留 greedy。Qwen3.8 维持已有模式预设。Qwen3-VL、2507 等其他变体不按架构名猜测模式或套用这些预设。模型卡缺失、改写或 base_model 不匹配时仍使用通用默认；调用方可显式设置模式与采样字段。

这是模型默认参数和提示模板行为的变化，不改变采样算法。它可能使原先默认 non-thinking 的 Qwen3 请求改为更长的 thinking 生成，输出、耗时和资源消耗均会变化；需要固定行为的调用方应显式传 `enable_thinking` 和采样参数。性能与质量须在实际任务和设备上分别衡量，不能以单次回答断言最优。
