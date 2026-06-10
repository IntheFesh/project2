# Resume Bullets, Email Templates & Anti-Pushback Prep (VLA-Collapse-Recover)

For Yue's use only. Update `<TODO>` placeholders before sending.

---

## A. Resume Bullets

### A.1 中文 · 国内求职版(LLM / Agent / RAG 工程岗)

**项目名:VLA-Collapse-Recover · 视觉语言动作模型扰动鲁棒性诊断框架**
*(单 GPU、< 1 GPU-day · 个人项目 · GitHub: IntheFesh/project2)*

- **端到端搭建** SmolVLA(610M 参数 VLA 模型)在 LIBERO-Plus 基准上的"扰动塌陷 - LoRA 恢复 - 配对统计"全流程评测管线;1,790 条 per-episode 配对数据,5 个扰动家族 × 2 个难度等级。
- **三组预注册配对假设检验** + Holm-Bonferroni 多重比较校正:验证 LoRA + 标准增广带来 +7.4pp 鲁棒性提升(95% CI [+2.8, +11.9],McNemar p ≈ 0.0018);**关键发现:** 改进可迁移至训练时**未见过**的扰动家族(layout +15.0pp,p ≈ 0.0072),而扰动家族对齐的针对性增广**并未带来系统性优势**(三个 Holm 校正后假设全部不显著)。
- **工程实现**:LeRobot 0.5.2 + PEFT-LoRA + crash-safe 断点续跑评测,自定义动态超时按 cell 规模自适应(解决 noise 家族 episode 跑满 max_steps 的实际工程问题),Phase 2-5 全量结果可一行命令(`make stats`)字节级精确重现。
- **诚实标注三条负向发现**(viewpoint 全 0%、noise 在 LoRA 后退化、C-lighting 不及 B),保留为科学结论而非粉饰为局限性。

### A.2 中文 · 简短版(放在简历第二项目位,2-3 行)

**VLA-Collapse-Recover** · 视觉语言动作模型扰动鲁棒性诊断 · [GitHub](https://github.com/IntheFesh/project2)
- SmolVLA + LoRA 端到端管线;1,790 配对 episode,三组预注册配对统计 + Holm 校正
- 发现 LoRA 改善可迁移至 held-out 家族(+15pp,p≈0.007),增广家族对齐无系统性优势(全 Holm-p>0.05)
- 技术栈:LeRobot · PEFT-LoRA · paired bootstrap + McNemar + Holm-Bonferroni · crash-safe eval harness

### A.3 English · PhD / RA Application

**VLA-Collapse-Recover — diagnostic eval framework for vision-language-action models**
*(single RTX 5090, < 1 GPU-day · solo project · [github.com/IntheFesh/project2](https://github.com/IntheFesh/project2))*

- Built end-to-end LIBERO-Plus collapse-and-recovery pipeline for SmolVLA (LoRA r=16/α=32; 1,790 paired episodes across 5 perturbation families × 2 difficulty levels).
- **Designed and pre-registered three paired hypotheses** with Holm-Bonferroni correction. **Finding:** LoRA + standard photometric augmentation lifts robustness by +7.4 pp pooled (95% CI [+2.8, +11.9], McNemar p ≈ 0.0018) and transfers +15.0 pp to a never-augmented `layout` family (held-out, p ≈ 0.0072); perturbation-family-matched augmentation provides no systematic gain over generic augmentation (all three Holm-corrected p > 0.05).
- Implemented a crash-safe, resume-able evaluation harness with per-cell adaptive timeouts; full statistical pipeline regenerates byte-identically via `make stats`.
- **Preserved three honest negative findings as conclusions, not bugs:** viewpoint stays at 0% under all conditions (2-D augmentation cannot confer 3-D viewpoint invariance, flagged up-front); noise *worsens* under LoRA (24.4% → 3.3%); C's lighting underperforms B's at the chosen augmentation magnitude.
- Stack: SmolVLA · LeRobot 0.5.2 · PEFT-LoRA · paired bootstrap + McNemar + Holm-Bonferroni.

### A.4 English · LinkedIn / Portfolio Short Form (3 lines)

**VLA-Collapse-Recover** — Diagnostic eval framework for vision-language-action models. SmolVLA + LoRA, single GPU, < 1 GPU-day. 1,790 paired episodes, three pre-registered hypotheses with Holm-Bonferroni correction.

**Key finding:** LoRA's task-representation improvement transfers across perturbation families independently of which family was augmented (+15 pp on held-out, p ≈ 0.007); perturbation-family-matched augmentation provides no systematic gain over generic. Diagnostic, not method.

**Stack:** SmolVLA · LeRobot · PEFT · paired bootstrap + McNemar + Holm-Bonferroni · GitHub Actions CI.

---

## B. Email Templates

### B.1 Cold email to a PhD advisor (English, embodied AI / VLA lab)

> **Subject:** Diagnostic study of LoRA fine-tuning in VLAs — `<lab>` PhD interest
>
> Dear Prof. `<TODO: name>`,
>
> I'm `<TODO: name>`, an M.A. Statistics student (US) with a B.S. in Information and Computational Science (China). I'm writing because your `<TODO: specific paper / direction>` directly motivates a question I just spent a month investigating, and I'd love your reaction before applying to your PhD program this cycle.
>
> The question: does perturbation-targeted LoRA fix the visual representation of an open VLA, or does it patch symptoms on the augmented families? I ran a paired three-condition experiment on SmolVLA + LIBERO-Plus (1,790 episodes, McNemar + Holm-corrected paired bootstrap, all on a single RTX 5090 in under a GPU-day). Headline: LoRA + photometric augmentation does lift robustness broadly (+7.4 pp pooled, p ≈ 0.002) and **transfers to a never-augmented held-out family** (+15 pp on `layout`, p ≈ 0.007) — yet perturbation-family-matched augmentation provides no systematic advantage over generic augmentation (three Holm-corrected family-level tests all fail to reject). The improvement seems to be about LoRA's effect on task representation, not augmentation-family matching.
>
> The single-page summary is attached; the full case study is at https://github.com/IntheFesh/project2/blob/main/report/case_study.md and the repo (code + raw 1,790-row CSV + `make stats` reproduction) is at https://github.com/IntheFesh/project2.
>
> I'd be grateful for any reaction, including "you're missing prior work on X" or "the natural follow-up is Y". If you have an opening for a fall-`<YYYY>` PhD student interested in embodied AI / VLA diagnostics, I'd very much like to be considered.
>
> Thank you for your time.
>
> Best, `<TODO: name>`

### B.2 Cold email to a Chinese employer (中文, LLM/Agent/RAG 岗)

> **主题:** 应聘 `<TODO: 岗位名>` · 简历与项目附件
>
> 您好,`<TODO: 称呼>`:
>
> 我是 `<TODO: 姓名>`,统计学硕士在读,有信息与计算科学本科背景。看到贵司在招 `<TODO: 岗位>`,我把简历附上,顺便简单介绍我最近做完的两个端到端项目,看是否匹配:
>
> 1. **PolicyArena**(LLM Agent 策略合规)· Qwen3-8B + QLoRA + LangGraph + RAG · 把策略合规成功率从 53.1% 提升到 96.9%(Holm-Bonferroni 校正后 p < 0.001),全流程 Langfuse 可观测、Docker Compose 一键部署。
>
> 2. **VLA-Collapse-Recover**(具身智能模型扰动鲁棒性诊断)· SmolVLA + LoRA + LIBERO-Plus · 单 GPU、< 1 GPU-day、1,790 配对 episode、三组预注册配对假设检验。**核心发现:** LoRA 改善可迁移至训练未见的扰动家族(+15pp,p≈0.007),而扰动对齐的针对性增广**并未带来系统性优势**——这是诚实的负向发现,我在 README 里保留为结论而非粉饰。
>
> 我的工程能力体现在:能独立把"研究问题 → 数据/训练管线 → 评测 → 配对统计 → 可复现仓库"全栈走完,中间踩过的工程坑(SR=0/10 调到 64%、磁盘撑爆、断点续跑等)都在 commit log 里有迹可循;统计背景让我在小样本设计、配对假设检验、置信区间报告上不会随便。
>
> 期待沟通的机会,可微信 `<TODO: 微信号>` / 邮箱 `<TODO: email>`。
>
> 谢谢您的时间。
>
> `<TODO: 姓名>` · `<TODO: 日期>`

### B.3 Follow-up (English, 7 days after no reply)

> **Subject:** Re: `<original subject>`
>
> Dear Prof. `<TODO: name>`,
>
> A short follow-up to my email of `<date>` in case it ended up in a busy thread. The summary one-pager is attached again; I'd value any reaction at all, including a one-line "not a fit this cycle."
>
> Best, `<TODO: name>`

---

## C. Anti-Pushback Prep · "他们可能会怎么质疑,我怎么回应"

面试时被 push back 是常态。**不要表现得受冒犯**;每一条 push back 都是机会展示你比简历更深。下面列我能想到的最常见质疑,每条都给"诚实承认 + 加分回应"的双层结构。

### C.1 "Single seed 不算 result,你不能 publish 这个。"

**承认:** 您说得对,单 seed 是这个项目最大的方法学局限。LIBERO-Plus 的 init state 是确定的,所以*配对*统计在单 seed 内仍然有效——我能告诉您在这组确定 init state 下 B 和 C 谁更好,paired bootstrap CI 也是合法的。但您说的对,seed 间方差我没有报。

**加分:** 三 seed 扩展的预算账我算过——大约再加 10 GPU-h,完全在我原始 ≤5 GPU-day 的预算内。我没做是因为这个项目从一开始就定位为 portfolio 而非 publication;如果我们要往会议投,我会先把它做到 3 seed,这是 ROI 最高的下一步。

### C.2 "C 没有打赢 B,你这个 method 不 work 啊?"

**承认:** 是的,如果用"method paper"的标准衡量,这个项目没产出"我们的方法赢了 baseline"的结论。在三个 in-distribution 家族上 C 和 B 的 paired test 全部 Holm-不显著,lighting 甚至是 C 比 B 差 10.8pp。

**加分:** 但这个项目从开始就**不是 method paper**,是 diagnostic。我们想问的问题是"LoRA 改善的机制是什么"。答案 H1+H2+H3 联合读出来是:**改善是真的,但不来自 augmentation family matching,来自 LoRA 对 task representation 本身的改变**(因为它能迁移到从未增广过的 layout)。这是个**有发表价值的发现**(LoRA 微调的机制可能比 augmentation literature 想的更"非定向"),只是它不长 method paper 的样子。如果您觉得这个 framing 更值得探索,我可以把后续工作放到这个方向。

### C.3 "你那个 viewpoint 全 0%,是不是 LoRA 没训对?"

**承认:** Viewpoint 在 A / B / C 三条件下都是 0%,我也确实想过这是不是 bug。但有四条独立证据排除"训练 / 加载没对"——B/C 在 lighting / texture / layout 上都从 base 显著抬升;adapter 加载链路经过 lerobot-eval 端到端验证;loss 正常下降;runtime per step 与 base 一致。

**加分:** 真正的解释是 C 的 viewpoint 增广是个**2-D image-space proxy**(`RandomPerspective` + `RandomAffine`),它**结构上不能**赋予 policy 3-D 视角不变性——3D 相机移动改变了 occlusion / parallax / 光照,这些 2-D warp 都给不了。我在 `data/augment/visual_aug.py` 的代码注释里**事先标注过**这是个 proxy 而非 faithful augmentation;0% 的结果正是这个 proxy 假设*应该*预测的结果。真正解决 viewpoint 需要 simulator 重渲染(LIBERO 在 robosuite 里完全能做,只是单 GPU 预算内做不下)。这是个干净的、可证伪的、可后续工作的结论。

### C.4 "你为什么只测 LIBERO-spatial 一个 suite?"

**承认:** 是的,这是 generalization 上的硬限制——只有 spatial 一个 suite,任务都是"pick up the black bowl + somewhere"这一类。在 object / goal / 10 上 LoRA 的迁移模式可能不一样。

**加分:** 单 suite 是预算决策——432 episode 的训练数据 + 4-6 小时 / 条件的评测,跑完一个 suite 已经吃掉一天 GPU。我选 spatial 是因为 LIBERO-Plus 的 spatial cell 选择器(`select_cell_task_ids`)是 deterministic 的、`init_states=True` 让配对评测严格有效——这是配对统计能用的前提。Cross-suite 是我列的最优先 follow-up 之一。

### C.5 "为什么不直接用 LoRA-Plus / DoRA / 别的新 PEFT 方法?"

**承认:** 用的是 baseline LoRA(r=16, α=32),不是最新的 PEFT 变体。

**加分:** 因为这个项目是 *diagnostic*,我希望 LoRA 是个尽可能"标准"的轮子,这样发现的现象可以被读为 "LoRA 微调的普遍性质",而不是 "我们这个 specific 变体的副作用"。如果用 DoRA / VeRA / LoRA-FA,任何意外结果都会变成"是不是 DoRA 那个 magnitude vector 在搞鬼"——不利于诊断。Method comparison 是另一个项目;这个项目是 mechanism diagnosis。

### C.6 "你这个 noise 上 LoRA 让模型更差,听起来像 bug。"

**承认:** 第一眼看像 bug,我也这么想过。但 noise 在 B 上从 A 的 24.4% 降到 3.3% 是 **30 episode 的 McNemar 强烈显著**,B-only / A-only discordant 大幅倾向 A,不是噪声波动。

**加分:** 我**没有把它当成结论 claim**——README 和 case study 里都写明"open question, not a result we claim"。但它是个**可证伪的假设**:photometric augmentation 把 visual encoder 的激活向"clean-looking"先验偏移,additive sensor noise 在那个先验下破坏更严重。要验证只需要在 encoder 上 probe noise-frequency sensitivity 或做 augmentation magnitude ablation——下一步该做的事情很明确。承认 "we don't yet know why" 比假装 "we have an answer" 更科学。

### C.7 "你统计学硕士做这种东西能有什么独特视角?"

**(这条经常以表扬包装的形式出现:"很有意思你能从统计角度看这个。")**

**回应(不要太谦虚):** 我的统计训练在这个项目上的实际产出是**三件具体的事**:其一,*预注册* 三组假设(而不是事后看了数挑最显著的报);其二,*配对* 统计设计(LIBERO-Plus 的 deterministic init state 让 paired test 比 unpaired 在小样本上功效高一个量级——这正是 5-15pp 的 method gap 能被检出的原因);其三,*多重比较校正*(H2 的三个 family-level test 用 Holm-Bonferroni step-down,而不是 Bonferroni 太保守、也不是不校正)。这些在 robotics paper 里其实少见——大多数 VLA 评测就是 "我们方法 SR x%, baseline SR y%",连置信区间都不报。我觉得这是个真实的、未被充分利用的方法论 gap。

### C.8 "这个项目花了你多久?"

**诚实回答:** 端到端大概 `<TODO: 三周 / 一个月>`,其中 GPU 实际占用 < 1 day。最大的时间花费**不在训练或评测**,在前期的 environment 调通(SmolVLA + LIBERO-Plus + LeRobot 三个 repo 版本对齐踩过不少坑,Phase 1 的 SR=0/10 调到 64% 调了几天)和后期的写作(README 重写、案例研究、技术报告)。这两件事都是真实研究工程的一部分,不是浪费的时间。

### C.9 "如果让你再做一次,你会怎么做不一样?"

**诚实回答,展示反思能力:** 三件事。其一,**augmentation magnitude 早做 ablation**——C-lighting 不及 B 大概率是因为我把幅度直接锁在 level-5 全幅度,该先在 ±0.1/0.2/0.3/0.4 做 grid。其二,**3 seed 从一开始就做**——单 seed 在 portfolio 上够,但任何认真的 publication 一定会被要求,延后做不如先做。其三,**Probe 2(language conditioning)和这次的 Probe 1 一起做**——我现在只回答了"LoRA 改了什么 representation",没回答"是 visual representation 还是 language conditioning";单独再开一轮成本比当时一起做高。

### C.10 "你为什么不接着把它做完到能投 CoRL/RSS?"

**承认:** 没接着投的两个真实原因。其一,我求职:PhD 比例是 3,工作是 7;portfolio 项目对求职的边际收益在"完成 + 诚实"这一档就饱和了,继续投入 ROI 急剧下降。其二,我的另一个项目 PolicyArena(LLM Agent 策略合规)在更接近我求职方向的 LLM/RAG 工程上,边际收益更高,优先继续打磨。

**加分:** 但**如果**我决定走 PhD 这条路,这个项目是个干净的 starting point——补 3 seed + 加 Probe 2 + cross-suite,可能在 6-8 周内做到一个真正的 venue。今天告诉您这个 timeline 不是空谈,是我在 README 的 "Future work" 里已经写下来的具体路线。

---

## D. 一些 meta 准备(自己看)

- **不要在 push back 时变小**:你统计学背景在这个项目上是真实价值,该说就说,不用过度谦逊。但**也不要装大**——不要把"diagnostic finding"包装成"我们发明了一个新方法",reviewer 和招聘官识别这个的速度比你想象的快。
- **每条 push back 都先承认对的部分**:这是高情商沟通,且会让你"加分"那段听起来更可信。
- **三条 negative findings 是这个项目的 *moat*,不是弱点**——任何懂 robotics 的人看到"诚实保留 viewpoint 0% / noise 退化 / C-lighting 不及 B"会立刻知道你不是在卖弄。这种诚实在工业界和学术界都是 over-indexed-on 的信号。
- **PolicyArena 优先**:如果只能讲一个项目,讲 PolicyArena;VCR 作为"补充"出现。两个都是 portfolio,但 PolicyArena 离工业界更近。

---

*Last updated: `<TODO: date>`. Numbers verified against `analysis/runs/phase5_summary.md`.*
