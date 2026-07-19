# sglang-kimi

## Git workflow

- `k3-track`（tracking branch）：直接 commit & push，不开 PR。
- `kimi-k3`（主 dev 分支）：默认走 PR —— 改动在单独 worktree 的 feature branch 上做，开 PR 合入，不直接 push 到 `kimi-k3`。

## K3 Track — plans & progress tracking

Kimi K3 的计划、进度、实验记录不在代码分支里，而在本 repo 的独立 orphan branch `k3-track`：

- 线上浏览: https://github.com/DarkSharpness/sglang-kimi/tree/k3-track
- 本地 worktree（**强制**）: `git worktree add ../sglang-kimi-k3-track k3-track`（已存在则直接 cd 过去）。一切 tracking 读写只在该 worktree 进行 —— 不要在代码 worktree checkout `k3-track`，也不要把 tracking 文件写进代码分支。

### Journals — 操作流水账（供 agents retrieve context）

任何有意义的操作（实验跑完、debug 结论、设计定论、PR 开/合、状态变化等）都可以在 `k3-track` 分支的 `journals/` 目录下新增一个文件并直接 commit & push，纯流水账地讲一下做了什么：

- 文件名：`<date>-<time>-<author>-<commit>-<descript>.md`，例：`2026-07-18-1954-lsyin-1ccf020d-kda-fused-decode.md`（commit 为相关 code commit 短 sha）
- 内容不要求结构化模板，讲清做了什么即可
- 只新增文件，不改旧 journal；一个操作一个文件（天然无 merge 冲突）
- journals 是轻量通道，不受下面「更新模式」的人工审核约束，agent 可直接落

### 更新模式（AI 必须遵守）

1. **AI 检测 + 主动提醒**：在代码 worktree 陪 human 工作时，如果发现 tracking 可能需要更新（实验出了结果、todo 完成、新方向确定、状态变化），主动提醒 human，但不要自行更新。
2. **人提供 prompt**：human 明确说了要更新什么，AI 才去动 k3-track worktree。
3. **AI 更新，人审核**：AI 按 k3-track 根目录 `CLAUDE.md` 的路由与格式写好，交 human 审核，审核通过后才 commit / push。

benchmark 数字必须绑定 code commit（`data@YYYY-MM-DD-<sha8>` round），没有 binding 的数字不落库。
