# sglang-kimi

## K3 Track — plans & progress tracking

Kimi K3 的计划、进度、实验记录不在代码分支里，而在本 repo 的独立 orphan branch `k3-track`：

- 线上浏览: https://github.com/DarkSharpness/sglang-kimi/tree/k3-track
- 本地 worktree（**强制**）: `git worktree add ../sglang-kimi-k3-track k3-track`（已存在则直接 cd 过去）。一切 tracking 读写只在该 worktree 进行 —— 不要在代码 worktree checkout `k3-track`，也不要把 tracking 文件写进代码分支。

### 更新模式（AI 必须遵守）

1. **AI 检测 + 主动提醒**：在代码 worktree 陪 human 工作时，如果发现 tracking 可能需要更新（实验出了结果、todo 完成、新方向确定、状态变化），主动提醒 human，但不要自行更新。
2. **人提供 prompt**：human 明确说了要更新什么，AI 才去动 k3-track worktree。
3. **AI 更新，人审核**：AI 按 k3-track 根目录 `CLAUDE.md` 的路由与格式写好，交 human 审核，审核通过后才 commit / push。

benchmark 数字必须绑定 code commit（`data@YYYY-MM-DD-<sha8>` round），没有 binding 的数字不落库。
