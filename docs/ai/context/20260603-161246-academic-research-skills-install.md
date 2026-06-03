# academic-research-skills 本地安装记录

## 背景

- 用户要求将 `https://github.com/Imbad0202/academic-research-skills.git` 安装为本地 skill。
- 该仓库默认分支为 `main`，HEAD 为 `83161923589b0a5f2aa77fef620d97fdcb0247a1`。
- 仓库不是单一 skill，而是包含 4 个带 `SKILL.md` 的 skill 目录。

## 安装内容

已通过 Codex `skill-installer` 安装到 `/home/rpartx3080/.codex/skills/`：

- `academic-pipeline`
- `deep-research`
- `academic-paper`
- `academic-paper-reviewer`

## 取舍

- 因用户提供的是完整 GitHub 仓库链接，且仓库内 4 个目录都符合 skill 结构，所以按 skill 集合一次性安装。
- 没有覆盖历史 `docs/ai/context/` 文件；本记录为新增上下文。

## 后续

- 需要重启 Codex 才能让新安装的 skills 出现在后续会话的可用 skill 列表中。
