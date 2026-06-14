# Codex 升级计划

## 背景

- 用户要求将当前 Codex 升级到最新版本。
- 当前项目要求所有计划、上下文和决策写入 `docs/ai/context/`，且只能新增文件。
- 已按会话要求读取 `using-superpowers` skill。

## 当前状态

- 当前版本：`codex-cli 0.134.0`
- 可执行文件：`/home/rpartx3080/.local/bin/codex`
- 安装形态：standalone，`current` 指向 `/home/rpartx3080/.codex/packages/standalone/releases/0.134.0-x86_64-unknown-linux-musl`
- 本机没有 `npm` 或 `pipx`，不适合使用 npm/pipx 路线升级。

## 方案

1. 使用本机 CLI 提供的 `codex update` 子命令升级 standalone 安装。
2. 升级后运行 `codex --version` 验证版本。
3. 检查 `/home/rpartx3080/.codex/packages/standalone/current` 是否指向新 release。
4. 新建结果文档记录升级前后版本、命令和验证结果。

## 权衡

- 优先使用 Codex 自带 updater，避免引入 npm 或改变安装形态。
- 不删除旧 release，保留回退可能性。
- 本次不修改项目代码。
