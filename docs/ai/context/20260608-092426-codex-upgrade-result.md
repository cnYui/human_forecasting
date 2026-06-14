# Codex 升级结果

## 结果

- 升级前版本：`codex-cli 0.134.0`
- 升级后版本：`codex-cli 0.137.0`
- 最新 release：`rust-v0.137.0`
- 安装路径：`/home/rpartx3080/.codex/packages/standalone/releases/0.137.0-x86_64-unknown-linux-musl`
- 可见命令：`/home/rpartx3080/.local/bin/codex`

## 执行记录

初次执行 `codex update` 失败：

```text
Could not find SHA-256 digest for codex-package-x86_64-unknown-linux-musl.tar.gz in codex-package_SHA256SUMS.
```

排查结果：

- release 清单实际包含 `codex-package-x86_64-unknown-linux-musl.tar.gz` 的 SHA-256。
- 本机 `/usr/bin/awk` 是 `mawk 1.3.4`，对 installer 使用的 `{64}` 正则不兼容。
- 使用临时目录中的 BusyBox `awk` 包装后重新执行 `codex update` 成功。
- 该临时 `awk` 只影响本次升级命令的 `PATH`，未修改系统 `/usr/bin/awk`。

## 验证

```text
$ codex --version
codex-cli 0.137.0

$ readlink -f /home/rpartx3080/.codex/packages/standalone/current
/home/rpartx3080/.codex/packages/standalone/releases/0.137.0-x86_64-unknown-linux-musl

$ readlink -f /home/rpartx3080/.local/bin/codex
/home/rpartx3080/.codex/packages/standalone/releases/0.137.0-x86_64-unknown-linux-musl/bin/codex
```

`codex doctor --summary --ascii --no-color` 结果：

```text
Codex Doctor v0.137.0 · linux-x86_64
17 ok | 1 idle | 0 warn | 0 fail ok
```

## 后续注意

- 当前 Codex 进程仍是升级前启动的会话；需要重启 Codex 才能让当前交互进程使用新版本能力。
- 旧 release `0.134.0-x86_64-unknown-linux-musl` 已保留，未删除。
