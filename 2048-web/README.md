# 2048-web（第三方代码，仅作教师使用）

这个目录是 [game-difficulty/2048EndgameTablebase](https://github.com/game-difficulty/2048EndgameTablebase)
浏览器版本的**原样拷贝**，用于：

1. 作为 `ntuple2048` 的专家教师（`python -m ntuple2048 distill`）；
2. 保留了可复现的 WASM 二进制，避免依赖上游仓库的可用性。

## 文件来源

| 文件 | 来源 | 说明 |
|------|------|------|
| `ai_core.wasm` | 上游构建产物 | C++ 编译的搜索 + 残局表基（data 段 2.8 MB） |
| `ai_core.js` | 上游构建产物 | Emscripten/embind 胶水 |
| `main.js` / `worker.js` / `index.html` | 上游源码 | 网页 UI 与调度逻辑 |
| `expert_core.mjs` | **本项目新增** | 从 `worker.js` 第 24–724 行抽出的 `L3Manager` + `CoreAILogic`，仅加了 `setCore()` 与静音 `console`，供 Node 直接调用 |
| `teacher_server.mjs` | **本项目新增** | 批量自对弈、输出 `T2048V1` 二进制轨迹 |
| `package.json` | **本项目新增** | 仅用于声明 `"type": "module"`，让 Node 能以 ESM 加载 `ai_core.js` |

## 许可

`ai_core.wasm` / `ai_core.js` / `main.js` / `worker.js` / `index.html` 的版权与许可
归上游项目所有，本仓库仅为互操作与离线复用而附带。若上游后续提供明确许可条款，
以该条款为准；如需在商业场景使用，请先联系上游作者。
