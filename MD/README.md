# MD — 文件索引

本目錄存放實驗設計、硬體操作、程式架構等技術備忘錄。

---

## QICK 參考資料

| 文件 | 說明 |
| --- | --- |
| [QICK_ASMv2_zh.md](QICK_ASMv2_zh.md) | QICK ASMv2 指令完整參考（中文）— `_initialize` / `_body` 所有可用方法 |
| [rb_asmv2_register.md](rb_asmv2_register.md) | RB 實驗中 asmv2 register 的分配與使用方式 |

---

## 服務 & 遠端控制

| 文件 | 說明 |
| --- | --- |
| [service_guide.md](service_guide.md) | FastAPI 服務使用說明、多使用者協調方案（硬體鎖、per-qubit 鎖、預約機制） |

---

## 實驗參數 & 校正流程

| 文件 | 說明 |
| --- | --- |
| [experiment_params.md](experiment_params.md) | 所有實驗的參數列表（對應 `new_single_qb_cal.ipynb`），含 sweep 設定與 config 更新 |
| [autocal.md](autocal.md) | `AutoCalibrate` 自動校正流程說明（輸入、7 步驟、輸出） |
| [2QRB_design.md](2QRB_design.md) | 雙 qubit RB 設計文件 — symplectic formalism、Clifford group 建構 |
| [rb_asm_openloop_refactor.md](rb_asm_openloop_refactor.md) | `s015_RB_asm` open-loop 重構備忘錄 |

---

## TWPA

| 文件 | 說明 |
| --- | --- |
| [TWPA.md](TWPA.md) | AI-TWPA-C 完整參考手冊：校正流程、`TWPAGain.analyze()` 計算說明、官方 Scoring Notebook 格式對照 |

---

## 分析方法 & 硬體補償

| 文件 | 說明 |
| --- | --- |
| [flux_filter.md](flux_filter.md) | Flux pulse 失真補償備忘錄（`flux_filter.py` 移植說明） |
| [qubit_spec_predistorted.md](qubit_spec_predistorted.md) | Predistorted qubit spec 備忘錄（`qubit_spec_predistorted.py` 移植說明） |

---

## GUI & 產品設計

| 文件 | 說明 |
| --- | --- |
| [gui.md](gui.md) | QICK GUI 產品設計、API productization roadmap，以及新增 experiment 到 GUI 的流程。 |

---

## 開發規範

| 文件 | 說明 |
| --- | --- |
| [function_writing_example.md](function_writing_example.md) | 函式撰寫規範（NumPy docstring 格式範例） |
| [modify.md](modify.md) | Docstring 轉換進度追蹤（NumPy style） |

---

## 專案根目錄其他文件

| 文件 | 說明 |
| --- | --- |
| [../Readme.md](../Readme.md) | `qick_workspace/` 程式撰寫規範（命名、型別、docstring） |
| [../CLAUDE.md](../CLAUDE.md) | `QickworkspaceV2/` 套件架構說明（給 Claude Code 使用） |
| [../QickworkspaceV2/CHECKPOINT.md](../QickworkspaceV2/CHECKPOINT.md) | `QickworkspaceV2` 套件建置進度（CP1–CP11，已完成） |
