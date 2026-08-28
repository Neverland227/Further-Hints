# Exact Affine Hints 下离散 Secret Prior 实验框架：实现与服务器交接报告

## 1. 当前结论

本地已完成代码构建、精确代数验证和极小 smoke 验证；正式 30-instance
统计实验、B2/B4 norm-aware 候选实验以及 50-instance confirmation 尚未
运行。它们按用户要求留给 64 核、128 GiB 的服务器。

因此当前**不作** `STOP`、`MODELING RESULT` 或
`MEASURABLE SECURITY-ESTIMATOR EFFECT` 的科学分类。任何一种分类都必须
由服务器的 gate-eligible 数据决定；本地 smoke 不能晋升为论文证据。

## 2. Implementation status

| 部分 | 状态 | 说明 |
|---|---|---|
| Phase -1 analytic estimator | implemented | rank-only capacity、三种 `d_eff`、连续/整数 beta sensitivity、完整 work proxy；纯信息 proxy 不进入 beta |
| Phase 0 exact algebra/posterior | implemented | prime、prime-power、partial elimination、fixed-weight、CBD、表示不变性 |
| Phase 1 synthetic selectivity | implemented | 六类 H、pivot 策略、大模数 histogram/character/collision、小模数 survivor、projective/valuation、fixed-weight `D_I`、CBD AUC/PR、list-cluster bootstrap |
| B0 | partial | no-hint embedding/certificate 已实现；完整 measured campaign 留给可选 backend |
| B1/B3 | implemented | rank-only elimination 与 exact-prior postfilter |
| B2/B4 | implemented_optional_backend | MN Eq. (20)、Algorithm-1 adaptation、LLL/small BKZ、项目内硬节点上限枚举、候选提取；服务器需 fpylll |
| B2_PROXY | implemented | 仅诊断，Gate 1 明确排除 |
| Phase 2 | implemented, gate-locked | calibration/confirmation 分离、两 arm 分别选点、confirmation 冻结、aggregate-success expected work、paired cluster CI |
| Phase 3 | implemented, gate-locked | tiny factor graph / min-fill / MITM；`n <= 28` |
| Phase 4 | partial | empirical calibration JSON 与有限 common-prefix integration；当前论文 estimator 直接 adapter 不可用 |
| G6K | deferred_out_of_scope | disabled stub only |
| DDGR/Cao cross-check | unavailable | 不使用弱实现冒充命名 baseline |

## 3. Source-to-code map

精确映射见 `docs/SOURCE_MAP.md`。核心边界如下：

- MN Section 3 用于 mod-q 消元；
- MN Section 5 Eq. (20) 用于 full-secret/norm-aware lattice；
- local working-paper Section 5.3 Lemma 3 只用于 `G=P K F_I` 的合成结构；
- partial elimination、posterior coupling、硬节点枚举和统计 gate 属于项目推导或诊断；
- matched norm shell 是 proxy，不是 MN/DDGR/Cao。

## 4. Exact posterior correctness

本地最终测试：

- `32 passed`；
- `3 skipped`，均为仅在 fpylll 存在时运行的 MN Algorithm-1 / bounded-enumeration 测试；
- Phase-0 smoke：6 个 exact checks，0 failures，Gate 0 = PASS；
- 覆盖 `q=17` 与 `q=32`，以及 `r_elim=0,1,2`；
- reconstruction、hint invariant、LWE residual identity、posterior equality、
  representation invariance 全部通过；整个可承受支持集上的 normalized
  posterior total variation 为 0。

测试过程中发现并修复了一处 signed/mod-q 表示错误：重构后的 `q-1` 必须
center-lift 为 `-1` 后才能送入 ternary/CBD prior。修复后全套测试通过。

## 5. Rank-only reference validation

小模数 rank-only 单元测试以 40,000 个候选检查
`(3/q)^r` 的条件参考，容差内通过。该公式只对相关 uniform-image 条件成立，
不用于 structured H 或 norm-aware candidate distribution 的无条件预测。

Phase-1 smoke 使用 2 个 instance/list cluster、每 list 300 个候选，仅验证
runner 和结构诊断。它显示 dense 与 `coded_dual_G_transpose` 的 pass rate
可能明显不同，但样本极小且不是 gate-eligible，故不报告为科学效应。

## 6. Norm-aware candidate distribution 与 exact prior beyond norm

本地 Windows 没有 fpylll，因此没有生成 B2 candidate list，也没有估计
`B_disc beyond norm`。服务器必须先通过：

1. MN zero-block/transform invariants；
2. bounded enumeration 的整数范数和硬 node-limit 检查；
3. true-candidate presence 与 true retention 分离报告；
4. 至少两类 H 上的 list-cluster CI。

若 B2 不完整、node-limit 截断或资源失败，Gate 1 必须是
`BLOCKED_B2_UNAVAILABLE`，不能用 B2_PROXY 代替。

## 7. Structured H vs dense H

实现记录 rank/unit minor、行列 weight、puncturing rule、information set、
activation depth、character bias、collision、selectivity 和 correlation。

`coded_dual_G_transpose` 的代数形式来自 `G=P K F_I`，但默认 prefix
puncturing 是 `PROJECT_DIAGNOSTIC`，不声称复现 Carrier production code。

## 8. Candidate correlations

prime q 下记录 projective class、最大 class、collinear-pair fraction；
collinear/non-collinear joint pass rate 和 list-level overdispersion；
prime-power 下记录最小 p-adic valuation。主要 CI 按 instance/list cluster
bootstrap，不把候选当独立样本。

## 9. Partial elimination

所有 `r_elim=0,...,r` 使用同一接口。输出同时记录：

- residual dimension；
- pivot fill；
- mean/max activation depth；
- remaining checks；
- exact prior selectivity；
- complete expected work（在 Phase 2）。

## 10. Estimator vs toy measured data

Phase -1 smoke 已验证执行链，但被标成 `NOT_EVALUATED_SMOKE`。另一个测试中
发现低 beta 渐近 root-Hermite 公式非单调；实现已限制为 `beta >= 40`，并
禁止 same-second-moment/CBD entropy proxy 转换为 beta 或 work。

正式 predicted/measured direction、delta beta 和 expected-work 对比均待
服务器 Phase 2 数据。

## 11. Full cost breakdown

Phase 2 分解：basis、reduction、candidate generation、predicate、verification，
并在完整 confirmation cluster 上以
`W=mean(T_total)/mean(success)` 比较；失败实例不会因单次 work 为无穷而被
删掉。exact prior 当前是 `leaf_postfilter`；未实现
sieve-database filtering 或 sieve-internal partitioning。

## 12. Calibration / confirmation

参数只从 YAML 的有限 grid 选择。baseline 与 exact-prior arm 在 calibration
上分别最小化 aggregate expected work。当前冻结网格为 320 个 calibration
task，confirmation 最多 200 个 task。随后写入
`confirmation_preregistration.json`；恢复运行时该文件不会被重选或覆盖。

## 13. Failures / unavailable

完整列表见 `DEFERRED_OUT_OF_SCOPE.md`。本地 fpylll 测试被显式 skip；没有
伪造 lattice output。所有正式失败写入 `failures.jsonl`，resource limit 不会
被静默删除。

当前 `D:/Further Hints` 不是 Git checkout，因此本地 manifest 中
`git_commit`/`dirty_state` 为 `null`。服务器运行前应把该目录置于受版本控制
的 checkout（或至少归档整个项目目录与 config hash）；代码在 Git checkout
中会自动记录 commit 与 dirty state。manifest 还会独立记录排除 generated
results 后的 `source_tree_sha256`，作为 loose-source drift 的第二道检查。

## 14. 服务器资源策略

- synthetic：48 workers，1 BLAS thread/worker，2 tasks 后退出；
- lattice：6 workers，1 task 后退出，10 GiB/worker，900 s/task；
- Sage：默认禁用；如未来局部启用，最多 4 workers、每进程 1 task、12 GiB、
  600 s，然后销毁进程；
- 预留至少 24 GiB 给系统、文件缓存与父进程；
- 不自动增加 beta、radius、dimension、candidate budget 或 worker 数。

BLAS/OpenMP 限制会在 spawn 子进程导入 NumPy/fpylll **之前**写入环境，避免
48 个 worker 各自再创建 64 个 native threads。Gate 0、Gate -1 和 Gate 1
均读取最新的真实 decision artifact；不能再由 YAML 布尔值伪造通过。

正式工作量上界：`dense_smallq` 约 8,640 个 instance task、最多约
1.04e8 个 synthetic candidates；`coded_dual_q3329` 600 个 task、最多
2.4e7 个 candidates；B2 为 120 个短寿命 lattice task。该分配用于 64-core
Turin，而非本地 16-thread Windows 机器。

详见 `docs/SERVER_RUNBOOK.md`。

## 15. 可以安全声称什么

当前只可声称：

> 已实现并本地验证 exact affine elimination 与 discrete-prior posterior 的
> 一致性；已构建带硬 gate、cluster-level statistics 和服务器 resource
> guards 的可复现实验框架。

已核验的本地 smoke artifacts（固定示例；服务器运行将生成新的 run ID）：

- `results/phase_minus1/20260828T043031Z-897e932d`：`NOT_EVALUATED_SMOKE`；
- `results/phase0/20260828T043031Z-0eb29265`：Gate 0 `PASS`；
- `results/phase1/20260828T043039Z-d016763f`：0 failures，Gate 1
  `NOT_EVALUATED_SMOKE`；
- `results/phase4/20260828T043046Z-aa004466`：因尚无 Phase 2 数据而
  `UNAVAILABLE`，所有 finite-grid 点均验证为 `EXTRAPOLATION`。

不能声称 exact prior 在 norm-aware geometry 之外有或没有边际收益。

## 16. 不能声称什么

- 不能声称 cryptographic-scale break；
- 不能声称 practical Kyber/ML-KEM attack；
- 不能从 smoke 推断 asymptotic speedup；
- 不能把 proxy 当 theorem/B2；
- 不能把 estimator delta 当 measured speedup；
- 不能假设 candidate independence；
- 不能把本地 Working Paper PDF 当当前 authoritative manuscript。
