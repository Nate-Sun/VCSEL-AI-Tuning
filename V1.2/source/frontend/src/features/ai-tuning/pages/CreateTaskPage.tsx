import { useReducer, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  authorizeThenStartV11Task,
  countV11Sweep,
  createV11Task,
  freezeV11B0,
  getV11HardwareSnapshot,
  scanV11Mapping,
  validateV11Parameters,
} from "../api/client";
import { MutationButton } from "../components/MutationButton";
import { StagedLiveAuthority } from "../components/CapabilityPanel";
import { useV11RuntimeQueries } from "../hooks/useV11RuntimeQueries";
import { deriveV11WizardGate } from "../hooks/useV11RuntimeGate";
import {
  parameterNameSchema,
  type V11HardwareSnapshot,
  type V11MappingScan,
  type V11ParameterValidation,
  type V11SweepCount,
} from "../schemas/contracts";
import {
  canContinueMapping,
  initialWizardState,
  isValidSettleTime,
  supportedWizardSweepHardware,
  wizardReducer,
} from "../store/wizardReducer";
import {
  buildV11AuthorizationRequest,
  buildV11CreateTaskRequest,
  createV11TaskId,
  isCurrentSnapshotApproved,
} from "../store/wizardWorkflow";

const STEPS = ["Task", "Mapping", "Snapshot / B0", "Script / RegMap", "Ranges", "Profile", "Sweep", "Authorize"];
const REFERENCE_FIR = [0, 9, -28, 104, -8, 1, -3];
const FIR_LABELS = ["Pre-3", "Pre-2", "Pre-1", "Main", "Post-1", "Post-2", "Post-3"];

function displayValue(value: unknown, fallback = "Unavailable"): string {
  return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
}

export function CreateTaskPage() {
  const [params] = useSearchParams();
  const preferred = params.get("type") === "SWEEP" ? "SWEEP" : "AI_OPTIMIZATION";
  const [state, dispatch] = useReducer(wizardReducer, {
    ...initialWizardState,
    taskType: preferred,
    sweepMode: preferred === "SWEEP" ? "SINGLE" : null,
    sweepSeed: preferred === "SWEEP" ? 1042 : null,
  });
  const [pending, setPending] = useState(false);
  const [approvedSnapshotToken, setApprovedSnapshotToken] = useState<string | null>(null);
  const [b0Frozen, setB0Frozen] = useState(false);
  const [snapshot, setSnapshot] = useState<V11HardwareSnapshot | null>(null);
  const [scan, setScan] = useState<V11MappingScan | null>(null);
  const [validatedRanges, setValidatedRanges] = useState<V11ParameterValidation | null>(null);
  const [sweepCount, setSweepCount] = useState<V11SweepCount | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { capabilities, activeTask } = useV11RuntimeQueries();
  const runtimeReady = capabilities.isSuccess && activeTask.isSuccess;
  const active = activeTask.data ?? null;
  const gate = runtimeReady && capabilities.data
    ? deriveV11WizardGate(capabilities.data, active, state.taskId)
    : null;
  const ranges = state.selectedParameters.map((name) => ({ name, ...state.ranges[name] }));
  const profileOptions = capabilities.data?.profile_options ?? [];
  const stagedLive = capabilities.data?.live_ai_profile === "PILOT_3_THEN_FAST24";
  const b0Approved = isCurrentSnapshotApproved(
    approvedSnapshotToken,
    snapshot?.snapshot_token ?? null,
  );

  const disabledReason = !gate
    ? capabilities.isError || activeTask.isError
      ? "V1.1 runtime capabilities or active-task state is unavailable. Mutations remain locked."
      : "Waiting for V1.1 runtime and active-task state."
    : !gate.canMutate
      ? gate.reason ?? "Mutations are locked."
      : active && active.task_id !== state.taskId
        ? gate.reason ?? "Another task is active. Finish or recover it before continuing this draft."
        : !gate.canCreateDraft && !state.taskId
          ? gate.reason ?? "Another task is active."
          : null;

  async function createAndFreezeDraft() {
    if (!snapshot || !validatedRanges || !b0Approved) {
      throw new Error("Snapshot, B0 approval, and validated ranges are required");
    }
    const profile = profileOptions.find((option) => option.name === state.profile);
    if (!profile) throw new Error("Selected profile is unavailable from the server");

    let taskId = state.taskId;
    if (!taskId) {
      taskId = createV11TaskId();
      const request = buildV11CreateTaskRequest({
        taskId,
        taskType: state.taskType,
        channelMapping: snapshot.b0.channel_mapping,
        mappingConfirmed: state.mapping.operator_confirmed,
        parameters: validatedRanges.parameters.map(({ name, min, max, step }) => ({ name, min, max, step })),
        depth: state.profile,
        candidateBudget: profile.candidate_budget,
        advancedIterationLimit: state.profile === "ADVANCED" ? profile.max_alternating_rounds : null,
        sweepHardware: state.sweepHardware,
        sweepMode: state.sweepMode,
        sweepSeed: state.sweepSeed,
        maxPoints: state.maxPoints,
        settleTimeS: state.settleTimeS,
        saveDataset: state.saveDataset,
      });
      const created = await createV11Task(request);
      if (created.state !== "DRAFT") {
        throw new Error(`Task creation returned ${created.state}; B0 was not frozen`);
      }
      taskId = created.task_id;
      dispatch({ type: "SET_TASK_ID", value: taskId });
    }

    const frozen = await freezeV11B0(taskId, {
      snapshot_token: snapshot.snapshot_token,
      operator_confirmed: true,
    });
    if (frozen.state !== "B0_FROZEN") {
      throw new Error(`B0 freeze returned ${frozen.state}`);
    }
    setB0Frozen(true);
    await activeTask.refetch();
  }

  async function next() {
    setError(null);
    setPending(true);
    try {
      if (state.step === 1) {
        const captured = await getV11HardwareSnapshot();
        setSnapshot(captured);
        setApprovedSnapshotToken(null);
        setScan(null);
        setValidatedRanges(null);
        setSweepCount(null);
        dispatch({
          type: "SET_MAPPING",
          value: { ...captured.b0.channel_mapping, operator_confirmed: false },
        });
      } else if (state.step === 5) {
        if (!snapshot) throw new Error("A hardware snapshot is required");
        const validated = await validateV11Parameters(snapshot.snapshot_token, ranges);
        if (validated.snapshot_token !== snapshot.snapshot_token || validated.mapping_hash !== snapshot.mapping_hash) {
          setValidatedRanges(null);
          throw new Error("Preflight snapshot changed during parameter validation");
        }
        setValidatedRanges(validated);
      } else if (state.step === 6 && state.taskType === "AI_OPTIMIZATION") {
        await createAndFreezeDraft();
      } else if (state.step === 7 && state.taskType === "SWEEP") {
        if (!state.sweepMode || state.sweepSeed === null) throw new Error("Sweep mode and seed are required");
        if (!isValidSettleTime(state.settleTimeS)) throw new Error("settle_time_s must be a finite nonnegative number");
        if (!snapshot) throw new Error("A hardware snapshot is required");
        const count = await countV11Sweep({
          snapshot_token: snapshot.snapshot_token,
          hardware: "I2C_SWEEP",
          mode: state.sweepMode === "SINGLE" ? "SINGLE_PARAMETER" : "FULL_COMBINATION",
          parameters: ranges,
          seed: state.sweepSeed,
          max_points: state.maxPoints,
        });
        if (count.snapshot_token !== snapshot.snapshot_token || count.mapping_hash !== snapshot.mapping_hash) {
          setSweepCount(null);
          throw new Error("Preflight snapshot changed during Sweep counting");
        }
        setSweepCount(count);
        await createAndFreezeDraft();
      }
      dispatch({
        type: "SET_STEP",
        value: state.step === 6 && state.taskType === "AI_OPTIMIZATION" ? 8 : state.step + 1,
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Step validation failed");
    } finally {
      setPending(false);
    }
  }

  function approveB0() {
    if (!snapshot || disabledReason) return;
    setApprovedSnapshotToken(snapshot.snapshot_token);
    setError(null);
  }

  async function scanSources() {
    if (disabledReason) return;
    setPending(true);
    setError(null);
    try {
      if (!snapshot) throw new Error("A hardware snapshot is required");
      const result = await scanV11Mapping(snapshot.snapshot_token);
      if (result.snapshot_token !== snapshot.snapshot_token || result.mapping_hash !== snapshot.mapping_hash) {
        setScan(null);
        throw new Error("Preflight snapshot changed during mapping scan");
      }
      const available = new Set(
        result.parameters
          .filter((parameter) => parameter.status === "AVAILABLE" && parameterNameSchema.safeParse(parameter.name).success)
          .map((parameter) => parameter.name),
      );
      dispatch({
        type: "SET_PARAMETERS",
        value: state.selectedParameters.filter((name) => available.has(name)),
      });
      setScan(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Script / RegMap scan failed");
    } finally {
      setPending(false);
    }
  }

  async function start() {
    if (!state.taskId || !b0Frozen) return;
    setPending(true);
    setError(null);
    try {
      const started = await authorizeThenStartV11Task(
        state.taskId,
        buildV11AuthorizationRequest(state.taskId),
      );
      navigate(`/runs/${started.task_type === "SWEEP" ? "sweep" : "optimization"}/${started.task_id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Authorization or start failed");
    } finally {
      setPending(false);
    }
  }

  const stepBlocked =
    (state.step === 2 && !canContinueMapping(state.mapping))
    || (state.step === 3 && !b0Approved)
    || (state.step === 4 && (!scan || state.selectedParameters.length === 0))
    || (state.step === 5 && state.selectedParameters.length === 0)
    || (state.step === 6 && !profileOptions.some((option) => option.name === state.profile))
    || (state.step === 7 && state.taskType === "SWEEP" && !isValidSettleTime(state.settleTimeS));

  return (
    <div className="page-stack wizard-page">
      <header className="page-heading">
        <div>
          <div className="eyebrow">GUIDED OPERATION · 8 STEPS</div>
          <h1>Create tuning task</h1>
          <p>创建调优任务 · preflight evidence is complete before the Draft is created</p>
        </div>
        <span className="wizard-counter">{String(state.step).padStart(2, "0")} / 08</span>
      </header>
      <ol className="wizard-rail" aria-label="Task setup progress">
        {STEPS.map((label, index) => (
          <li className={index + 1 === state.step ? "current" : index + 1 < state.step ? "done" : ""} key={label}>
            <span>{index + 1}</span><small>{label}</small>
          </li>
        ))}
      </ol>

      <section className="wizard-canvas">
        {state.step === 1 ? (
          <>
            <div className="section-title"><span>STEP 01</span><h2>Choose task / 选择任务</h2></div>
            <div className="choice-grid">
              {(["AI_OPTIMIZATION", "SWEEP"] as const)
                .filter((type) => capabilities.data?.campaign_types.includes(type) ?? true)
                .map((type) => (
                <label className={state.taskType === type ? "choice-card selected" : "choice-card"} key={type}>
                  <input type="radio" name="task-type" checked={state.taskType === type} onChange={() => dispatch({ type: "SELECT_TASK_TYPE", value: type })} />
                  <span>{type === "AI_OPTIMIZATION" ? "AI Optimization" : "Sweep"}</span>
                  <small>{type === "AI_OPTIMIZATION" ? "Alternating I2C + FIR search" : "Deterministic parameter characterization"}</small>
                </label>
              ))}
            </div>
            <p className="helper-copy">Continue reads the canonical V1.1 hardware snapshot. No task exists yet.</p>
          </>
        ) : null}

        {state.step === 2 ? (
          <>
            <div className="section-title"><span>STEP 02</span><h2>Hardware mapping / 硬件映射</h2></div>
            <div className="form-grid">
              <label>I2C channel<input readOnly value={state.mapping.i2c_channel} /></label>
              <label>BERT port<input readOnly value={state.mapping.bert_port} /></label>
              <label>BERT channel<input readOnly value={state.mapping.bert_channel} /></label>
              <label>TDECQ source<input readOnly value={state.mapping.dca_tdecq_source} /></label>
              <label>TDECQ equalizer function<input readOnly value={`F${state.mapping.dca_tdecq_equalizer_function}`} /></label>
              <label>Outer ER source<input readOnly value={state.mapping.dca_outer_er_source} /></label>
            </div>
            <label className="confirm-line">
              <input
                type="checkbox"
                checked={state.mapping.operator_confirmed}
                onChange={(event) => dispatch({ type: "SET_MAPPING", value: { ...state.mapping, operator_confirmed: event.target.checked } })}
              />
              <span>确认这些通道属于同一条物理链路<br /><small>These channels belong to the same physical link.</small></span>
            </label>
          </>
        ) : null}

        {state.step === 3 ? (
          <>
            <div className="section-title"><span>STEP 03</span><h2>Current snapshot & B0 / 当前快照与基线</h2></div>
            {snapshot ? (
              <>
                <div className="snapshot-grid">
                  <article><small>DCA FINGERPRINT</small><strong>{displayValue(snapshot.b0.dca_fingerprint.identity)}</strong><span>TDECQ {snapshot.b0.channel_mapping.dca_tdecq_source} · F{snapshot.b0.channel_mapping.dca_tdecq_equalizer_function} · ER {snapshot.b0.channel_mapping.dca_outer_er_source}</span></article>
                  <article><small>BERT READBACK</small><strong>Port {snapshot.b0.fir.port} · Channel {snapshot.b0.fir.channel}</strong><span>Pattern {snapshot.b0.fir.pattern}</span><span>Upper / Lower NLC: {snapshot.b0.fir.upper_nlc} / {snapshot.b0.fir.lower_nlc}</span></article>
                  <article><small>I2C READBACK</small><strong>Channel {snapshot.b0.channel_mapping.i2c_channel}</strong><span>{Object.entries(snapshot.b0.i2c_fields).map(([name, value]) => `${name} ${value}`).join(" · ")}</span><span>Snapshot token: {snapshot.snapshot_token}</span></article>
                </div>
                <div className="fir-editor" aria-label="Current FIR readback">
                  {snapshot.b0.fir.taps.map((tap, index) => <div key={FIR_LABELS[index]}><small>{FIR_LABELS[index]}</small><strong>{tap}</strong></div>)}
                </div>
              </>
            ) : <div className="alert alert-fault">Hardware snapshot unavailable. B0 cannot be approved.</div>}
            <p className="helper-copy">Reference only (not B0): {REFERENCE_FIR.join(" / ")} · Upper/Lower 6/6</p>
            <MutationButton pending={pending} disabled={b0Approved || !snapshot} disabledReason={disabledReason} onClick={approveB0}>
              {b0Approved ? "B0 CANDIDATE APPROVED" : "Approve current snapshot as B0 / 确认为本次 B0"}
            </MutationButton>
            <p className="helper-copy">The server freezes this approved snapshot only after the complete task request is created.</p>
          </>
        ) : null}

        {state.step === 4 ? (
          <>
            <div className="section-title"><span>STEP 04</span><h2>Script / RegMap scan</h2></div>
            <MutationButton pending={pending} disabledReason={disabledReason} onClick={() => void scanSources()}>Scan server mapping / 扫描服务端映射</MutationButton>
            <p className="helper-copy">The canonical endpoint accepts no body. Manual register addresses, uploads, and raw commands are not sent.</p>
            {scan ? (
              <div className="parameter-grid">
                {scan.parameters.map((parameter) => {
                  const safeName = parameterNameSchema.safeParse(parameter.name);
                  const available = parameter.status === "AVAILABLE" && safeName.success;
                  return (
                    <label key={parameter.name} aria-disabled={!available}>
                      <input
                        type="checkbox"
                        disabled={!available}
                        checked={available && state.selectedParameters.includes(safeName.data)}
                        onChange={() => safeName.success && dispatch({ type: "TOGGLE_PARAMETER", value: safeName.data })}
                      />
                      <span>{parameter.name}<small>current {parameter.current_value ?? "—"} · script order {parameter.script_order ?? "—"}</small></span>
                      <small>{available ? "AVAILABLE" : `${parameter.status} · ${parameter.reason ?? "server rejected"}`}</small>
                    </label>
                  );
                })}
              </div>
            ) : <div className="notice-warning">Run the server scan before continuing.</div>}
          </>
        ) : null}

        {state.step === 5 ? (
          <>
            <div className="section-title"><span>STEP 05</span><h2>Discrete ranges / 离散安全范围</h2></div>
            <div className="range-table">
              <div className="range-head"><span>Parameter</span><span>Min</span><span>Max</span><span>Step</span><span>Server grid</span></div>
              {ranges.map((range) => (
                <div key={range.name}>
                  <strong>{range.name}</strong>
                  {(["min", "max", "step"] as const).map((field) => (
                    <input key={field} aria-label={`${range.name} ${field}`} type="number" value={range[field]} onChange={(event) => {
                      dispatch({ type: "SET_RANGE", name: range.name, field, value: Number(event.target.value) });
                      setValidatedRanges(null);
                    }} />
                  ))}
                  <code>{validatedRanges?.parameters.find((item) => item.name === range.name)?.values.join(", ") ?? "Validated by server on Continue"}</code>
                </div>
              ))}
            </div>
          </>
        ) : null}

        {state.step === 6 ? (
          <>
            <div className="section-title"><span>STEP 06</span><h2>Profile / 执行档位</h2></div>
            {profileOptions.length ? (
              <div className="profile-grid">
                {profileOptions.map((option) => (
                  <label className={state.profile === option.name ? "profile-card selected" : "profile-card"} key={option.name}>
                    <input type="radio" name="profile" checked={state.profile === option.name} onChange={() => dispatch({ type: "SET_PROFILE", value: option.name })} />
                    <b>{option.name === "PILOT_3_THEN_FAST24" ? "PILOT_3 → FAST24" : option.name}</b><strong>{option.candidate_budget}</strong><small>server budget · max {option.max_alternating_rounds} alternating rounds</small>
                    {option.name === "PILOT_3_THEN_FAST24" && capabilities.data ? <StagedLiveAuthority capability={capabilities.data} /> : null}
                  </label>
                ))}
              </div>
            ) : <div className="alert alert-fault">Profile budgets unavailable from /capabilities.</div>}
            <label className={state.saveDataset ? "dataset-toggle enabled" : "dataset-toggle"}>
              <input
                type="checkbox"
                checked={state.saveDataset}
                disabled={Boolean(state.taskId) || stagedLive}
                onChange={(event) => dispatch({ type: "SET_SAVE_DATASET", value: event.target.checked })}
              />
              <span>
                <b>保存实验数据 / Save dataset</b>
                {state.saveDataset ? (
                  <small>开启：保存原始眼图密度矩阵、原始眼图 PNG、TDECQ 均衡眼图 PNG 和完整测量 record；可能增加 Round Time 和磁盘占用。 / Enabled: saves the raw density matrix, both eye PNGs, and the full measurement record; Round Time and disk use may increase.</small>
                ) : (
                  <small>关闭：不保存眼图密度矩阵和 PNG，不执行额外 DCA 数据采集；仍保留安全运行和恢复所需的最小记录。 / Disabled: no density matrix, PNG, or extra DCA dataset acquisition; minimum safety and recovery records remain.</small>
                )}
              </span>
              <strong>{state.saveDataset ? "ON" : "OFF"}</strong>
            </label>
          </>
        ) : null}

        {state.step === 7 && state.taskType === "SWEEP" ? (
          <>
            <div className="section-title"><span>STEP 07</span><h2>Sweep configuration / 扫描配置</h2></div>
            <div className="sweep-config">
              <h3>Hardware · select exactly one</h3>
              <div className="segmented">
                {supportedWizardSweepHardware.map((hardware) => <label key={hardware}><input type="radio" name="sweep-hardware" checked={state.sweepHardware === hardware} onChange={() => { dispatch({ type: "SET_SWEEP_HARDWARE", value: hardware }); setSweepCount(null); }} />I2C</label>)}
              </div>
              <p className="helper-copy">BERT FIR Sweep remains locked until FIR-tap parameters are exposed by the V1.1 mapping contract.</p>
              <h3>Sweep mode</h3>
              <div className="segmented">
                {(["SINGLE", "FULL"] as const).map((mode) => <label key={mode}><input type="radio" name="sweep-mode" checked={state.sweepMode === mode} onChange={() => { dispatch({ type: "SET_SWEEP_MODE", value: mode }); setSweepCount(null); }} />{mode === "SINGLE" ? "Single-parameter" : "Full-combination"}</label>)}
              </div>
              <div className="form-grid">
                <label>Seed<input type="number" value={state.sweepSeed ?? 1042} onChange={(event) => { dispatch({ type: "SET_SWEEP_SEED", value: Number(event.target.value) }); setSweepCount(null); }} /></label>
                <label>Max points<input type="number" value={state.maxPoints} onChange={(event) => { dispatch({ type: "SET_MAX_POINTS", value: Number(event.target.value) }); setSweepCount(null); }} /></label>
                <label>
                  Settle time / 稳定等待 (s)
                  <input
                    aria-label="Settle time (seconds)"
                    type="number"
                    min="0"
                    step="0.01"
                    value={Number.isFinite(state.settleTimeS) ? state.settleTimeS : ""}
                    onChange={(event) => dispatch({ type: "SET_SETTLE_TIME", value: event.currentTarget.valueAsNumber })}
                  />
                  <small>Default: 0.1 s · finite and nonnegative</small>
                </label>
                <div className="count-card"><span>POINTS / SKIPPED_CONSTRAINT</span><b>{sweepCount ? `${sweepCount.point_count} / ${sweepCount.skipped_constraint_count}` : "Calculated by server on Continue"}</b></div>
              </div>
            </div>
          </>
        ) : null}

        {state.step === 8 ? (
          <>
            <div className="section-title"><span>STEP 08</span><h2>FULL_CAMPAIGN authorization</h2></div>
            <div className="authorization-summary">
              <dl>
                <div><dt>Task ID</dt><dd>{state.taskId ?? "UNAVAILABLE"}</dd></div>
                <div><dt>Task Type</dt><dd>{state.taskType}</dd></div>
                <div><dt>B0 / rollback target</dt><dd>{snapshot?.snapshot_token ?? "UNAVAILABLE"} · {b0Frozen ? "B0_FROZEN" : "NOT FROZEN"}</dd></div>
                <div><dt>Mapping</dt><dd>I2C {state.mapping.i2c_channel} · BERT {state.mapping.bert_port}/{state.mapping.bert_channel} · TDECQ {state.mapping.dca_tdecq_source}/F{state.mapping.dca_tdecq_equalizer_function} · ER {state.mapping.dca_outer_er_source}</dd></div>
                <div><dt>Parameter ranges / step</dt><dd>{ranges.map((range) => `${range.name} ${range.min}:${range.step}:${range.max}`).join(" · ")}</dd></div>
                <div><dt>Profile / budget</dt><dd>{state.profile} · {profileOptions.find((option) => option.name === state.profile)?.candidate_budget ?? "UNAVAILABLE"} candidates</dd></div>
                <div><dt>Dataset saving</dt><dd>{state.saveDataset ? "ENABLED · LOCKED" : "DISABLED · LOCKED"}</dd></div>
                {state.taskType === "SWEEP" ? <div><dt>Sweep</dt><dd>{state.sweepHardware} · {state.sweepMode} · seed {state.sweepSeed ?? "—"} · max {state.maxPoints} · settle {state.settleTimeS} s · points {sweepCount?.point_count ?? "UNAVAILABLE"}</dd></div> : null}
                <div><dt>Authorization TTL</dt><dd>30 minutes from server request submission</dd></div>
              </dl>
              <p className="live-warning">One click authorizes, verifies AUTHORIZED, then starts. A failed authorization never calls start.</p>
            </div>
          </>
        ) : null}

        {error ? <div className="alert alert-fault" role="alert">{error}</div> : null}
        <footer className="wizard-actions">
          <button className="text-button" disabled={state.step === 1 || pending || Boolean(state.taskId)} onClick={() => dispatch({ type: "SET_STEP", value: state.step === 8 && state.taskType === "AI_OPTIMIZATION" ? 6 : state.step - 1 })}>← Back / 返回</button>
          {state.step < 8 ? (
            <MutationButton pending={pending} disabled={stepBlocked} disabledReason={disabledReason} onClick={() => void next()}>Continue / 继续 →</MutationButton>
          ) : (
            <MutationButton pending={pending} pendingLabel="Authorizing… / 正在授权…" disabled={!state.taskId || !b0Frozen} disabledReason={!gate?.canStart ? gate?.reason ?? "Start unavailable." : null} onClick={() => void start()}>Confirm and Start / 确认并启动</MutationButton>
          )}
        </footer>
      </section>
    </div>
  );
}
