import { useReducer, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  authorizeAndStart,
  captureSnapshot,
  createTask,
  getSweepCount,
  saveBaseline,
  scanConfiguration,
  validateParameterRanges,
} from "../api/client";
import { MutationButton } from "../components/MutationButton";
import { useRuntimeQueries } from "../hooks/useRuntimeQueries";
import { deriveRuntimeGate } from "../hooks/useRuntimeGate";
import type { ConfigurationScan, HardwareSnapshot, RangeValidation, SweepCount } from "../schemas/contracts";
import { canContinueMapping, initialWizardState, wizardReducer } from "../store/wizardReducer";

const STEPS = ["Task", "Mapping", "Snapshot / B0", "Script / RegMap", "Ranges", "Profile", "Sweep", "Authorize"];
const REFERENCE_FIR = [0, 9, -28, 104, -8, 1, -3];
const FIR_LABELS = ["Pre-3", "Pre-2", "Pre-1", "Main", "Post-1", "Post-2", "Post-3"];

export function CreateTaskPage() {
  const [params] = useSearchParams();
  const preferred = params.get("type") === "SWEEP" ? "SWEEP" : "AI_OPTIMIZATION";
  const [state, dispatch] = useReducer(wizardReducer, { ...initialWizardState, taskType: preferred, sweepMode: preferred === "SWEEP" ? "SINGLE" : null });
  const [pending, setPending] = useState(false);
  const [b0Saved, setB0Saved] = useState(false);
  const [snapshot, setSnapshot] = useState<HardwareSnapshot | null>(null);
  const [scan, setScan] = useState<ConfigurationScan | null>(null);
  const [validatedRanges, setValidatedRanges] = useState<RangeValidation | null>(null);
  const [sweepCount, setSweepCount] = useState<SweepCount | null>(null);
  const [catalogScript, setCatalogScript] = useState("tx_tuning_v11.py");
  const [scriptUpload, setScriptUpload] = useState<File | null>(null);
  const [regMapUpload, setRegMapUpload] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { capabilities, campaigns } = useRuntimeQueries();
  const runtimeReady = capabilities.isSuccess && campaigns.isSuccess;
  const active = campaigns.data?.find((campaign) => !["COMPLETED", "FAILED"].includes(campaign.state)) ?? null;
  const gate = runtimeReady && capabilities.data ? deriveRuntimeGate(capabilities.data, active) : null;
  const ranges = state.selectedParameters.map((name) => ({ name, ...state.ranges[name] }));
  const profileOptions = capabilities.data?.profile_options ?? [];

  async function next() {
    setError(null);
    setPending(true);
    try {
      if (state.step === 1 && !state.taskId) {
        const draft = await createTask({ task_type: state.taskType });
        dispatch({ type: "SET_TASK_ID", value: draft.task_id });
      } else if (state.step === 2 && state.taskId) {
        setSnapshot(await captureSnapshot(state.taskId, { mapping: state.mapping }));
      } else if (state.step === 5 && state.taskId) {
        setValidatedRanges(await validateParameterRanges(state.taskId, { ranges }));
      } else if (state.step === 7 && state.taskId && state.taskType === "SWEEP") {
        const count = await getSweepCount(state.taskId, {
          hardware: state.sweepHardware,
          mode: state.sweepMode,
          seed: state.sweepSeed,
          max_points: state.maxPoints,
          ranges,
        });
        setSweepCount(count);
        if (count.over_limit) throw new Error(`Sweep exceeds server limit (${count.legal_count} > ${count.limit}).`);
      }
      dispatch({ type: "SET_STEP", value: state.step === 6 && state.taskType === "AI_OPTIMIZATION" ? 8 : state.step + 1 });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Step validation failed");
    } finally {
      setPending(false);
    }
  }

  async function saveB0() {
    if (!state.taskId || disabledReason) return;
    setPending(true);
    setError(null);
    try { await saveBaseline(state.taskId, { source: "CURRENT_VALIDATED_SNAPSHOT" }); setB0Saved(true); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "B0 save failed"); }
    finally { setPending(false); }
  }

  async function scanSources() {
    if (!state.taskId || disabledReason) return;
    setPending(true);
    setError(null);
    try {
      const result = await scanConfiguration(state.taskId, { catalogScript, scriptUpload, regMapUpload });
      const available = new Set(result.parameters.filter((parameter) => parameter.available).map((parameter) => parameter.name));
      dispatch({ type: "SET_PARAMETERS", value: state.selectedParameters.filter((name) => available.has(name)) });
      setScan(result);
    }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Script / RegMap scan failed"); }
    finally { setPending(false); }
  }

  async function start() {
    if (!state.taskId) return;
    setPending(true);
    setError(null);
    try {
      const started = await authorizeAndStart(state.taskId, {
        authorization_mode: "FULL_CAMPAIGN",
        mapping: state.mapping,
        parameters: state.selectedParameters,
        ranges,
        profile: state.profile,
        ...(state.taskType === "SWEEP" ? { sweep: { hardware: state.sweepHardware, mode: state.sweepMode, seed: state.sweepSeed, max_points: state.maxPoints } } : {}),
      });
      navigate(`/runs/${started.task_type === "SWEEP" ? "sweep" : "optimization"}/${started.campaign_id}`);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Authorization failed"); setPending(false); }
  }

  const stepBlocked = (state.step === 2 && !canContinueMapping(state.mapping)) || (state.step === 3 && !b0Saved) || (state.step === 4 && (!scan || state.selectedParameters.length === 0)) || (state.step === 6 && !profileOptions.some((option) => option.name === state.profile));
  const disabledReason = !gate
    ? capabilities.isError || campaigns.isError
      ? "Runtime or campaign state is unavailable. Mutations remain locked."
      : "Waiting for runtime and campaign state."
    : !gate.canMutate
      ? gate.reason ?? "Mutations are locked."
      : active && active.state !== "DRAFT"
        ? "Another task is active. Finish or recover it before continuing this draft."
        : !gate.canCreateDraft && !state.taskId
          ? gate.reason ?? "Another task is active."
          : null;

  return (
    <div className="page-stack wizard-page">
      <header className="page-heading"><div><div className="eyebrow">GUIDED OPERATION · 8 STEPS</div><h1>Create tuning task</h1><p>创建调优任务 · every write goes through validated FastAPI endpoints</p></div><span className="wizard-counter">{String(state.step).padStart(2, "0")} / 08</span></header>
      <ol className="wizard-rail" aria-label="Task setup progress">{STEPS.map((label, index) => <li className={index + 1 === state.step ? "current" : index + 1 < state.step ? "done" : ""} key={label}><span>{index + 1}</span><small>{label}</small></li>)}</ol>

      <section className="wizard-canvas">
        {state.step === 1 ? <><div className="section-title"><span>STEP 01</span><h2>Choose task / 选择任务</h2></div><div className="choice-grid">{(["AI_OPTIMIZATION", "SWEEP"] as const).map((type) => <label className={state.taskType === type ? "choice-card selected" : "choice-card"} key={type}><input type="radio" name="task-type" disabled={Boolean(state.taskId)} checked={state.taskType === type} onChange={() => dispatch({ type: "SELECT_TASK_TYPE", value: type })} /><span>{type === "AI_OPTIMIZATION" ? "AI Optimization" : "Sweep"}</span><small>{state.taskId ? "Task type locked after server Draft creation" : type === "AI_OPTIMIZATION" ? "Alternating I2C + FIR search" : "Deterministic parameter characterization"}</small></label>)}</div></> : null}

        {state.step === 2 ? <><div className="section-title"><span>STEP 02</span><h2>Hardware mapping / 硬件映射</h2></div><div className="form-grid">{([['i2c_channel','I2C channel',['1','2','3']],['bert_channel','BERT lane',['0','1','2','3']],['dca_source','DCA source',['CH1A','CH1B']]] as const).map(([key,label,options]) => <label key={key}>{label}<select value={state.mapping[key]} onChange={(event) => { setSnapshot(null); setB0Saved(false); setScan(null); setValidatedRanges(null); setSweepCount(null); dispatch({ type: "SET_MAPPING", value: { ...state.mapping, [key]: event.target.value, same_link_confirmed: false } }); }}>{options.map((value) => <option key={value}>{value}</option>)}</select></label>)}</div><label className="confirm-line"><input type="checkbox" checked={state.mapping.same_link_confirmed} onChange={(event) => dispatch({ type: "SET_MAPPING", value: { ...state.mapping, same_link_confirmed: event.target.checked } })} /><span>以上三个通道属于同一条物理链路<br /><small>These three channels belong to the same physical link.</small></span></label></> : null}

        {state.step === 3 ? <><div className="section-title"><span>STEP 03</span><h2>Current snapshot & B0 / 当前快照与基线</h2></div>{snapshot ? <><div className="snapshot-grid"><article><small>DCA IDENTITY</small><strong>{snapshot.dca.identity}</strong><span>{snapshot.dca.source} · {snapshot.dca.wavelength} · Pattern {snapshot.dca.pattern} · Trigger {snapshot.dca.trigger}</span><span>TDECQ source: {snapshot.dca.tdecq_source} · ER source: {snapshot.dca.er_source}</span></article><article><small>BERT IDENTITY</small><strong>{snapshot.bert.identity} · Port {snapshot.bert.port}</strong><span>Pattern {snapshot.bert.pattern} · FIR readback available</span><span>Upper / Lower NLC: {snapshot.bert.upper_nlc} / {snapshot.bert.lower_nlc}</span></article><article><small>I2C IDENTITY</small><strong>{snapshot.i2c.identity} · Bus {snapshot.i2c.bus}</strong><span>{Object.entries(snapshot.i2c.values).map(([name, value]) => `${name} ${value}`).join(" · ")}</span><span>Snapshot ID: {snapshot.snapshot_id}</span></article></div><div className="fir-editor" aria-label="Current FIR readback">{snapshot.bert.fir.map((tap, index) => <div key={FIR_LABELS[index]}><small>{FIR_LABELS[index]}</small><strong>{tap}</strong></div>)}</div><div className="locked-row"><span>Upper NLC <b>{snapshot.bert.upper_nlc}</b> · READBACK</span><span>Lower NLC <b>{snapshot.bert.lower_nlc}</b> · READBACK</span></div></> : <div className="alert alert-fault">Hardware snapshot unavailable. B0 cannot be saved.</div>}<p className="helper-copy">Reference only (not B0): {REFERENCE_FIR.join(" / ")} · Upper/Lower 6/6</p>{snapshot?.simulated ? <p className="simulation-note">SIMULATED SNAPSHOT · values are supplied by the FAKE API scenario.</p> : null}<MutationButton pending={pending} disabled={b0Saved || !snapshot} disabledReason={disabledReason} onClick={() => void saveB0()}>{b0Saved ? "B0 SAVED · VERIFIED" : "Save current state as B0 / 保存为本次 B0"}</MutationButton></> : null}

        {state.step === 4 ? <><div className="section-title"><span>STEP 04</span><h2>Script / RegMap scan</h2></div><div className="form-grid"><label>Script<select value={catalogScript} onChange={(event) => { setCatalogScript(event.target.value); setScan(null); }}><option>tx_tuning_v11.py</option></select></label><label>Upload Script<input type="file" accept=".py,.json,.yaml,.yml" onChange={(event) => { setScriptUpload(event.target.files?.[0] ?? null); setScan(null); }} /></label><label>Upload RegMap<input type="file" accept=".json,.yaml,.yml" onChange={(event) => { setRegMapUpload(event.target.files?.[0] ?? null); setScan(null); }} /></label></div><MutationButton pending={pending} disabledReason={disabledReason} onClick={() => void scanSources()}>Scan selected sources / 扫描所选文件</MutationButton><p className="helper-copy">Only parsed, allow-listed parameters are selectable. Manual register addresses and raw commands do not exist in this UI.</p>{scan ? <><div className="parameter-grid">{scan.parameters.map((parameter) => <label key={parameter.name} aria-disabled={!parameter.available}><input type="checkbox" disabled={!parameter.available} checked={parameter.available && state.selectedParameters.includes(parameter.name)} onChange={() => dispatch({ type: "TOGGLE_PARAMETER", value: parameter.name })} /><span>{parameter.name}<small>current {parameter.current} · {parameter.address_summary} · encoding {parameter.encoding_range}</small></span><small>{parameter.available ? "AVAILABLE" : `UNAVAILABLE · ${parameter.unavailable_reason ?? "server rejected"}`}</small></label>)}</div>{scan.rejected_names.length ? <div className="notice-warning">Rejected by safe allowlist: {scan.rejected_names.join(", ")}</div> : null}</> : <div className="notice-warning">Run the server scan before continuing.</div>}</> : null}

        {state.step === 5 ? <><div className="section-title"><span>STEP 05</span><h2>Discrete ranges / 离散范围</h2></div><div className="range-table"><div className="range-head"><span>Parameter</span><span>Min</span><span>Max</span><span>Step</span><span>Server grid</span></div>{ranges.map((range) => <div key={range.name}><strong>{range.name}</strong>{(["min", "max", "step"] as const).map((field) => <input key={field} aria-label={`${range.name} ${field}`} type="number" value={range[field]} onChange={(event) => { dispatch({ type: "SET_RANGE", name: range.name, field, value: Number(event.target.value) }); setValidatedRanges(null); }} />)}<code>{validatedRanges?.grids[range.name]?.values.join(", ") ?? "Validated by server on Continue"}</code></div>)}</div>{Object.values(validatedRanges?.grids ?? {}).some((grid) => grid.b0_outside) ? <div className="notice-warning">B0 outside range / B0 位于所选范围之外：启动时仍将其保留为回滚目标。</div> : null}</> : null}

        {state.step === 6 ? <><div className="section-title"><span>STEP 06</span><h2>Profile / 执行档位</h2></div>{profileOptions.length ? <div className="profile-grid">{profileOptions.map((option) => <label className={state.profile === option.name ? "profile-card selected" : "profile-card"} key={option.name}><input type="radio" name="profile" checked={state.profile === option.name} onChange={() => dispatch({ type: "SET_PROFILE", value: option.name })} /><b>{option.name}</b><strong>{option.candidate_budget}</strong><small>server budget · max {option.max_alternating_rounds} alternating rounds</small></label>)}</div> : <div className="alert alert-fault">Profile budgets unavailable from /capabilities.</div>}</> : null}

        {state.step === 7 && state.taskType === "SWEEP" ? <><div className="section-title"><span>STEP 07</span><h2>Sweep configuration / 扫描配置</h2></div><div className="sweep-config"><h3>Hardware — select exactly one</h3><div className="segmented">{(["I2C","BERT_FIR"] as const).map((hardware) => <label key={hardware}><input type="radio" name="sweep-hardware" checked={state.sweepHardware === hardware} onChange={() => { dispatch({ type: "SET_SWEEP_HARDWARE", value: hardware }); setSweepCount(null); }} />{hardware === "BERT_FIR" ? "BERT FIR" : "I2C"}</label>)}</div>{state.sweepHardware === "BERT_FIR" && snapshot ? <><div className="fir-editor">{snapshot.bert.fir.map((tap,index) => <div key={FIR_LABELS[index]}><small>{FIR_LABELS[index]}</small><strong>{tap}</strong></div>)}</div><div className="locked-row"><span>NLC {snapshot.bert.upper_nlc} / {snapshot.bert.lower_nlc} · LOCKED TO B0</span><span>Main + L1 constraints · SERVER VALIDATED</span></div></> : null}<h3>Sweep mode</h3><div className="segmented">{(["SINGLE","FULL"] as const).map((mode) => <label key={mode}><input type="radio" name="sweep-mode" checked={state.sweepMode === mode} onChange={() => { dispatch({ type: "SET_SWEEP_MODE", value: mode }); setSweepCount(null); }} />{mode === "SINGLE" ? "Single-parameter" : "Full-combination"}</label>)}</div>{state.sweepMode === "FULL" ? <div className="form-grid"><label>Seed<input type="number" value={state.sweepSeed ?? 1042} onChange={(event) => { dispatch({ type: "SET_SWEEP_SEED", value: Number(event.target.value) }); setSweepCount(null); }} /></label><label>Max points<input type="number" value={state.maxPoints} onChange={(event) => { dispatch({ type: "SET_MAX_POINTS", value: Number(event.target.value) }); setSweepCount(null); }} /></label><div className="count-card"><span>Original / Legal / SKIPPED_CONSTRAINT</span><b>{sweepCount ? `${sweepCount.original_count} / ${sweepCount.legal_count} / ${sweepCount.skipped_count}` : "Calculated by server on Continue"}</b></div></div> : null}</div></> : null}

        {state.step === 8 ? <><div className="section-title"><span>STEP 08</span><h2>FULL_CAMPAIGN authorization</h2></div><div className="authorization-summary"><dl><div><dt>Task Type</dt><dd>{state.taskType}</dd></div><div><dt>B0 / rollback target</dt><dd>{snapshot?.snapshot_id ?? "UNAVAILABLE"} · {b0Saved ? "VERIFIED" : "NOT VERIFIED"}</dd></div><div><dt>Mapping</dt><dd>I2C {state.mapping.i2c_channel} · BERT {state.mapping.bert_channel} · {state.mapping.dca_source}</dd></div><div><dt>DCA configuration</dt><dd>{snapshot ? `${snapshot.dca.source} · ${snapshot.dca.wavelength} · ${snapshot.dca.pattern} · ${snapshot.dca.trigger}` : "UNAVAILABLE"}</dd></div><div><dt>Parameter ranges / step</dt><dd>{ranges.map((range) => `${range.name} ${range.min}:${range.step}:${range.max}`).join(" · ")}</dd></div><div><dt>Profile / budget</dt><dd>{state.profile} · {profileOptions.find((option) => option.name === state.profile)?.candidate_budget ?? "UNAVAILABLE"} candidates</dd></div>{state.taskType === "SWEEP" ? <div><dt>Sweep</dt><dd>{state.sweepHardware} · {state.sweepMode} · seed {state.sweepSeed ?? "—"} · max {state.maxPoints} · legal {sweepCount?.legal_count ?? "UNAVAILABLE"}</dd></div> : null}<div><dt>Authorization ID</dt><dd>Generated by FastAPI after this single confirmation</dd></div></dl><p className="live-warning">One click sends one authorization request. The UI never assumes a hardware write succeeded.</p></div></> : null}

        {error ? <div className="alert alert-fault" role="alert">{error}</div> : null}
        <footer className="wizard-actions"><button className="text-button" disabled={state.step === 1 || pending} onClick={() => dispatch({ type: "SET_STEP", value: state.step === 8 && state.taskType === "AI_OPTIMIZATION" ? 6 : state.step - 1 })}>← Back / 返回</button>{state.step < 8 ? <MutationButton pending={pending} disabled={stepBlocked} disabledReason={disabledReason ?? (state.step === 3 && !b0Saved ? "Save and verify B0 before continuing." : null)} onClick={() => void next()}>Continue / 继续 →</MutationButton> : <MutationButton pending={pending} pendingLabel="Authorizing… / 正在授权…" disabledReason={!gate?.canStart ? gate?.reason ?? "Start unavailable." : null} onClick={() => void start()}>Authorize and Start / 授权并启动</MutationButton>}</footer>
      </section>
    </div>
  );
}
