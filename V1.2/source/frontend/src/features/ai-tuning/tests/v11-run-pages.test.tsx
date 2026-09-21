import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { App } from "../../../app/App";
import { server } from "../../../mocks/server";
import { v11SweepPointsResponseFixture } from "../fixtures/v11SweepPoints";
import type { V11TaskDocument } from "../schemas/contracts";

const baseConfig = {
  task_id: "task-v11",
  task_type: "AI_OPTIMIZATION",
  channel_mapping: {
    i2c_channel: 0,
    bert_port: "port1",
    bert_channel: "0",
    dca_tdecq_source: "FUNC1",
    dca_tdecq_equalizer_function: 1,
    dca_outer_er_source: "CHAN1A",
    operator_confirmed: true,
  },
  parameters: [{ name: "eq", min: 18, max: 22, step: 2 }],
  depth: "STANDARD",
  candidate_budget: 64,
  seed: 1042,
  settle_time_s: 0.1,
  save_dataset: false,
} satisfies V11TaskDocument["config"];

function task(
  taskId: string,
  taskType: "AI_OPTIMIZATION" | "SWEEP",
  state: V11TaskDocument["state"] = "RUNNING",
): V11TaskDocument {
  return {
    task_id: taskId,
    task_type: taskType,
    state,
    runtime_mode: "FAKE",
    product_version: "1.1.0",
    schema_version: 2,
    b0_hash: "b".repeat(64),
    configuration_fingerprint: "c".repeat(64),
    config: {
      ...baseConfig,
      task_id: taskId,
      task_type: taskType,
      ...(taskType === "SWEEP" ? {
        sweep_hardware: "I2C_SWEEP",
        sweep_mode: "SINGLE_PARAMETER",
        max_points: 3,
      } : {}),
    },
    measurement_status: "SUCCESS",
    dataset_status: "DISABLED",
    dataset_error: null,
    dataset_files: [],
    created_at: "2026-07-29T10:00:00Z",
    updated_at: "2026-07-29T10:01:00Z",
  };
}

const observation = {
  observation_id: "observation-v11-1",
  sequence: 1,
  phase: "I2C_COARSE",
  status: "COMPLETED",
  valid: true,
  requested: { eq: 18 },
  applied: { eq: 18 },
  readback: { eq: 18 },
  tdecq: 3.2,
  er: 3.0,
  raw_response: {},
  error_reason: null,
  created_at: "2026-07-29T10:01:00Z",
};

function handlers(
  taskId: string,
  taskType: "AI_OPTIMIZATION" | "SWEEP",
  state: V11TaskDocument["state"] = "RUNNING",
  taskDocument: V11TaskDocument = task(taskId, taskType, state),
) {
  return [
    http.get("/api/v1.1/tasks/:taskId", () => HttpResponse.json(taskDocument)),
    http.get("/api/v1.1/tasks/:taskId/observations", () => HttpResponse.json([observation])),
    http.get("/api/v1.1/tasks/:taskId/best", () => HttpResponse.json({
      observation_id: observation.observation_id,
      i2c: { eq: 18 },
      fir: [0, 9, -28, 104, -8, 1, -3],
      upper_nlc: 6,
      lower_nlc: 6,
      pattern: "SSPRQ",
      bert_port: "port1",
      bert_channel: "0",
      dca_fingerprint: "fake",
      result: { tdecq: 3.2 },
      updated_at: "2026-07-29T10:01:00Z",
    })),
    http.get("/api/v1.1/tasks/:taskId/events", () => HttpResponse.json([{
      task_id: taskId,
      sequence: 1,
      event_type: "OBSERVATION_RECORDED",
      from_state: "AUTHORIZED",
      to_state: state,
      payload: {},
      product_version: "1.1.0",
      schema_version: 2,
      campaign_type: taskType,
      runtime_mode: "FAKE",
      b0_hash: "b".repeat(64),
      config_fingerprint: "c".repeat(64),
      created_at: "2026-07-29T10:01:00Z",
    }])),
  ];
}

describe("V1.1 task pages", () => {
  it("renders Optimization state, observations, best, and events only from V1.1", async () => {
    let legacyTaskCalls = 0;
    server.use(
      ...handlers("task-v11-opt", "AI_OPTIMIZATION"),
      http.get("/api/campaigns/:campaignId", () => {
        legacyTaskCalls += 1;
        return HttpResponse.json({ detail: { code: "not_found" } }, { status: 404 });
      }),
      http.get("/api/campaigns/:campaignId/events/stream", () => {
        legacyTaskCalls += 1;
        return HttpResponse.json({}, { status: 404 });
      }),
    );
    window.history.pushState({}, "", "/runs/optimization/task-v11-opt");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "task-v11-opt" })).toBeVisible();
    expect((await screen.findAllByText("observation-v11-1")).length).toBeGreaterThan(0);
    expect(await screen.findByText("OBSERVATION_RECORDED")).toBeVisible();
    await waitFor(() => expect(legacyTaskCalls).toBe(0));
    expect(screen.getByRole("button", { name: /Pause/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Resume/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Abort/i })).toBeEnabled();
    expect(screen.getByText("Measurement: SUCCESS")).toBeVisible();
    expect(screen.getByText("Dataset saving: DISABLED")).toBeVisible();
    expect(screen.queryByRole("img", { name: /eye/i })).not.toBeInTheDocument();
  });

  it.each(["PARTIAL", "FAILED"] as const)("keeps measurement SUCCESS when dataset saving is %s", async (datasetStatus) => {
    const taskId = `task-v11-${datasetStatus.toLowerCase()}`;
    const partialTask: V11TaskDocument = {
      ...task(taskId, "AI_OPTIMIZATION"),
      config: { ...task(taskId, "AI_OPTIMIZATION").config, save_dataset: true },
      dataset_status: datasetStatus,
      dataset_error: `${datasetStatus}: TDECQ equalized eye was not persisted`,
      dataset_files: [{
        file_id: "partial-density",
        kind: "EYE_DENSITY_NPY",
        name: "density.npy",
        url: `/api/v1.1/tasks/${taskId}/datasets/density.npy`,
        size_bytes: 2048,
        point_id: null,
        observation_id: "observation-v11-1",
      }],
    };
    server.use(...handlers(taskId, "AI_OPTIMIZATION", "RUNNING", partialTask));
    window.history.pushState({}, "", `/runs/optimization/${taskId}`);

    render(<App />);

    expect(await screen.findByText("Measurement: SUCCESS")).toBeVisible();
    expect(screen.getByText(`Dataset saving: ${datasetStatus}`)).toBeVisible();
    expect(screen.getByText(/TDECQ equalized eye was not persisted/)).toBeVisible();
    expect(screen.getByRole("link", { name: "Download density.npy" })).toHaveAttribute("download");
  });

  it("renders stable point IDs, server global best, and persisted round timing", async () => {
    let legacyTaskCalls = 0;
    server.use(
      ...handlers("task-v11-sweep", "SWEEP"),
      http.get("/api/v1.1/tasks/:taskId/sweep-points", () => HttpResponse.json(v11SweepPointsResponseFixture)),
      http.get("/api/campaigns/:campaignId/events/stream", () => {
        legacyTaskCalls += 1;
        return HttpResponse.json({}, { status: 404 });
      }),
    );
    window.history.pushState({}, "", "/runs/sweep/task-v11-sweep");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "task-v11-sweep" })).toBeVisible();
    expect(await screen.findByText(/I2C_SWEEP/)).toBeVisible();
    expect(screen.getByText("point-00000001")).toBeVisible();
    expect(screen.getByText("point-00000004")).toBeVisible();
    expect(screen.getByText(/GLOBAL BEST \/ SERVER/)).toBeVisible();
    expect(screen.getAllByText("8.200000 s")[0]).toBeVisible();
    expect(screen.getAllByText("8.212000 s")[0]).toBeVisible();
    expect(screen.getByText("0.100000 s")).toBeInTheDocument();
    expect(screen.getByText("0.100120 s")).toBeInTheDocument();
    expect(screen.getByText("8.086607 s")).toBeInTheDocument();
    expect(screen.getByText("VERIFIED")).toBeInTheDocument();
    expect(screen.queryByLabelText("Filter parameter")).not.toBeInTheDocument();
    await waitFor(() => expect(legacyTaskCalls).toBe(0));
  });

  it("disables all run controls for a terminal V1.1 task", async () => {
    server.use(...handlers("task-v11-terminal", "AI_OPTIMIZATION", "COMPLETED"));
    window.history.pushState({}, "", "/runs/optimization/task-v11-terminal");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "task-v11-terminal" })).toBeVisible();
    expect(screen.getByRole("button", { name: /Pause/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Resume/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Abort/i })).toBeDisabled();
  });

  it("renders the V1.1 final report from the task-scoped endpoint", async () => {
    let legacyReportCalls = 0;
    server.use(
      http.get("/api/v1.1/tasks/:taskId/final-report", () => HttpResponse.json({
        ...task("task-v11-result", "AI_OPTIMIZATION", "COMPLETED"),
        observations: [observation],
        events: [{ result: "persisted" }],
      })),
      http.get("/api/campaigns/:campaignId/report", () => {
        legacyReportCalls += 1;
        return HttpResponse.json({}, { status: 404 });
      }),
    );
    window.history.pushState({}, "", "/tasks/task-v11-result/result");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "task-v11-result" })).toBeVisible();
    expect(screen.getByText("observation-v11-1")).toBeVisible();
    expect(screen.getByRole("heading", { name: "COMPLETED" })).toBeVisible();
    await waitFor(() => expect(legacyReportCalls).toBe(0));
  });

  it("shows server PNG views and NPY/record downloads for a saved dataset", async () => {
    const savedTask = task("task-v11-dataset", "AI_OPTIMIZATION", "COMPLETED");
    server.use(
      http.get("/api/v1.1/tasks/:taskId/final-report", () => HttpResponse.json({
        ...savedTask,
        config: { ...savedTask.config, save_dataset: true },
        dataset_status: "SUCCESS",
        dataset_files: [
          { file_id: "raw-eye", kind: "RAW_EYE_PNG", name: "raw_eye.png", url: "/artifacts/task-v11-dataset/raw_eye.png", size_bytes: 1200, point_id: "point-1", observation_id: null },
          { file_id: "tdecq-eye", kind: "TDECQ_EQUALIZED_EYE_PNG", name: "tdecq_eye.png", url: "/artifacts/task-v11-dataset/tdecq_eye.png", size_bytes: 1300, point_id: "point-1", observation_id: null },
          { file_id: "density", kind: "EYE_DENSITY_NPY", name: "density.npy", url: "/api/v1.1/tasks/task-v11-dataset/datasets/density.npy", size_bytes: 2048, point_id: "point-1", observation_id: null },
          { file_id: "record", kind: "MEASUREMENT_RECORD_JSON", name: "record.json", url: "/api/v1.1/tasks/task-v11-dataset/datasets/record.json", size_bytes: 512, point_id: "point-1", observation_id: null },
        ],
        observations: [observation],
        events: [],
      })),
    );
    window.history.pushState({}, "", "/tasks/task-v11-dataset/result");

    render(<App />);

    expect(await screen.findByText("Dataset saving: SUCCESS")).toBeVisible();
    expect(screen.getByRole("img", { name: "raw_eye.png preview" })).toBeVisible();
    expect(screen.getByRole("img", { name: "tdecq_eye.png preview" })).toBeVisible();
    expect(screen.getByRole("link", { name: "View raw_eye.png" })).toHaveAttribute("href", "/artifacts/task-v11-dataset/raw_eye.png");
    expect(screen.getByRole("link", { name: "Download density.npy" })).toHaveAttribute("download");
    expect(screen.getByRole("link", { name: "Download record.json" })).toHaveAttribute("download");
  });
});
