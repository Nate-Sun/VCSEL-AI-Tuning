import { z } from "zod";
import {
  apiErrorSchema,
  campaignReportSchema,
  campaignSchema,
  capabilitySchema,
  observationSchema,
  runtimeHealthSchema,
  v11BestStateSchema,
  v11FinalReportSchema,
  v11HardwareSnapshotSchema,
  v11HistoryApplyReceiptSchema,
  v11HistoryRecoveryResultSchema,
  v11MappingScanSchema,
  v11ObservationSchema,
  v11ParameterValidationSchema,
  v11RuntimeCapabilitiesSchema,
  v11SweepCountSchema,
  v11SweepPointPageSchema,
  v11TaskDocumentSchema,
  v11TaskEventSchema,
} from "../schemas/contracts";
import type {
  V11ApplyHistoryResultRequest,
  V11AuthorizeTaskRequest,
  V11CreateTaskRequest,
  V11ParameterSelection,
  V11SaveB0Request,
  V11SweepCountRequest,
} from "../schemas/contracts";

export const ALLOWED_GET_ENDPOINTS = [
  "/health",
  "/capabilities",
  "/api/campaigns",
  "/api/campaigns/{id}",
  "/api/campaigns/{id}/observations",
  "/api/campaigns/{id}/best",
  "/api/campaigns/{id}/report",
  "/api/campaigns/{id}/events/stream",
] as const;

export const ALLOWED_POST_ENDPOINTS = [
  "/api/campaigns/{id}/pause",
  "/api/campaigns/{id}/resume",
  "/api/campaigns/{id}/abort",
] as const;

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(
  /\/$/,
  "",
) ?? "";

export class ApiResponseError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly code?: string,
    readonly fields?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiResponseError";
  }

  static fromResponse(status: number, payload: unknown): ApiResponseError {
    const parsed = apiErrorSchema.safeParse(payload);
    if (parsed.success) {
      return new ApiResponseError(
        parsed.data.detail.message,
        status,
        parsed.data.detail.code,
        parsed.data.detail.fields,
      );
    }
    if (payload && typeof payload === "object" && "detail" in payload && Array.isArray(payload.detail)) {
      const details = payload.detail as Array<{ loc?: unknown; msg?: unknown; type?: unknown }>;
      const message = details
        .map((item) => typeof item.msg === "string" ? item.msg : null)
        .filter((item): item is string => Boolean(item))
        .join("; ");
      return new ApiResponseError(message || "Request validation failed", status, "validation_error", { detail: details });
    }
    return new ApiResponseError(`Request failed with HTTP ${status}`, status);
  }
}

function campaignPath(campaignId: string): string {
  if (!/^[A-Za-z0-9_.-]+$/.test(campaignId)) {
    throw new TypeError("Campaign ID contains unsupported characters");
  }
  return `/api/campaigns/${encodeURIComponent(campaignId)}`;
}

function v11TaskPath(taskId: string): string {
  return `/api/v1.1/tasks/${encodeURIComponent(taskId)}`;
}

export async function requestJson<TOutput>(
  method: "GET" | "POST",
  path: string,
  schema: z.ZodType<TOutput, z.ZodTypeDef, unknown>,
  body?: unknown,
  options?: Pick<RequestInit, "cache">,
): Promise<TOutput> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    ...options,
    headers: {
      Accept: "application/json",
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const payload: unknown = await response.json().catch(() => undefined);
  if (!response.ok) {
    throw ApiResponseError.fromResponse(response.status, payload);
  }
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    throw new ApiResponseError(`Invalid response from ${method} ${path}`);
  }
  return parsed.data;
}

function getJson<TOutput>(path: string, schema: z.ZodType<TOutput, z.ZodTypeDef, unknown>) {
  return requestJson("GET", path, schema);
}

const campaignActionSchema = z
  .object({ campaign_id: z.string().min(1), state: z.string().min(1) })
  .passthrough();

export function getHealth() {
  return getJson("/health", runtimeHealthSchema);
}

export function getCapabilities() {
  return getJson("/capabilities", capabilitySchema);
}

export function getCampaigns(limit = 50, offset = 0, mockScenario?: string | null) {
  const scenarioQuery = mockScenario ? `&scenario=${encodeURIComponent(mockScenario)}` : "";
  return getJson(
    `/api/campaigns?limit=${limit}&offset=${offset}${scenarioQuery}`,
    z.array(campaignSchema),
  );
}

export function getCampaign(campaignId: string) {
  return getJson(campaignPath(campaignId), campaignSchema);
}

export function getObservations(campaignId: string) {
  return getJson(
    `${campaignPath(campaignId)}/observations`,
    z.array(observationSchema),
  );
}

export function getBestObservation(campaignId: string) {
  return getJson(
    `${campaignPath(campaignId)}/best`,
    observationSchema.nullable(),
  );
}

export function getCampaignReport(campaignId: string) {
  return getJson(`${campaignPath(campaignId)}/report`, campaignReportSchema);
}

export function pauseCampaign(campaignId: string) {
  return requestJson("POST", `${campaignPath(campaignId)}/pause`, campaignActionSchema);
}

export function resumeCampaign(campaignId: string) {
  return requestJson("POST", `${campaignPath(campaignId)}/resume`, campaignActionSchema);
}

export function abortCampaign(campaignId: string) {
  return requestJson("POST", `${campaignPath(campaignId)}/abort`, campaignActionSchema);
}

export function getV11HardwareSnapshot() {
  return requestJson(
    "GET",
    "/api/v1.1/hardware/snapshot",
    v11HardwareSnapshotSchema,
    undefined,
    { cache: "no-store" },
  );
}

export function getV11RuntimeCapabilities() {
  return getJson("/api/v1.1/runtime/capabilities", v11RuntimeCapabilitiesSchema);
}

export function getV11ActiveTask() {
  return getJson("/api/v1.1/tasks/active", v11TaskDocumentSchema.nullable());
}

export function scanV11Mapping(snapshotToken: string) {
  return requestJson(
    "POST",
    "/api/v1.1/mapping/scan",
    v11MappingScanSchema,
    { snapshot_token: snapshotToken },
  );
}

export function validateV11Parameters(
  snapshotToken: string,
  parameters: V11ParameterSelection[],
) {
  return requestJson(
    "POST",
    "/api/v1.1/parameters/validate",
    v11ParameterValidationSchema,
    { snapshot_token: snapshotToken, parameters },
  );
}

export function countV11Sweep(body: V11SweepCountRequest) {
  return requestJson(
    "POST",
    "/api/v1.1/sweeps/count",
    v11SweepCountSchema,
    body,
  );
}

export function createV11Task(body: V11CreateTaskRequest) {
  return requestJson("POST", "/api/v1.1/tasks", v11TaskDocumentSchema, body);
}

export function freezeV11B0(taskId: string, body: V11SaveB0Request) {
  return requestJson(
    "POST",
    `${v11TaskPath(taskId)}/b0`,
    v11TaskDocumentSchema,
    body,
  );
}

export function authorizeV11Task(taskId: string, body: V11AuthorizeTaskRequest) {
  return requestJson(
    "POST",
    `${v11TaskPath(taskId)}/authorization`,
    v11TaskDocumentSchema,
    body,
  );
}

export function startV11Task(taskId: string) {
  return requestJson(
    "POST",
    `${v11TaskPath(taskId)}/start`,
    v11TaskDocumentSchema,
  );
}

export async function authorizeThenStartV11Task(
  taskId: string,
  authorization: V11AuthorizeTaskRequest,
) {
  const authorized = await authorizeV11Task(taskId, authorization);
  if (authorized.state !== "AUTHORIZED") {
    throw new ApiResponseError(
      `Authorization returned ${authorized.state}; start was not requested`,
      undefined,
      "authorization_state_mismatch",
      { expected: "AUTHORIZED", actual: authorized.state },
    );
  }
  const started = await startV11Task(taskId);
  if (started.state !== "RUNNING") {
    throw new ApiResponseError(
      `Start returned ${started.state}`,
      undefined,
      "start_state_mismatch",
      { expected: "RUNNING", actual: started.state },
    );
  }
  return started;
}

export function getV11Task(taskId: string) {
  return getJson(v11TaskPath(taskId), v11TaskDocumentSchema);
}

export function getV11TaskEvents(taskId: string) {
  return getJson(`${v11TaskPath(taskId)}/events`, z.array(v11TaskEventSchema));
}

export function getV11Observations(taskId: string) {
  return getJson(`${v11TaskPath(taskId)}/observations`, z.array(v11ObservationSchema));
}

export function getV11FinalReport(taskId: string) {
  return getJson(`${v11TaskPath(taskId)}/final-report`, v11FinalReportSchema);
}

export function pauseV11Task(taskId: string) {
  return requestJson("POST", `${v11TaskPath(taskId)}/pause`, v11TaskDocumentSchema);
}

export function resumeV11Task(taskId: string) {
  return requestJson("POST", `${v11TaskPath(taskId)}/resume`, v11TaskDocumentSchema);
}

export function abortV11Task(taskId: string, reason: string) {
  return requestJson("POST", `${v11TaskPath(taskId)}/abort`, v11TaskDocumentSchema, { reason });
}

export function recoverV11Task(taskId: string, action: "RESUME" | "ABORT_RECOVERED") {
  return requestJson(
    "POST",
    `${v11TaskPath(taskId)}/recovery`,
    v11TaskDocumentSchema,
    { action, operator_confirmed: true },
  );
}

export function getV11SweepPoints(taskId: string, offset = 0, limit = 100) {
  return getJson(
    `${v11TaskPath(taskId)}/sweep-points?offset=${offset}&limit=${limit}`,
    v11SweepPointPageSchema,
  );
}

export function getV11BestState(taskId: string) {
  return getJson(`${v11TaskPath(taskId)}/best`, v11BestStateSchema.nullable());
}

export function v11SweepExportUrl(taskId: string, format: "csv" | "json") {
  return `${API_BASE}${v11TaskPath(taskId)}/sweep-results.${format}`;
}

export const sweepExportUrl = v11SweepExportUrl;

export function applyV11HistoryResult(body: V11ApplyHistoryResultRequest) {
  return requestJson(
    "POST",
    "/api/v1.1/history/apply",
    v11HistoryApplyReceiptSchema,
    body,
  );
}

export function recoverV11History(recoveryId: string) {
  return requestJson(
    "POST",
    `/api/v1.1/recovery/history/${encodeURIComponent(recoveryId)}`,
    v11HistoryRecoveryResultSchema,
    { operator_confirmed: true },
  );
}

export function eventStreamUrl(campaignId: string): string {
  return `${API_BASE}${campaignPath(campaignId)}/events/stream`;
}
