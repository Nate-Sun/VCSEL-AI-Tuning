import { z } from "zod";
import {
  apiErrorSchema,
  authorizationResultSchema,
  baselineResultSchema,
  campaignReportSchema,
  campaignSchema,
  capabilitySchema,
  configurationScanSchema,
  hardwareSnapshotSchema,
  historyApplySchema,
  observationSchema,
  rangeValidationSchema,
  recoverySchema,
  runtimeHealthSchema,
  sweepCountSchema,
  sweepPageSchema,
  taskTypeSchema,
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

export const TEMPORARY_V11_ENDPOINTS = {
  createTask: "/api/v1.1/tasks",
  snapshot: "/api/v1.1/tasks/{id}/snapshot",
  scanConfiguration: "/api/v1.1/tasks/{id}/scan-configuration",
  saveBaseline: "/api/v1.1/tasks/{id}/baseline",
  validateRanges: "/api/v1.1/tasks/{id}/validate-ranges",
  sweepCount: "/api/v1.1/tasks/{id}/sweep-count",
  authorizeAndStart: "/api/v1.1/tasks/{id}/authorize-start",
  recovery: "/api/v1.1/campaigns/{id}/recovery",
  restoreBaseline: "/api/v1.1/campaigns/{id}/recovery/restore-baseline",
  continueCampaign: "/api/v1.1/campaigns/{id}/recovery/continue",
  endCampaign: "/api/v1.1/campaigns/{id}/recovery/end",
  sweepPage: "/api/v1.1/campaigns/{id}/sweep-points",
  sweepExport: "/api/v1.1/campaigns/{id}/sweep-export",
  applyHistory: "/api/v1.1/campaigns/{id}/apply",
  applyStatus: "/api/v1.1/operations/{id}",
} as const;

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

function resourcePath(template: string, id: string): string {
  return template.replace("{id}", encodeURIComponent(id));
}

export async function requestJson<TOutput>(
  method: "GET" | "POST",
  path: string,
  schema: z.ZodType<TOutput, z.ZodTypeDef, unknown>,
  body?: unknown,
): Promise<TOutput> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
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

const taskDocumentSchema = z
  .object({
    task_id: z.string().min(1),
    task_type: taskTypeSchema,
    state: z.string().min(1),
  })
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

export function createTask(body: { task_type: "AI_OPTIMIZATION" | "SWEEP" }) {
  return requestJson("POST", TEMPORARY_V11_ENDPOINTS.createTask, taskDocumentSchema, body);
}

export function captureSnapshot(taskId: string, body: unknown = {}) {
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.snapshot, taskId), hardwareSnapshotSchema, body);
}

export interface ConfigurationScanSource {
  catalogScript: string;
  scriptUpload?: File | null;
  regMapUpload?: File | null;
}

export async function scanConfiguration(taskId: string, source: ConfigurationScanSource = { catalogScript: "tx_tuning_v11.py" }) {
  const body = {
    script: source.scriptUpload ? { kind: "UPLOAD", filename: source.scriptUpload.name, content: await source.scriptUpload.text() } : { kind: "CATALOG", id: source.catalogScript },
    regmap: source.regMapUpload ? { kind: "UPLOAD", filename: source.regMapUpload.name, content: await source.regMapUpload.text() } : { kind: "SCRIPT_DEFAULT" },
  };
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.scanConfiguration, taskId), configurationScanSchema, body);
}

export function saveBaseline(taskId: string, body: unknown) {
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.saveBaseline, taskId), baselineResultSchema, body);
}

export function validateParameterRanges(taskId: string, body: unknown) {
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.validateRanges, taskId), rangeValidationSchema, body);
}

export function getSweepCount(taskId: string, body: unknown) {
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.sweepCount, taskId), sweepCountSchema, body);
}

export function authorizeAndStart(taskId: string, body: unknown) {
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.authorizeAndStart, taskId), authorizationResultSchema, body);
}

export function getRecovery(campaignId: string) {
  return getJson(resourcePath(TEMPORARY_V11_ENDPOINTS.recovery, campaignId), recoverySchema);
}

export function restoreBaseline(campaignId: string) {
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.restoreBaseline, campaignId), recoverySchema, {});
}

export function continueRecoveredCampaign(campaignId: string) {
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.continueCampaign, campaignId), campaignActionSchema, {});
}

export function endRecoveredCampaign(campaignId: string) {
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.endCampaign, campaignId), campaignActionSchema, {});
}

export function getSweepPage(campaignId: string, query = "page=1&page_size=25") {
  return getJson(`${resourcePath(TEMPORARY_V11_ENDPOINTS.sweepPage, campaignId)}?${query}`, sweepPageSchema);
}

export function sweepExportUrl(campaignId: string, format: "csv" | "json") {
  return `${API_BASE}${resourcePath(TEMPORARY_V11_ENDPOINTS.sweepExport, campaignId)}?format=${format}`;
}

export function applyHistoryResult(campaignId: string) {
  return requestJson("POST", resourcePath(TEMPORARY_V11_ENDPOINTS.applyHistory, campaignId), historyApplySchema, {});
}

export function getHistoryApplyStatus(operationId: string) {
  return getJson(resourcePath(TEMPORARY_V11_ENDPOINTS.applyStatus, operationId), historyApplySchema);
}

export function eventStreamUrl(campaignId: string): string {
  return `${API_BASE}${campaignPath(campaignId)}/events/stream`;
}
