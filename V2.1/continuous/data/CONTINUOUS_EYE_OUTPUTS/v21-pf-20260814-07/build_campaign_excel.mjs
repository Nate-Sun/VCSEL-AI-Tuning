import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const worktree = "C:/Users/NateS/Documents/VCSEL/.worktrees/v2-1-continuous-eye-optimization";
const campaignId = "v21-pf-20260814-07";
const outputDir = path.join(worktree, "outputs", campaignId);
const logPath = path.join(worktree, `${campaignId}-commission.log`);
const campaignRoot = path.join(
  worktree,
  "artifacts",
  "v2.1",
  "continuous",
  "campaigns",
  campaignId,
);
const observationsDir = path.join(campaignRoot, "observations");
const targetTdecq = 2.0;

async function readFinalReport() {
  const source = await fs.readFile(logPath);
  const encoding = source[0] === 0xff && source[1] === 0xfe ? "utf16le" : "utf8";
  const lines = source
    .toString(encoding)
    .replace(/^\uFEFF/, "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .reverse();
  for (const line of lines) {
    if (!line.startsWith("{")) continue;
    try {
      const value = JSON.parse(line);
      if (value.campaign_id === campaignId && Array.isArray(value.round_records)) {
        return value;
      }
    } catch {
      // Continue until the final JSON report line is found.
    }
  }
  throw new Error(`Final campaign report not found in ${logPath}`);
}

async function observationEvidence(observationId) {
  const filePath = path.join(observationsDir, `${observationId}.json`);
  const observation = JSON.parse(await fs.readFile(filePath, "utf8"));
  const evidence = observation.evidence ?? {};
  const original = evidence.png_path ?? "";
  const kimi = evidence.kimi_png_path ?? "";
  return { original, kimi };
}

const report = await readFinalReport();
const baselineRecord = report.round_records[0];
if (!baselineRecord) throw new Error("Campaign report has no round records");

const baselineEvidence = await observationEvidence(baselineRecord.before_observation_id);
const baselineParams = report.adapter_identities?.i2c?.readback ?? {};
const rawRows = [
  [
    "Baseline",
    0,
    "VALID",
    baselineRecord.before_metrics.tdecq_db,
    baselineRecord.before_metrics.outer_er_db,
    null,
    null,
    null,
    "",
    baselineParams.eq ?? null,
    baselineParams.gain_ctrl ?? null,
    baselineParams.ibias_dac ?? null,
    baselineParams.res_bw ?? null,
    baselineParams.res_gain ?? null,
    baselineParams.vga_cur ?? null,
    baselineParams.efin_cur ?? null,
    baselineParams.vbias_ref ?? null,
    baselineParams.vref ?? null,
    null,
    new Date(report.started_at),
    null,
    baselineRecord.before_observation_id,
    baselineEvidence.original,
    baselineEvidence.kimi,
  ],
];

for (const round of report.round_records) {
  const evidence = await observationEvidence(round.after_observation_id);
  const params = round.readback_params ?? {};
  rawRows.push([
    `Round ${round.round_index}`,
    round.round_index,
    round.measurement_status,
    round.after_metrics.tdecq_db,
    round.after_metrics.outer_er_db,
    null,
    null,
    null,
    Object.entries(round.requested_params ?? {})
      .map(([key, value]) => `${key}=${value}`)
      .join(", "),
    params.eq ?? null,
    params.gain_ctrl ?? null,
    params.ibias_dac ?? null,
    params.res_bw ?? null,
    params.res_gain ?? null,
    params.vga_cur ?? null,
    params.efin_cur ?? null,
    params.vbias_ref ?? null,
    params.vref ?? null,
    round.stage_timings_s?.total ?? null,
    new Date(round.started_at),
    new Date(round.finished_at),
    round.after_observation_id,
    evidence.original,
    evidence.kimi,
  ]);
}

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const raw = workbook.worksheets.add("Round Data");
summary.showGridLines = false;
raw.showGridLines = false;

const navy = "#17324D";
const blue = "#2F75B5";
const lightBlue = "#D9EAF7";
const green = "#E2F0D9";
const red = "#FCE4D6";
const amber = "#FFF2CC";
const lightGray = "#F3F6F8";
const textColor = "#203040";

// Round Data: one row per measured point, preserving source values.
const rawHeaders = [
  "Point",
  "Round",
  "Status",
  "TDECQ (dB)",
  "Outer ER (dB)",
  "Δ TDECQ vs Baseline (dB)",
  "Result vs Baseline",
  "Target < 2.0 dB",
  "Kimi Requested Params",
  "eq",
  "gain_ctrl",
  "ibias_dac",
  "res_bw",
  "res_gain",
  "vga_cur",
  "efin_cur",
  "vbias_ref",
  "vref",
  "Duration (s)",
  "Started (UTC)",
  "Finished (UTC)",
  "Observation ID",
  "Original Eye PNG",
  "Kimi Input PNG",
];
raw.getRange("A1:X1").values = [rawHeaders];
raw.getRange(`A2:X${rawRows.length + 1}`).values = rawRows;

for (let row = 2; row <= rawRows.length + 1; row += 1) {
  raw.getRange(`F${row}`).formulas = [[`=IF(D${row}="","",D${row}-$D$2)`]];
  raw.getRange(`G${row}`).formulas = [[
    `=IF(D${row}="","INVALID",IF(D${row}<$D$2,"BETTER",IF(D${row}=$D$2,"SAME","WORSE")))`,
  ]];
  raw.getRange(`H${row}`).formulas = [[
    `=IF(D${row}="","NO DATA",IF(D${row}<'Summary'!$B$4,"PASS","FAIL"))`,
  ]];
}

raw.getRange("A1:X1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: navy },
};
raw.getRange("A2:X12").format.font = { color: textColor };
raw.getRange("A2:X2").format.fill = lightBlue;
raw.getRange("A2:X12").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E1E8" },
  bottom: { style: "thin", color: "#D9E1E8" },
};
raw.getRange("B2:B12").format.numberFormat = "0";
raw.getRange("D2:F12").format.numberFormat = "0.000";
raw.getRange("J2:S12").format.numberFormat = "0.000";
raw.getRange("T2:U12").format.numberFormat = "yyyy-mm-dd hh:mm:ss";
raw.getRange("A1:X12").format.verticalAlignment = "center";
raw.getRange("I2:I12").format.wrapText = true;
raw.getRange("V2:X12").format.wrapText = true;
raw.getRange("C2:C12").conditionalFormats.add("containsText", {
  text: "INVALID",
  format: { fill: red, font: { bold: true, color: "#9C0006" } },
});
raw.getRange("G2:G12").conditionalFormats.add("containsText", {
  text: "BETTER",
  format: { fill: green, font: { bold: true, color: "#375623" } },
});
raw.getRange("G2:G12").conditionalFormats.add("containsText", {
  text: "WORSE",
  format: { fill: red, font: { color: "#9C0006" } },
});
raw.getRange("H2:H12").conditionalFormats.add("containsText", {
  text: "PASS",
  format: { fill: green, font: { bold: true, color: "#375623" } },
});
raw.getRange("D2:D12").conditionalFormats.add("cellIs", {
  operator: "lessThan",
  formula: targetTdecq,
  format: { fill: green, font: { bold: true, color: "#375623" } },
});

raw.getRange("A:A").format.columnWidth = 13;
raw.getRange("B:B").format.columnWidth = 8;
raw.getRange("C:C").format.columnWidth = 11;
raw.getRange("D:H").format.columnWidth = 16;
raw.getRange("I:I").format.columnWidth = 34;
raw.getRange("J:R").format.columnWidth = 11;
raw.getRange("S:S").format.columnWidth = 13;
raw.getRange("T:U").format.columnWidth = 21;
raw.getRange("V:V").format.columnWidth = 48;
raw.getRange("W:X").format.columnWidth = 52;
raw.getRange("1:1").format.rowHeight = 34;
raw.getRange("2:12").format.rowHeight = 30;
raw.freezePanes.freezeRows(1);
raw.freezePanes.freezeColumns(3);
const rawTable = raw.tables.add("A1:X12", true, "CampaignRoundData");
rawTable.style = "TableStyleMedium2";

// Summary: compact campaign outcome plus chart-ready trend table.
summary.getRange("A1:N1").merge();
summary.getRange("A1").values = [["V2.1 Continuous Eye Optimization — 10-Round Review"]];
summary.getRange("A1:N1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 18 },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
summary.getRange("A1:N1").format.rowHeight = 34;

summary.getRange("A3:B9").values = [
  ["Campaign ID", campaignId],
  ["Target TDECQ (dB)", targetTdecq],
  ["State", report.state],
  ["Stop reason", report.stop_reason],
  ["Completed rounds", report.completed_rounds],
  ["B0 restored", report.b0_restored ? "YES" : "NO"],
  ["Runtime (s)", (new Date(report.finished_at) - new Date(report.started_at)) / 1000],
];
summary.getRange("A3:A9").format = {
  fill: lightBlue,
  font: { bold: true, color: navy },
};
summary.getRange("A3:B9").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E1E8" },
  outside: { style: "thin", color: "#B4C6D7" },
};
summary.getRange("B4:B4").format.numberFormat = "0.00";
summary.getRange("B9:B9").format.numberFormat = "0.0";

summary.getRange("D3:E9").values = [
  ["Baseline TDECQ (dB)", null],
  ["Best valid TDECQ (dB)", null],
  ["Best change vs baseline (dB)", null],
  ["Valid rounds", null],
  ["Invalid rounds", null],
  ["Target reached", null],
  ["Overall conclusion", null],
];
summary.getRange("E3").formulas = [["='Round Data'!D2"]];
summary.getRange("E4").formulas = [["=MIN('Round Data'!D3:D12)"]];
summary.getRange("E5").formulas = [["=E4-E3"]];
summary.getRange("E6").formulas = [["=COUNT('Round Data'!D3:D12)"]];
summary.getRange("E7").formulas = [["=COUNTIF('Round Data'!C3:C12,\"INVALID\")"]];
summary.getRange("E8").formulas = [["=IF(MIN('Round Data'!D2:D12)<B4,\"YES\",\"NO\")"]];
summary.getRange("E9").formulas = [["=IF(E8=\"YES\",\"TARGET MET\",\"NO CANDIDATE BEAT BASELINE\")"]];
summary.getRange("D3:D9").format = {
  fill: lightGray,
  font: { bold: true, color: navy },
};
summary.getRange("D3:E9").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E1E8" },
  outside: { style: "thin", color: "#B4C6D7" },
};
summary.getRange("E3:E5").format.numberFormat = "0.000";
summary.getRange("E9").format.wrapText = true;
summary.getRange("E8:E9").conditionalFormats.add("containsText", {
  text: "NO",
  format: { fill: red, font: { bold: true, color: "#9C0006" } },
});
summary.getRange("E8:E9").conditionalFormats.add("containsText", {
  text: "TARGET MET",
  format: { fill: green, font: { bold: true, color: "#375623" } },
});

summary.getRange("A11:D11").values = [["Point", "TDECQ (dB)", "Outer ER (dB)", "Target (dB)"]];
summary.getRange("A12:D22").values = rawRows.map((row) => [
  row[0],
  row[3],
  row[4],
  targetTdecq,
]);
summary.getRange("A11:D11").format = {
  fill: blue,
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
};
summary.getRange("A12:D22").format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E1E8" },
  bottom: { style: "thin", color: "#D9E1E8" },
};
summary.getRange("B12:D22").format.numberFormat = "0.000";
summary.getRange("B12:B22").conditionalFormats.add("cellIs", {
  operator: "lessThan",
  formula: targetTdecq,
  format: { fill: green, font: { bold: true, color: "#375623" } },
});
const trendChart = summary.charts.add("line", summary.getRange("A11:D22"));
trendChart.title = "TDECQ and ER by Point (dB)";
trendChart.hasLegend = true;
trendChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
trendChart.yAxis = { numberFormatCode: "0.00", min: 0, max: 5.5 };
trendChart.setPosition("G3", "N20");

summary.getRange("A24:N25").merge();
summary.getRange("A24").values = [[
  "Interpretation: lower TDECQ is better. Blank TDECQ cells represent INVALID measurements and are not treated as zero. The campaign completed normally, restored B0, and did not reach the 2.0 dB target.",
]];
summary.getRange("A24:N25").format = {
  fill: amber,
  font: { color: "#7F6000" },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: "#D6B656" },
};

summary.getRange("A27:N28").merge();
summary.getRange("A27").values = [[
  `Source: ${logPath} | Raw evidence: ${campaignRoot}`,
]];
summary.getRange("A27:N28").format = {
  fill: lightGray,
  font: { color: "#5B6570", italic: true, size: 9 },
  wrapText: true,
  verticalAlignment: "center",
};

summary.getRange("A:A").format.columnWidth = 24;
summary.getRange("B:B").format.columnWidth = 20;
summary.getRange("C:C").format.columnWidth = 16;
summary.getRange("D:D").format.columnWidth = 27;
summary.getRange("E:E").format.columnWidth = 29;
summary.getRange("F:F").format.columnWidth = 3;
summary.getRange("G:N").format.columnWidth = 12;
summary.getRange("3:9").format.rowHeight = 25;
summary.getRange("11:22").format.rowHeight = 22;
summary.getRange("24:25").format.rowHeight = 30;
summary.getRange("27:28").format.rowHeight = 25;
summary.freezePanes.freezeRows(1);

await fs.mkdir(outputDir, { recursive: true });

const summaryCheck = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:N28",
  include: "values,formulas",
  tableMaxRows: 28,
  tableMaxCols: 14,
  maxChars: 9000,
});
console.log(summaryCheck.ndjson);

const rawCheck = await workbook.inspect({
  kind: "table",
  range: "Round Data!A1:X12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 24,
  maxChars: 12000,
});
console.log(rawCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const summaryPreview = await workbook.render({
  sheetName: "Summary",
  range: "A1:N28",
  scale: 1.25,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "summary_preview.png"),
  new Uint8Array(await summaryPreview.arrayBuffer()),
);

const rawPreviewLeft = await workbook.render({
  sheetName: "Round Data",
  range: "A1:L12",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "round_data_left_preview.png"),
  new Uint8Array(await rawPreviewLeft.arrayBuffer()),
);

const rawPreviewRight = await workbook.render({
  sheetName: "Round Data",
  range: "M1:X12",
  scale: 1,
  format: "png",
});
await fs.writeFile(
  path.join(outputDir, "round_data_right_preview.png"),
  new Uint8Array(await rawPreviewRight.arrayBuffer()),
);

const output = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, `${campaignId}-10-round-results.xlsx`);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, summaryPreview: path.join(outputDir, "summary_preview.png") }));
