import type {
  V11DatasetFile,
  V11DatasetStatus,
  V11MeasurementStatus,
} from "../schemas/contracts";

interface DatasetStatusPanelProps {
  saveDataset: boolean;
  measurementStatus: V11MeasurementStatus;
  datasetStatus: V11DatasetStatus;
  datasetError: string | null;
  files: V11DatasetFile[];
}

const PNG_KINDS = new Set<V11DatasetFile["kind"]>([
  "RAW_EYE_PNG",
  "TDECQ_EQUALIZED_EYE_PNG",
]);

function formatSize(size: number | null | undefined): string {
  if (size === null || size === undefined) return "Size unavailable";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export function DatasetStatusPanel({
  saveDataset,
  measurementStatus,
  datasetStatus,
  datasetError,
  files,
}: DatasetStatusPanelProps) {
  const effectiveDatasetStatus = saveDataset ? datasetStatus : "DISABLED";
  const disabled = effectiveDatasetStatus === "DISABLED";
  const hasDatasetWarning = effectiveDatasetStatus === "PARTIAL" || effectiveDatasetStatus === "FAILED";

  return (
    <section className={`panel dataset-panel dataset-${effectiveDatasetStatus.toLowerCase()}`} aria-labelledby="dataset-status-title">
      <div className="dataset-panel-heading">
        <div>
          <div className="panel-kicker">MEASUREMENT + DATASET EVIDENCE</div>
          <h2 id="dataset-status-title">实验数据 / Experiment data</h2>
        </div>
        <span className={disabled ? "dataset-lock neutral" : "dataset-lock"}>IMMUTABLE TASK SETTING</span>
      </div>

      <div className="dataset-status-grid">
        <strong className={measurementStatus === "SUCCESS" ? "measurement-success" : measurementStatus === "FAILED" ? "measurement-failed" : ""}>
          Measurement: {measurementStatus}
        </strong>
        <strong className={hasDatasetWarning ? "dataset-warning-label" : ""}>
          Dataset saving: {effectiveDatasetStatus}
        </strong>
      </div>

      {disabled ? (
        <p className="dataset-disabled-copy">未请求眼图密度矩阵、PNG 或额外 DCA 数据采集；仍保留安全运行和恢复所需的最小记录。 / Density matrices, eye PNGs, and extra DCA dataset acquisition were not requested; minimum safety and recovery records remain.</p>
      ) : (
        <>
          {hasDatasetWarning ? (
            <p className="dataset-warning" role="status">数据保存警告 / Dataset warning: {datasetError ?? "The server did not provide an error detail."}</p>
          ) : datasetError ? (
            <p className="dataset-warning" role="status">数据保存提示 / Dataset notice: {datasetError}</p>
          ) : null}

          {files.length > 0 ? (
            <div className="dataset-file-grid">
              {files.map((file) => {
                const isPng = PNG_KINDS.has(file.kind);
                const isMockPreview = file.url.startsWith("/artifacts/mock/");
                const association = file.point_id ?? file.observation_id ?? "task-level";
                return (
                  <article className={isPng ? "dataset-file dataset-file-image" : "dataset-file"} key={file.file_id}>
                    {isPng ? (
                      <div className={isMockPreview ? "dataset-thumbnail-shell mock" : "dataset-thumbnail-shell"}>
                        <img className="dataset-thumbnail" src={file.url} alt={`${file.name} preview`} loading="lazy" />
                        {isMockPreview ? <span>FAKE PREVIEW</span> : null}
                      </div>
                    ) : <div className="dataset-file-mark">{file.kind === "EYE_DENSITY_NPY" ? "NPY" : "JSON"}</div>}
                    <div className="dataset-file-copy">
                      <small>{file.kind}</small>
                      <b>{file.name}</b>
                      <span>{association} · {formatSize(file.size_bytes)}</span>
                    </div>
                    {isPng ? (
                      <a className="secondary-button" href={file.url} target="_blank" rel="noreferrer" aria-label={`View ${file.name}`}>查看 / View</a>
                    ) : (
                      <a className="secondary-button" href={file.url} download aria-label={`Download ${file.name}`}>下载 / Download</a>
                    )}
                  </article>
                );
              })}
            </div>
          ) : (
            <p className="empty-state">Dataset saving is enabled; files have not been reported by the server.</p>
          )}
        </>
      )}
    </section>
  );
}
