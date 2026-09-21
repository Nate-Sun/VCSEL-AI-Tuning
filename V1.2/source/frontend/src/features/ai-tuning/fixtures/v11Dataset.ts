import type { V11DatasetFile } from "../schemas/contracts";

export const v11DatasetFilesFixture = [
  {
    file_id: "fake-eye-density-best",
    kind: "EYE_DENSITY_NPY",
    name: "eye_density.npy",
    url: "/artifacts/mock/v1.1/eye_density.npy",
    size_bytes: 524_288,
    point_id: null,
    observation_id: "observation-best-1",
  },
  {
    file_id: "fake-raw-eye-best",
    kind: "RAW_EYE_PNG",
    name: "raw_eye.png",
    url: "/artifacts/mock/v1.1/raw_eye.png",
    size_bytes: 98_304,
    point_id: null,
    observation_id: "observation-best-1",
  },
  {
    file_id: "fake-tdecq-eye-best",
    kind: "TDECQ_EQUALIZED_EYE_PNG",
    name: "tdecq_equalized_eye.png",
    url: "/artifacts/mock/v1.1/tdecq_equalized_eye.png",
    size_bytes: 106_496,
    point_id: null,
    observation_id: "observation-best-1",
  },
  {
    file_id: "fake-record-best",
    kind: "MEASUREMENT_RECORD_JSON",
    name: "record.json",
    url: "/api/v1.1/tasks/task-fake-011/dataset/record.json",
    size_bytes: 8_192,
    point_id: null,
    observation_id: "observation-best-1",
  },
] satisfies V11DatasetFile[];
