"""Minimal industrial Tkinter interface for continuous-eye optimization."""

from __future__ import annotations

from queue import Empty, Queue
import os
from pathlib import Path
from threading import Thread
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Protocol

from pydantic import ValidationError

from .artifacts import CampaignArtifactReader, CampaignProjection, MeasuredPoint
from .csv_export import export_campaign_csv
from .integrity import IntegrityResult, VALIDATED_COMMIT, VALIDATED_TAG
from .models import (
    DEFAULT_B0,
    B0Input,
    BenchSettings,
    DesktopPhase,
    DesktopResult,
    PreflightSummary,
    RunInputs,
)
from .session import DesktopSessionError
from .settings import BenchSettingsStore, DesktopSettingsError


B0_FIELDS = (
    ("ibias_dac", "IBIAS DAC"),
    ("vref", "VREF"),
    ("vbias_ref", "VBIAS REF"),
    ("vga_cur", "VGA CUR"),
    ("res_gain", "RES GAIN"),
    ("res_bw", "RES BW"),
    ("gain_ctrl", "GAIN CTRL"),
    ("efin_cur", "EFIN CUR"),
    ("eq", "EQ"),
)

DEFAULT_BENCH = BenchSettings(
    dca_visa_resource="TCPIP0::Nate-Rog::hislip0,4880::INSTR",
    i2c_com_port="COM7",
    i2c_address=0xA6,
)


class DesktopSessionPort(Protocol):
    @property
    def phase(self) -> DesktopPhase: ...

    def run_preflight(self, inputs: RunInputs) -> PreflightSummary: ...

    def approve_and_start(self) -> None: ...

    def cancel_approval(self) -> None: ...

    def request_stop(self) -> None: ...

    def progress(self): ...

    def wait(self, timeout: float | None = None) -> DesktopResult: ...


class DesktopApp:
    def __init__(
        self,
        root: tk.Tk,
        session: DesktopSessionPort,
        settings_store: BenchSettingsStore,
        integrity: IntegrityResult,
        *,
        artifact_reader_factory: Callable[[Path], CampaignArtifactReader] = CampaignArtifactReader,
    ) -> None:
        self.root = root
        self.session = session
        self.settings_store = settings_store
        self.integrity = integrity
        self._reader_factory = artifact_reader_factory
        self._preflight_queue: Queue[tuple[str, object]] = Queue()
        self._closing = False
        self._approval_dialog: tk.Toplevel | None = None
        self._projection: CampaignProjection | None = None
        self._photo_image: object | None = None
        self.status_text = tk.StringVar(value="就绪 / Ready")

        self.root.title("VCSEL Continuous Eye · V2.1 B")
        self.root.geometry("1060x720")
        self.root.minsize(920, 640)
        self.root.configure(background="#081014")
        self.root.protocol("WM_DELETE_WINDOW", self.request_close)
        self._configure_style()
        self.current_frame = self._build_settings_frame()
        self.current_frame.pack(fill="both", expand=True)

        if not integrity.verified:
            details = "; ".join(integrity.errors) or "build manifest unavailable"
            self.status_text.set(f"真实运行已锁定 / LIVE locked: {details}")
            self._preflight_button.configure(state="disabled")

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("App.TFrame", background="#081014")
        style.configure("Panel.TFrame", background="#101b20")
        style.configure(
            "Title.TLabel",
            background="#081014",
            foreground="#e8f6f2",
            font=("Bahnschrift", 24, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#081014",
            foreground="#7f9da1",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Panel.TLabel",
            background="#101b20",
            foreground="#c8d8d8",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "PanelHeader.TLabel",
            background="#101b20",
            foreground="#66e0c2",
            font=("Bahnschrift", 11, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background="#081014",
            foreground="#e2b65c",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "TEntry",
            fieldbackground="#17262c",
            foreground="#e8f6f2",
            insertcolor="#66e0c2",
            bordercolor="#294047",
            padding=6,
        )
        style.configure(
            "Primary.TButton",
            background="#3ac8a6",
            foreground="#04100d",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(18, 10),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#66e0c2"), ("disabled", "#314b4b")],
            foreground=[("disabled", "#809291")],
        )
        style.configure(
            "Stop.TButton",
            background="#dca94f",
            foreground="#161005",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(18, 10),
        )
        style.configure(
            "Panel.TLabelframe",
            background="#101b20",
            bordercolor="#294047",
            relief="solid",
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background="#101b20",
            foreground="#66e0c2",
            font=("Bahnschrift", 10, "bold"),
        )
        style.configure(
            "Treeview",
            background="#101b20",
            fieldbackground="#101b20",
            foreground="#d9e7e5",
            rowheight=28,
        )
        style.configure(
            "Treeview.Heading",
            background="#17262c",
            foreground="#66e0c2",
            font=("Bahnschrift", 9, "bold"),
        )

    def _build_settings_frame(self) -> ttk.Frame:
        frame = ttk.Frame(self.root, style="App.TFrame", name="settings", padding=28)
        frame.columnconfigure(0, weight=1)
        ttk.Label(
            frame,
            text="CONTINUOUS EYE",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text=(
                "V2.1 · validated B engine  |  "
                f"{VALIDATED_TAG}  |  {VALIDATED_COMMIT[:12]}"
            ),
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 22))

        content = ttk.Frame(frame, style="App.TFrame")
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)

        run_panel = ttk.LabelFrame(
            content,
            text="  运行输入 / RUN INPUT  ",
            style="Panel.TLabelframe",
            padding=18,
        )
        run_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        run_panel.columnconfigure(1, weight=1)
        self._api_key = tk.StringVar()
        self._max_rounds = tk.StringVar(value="50")
        self._target = tk.StringVar(value="2.8")
        self._labeled_entry(run_panel, 0, "Kimi API Key", self._api_key, "api_key", show="•")
        self._labeled_entry(run_panel, 1, "优化轮数 / Rounds", self._max_rounds, "max_rounds")
        self._labeled_entry(run_panel, 2, "目标 TDECQ (dB)", self._target, "target_tdecq")

        b0_panel = ttk.LabelFrame(
            content,
            text="  B0 · I2C BASELINE  ",
            style="Panel.TLabelframe",
            padding=18,
        )
        b0_panel.grid(row=0, column=1, sticky="nsew")
        b0_panel.columnconfigure(1, weight=1)
        self._b0_variables: dict[str, tk.StringVar] = {}
        defaults = DEFAULT_B0.model_dump()
        for row, (field, label) in enumerate(B0_FIELDS):
            variable = tk.StringVar(value=str(defaults[field]))
            self._b0_variables[field] = variable
            self._labeled_entry(b0_panel, row, label, variable, f"b0_{field}")

        bench = self._load_bench_settings()
        bench_panel = ttk.LabelFrame(
            frame,
            text="  实验台连接 / BENCH ENDPOINTS  ",
            style="Panel.TLabelframe",
            padding=14,
        )
        bench_panel.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        for column in range(6):
            bench_panel.columnconfigure(column, weight=1 if column in {1, 3, 5} else 0)
        self._dca_resource = tk.StringVar(value=bench.dca_visa_resource)
        self._com_port = tk.StringVar(value=bench.i2c_com_port)
        self._i2c_address = tk.StringVar(value=f"0x{bench.i2c_address:02X}")
        self._inline_entry(bench_panel, 0, "DCA VISA", self._dca_resource, "dca_resource")
        self._inline_entry(bench_panel, 2, "COM", self._com_port, "com_port")
        self._inline_entry(bench_panel, 4, "I2C", self._i2c_address, "i2c_address")

        footer = ttk.Frame(frame, style="App.TFrame")
        footer.grid(row=4, column=0, sticky="ew", pady=(22, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer,
            textvariable=self.status_text,
            style="Status.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            footer,
            text="打开历史结果 / Open history",
            name="open_history",
            command=self._open_history,
        ).grid(row=0, column=1, sticky="e", padx=(12, 10))
        self._preflight_button = ttk.Button(
            footer,
            text="运行 PRE-FLIGHT",
            style="Primary.TButton",
            name="preflight",
            command=self._start_preflight,
        )
        self._preflight_button.grid(row=0, column=2, sticky="e")
        return frame

    @staticmethod
    def _labeled_entry(
        parent: ttk.Widget,
        row: int,
        text: str,
        variable: tk.StringVar,
        name: str,
        *,
        show: str = "",
    ) -> ttk.Entry:
        ttk.Label(parent, text=text, style="Panel.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 12), pady=4
        )
        entry = ttk.Entry(parent, textvariable=variable, name=name, show=show)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        return entry

    @staticmethod
    def _inline_entry(
        parent: ttk.Widget,
        column: int,
        text: str,
        variable: tk.StringVar,
        name: str,
    ) -> None:
        ttk.Label(parent, text=text, style="Panel.TLabel").grid(
            row=0, column=column, sticky="w", padx=(0, 7)
        )
        ttk.Entry(parent, textvariable=variable, name=name).grid(
            row=0,
            column=column + 1,
            sticky="ew",
            padx=(0, 18 if column < 4 else 0),
        )

    def _load_bench_settings(self) -> BenchSettings:
        try:
            return self.settings_store.load() or DEFAULT_BENCH
        except DesktopSettingsError as error:
            self.status_text.set(f"设置损坏，使用验证默认值 / {error}")
            return DEFAULT_BENCH

    def collect_inputs(self) -> RunInputs:
        try:
            address = int(self._i2c_address.get().strip(), 0)
            b0 = {
                field: int(self._b0_variables[field].get().strip(), 10)
                for field, _ in B0_FIELDS
            }
            return RunInputs(
                api_key=self._api_key.get(),
                max_rounds=int(self._max_rounds.get().strip(), 10),
                target_tdecq_db=float(self._target.get().strip()),
                b0=B0Input.model_validate(b0),
                bench=BenchSettings(
                    dca_visa_resource=self._dca_resource.get(),
                    i2c_com_port=self._com_port.get(),
                    i2c_address=address,
                ),
            )
        except (ValueError, ValidationError) as error:
            raise DesktopSessionError(f"输入无效 / Invalid input: {error}") from error

    def _start_preflight(self) -> None:
        try:
            inputs = self.collect_inputs()
            self.settings_store.save(inputs.bench)
        except (DesktopSessionError, DesktopSettingsError) as error:
            self.status_text.set(str(error))
            return
        self._preflight_button.configure(state="disabled")
        self.status_text.set("PRE-FLIGHT：验证 DCA、I2C/B0 与 Kimi…")

        def work() -> None:
            try:
                summary = self.session.run_preflight(inputs)
            except Exception as error:
                self._preflight_queue.put(("error", error))
            else:
                self._preflight_queue.put(("result", summary))

        Thread(target=work, name="vcsel-desktop-preflight", daemon=False).start()
        self.root.after(40, self._poll_preflight)

    def _poll_preflight(self) -> None:
        try:
            kind, value = self._preflight_queue.get_nowait()
        except Empty:
            if self.root.winfo_exists():
                self.root.after(40, self._poll_preflight)
            return
        if kind == "error":
            self.status_text.set(f"PRE-FLIGHT 失败 / Failed: {value}")
            if self.integrity.verified:
                self._preflight_button.configure(state="normal")
            return
        summary = value
        if not isinstance(summary, PreflightSummary):
            self.status_text.set("PRE-FLIGHT 返回格式无效 / Invalid result")
            return
        if not summary.approvable:
            self.status_text.set(f"PRE-FLIGHT 未通过 / Rejected: {summary.details}")
            return
        self.status_text.set("PRE-FLIGHT 通过，等待明确批准 / Awaiting approval")
        self._show_approval(summary)

    def _show_approval(self, summary: PreflightSummary) -> None:
        dialog = tk.Toplevel(self.root, name="approval")
        self._approval_dialog = dialog
        dialog.title("PRE-FLIGHT APPROVAL")
        dialog.configure(background="#101b20")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        body = ttk.Frame(dialog, style="Panel.TFrame", padding=22)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="PRE-FLIGHT VERIFIED", style="PanelHeader.TLabel").pack(
            anchor="w", pady=(0, 12)
        )
        gates = (
            ("DCA identity + configuration", summary.dca_verified),
            ("I2C readback equals B0", summary.i2c_readback_verified),
            ("Kimi response valid", summary.kimi_proposal_valid),
            ("DUT write = false", not summary.dut_write_performed),
            ("BERT contacted = false", not summary.bert_contacted),
        )
        for label, passed in gates:
            ttk.Label(
                body,
                text=f"{'✓' if passed else '✕'}  {label}",
                style="Panel.TLabel",
            ).pack(anchor="w", pady=3)
        ttk.Separator(body).pack(fill="x", pady=14)
        ttk.Label(
            body,
            text="批准后将开始真实 I2C 优化。运行中仅显示轮次。",
            style="Panel.TLabel",
        ).pack(anchor="w")
        actions = ttk.Frame(body, style="Panel.TFrame", name="actions")
        actions.pack(fill="x", pady=(18, 0))
        ttk.Button(
            actions,
            text="取消 / Cancel",
            command=self._cancel_approval,
        ).pack(side="left")
        ttk.Button(
            actions,
            text="批准并开始 / APPROVE & START",
            style="Primary.TButton",
            name="approve_start",
            command=self._approve_start,
        ).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", self._cancel_approval)

    def _cancel_approval(self) -> None:
        try:
            self.session.cancel_approval()
        except DesktopSessionError:
            pass
        if self._approval_dialog is not None:
            self._approval_dialog.destroy()
            self._approval_dialog = None
        self.status_text.set("已取消；PRE-FLIGHT 会话不会执行写入 / Cancelled")

    def _approve_start(self) -> None:
        try:
            self.session.approve_and_start()
        except Exception as error:
            self.status_text.set(f"无法开始 / Start failed: {error}")
            return
        if self._approval_dialog is not None:
            self._approval_dialog.destroy()
            self._approval_dialog = None
        self._show_running()
        self.root.after(100, self._poll_run)

    def _show_running(self) -> None:
        self.current_frame.pack_forget()
        frame = ttk.Frame(self.root, style="App.TFrame", name="running", padding=40)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        center = ttk.Frame(frame, style="Panel.TFrame", padding=48)
        center.grid(row=0, column=0)
        ttk.Label(
            center,
            text="OPTIMIZATION RUNNING",
            style="PanelHeader.TLabel",
        ).pack(pady=(0, 18))
        self._progress_text = tk.StringVar(value="Round 0 / —")
        ttk.Label(
            center,
            textvariable=self._progress_text,
            style="Panel.TLabel",
            name="round_progress",
        ).pack(pady=10)
        self._stop_button = ttk.Button(
            center,
            text="安全停止 / SAFE STOP",
            style="Stop.TButton",
            name="safe_stop",
            command=self._safe_stop,
        )
        self._stop_button.pack(pady=(22, 0))
        frame.pack(fill="both", expand=True)
        self.current_frame = frame

    def _safe_stop(self) -> None:
        try:
            self.session.request_stop()
        except Exception as error:
            self._progress_text.set(f"停止请求失败 / Stop failed: {error}")
            return
        self._stop_button.configure(state="disabled")
        self._progress_text.set("正在安全停止并恢复 B0… / Restoring B0…")

    def _poll_run(self) -> None:
        try:
            progress = self.session.progress()
        except Exception as error:
            self._show_failure(f"进度证据无效 / Invalid progress: {error}")
            return
        if progress.phase in {DesktopPhase.RUNNING, DesktopPhase.STOPPING}:
            if progress.phase is DesktopPhase.RUNNING:
                self._progress_text.set(
                    f"Round {progress.current_round} / {progress.max_rounds}"
                )
            if self.root.winfo_exists():
                self.root.after(250, self._poll_run)
            return
        try:
            result = self.session.wait(timeout=0)
        except TimeoutError:
            self.root.after(100, self._poll_run)
            return
        except Exception as error:
            self._show_failure(f"运行失败 / Run failed: {error}")
            return
        self._show_result(result)
        if self._closing:
            self.root.after(100, self.root.destroy)

    def _show_failure(self, message: str) -> None:
        self.current_frame.pack_forget()
        frame = ttk.Frame(self.root, style="App.TFrame", padding=32, name="failure")
        ttk.Label(frame, text="RUN FAILED", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text=message, style="Status.TLabel", wraplength=900).pack(
            anchor="w", pady=(18, 0)
        )
        frame.pack(fill="both", expand=True)
        self.current_frame = frame

    def _open_history(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.root,
            title="Open desktop campaign evidence",
        )
        if not selected:
            return
        root = Path(selected)
        self._show_result(
            DesktopResult(
                campaign_id=root.name,
                phase=DesktopPhase.COMPLETED,
                campaign_root=root,
                b0_restored=False,
                message="HISTORICAL_VIEW",
            )
        )

    def _show_result(self, result: DesktopResult) -> None:
        try:
            projection = self._reader_factory(result.campaign_root).read()
        except Exception as error:
            self._show_failure(f"结果证据无法读取 / Result evidence error: {error}")
            return
        self._projection = projection
        self.current_frame.pack_forget()
        frame = ttk.Frame(self.root, style="App.TFrame", padding=24, name="result")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(2, weight=1)
        ttk.Label(frame, text="RESULT EVIDENCE", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        summary = (
            f"{projection.terminal_state}  ·  {len(projection.points) - 1} rounds  ·  "
            f"B0 restored={str(projection.b0_restored).lower()}  ·  "
            f"evidence={str(projection.evidence_complete).lower()}"
        )
        ttk.Label(frame, text=summary, style="Status.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 14)
        )
        columns = ("round", "requested", "readback", "tdecq", "er", "status", "best")
        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            name="points",
            selectmode="browse",
        )
        headings = ("Round", "Requested", "Full readback", "TDECQ", "ER", "Status", "Best")
        widths = (60, 180, 260, 80, 70, 80, 55)
        for column, heading, width in zip(columns, headings, widths, strict=True):
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor="center" if width < 100 else "w")
        for point in projection.points:
            tree.insert(
                "",
                "end",
                iid=str(point.round_index),
                values=(
                    "B0" if point.row_type == "B0" else point.round_index,
                    self._params_text(point.requested_params),
                    self._params_text(point.readback_params),
                    "—" if point.tdecq_db is None else point.tdecq_db,
                    "—" if point.outer_er_db is None else point.outer_er_db,
                    point.measurement_status.value,
                    "★" if point.round_index == projection.best_round_index else "",
                ),
            )
        tree.grid(row=2, column=0, sticky="nsew", padx=(0, 14))
        tree.bind("<<TreeviewSelect>>", self._select_result_point)
        self._result_tree = tree

        detail = ttk.Frame(frame, style="Panel.TFrame", padding=14)
        detail.grid(row=2, column=1, sticky="nsew")
        self._eye_label = ttk.Label(detail, text="Select a point", style="Panel.TLabel")
        self._eye_label.pack(fill="both", expand=True)
        self._kimi_text = tk.Text(
            detail,
            height=8,
            background="#081014",
            foreground="#c8d8d8",
            insertbackground="#66e0c2",
            relief="flat",
            wrap="word",
            name="kimi",
        )
        self._kimi_text.pack(fill="x", pady=(12, 0))
        self._kimi_text.configure(state="disabled")

        actions = ttk.Frame(frame, style="App.TFrame")
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ttk.Button(actions, text="打开证据文件夹 / Open folder", command=self._open_result_folder).pack(side="left")
        ttk.Button(
            actions,
            text="导出 CSV / Export CSV",
            style="Primary.TButton",
            command=self._export_csv,
        ).pack(side="right")
        frame.pack(fill="both", expand=True)
        self.current_frame = frame
        if tree.get_children():
            first = tree.get_children()[0]
            tree.selection_set(first)
            self._select_result_point()

    @staticmethod
    def _params_text(params: dict[str, int]) -> str:
        return "  ".join(f"{name}={value}" for name, value in params.items())

    def _select_result_point(self, _event: object | None = None) -> None:
        if self._projection is None:
            return
        selected = self._result_tree.selection()
        if not selected:
            return
        round_index = int(selected[0])
        point = next(
            item for item in self._projection.points if item.round_index == round_index
        )
        self._render_eye(point)
        self._kimi_text.configure(state="normal")
        self._kimi_text.delete("1.0", "end")
        if point.kimi_events:
            import json

            self._kimi_text.insert(
                "1.0", json.dumps(point.kimi_events, ensure_ascii=False, indent=2)
            )
        else:
            self._kimi_text.insert("1.0", "No Kimi proposal events for this point.")
        self._kimi_text.configure(state="disabled")

    def _render_eye(self, point: MeasuredPoint) -> None:
        if point.eye_path is None:
            self._eye_label.configure(text="No persisted eye evidence", image="")
            self._photo_image = None
            return
        try:
            from PIL import Image, ImageTk

            with Image.open(point.eye_path) as source:
                image = source.copy()
            image.thumbnail((360, 260))
            photo = ImageTk.PhotoImage(image)
        except Exception as error:
            self._eye_label.configure(text=f"Eye preview unavailable: {error}", image="")
            self._photo_image = None
            return
        self._photo_image = photo
        self._eye_label.configure(image=photo, text="")

    def _open_result_folder(self) -> None:
        if self._projection is not None:
            os.startfile(self._projection.campaign_root)  # type: ignore[attr-defined]

    def _export_csv(self) -> None:
        if self._projection is None:
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export campaign CSV",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"),),
            initialfile=f"{self._projection.campaign_id}.csv",
        )
        if selected:
            export_campaign_csv(self._projection, selected)

    def request_close(self) -> None:
        phase = self.session.phase
        if phase is DesktopPhase.RUNNING:
            self._closing = True
            self._safe_stop()
            return
        if phase is DesktopPhase.STOPPING:
            self._closing = True
            return
        if phase is DesktopPhase.PREFLIGHT:
            self._closing = True
            self.status_text.set("等待 PRE-FLIGHT 安全结束 / Waiting for PRE-FLIGHT")
            return
        if phase is DesktopPhase.AWAITING_APPROVAL:
            self._cancel_approval()
        self.root.destroy()
