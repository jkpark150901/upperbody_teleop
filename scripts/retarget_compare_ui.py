"""Small launcher UI for comparing a Quest recording across hand retargeters."""
from __future__ import annotations

import pathlib
import socket
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_ANYDEX_DATA = _ROOT / "third_party" / "AnyDexRetarget" / "example" / "data"

# scripts/run_anydex_hand_recording.py's --control-port: a fixed local port
# is fine since only one compare run is normally active at a time.
_CONTROL_PORT = 6201


def _send_control(cmd: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(cmd.encode("utf-8"), ("127.0.0.1", _CONTROL_PORT))
    finally:
        sock.close()

_ANYDEX_ROBOTS = {
    "DG-5F current pipeline": None,
    "Inspire Hand": "inspire_hand",
    "Gaia Hand20": "gaia_hand20",
    "Wuji Hand": "wuji_hand",
    "Unitree Dex5": "unitree_dex5_hand",
    "Sharpa Hand": "sharpa_hand",
    "Shadow Hand": "shadow_hand",
    "Linker L20": "linker_l20",
}

_SHORT = {
    "inspire_hand": "inspire",
    "gaia_hand20": "gaia",
    "wuji_hand": "wuji",
    "unitree_dex5_hand": "unitree_dex5",
    "sharpa_hand": "sharpa",
    "shadow_hand": "shadow",
    "linker_l20": "linker_l20",
}

# Mingrui-Yu/retargeting, cloned at <repo root>/retargeting. A separate
# codebase from AnyDexRetarget: own CLI (retargeting_apps.main, hydra
# configs), own conda env (not yet created on this machine -- see its
# README's `conda create -n retargeting ...`), and as of this clone its
# third_party/utils_python submodule is NOT initialized
# (`git submodule update --init --recursive` needed before anything in it
# will import). It also only ships configs/robots for 2 hands -- no DG5F --
# so there is no "retargeted onto DG5F" option for this engine yet.
_RETARGETING_ROOT = _ROOT / "retargeting"
_RETARGETING_CONDA_ENV = "retargeting"
_RETARGETING_ROBOTS = {
    "Panda + Leap (Paxini)": "leap_paxini",
    "Panda + Shadow": "shadow",
}

_ENGINES = ("AnyDexRetarget", "retargeting (Mingrui-Yu)")


_LAUNCHED_TITLES: list[str] = []


def _stop_window(title: str) -> None:
    # Mirrors stop_all.bat's pattern: /T also kills each cmd window's child
    # python.exe (the actual compare process), not just the shell itself.
    subprocess.run(
        ["taskkill", "/FI", f"WINDOWTITLE eq {title}*", "/T", "/F"],
        capture_output=True,
    )


def _new_console(cmd: str, title: str, cwd: pathlib.Path = _ROOT) -> None:
    # Writing the launch command to a temp .bat and running THAT (instead of
    # handing cmd.exe a `/k "<command string with its own nested quotes>"`
    # via subprocess.Popen's argv list) sidesteps a real quoting mismatch:
    # Python's list-to-commandline escaping of embedded `"` doesn't survive
    # cmd.exe's own re-parsing of /k's argument, which silently mangles the
    # quoted `cd /d "..."` path and surfaces as "파일 이름, 디렉터리 이름
    # 또는 볼륨 레이블 구문이 잘못되었습니다." (confirmed the hard way).
    import tempfile

    fd, bat_path = tempfile.mkstemp(suffix=".bat", prefix="retarget_compare_")
    with open(fd, "w", encoding="mbcs") as f:
        f.write("@echo off\r\n")
        f.write(f"title {title}\r\n")
        f.write(f'cd /d "{cwd}"\r\n')
        f.write(f"{cmd}\r\n")
    kwargs = {}
    if hasattr(subprocess, "CREATE_NEW_CONSOLE"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(["cmd.exe", "/d", "/k", bat_path], cwd=str(cwd), **kwargs)
    if title not in _LAUNCHED_TITLES:
        _LAUNCHED_TITLES.append(title)


def _recording_options() -> list[str]:
    rec_dir = _ROOT / "recordings"
    if not rec_dir.exists():
        return []
    return [str(p.relative_to(_ROOT)) for p in sorted(rec_dir.glob("*.jsonl"))]


def _sanitize(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep).strip("_") or "recording"


class CompareUi(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Retarget Compare")
        self.geometry("760x400")

        recordings = _recording_options()
        default_recording = "recordings/quest_hand_latest.jsonl"
        if recordings and default_recording not in recordings:
            default_recording = recordings[0]

        self.recording_var = tk.StringVar(value=default_recording)
        self.engine_var = tk.StringVar(value=_ENGINES[0])
        self.robot_var = tk.StringVar(value="Inspire Hand")
        self.hand_var = tk.StringVar(value="right")
        self.optimizer_var = tk.StringVar(value="adaptive")
        self.start_var = tk.StringVar(value="0")
        self.duration_var = tk.StringVar(value="10")
        self.speed_var = tk.StringVar(value="1.0")
        self.loop_var = tk.BooleanVar(value=True)
        self.dg5f_source_var = tk.StringVar(value="retarget")
        self.pkl_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        pad = {"padx": 8, "pady": 5}
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Label(root, text="Recording JSONL").grid(row=0, column=0, sticky="w", **pad)
        self.recording_combo = ttk.Combobox(root, textvariable=self.recording_var, values=recordings, width=64)
        self.recording_combo.grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(root, text="Browse", command=self.browse_recording).grid(row=0, column=2, sticky="ew", **pad)

        ttk.Label(root, text="Engine").grid(row=1, column=0, sticky="w", **pad)
        self.engine_combo = ttk.Combobox(root, textvariable=self.engine_var, values=list(_ENGINES), state="readonly", width=30)
        self.engine_combo.grid(row=1, column=1, sticky="w", **pad)
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_changed)

        ttk.Label(root, text="Robot").grid(row=2, column=0, sticky="w", **pad)
        self.robot_combo = ttk.Combobox(root, textvariable=self.robot_var, values=list(_ANYDEX_ROBOTS), state="readonly", width=30)
        self.robot_combo.grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(root, text="Hand").grid(row=2, column=1, sticky="e", **pad)
        ttk.Combobox(root, textvariable=self.hand_var, values=["right", "left"], state="readonly", width=8).grid(
            row=2, column=2, sticky="w", **pad
        )

        ttk.Label(root, text="Optimizer").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(root, textvariable=self.optimizer_var, values=["adaptive", "vector"], state="readonly", width=12).grid(
            row=3, column=1, sticky="w", **pad
        )
        ttk.Label(root, text="DG-5F source").grid(row=3, column=1, sticky="e", **pad)
        ttk.Combobox(
            root,
            textvariable=self.dg5f_source_var,
            values=["retarget", "retargeted"],
            state="readonly",
            width=12,
        ).grid(row=3, column=2, sticky="w", **pad)

        time_row = ttk.Frame(root)
        time_row.grid(row=4, column=1, columnspan=2, sticky="w", **pad)
        ttk.Label(root, text="Window").grid(row=4, column=0, sticky="w", **pad)
        ttk.Label(time_row, text="start").pack(side="left")
        ttk.Entry(time_row, textvariable=self.start_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(time_row, text="duration").pack(side="left")
        ttk.Entry(time_row, textvariable=self.duration_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(time_row, text="speed").pack(side="left")
        ttk.Entry(time_row, textvariable=self.speed_var, width=8).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(time_row, text="loop", variable=self.loop_var).pack(side="left")

        ttk.Label(root, text="AnyDex PKL").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(root, textvariable=self.pkl_var, width=64).grid(row=5, column=1, sticky="ew", **pad)
        ttk.Button(root, text="Prepare PKL", command=self.prepare_pkl).grid(row=5, column=2, sticky="ew", **pad)

        actions = ttk.Frame(root)
        actions.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(15, 4))
        ttk.Button(actions, text="Run Selected Robot", command=self.run_selected).pack(side="left", padx=8)
        ttk.Button(actions, text="Run DG-5F", command=self.run_dg5f).pack(side="left", padx=8)
        ttk.Button(actions, text="Open Data Folder", command=self.open_data_folder).pack(side="left", padx=8)
        ttk.Button(actions, text="Stop All", command=self.stop_all).pack(side="left", padx=8)
        ttk.Button(actions, text="Pause/Resume", command=self.toggle_pause).pack(side="left", padx=8)

        ttk.Label(root, textvariable=self.status_var).grid(row=7, column=0, columnspan=3, sticky="w", padx=8, pady=12)
        root.columnconfigure(1, weight=1)

        self.refresh_output_path()
        for var in (self.recording_var, self.start_var, self.duration_var):
            var.trace_add("write", lambda *_: self.refresh_output_path())

    def _on_engine_changed(self, _event=None) -> None:
        if self.engine_var.get() == "AnyDexRetarget":
            self.robot_combo.configure(values=list(_ANYDEX_ROBOTS))
            if self.robot_var.get() not in _ANYDEX_ROBOTS:
                self.robot_var.set(next(iter(_ANYDEX_ROBOTS)))
        else:
            self.robot_combo.configure(values=list(_RETARGETING_ROBOTS))
            if self.robot_var.get() not in _RETARGETING_ROBOTS:
                self.robot_var.set(next(iter(_RETARGETING_ROBOTS)))

    def browse_recording(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(_ROOT / "recordings"),
            title="Select hand_process JSONL recording",
            filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")],
        )
        if path:
            p = pathlib.Path(path)
            try:
                self.recording_var.set(str(p.relative_to(_ROOT)))
            except ValueError:
                self.recording_var.set(str(p))

    def _recording_path(self) -> pathlib.Path:
        p = pathlib.Path(self.recording_var.get().strip())
        return p if p.is_absolute() else _ROOT / p

    def _duration_arg(self) -> list[str]:
        value = self.duration_var.get().strip()
        if not value:
            return []
        return ["--duration", value]

    def refresh_output_path(self) -> None:
        rec = pathlib.Path(self.recording_var.get().strip() or "recording").stem
        start = _sanitize(self.start_var.get().strip() or "0")
        dur = _sanitize(self.duration_var.get().strip() or "all")
        out = _ANYDEX_DATA / f"{_sanitize(rec)}_s{start}_d{dur}_anydex.pkl"
        try:
            self.pkl_var.set(str(out.relative_to(_ROOT)))
        except ValueError:
            self.pkl_var.set(str(out))

    def prepare_pkl(self) -> pathlib.Path:
        rec = self._recording_path()
        if not rec.exists():
            raise FileNotFoundError(rec)
        out = pathlib.Path(self.pkl_var.get().strip())
        out = out if out.is_absolute() else _ROOT / out
        cmd = [
            sys.executable,
            "-m",
            "scripts.convert_quest_recording_to_anydex_replay",
            "--input",
            str(rec),
            "--output",
            str(out),
            "--start-time",
            self.start_var.get().strip() or "0",
        ] + self._duration_arg()
        self.status_var.set("Preparing AnyDex PKL...")
        self.update_idletasks()
        proc = subprocess.run(cmd, cwd=str(_ROOT), text=True, capture_output=True)
        if proc.returncode != 0:
            self.status_var.set("PKL conversion failed")
            messagebox.showerror("PKL conversion failed", proc.stderr or proc.stdout)
            raise RuntimeError(proc.stderr or proc.stdout)
        self.status_var.set((proc.stdout.strip().splitlines() or ["PKL ready"])[-1])
        return out

    def _run_anydex_robot(self, robot_type: str) -> None:
        pkl = self.prepare_pkl()
        robot_short = _SHORT[robot_type]
        config = f"config/{self.optimizer_var.get()}/quest3/quest3_{robot_type}.yaml"
        play_rel = pkl
        try:
            play_arg = str(pkl.relative_to(_ROOT / "third_party" / "AnyDexRetarget" / "example"))
        except ValueError:
            play_arg = str(play_rel)
        loop = " --loop" if self.loop_var.get() else ""
        cmd = (
            f'"{sys.executable}" -m scripts.run_anydex_hand_recording '
            f'--config {config} --hand {self.hand_var.get()} --play "{play_arg}" '
            f'--speed {self.speed_var.get().strip() or "1.0"} --control-port {_CONTROL_PORT}{loop}'
        )
        _new_console(cmd, f"AnyDex {robot_short} {self.hand_var.get()}")
        self.status_var.set(f"Started {robot_short} with {play_arg}")

    def run_selected(self) -> None:
        if self.engine_var.get() != "AnyDexRetarget":
            robot_suffix = _RETARGETING_ROBOTS[self.robot_var.get()]
            self.run_retargeting_repo(robot_suffix)
            return
        robot_type = _ANYDEX_ROBOTS[self.robot_var.get()]
        if robot_type is None:
            self.run_dg5f()
            return
        try:
            self._run_anydex_robot(robot_type)
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))

    def run_retargeting_repo(self, robot_suffix: str) -> None:
        # Mingrui-Yu/retargeting has its own CLI/config system, not
        # AnyDexRetarget's -- robot choice is a hydra override selecting one
        # of configs/retargeting_profiles/vector_wrist_joint_panda_*.yaml.
        # There is no Quest-recording converter for this engine's input
        # schema (AVP .npz) yet, so this runs the repo's own bundled demo
        # trajectory (offline_retarget's default `data:` fixture) rather
        # than the Recording JSONL picked above -- good enough to confirm
        # the env/submodule are set up and to look at retargeting quality,
        # not yet a real side-by-side with the same Quest capture.
        if not _RETARGETING_ROOT.exists():
            messagebox.showerror("Missing repo", str(_RETARGETING_ROOT))
            return
        profile = f"vector_wrist_joint_panda_{robot_suffix}"
        run_name = f"compare_ui_{robot_suffix}"
        cmd = (
            f'call "C:\\Users\\admin\\miniforge3\\Scripts\\activate.bat" {_RETARGETING_CONDA_ENV} '
            f'&& python -m retargeting_apps.main app=offline_retarget '
            f'retargeting_profiles={profile} run_name={run_name} post.visualize.enabled=true'
        )
        _new_console(cmd, f"retargeting {robot_suffix}", cwd=_RETARGETING_ROOT)
        self.status_var.set(
            f"Started retargeting repo ({robot_suffix}) -- needs conda env "
            f"'{_RETARGETING_CONDA_ENV}' set up per its README; uses its bundled demo data, not the Recording JSONL above"
        )

    def run_dg5f(self) -> None:
        rec = self._recording_path()
        if not rec.exists():
            messagebox.showerror("Missing recording", str(rec))
            return
        port = "6100"
        physics_cmd = (
            f'"{sys.executable}" -m scripts.mujoco_physics_process '
            f'--port {port} --diagnose-latency --sim-config configs/sim_params.yaml'
        )
        duration = " ".join(self._duration_arg())
        source = self.dg5f_source_var.get()
        # --source retarget actually re-runs retargeting live (vs. "retargeted"
        # replaying already-retargeted joints from the recording), so only
        # then does --hand-retargeter/--anydex-mode matter -- pass them so the
        # UI's Optimizer dropdown selects DG5F's own yaml (configs/anydex/
        # dg5f_{side}_{mode}_quest3.yaml, via sim_params.yaml's
        # hand_retargeting.anydex_configs) instead of silently falling back to
        # sim_params.yaml's default "analytic" backend.
        retargeter_args = " --hand-retargeter anydex --anydex-mode " + self.optimizer_var.get() if source == "retarget" else ""
        sender_cmd = (
            f'"{sys.executable}" -m scripts.replay_quest_hand_to_mujoco '
            f'--file "{rec}" --send udp://127.0.0.1:{port} --sim-config configs/sim_params.yaml '
            f'--source {source}{retargeter_args} --start-time {self.start_var.get().strip() or "0"} '
            f'{duration} --speed {self.speed_var.get().strip() or "1.0"}'
            + (" --loop" if self.loop_var.get() else "")
        )
        _new_console(physics_cmd, "DG-5F physics compare")
        _new_console(sender_cmd, "DG-5F replay sender")
        self.status_var.set("Started DG-5F physics and replay sender")

    def open_data_folder(self) -> None:
        _ANYDEX_DATA.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer.exe", str(_ANYDEX_DATA)])

    def stop_all(self) -> None:
        if not _LAUNCHED_TITLES:
            self.status_var.set("Nothing launched yet")
            return
        for title in _LAUNCHED_TITLES:
            _stop_window(title)
        self.status_var.set(f"Stopped {len(_LAUNCHED_TITLES)} window(s)")
        _LAUNCHED_TITLES.clear()

    def toggle_pause(self) -> None:
        # Fire-and-forget UDP "toggle" to run_anydex_hand_recording.py's
        # --control-port listener; only meaningful for AnyDex robot runs
        # (DG-5F's mujoco_physics_process doesn't listen on this port).
        _send_control("toggle")
        self.status_var.set("Sent pause/resume toggle")


def main() -> None:
    app = CompareUi()
    app.mainloop()


if __name__ == "__main__":
    main()
