import os
import serial
import serial.tools.list_ports
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


class RobotHandController:

  def __init__(self, root):
    self.root = root
    self.root.title("Robot Hand Controller - Withrobot Style")
    self.root.geometry("900x820")
    self.root.resizable(True, True)

    # 시리얼 관련
    self.ser = None
    self.is_connected = False
    self.is_sending = False
    self.last_send_time = 0
    self.min_send_interval = 0.15  # 최소 전송 간격 (초) - 과부하 방지

    # 프로파일 제어 플래그
    self.stop_profile_flag = False
    self.is_profile_running = False

    # 현재 목표 각도 (도)
    self.target_angles = {
        "thumb": 0,
        "index": 0,
        "middle": 0,
        "ring": 0,
        "little": 0,
    }

    # 캘리브레이션 데이터 (Data값 → 실제 각도) - 확장 데이터 반영
    self.calib = {
        "thumb": [  # (data, angle)
            (0x0A, 0),
            (0x14, 18),
            (0x19, 27),
            (0x1E, 35),
            (0x23, 44),
            (0x28, 53),
            (0x2D, 62),
            (0x32, 72),
            (0x3C, 94),
            (0x46, 111),
        ],
        "index": [
            (0x0A, 0),
            (0x28, 60),
            (0x3C, 96),
            (0x50, 134),
            (0x64, 173),
            (0x82, 242),
            (0x95, 263),
            (0x9C, 279),
            (0xA2, 288),
            (0xA8, 292),
        ],
        "middle": [
            (0x0A, 0),
            (0x28, 52),
            (0x3C, 91),
            (0x50, 131),
            (0x64, 171),
            (0x82, 231),
            (0x8C, 250),
            (0x96, 268),
            (0x9E, 285),
            (0xA6, 296),
        ],
        "ring": [
            (0x0A, 0),
            (0x28, 55),
            (0x3C, 95),
            (0x50, 130),
            (0x64, 174),
            (0x82, 231),
            (0x8C, 254),
            (0x95, 269),
            (0x9E, 284),
            (0xA4, 293),
        ],
        "little": [
            (0x0A, 0),
            (0x28, 60),
            (0x3C, 99),
            (0x50, 138),
            (0x64, 177),
            (0x82, 235),
            (0x8C, 253),
            (0x94, 267),
            (0x9C, 281),
            (0xA4, 291),
        ],
    }

    # 최대 가동 범위 확장 반영
    self.max_angles = {
        "thumb": 111,
        "index": 292,
        "middle": 296,
        "ring": 293,
        "little": 291,
    }

    self.create_ui()
    self.update_port_list()

    # 피드백 수신 스레드
    self.running = True
    self.feedback_thread = threading.Thread(
        target=self.read_feedback, daemon=True
    )
    self.feedback_thread.start()

  def create_ui(self):
    # ===== 1. 상단: 포트 설정 =====
    port_frame = ttk.LabelFrame(self.root, text="Port Config", padding=10)
    port_frame.pack(fill="x", padx=10, pady=5)

    ttk.Label(port_frame, text="Device:").grid(row=0, column=0, sticky="w")
    self.port_var = tk.StringVar()
    self.port_combo = ttk.Combobox(
        port_frame, textvariable=self.port_var, width=15, state="readonly"
    )
    self.port_combo.grid(row=0, column=1, padx=5)

    ttk.Button(
        port_frame, text="Refresh", command=self.update_port_list
    ).grid(row=0, column=2, padx=5)

    ttk.Label(port_frame, text="Baudrate:").grid(
        row=0, column=3, sticky="w", padx=(20, 0)
    )
    self.baud_var = tk.StringVar(value="115200")
    ttk.Combobox(
        port_frame,
        textvariable=self.baud_var,
        values=["115200"],
        width=10,
        state="readonly",
    ).grid(row=0, column=4, padx=5)

    self.connect_btn = ttk.Button(
        port_frame, text="Open Port", command=self.toggle_connection
    )
    self.connect_btn.grid(row=0, column=5, padx=10)

    self.status_label = ttk.Label(
        port_frame, text="● Disconnected", foreground="red"
    )
    self.status_label.grid(row=0, column=6, padx=10)

    # ===== 2. 중간: 슬라이더 제어 =====
    control_frame = ttk.LabelFrame(
        self.root, text="Finger Position Control (Absolute)", padding=10
    )
    control_frame.pack(fill="x", padx=10, pady=5)

    self.sliders = {}
    self.angle_labels = {}

    fingers = [
        ("thumb", "엄지 (Thumb)"),
        ("index", "검지 (Index)"),
        ("middle", "중지 (Middle)"),
        ("ring", "약지 (Ring)"),
        ("little", "새끼 (Little)"),
    ]

    for i, (key, name) in enumerate(fingers):
      ttk.Label(control_frame, text=name, width=15).grid(
          row=i, column=0, sticky="w", pady=3
      )

      slider = ttk.Scale(
          control_frame,
          from_=0,
          to=self.max_angles[key],
          orient="horizontal",
          length=400,
          command=lambda val, k=key: self.on_slider_change(k, val),
      )
      slider.grid(row=i, column=1, padx=10, pady=3)
      self.sliders[key] = slider

      label = ttk.Label(control_frame, text="0.0°", width=8)
      label.grid(row=i, column=2, pady=3)
      self.angle_labels[key] = label

      # 라벨 등록 후 초기값 설정 (KeyError 방지)
      slider.set(0)

    # 전송 버튼 & 옵션
    btn_frame = ttk.Frame(control_frame)
    btn_frame.grid(row=5, column=0, columnspan=3, pady=10)

    self.btn_send_all = ttk.Button(
        btn_frame, text="Send All Positions", command=self.send_all_positions
    )
    self.btn_send_all.pack(side="left", padx=5)

    self.btn_all_open = ttk.Button(
        btn_frame, text="All Open (0°)", command=self.all_open
    )
    self.btn_all_open.pack(side="left", padx=5)

    self.btn_reset_cnt = ttk.Button(
        btn_frame,
        text="Reset Position Counter",
        command=self.send_reset,
    )
    self.btn_reset_cnt.pack(side="left", padx=5)

    # 속도/전류 설정
    param_frame = ttk.Frame(control_frame)
    param_frame.grid(row=6, column=0, columnspan=3, pady=5)

    ttk.Label(param_frame, text="Speed:").pack(side="left")
    self.speed_var = tk.IntVar(value=0x64)
    ttk.Spinbox(
        param_frame, from_=10, to=200, textvariable=self.speed_var, width=6
    ).pack(side="left", padx=5)

    ttk.Label(param_frame, text="Current:").pack(side="left", padx=(15, 0))
    self.current_var = tk.IntVar(value=0xC8)
    ttk.Spinbox(
        param_frame, from_=50, to=250, textvariable=self.current_var, width=6
    ).pack(side="left", padx=5)

    ttk.Label(param_frame, text="(권장: Speed 100, Current 200)").pack(
        side="left", padx=10
    )

    # ===== 3. 프로파일 제어 프레임 =====
    prof_frame = ttk.LabelFrame(
        self.root, text="Profile Sequence Control", padding=10
    )
    prof_frame.pack(fill="x", padx=10, pady=5)

    self.entry_prof_path = ttk.Entry(prof_frame)
    self.entry_prof_path.pack(side="left", fill="x", expand=True, padx=(0, 5))

    self.btn_browse = ttk.Button(
        prof_frame, text="Browse", width=8, command=self._browse_profile
    )
    self.btn_browse.pack(side="left", padx=2)

    self.btn_prof_start = ttk.Button(
        prof_frame, text="Start Profile", command=self.on_start_profile
    )
    self.btn_prof_start.pack(side="left", padx=2)

    self.btn_prof_stop = ttk.Button(
        prof_frame,
        text="Stop Profile",
        command=self.on_stop_profile,
        state="disabled",
    )
    self.btn_prof_stop.pack(side="left", padx=2)

    # ===== 4. 하단: 로그 =====
    log_frame = ttk.LabelFrame(
        self.root, text="Feedback Log (Realtime)", padding=5
    )
    log_frame.pack(fill="both", expand=True, padx=10, pady=5)

    self.log_text = scrolledtext.ScrolledText(
        log_frame, height=12, state="disabled", font=("Consolas", 9)
    )
    self.log_text.pack(fill="both", expand=True)

    ttk.Button(log_frame, text="Clear Log", command=self.clear_log).pack(
        anchor="e", pady=3
    )

  def update_port_list(self):
    ports = [p.device for p in serial.tools.list_ports.comports()]
    self.port_combo["values"] = ports
    if ports:
      self.port_combo.current(0)

  def toggle_connection(self):
    if not self.is_connected:
      port = self.port_var.get()
      if not port:
        messagebox.showerror("Error", "COM 포트를 선택하세요.")
        return
      try:
        self.ser = serial.Serial(
            port=port,
            baudrate=int(self.baud_var.get()),
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.1,
        )
        self.is_connected = True
        self.connect_btn.config(text="Close Port")
        self.status_label.config(text="● Connected", foreground="green")
        self.log("포트 연결 성공: " + port)
      except Exception as e:
        messagebox.showerror("Connection Error", str(e))
    else:
      self.disconnect()

  def disconnect(self):
    self.is_connected = False
    if self.ser and self.ser.is_open:
      self.ser.close()
    self.connect_btn.config(text="Open Port")
    self.status_label.config(text="● Disconnected", foreground="red")
    self.log("포트 연결 해제")

  def on_slider_change(self, finger, value):
    angle = float(value)
    self.target_angles[finger] = angle
    if finger in self.angle_labels:
      self.angle_labels[finger].config(text=f"{angle:.1f}°")

  def angle_to_data(self, finger, target_angle):
    """캘리브레이션 데이터를 이용해 각도 → Data 값 선형 보간"""
    points = self.calib[finger]
    if target_angle <= points[0][1]:
      return points[0][0]
    if target_angle >= points[-1][1]:
      return points[-1][0]

    for i in range(len(points) - 1):
      d1, a1 = points[i]
      d2, a2 = points[i + 1]
      if a1 <= target_angle <= a2:
        ratio = (target_angle - a1) / (a2 - a1)
        data = d1 + ratio * (d2 - d1)
        return int(round(data))
    return points[-1][0]

  def build_packet(self, finger, data_value):
    """11바이트 패킷 생성"""
    finger_map = {
        "thumb": 0x01,
        "index": 0x02,
        "middle": 0x04,
        "ring": 0x08,
        "little": 0x10,
    }
    pos = [0x00] * 6
    idx = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "little": 4}[finger]
    pos[idx] = data_value

    packet = bytes([
        0xFE,  # Hand = Right
        finger_map[finger],  # Finger
        self.speed_var.get() & 0xFF,  # Speed
        self.current_var.get() & 0xFF,  # Current
        pos[0],
        pos[1],
        pos[2],
        pos[3],
        pos[4],
        pos[5],
        0x01,  # Direction = Forward
    ])
    return packet

  def safe_send(self, packet, description=""):
    """보호 로직이 적용된 전송"""
    if not self.is_connected or not self.ser or not self.ser.is_open:
      self.log("[ERROR] 포트가 연결되지 않았습니다.")
      return False

    now = time.time()
    if now - self.last_send_time < self.min_send_interval:
      self.log(
          f"[PROTECT] 전송 간격이 너무 짧습니다. ({self.min_send_interval}s 대기)"
      )
      return False

    if self.is_sending:
      self.log("[PROTECT] 이미 전송 중입니다. 무시됨.")
      return False

    try:
      self.is_sending = True
      self.ser.write(packet)
      self.last_send_time = now
      hex_str = " ".join(f"{b:02X}" for b in packet)
      self.log(f"[TX] {hex_str}  {description}")
      return True
    except Exception as e:
      self.log(f"[ERROR] 전송 실패: {e}")
      return False
    finally:
      self.is_sending = False

  def send_all_positions(self):
    """현재 슬라이더 값을 모두 전송"""
    if not self.is_connected:
      messagebox.showwarning("Warning", "먼저 포트를 연결하세요.")
      return

    fingers = ["thumb", "index", "middle", "ring", "little"]
    for finger in fingers:
      angle = self.target_angles[finger]
      angle = max(0, min(angle, self.max_angles[finger]))
      data = self.angle_to_data(finger, angle)
      packet = self.build_packet(finger, data)
      success = self.safe_send(
          packet, f"{finger} → {angle:.1f}° (data={data})"
      )
      if success:
        time.sleep(0.2)
      else:
        time.sleep(0.05)

  def all_open(self):
    """모든 손가락 0°로"""
    for key in self.sliders:
      self.sliders[key].set(0)
      self.target_angles[key] = 0
      self.angle_labels[key].config(text="0.0°")
    self.send_all_positions()

  def send_reset(self):
    """위치 카운터 리셋 (Direction=3)"""
    packet = bytes(
        [0xFE, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03]
    )
    self.safe_send(packet, "RESET Position Counter")

  def _browse_profile(self):
    path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if path:
      self.entry_prof_path.delete(0, tk.END)
      self.entry_prof_path.insert(0, path)

  def on_start_profile(self):
    path = self.entry_prof_path.get().strip()
    if not os.path.exists(path):
      messagebox.showerror("Error", "프로파일 파일 경로가 올바르지 않습니다.")
      return
    if not self.is_connected:
      messagebox.showwarning("Warning", "먼저 포트를 연결하세요.")
      return

    self.stop_profile_flag = False
    self.is_profile_running = True
    self._set_ui_lock_for_profile(True)

    threading.Thread(
        target=self._run_profile_process, args=(path,), daemon=True
    ).start()

  def on_stop_profile(self):
    self.stop_profile_flag = True

  def _set_ui_lock_for_profile(self, locked):
    state = "disabled" if locked else "normal"
    for s in self.sliders.values():
      s.config(state=state)
    self.btn_send_all.config(state=state)
    self.btn_all_open.config(state=state)
    self.btn_reset_cnt.config(state=state)
    self.btn_browse.config(state=state)
    self.btn_prof_start.config(state=state)
    self.btn_prof_stop.config(state="normal" if locked else "disabled")

  def _run_profile_process(self, filepath):
    try:
      self.log("[PROFILE] 프로파일 시작: 0도 초기화")

      self.all_open()
      time.sleep(1.2)

      if self.stop_profile_flag:
        self.log("[PROFILE] 프로파일 정지됨.")
        return

      time_events = {}
      with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if not line or line.startswith("#"):
            continue
          parts = [p.strip() for p in line.split(",")]
          if len(parts) == 3:
            t_sec = float(parts[0])
            finger_name = parts[1].lower()
            angle = float(parts[2])

            if finger_name in self.max_angles:
              if t_sec not in time_events:
                time_events[t_sec] = []
              time_events[t_sec].append((finger_name, angle))

      sorted_times = sorted(time_events.keys())
      start_time = time.time()

      for t_target in sorted_times:
        if self.stop_profile_flag:
          break

        while (time.time() - start_time) < t_target:
          if self.stop_profile_flag:
            break
          time.sleep(0.01)

        if self.stop_profile_flag:
          break

        for finger_name, angle in time_events[t_target]:
          if self.stop_profile_flag:
            break

          angle_clamped = max(0, min(angle, self.max_angles[finger_name]))
          data_val = self.angle_to_data(finger_name, angle_clamped)

          def _update_ui(f=finger_name, a=angle_clamped):
            self.sliders[f].set(a)
            self.target_angles[f] = a
            self.angle_labels[f].config(text=f"{a:.1f}°")

          self.root.after(0, _update_ui)

          packet = self.build_packet(finger_name, data_val)
          self.safe_send(
              packet, f"[PROFILE {t_target}s] {finger_name} → {angle_clamped}°"
          )

          time.sleep(0.2)

      if self.stop_profile_flag:
        self.log("[PROFILE] 사용자에 의해 프로파일이 중지되었습니다.")
      else:
        self.log("[PROFILE] 프로파일 재생 완료.")

    except Exception as e:
      self.log(f"[ERROR] 프로파일 실행 중 오류: {e}")
      messagebox.showerror("Error", f"프로파일 실행 중 오류 발생: {e}")
    finally:
      self.is_profile_running = False
      self.root.after(0, lambda: self._set_ui_lock_for_profile(False))

  def read_feedback(self):
    """백그라운드에서 피드백 수신"""
    buffer = ""
    while self.running:
      if self.is_connected and self.ser and self.ser.is_open:
        try:
          data = self.ser.read(self.ser.in_waiting or 1)
          if data:
            buffer += data.decode("utf-8", errors="ignore")
            while "\n" in buffer or "," in buffer:
              if "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
              else:
                parts = buffer.split(",")
                if len(parts) >= 19:
                  line = ",".join(parts[:19])
                  buffer = ",".join(parts[19:])
                else:
                  break
              line = line.strip()
              if line:
                self.log(f"[RX] {line}")
        except Exception:
          pass
      time.sleep(0.02)

  def log(self, msg):
    def _log():
      self.log_text.config(state="normal")
      self.log_text.insert("end", f"{time.strftime('%H:%M:%S')} {msg}\n")
      self.log_text.see("end")
      self.log_text.config(state="disabled")

    self.root.after(0, _log)

  def clear_log(self):
    self.log_text.config(state="normal")
    self.log_text.delete("1.0", "end")
    self.log_text.config(state="disabled")

  def on_closing(self):
    self.running = False
    self.stop_profile_flag = True
    self.disconnect()
    self.root.destroy()


if __name__ == "__main__":
  root = tk.Tk()
  app = RobotHandController(root)
  root.protocol("WM_DELETE_WINDOW", app.on_closing)
  root.mainloop()