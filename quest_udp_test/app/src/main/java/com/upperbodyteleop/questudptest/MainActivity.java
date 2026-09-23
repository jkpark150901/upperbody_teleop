package com.upperbodyteleop.questudptest;

// Minimal standalone test: no OpenXR, no VR session, no hand tracking --
// just "can this device send UDP to the configured host". Runs as a plain
// 2D Android panel on Quest, sidestepping the OpenXR session/Guardian/focus
// issues XrHandsFB has been running into (see that sample's main.cpp for
// the full story). Sends the SAME wire format
// (devices/quest_hand/quest_hand_reader.py in the upperbody_teleop repo),
// with active=0 and all-zero joints, so scripts/quest_hand_monitor.py
// already recognizes it (as "SENDING, NOT TRACKED") with no receiver
// changes needed.
//
// Host/port: read once from /sdcard/quest_hand.cfg (same file/format
// XrHandsFB's ResolveHandUdpTarget uses -- "<ip> <port>", one line),
// falling back to the compiled-in default below if that's missing.

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.FileReader;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;

public class MainActivity extends Activity {
    private static final String DEFAULT_HOST = "192.168.8.156";
    private static final int DEFAULT_PORT = 5005;
    private static final String CONFIG_PATH = "/sdcard/quest_hand.cfg";

    // Wire format v1 -- see devices/quest_hand/quest_hand_reader.py's
    // module docstring for the authoritative spec. header=24B, 26 joints x
    // (3 pos + 4 quat) float32 = 728B, total 752B.
    private static final int MAGIC = 0x31485258; // "XRH1" little-endian
    private static final int N_JOINTS = 26;
    private static final int PACKET_SIZE = 24 + N_JOINTS * 28;

    private final AtomicInteger sentCount = new AtomicInteger(0);
    private final AtomicInteger errorCount = new AtomicInteger(0);
    private volatile boolean running = true;
    private volatile String lastError = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        TextView statusView = new TextView(this);
        statusView.setTextSize(20);
        statusView.setPadding(48, 48, 48, 48);
        setContentView(statusView);

        String[] target = resolveTarget();
        final String host = target[0];
        final int port = Integer.parseInt(target[1]);

        Thread sender = new Thread(() -> runSenderLoop(host, port), "udp-heartbeat");
        sender.setDaemon(true);
        sender.start();

        Handler handler = new Handler(Looper.getMainLooper());
        Runnable updateUi = new Runnable() {
            @Override
            public void run() {
                String text = "UDP test -> " + host + ":" + port
                        + "\nsent=" + sentCount.get()
                        + "  errors=" + errorCount.get();
                if (!lastError.isEmpty()) {
                    text += "\nlast error: " + lastError;
                }
                statusView.setText(text);
                handler.postDelayed(this, 250);
            }
        };
        handler.post(updateUi);
    }

    /** {host, portString}; falls back to the built-in default on any problem
     * reading/parsing CONFIG_PATH (missing file, bad format, etc). */
    private String[] resolveTarget() {
        try (BufferedReader r = new BufferedReader(new FileReader(CONFIG_PATH))) {
            String line = r.readLine();
            if (line != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length == 2) {
                    Integer.parseInt(parts[1]); // validate before returning
                    return parts;
                }
            }
        } catch (Exception e) {
            // fall through to default
        }
        return new String[]{DEFAULT_HOST, String.valueOf(DEFAULT_PORT)};
    }

    private void runSenderLoop(String host, int port) {
        DatagramSocket socket = null;
        try {
            socket = new DatagramSocket();
            InetAddress addr = InetAddress.getByName(host);
            int seqLeft = 0;
            int seqRight = 0;
            while (running) {
                send(socket, addr, port, /*hand=*/(byte) 0, seqLeft++);
                send(socket, addr, port, /*hand=*/(byte) 1, seqRight++);
                Thread.sleep(250);
            }
        } catch (Exception e) {
            lastError = e.toString();
        } finally {
            if (socket != null) {
                socket.close();
            }
        }
    }

    private void send(DatagramSocket socket, InetAddress addr, int port, byte hand, int seq) {
        ByteBuffer buf = ByteBuffer.allocate(PACKET_SIZE).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(MAGIC);
        buf.put((byte) 1); // version
        buf.put(hand); // 0=left, 1=right
        buf.put((byte) 0); // active = false (this is a connectivity test, not real tracking)
        buf.put((byte) 0); // reserved
        buf.putInt(seq);
        buf.putDouble(seq * 0.25); // t -- diagnostic only, not wall-clock
        buf.putInt(0); // tracked_mask = 0 (nothing tracked)
        for (int i = 0; i < N_JOINTS * 7; i++) {
            buf.putFloat(0.0f);
        }
        try {
            socket.send(new DatagramPacket(buf.array(), buf.array().length, addr, port));
            sentCount.incrementAndGet();
        } catch (Exception e) {
            errorCount.incrementAndGet();
            lastError = e.toString();
        }
    }

    @Override
    protected void onDestroy() {
        running = false;
        super.onDestroy();
    }
}
